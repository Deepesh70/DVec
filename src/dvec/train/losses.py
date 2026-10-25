"""Objective Loss Functions for Target Speaker Extraction.

Features:
- Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) Loss
- Asymmetric Complex Spectral Loss (heavier penalty for under-estimating target speech formants)
- Personal Voice Activity Detection (pVAD) Binary Cross-Entropy Loss
- Composite Multi-Task Training Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SNRLoss(nn.Module):
    """Scale-Dependent Signal-to-Noise Ratio (SNR) loss.
    
    Fixes alpha = 1.0, penalizing volume scaling errors and enforcing exact unity gain
    relative to the physical microphone input.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, est_targets: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            est_targets: (batch, time_samples) or (batch, 1, time_samples)
            targets: (batch, time_samples) or (batch, 1, time_samples)
        Returns:
            Scalar loss: -SNR (dB) averaged across batch
        """
        if est_targets.dim() == 3:
            est_targets = est_targets.squeeze(1)
        if targets.dim() == 3:
            targets = targets.squeeze(1)

        min_len = min(est_targets.shape[-1], targets.shape[-1])
        est_targets = est_targets[..., :min_len]
        targets = targets[..., :min_len]

        # Zero-mean normalization
        est_targets = est_targets - torch.mean(est_targets, dim=-1, keepdim=True)
        targets = targets - torch.mean(targets, dim=-1, keepdim=True)

        # Direct error with alpha = 1.0 (enforces physical volume preservation)
        noise = est_targets - targets

        signal_power = torch.sum(targets ** 2, dim=-1) + self.eps
        noise_power = torch.sum(noise ** 2, dim=-1) + self.eps

        snr = 10.0 * torch.log10(signal_power / noise_power)
        return -torch.mean(snr)


class SISDRLoss(nn.Module):
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss.
    
    Higher SI-SDR means better separation. We minimize -SI-SDR.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, est_targets: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            est_targets: (batch, time_samples) or (batch, 1, time_samples)
            targets: (batch, time_samples) or (batch, 1, time_samples)
        Returns:
            Scalar loss: -SI-SDR (dB) averaged across batch
        """
        if est_targets.dim() == 3:
            est_targets = est_targets.squeeze(1)
        if targets.dim() == 3:
            targets = targets.squeeze(1)

        # Truncate to matching length if needed
        min_len = min(est_targets.shape[-1], targets.shape[-1])
        est_targets = est_targets[..., :min_len]
        targets = targets[..., :min_len]

        # Zero-mean normalization
        est_targets = est_targets - torch.mean(est_targets, dim=-1, keepdim=True)
        targets = targets - torch.mean(targets, dim=-1, keepdim=True)

        # Scale projection factor: alpha = <est, target> / ||target||^2
        dot = torch.sum(est_targets * targets, dim=-1, keepdim=True)
        target_energy = torch.sum(targets ** 2, dim=-1, keepdim=True) + self.eps
        alpha = dot / target_energy

        s_target = alpha * targets
        e_noise = est_targets - s_target

        s_target_pow = torch.sum(s_target ** 2, dim=-1) + self.eps
        e_noise_pow = torch.sum(e_noise ** 2, dim=-1) + self.eps

        si_sdr = 10.0 * torch.log10(s_target_pow / e_noise_pow)
        return -torch.mean(si_sdr)


class AsymmetricComplexSpectralLoss(nn.Module):
    """Complex Spectral Loss with asymmetric penalty on magnitude under-estimation.
    
    Under-estimating target speech spectral energy creates unnatural muffled / choppy
    speech artifacts, which is penalized more heavily than minor noise leakage.
    """

    def __init__(self, under_est_penalty: float = 1.5, eps: float = 1e-8):
        super().__init__()
        self.under_est_penalty = under_est_penalty
        self.eps = eps

    def forward(
        self,
        est_r: torch.Tensor,
        est_i: torch.Tensor,
        target_r: torch.Tensor,
        target_i: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            est_r, est_i: Estimated real and imaginary STFT planes (B, 1, F, T)
            target_r, target_i: Target real and imaginary STFT planes (B, 1, F, T)
        """
        est_mag = torch.sqrt(est_r ** 2 + est_i ** 2 + self.eps)
        target_mag = torch.sqrt(target_r ** 2 + target_i ** 2 + self.eps)

        # Magnitude error
        mag_diff = est_mag - target_mag

        # Asymmetric weighting: heavier penalty if est_mag < target_mag
        weights = torch.where(mag_diff < 0.0, self.under_est_penalty, 1.0)
        mag_loss = torch.mean(weights * torch.abs(mag_diff))

        # Real & Imaginary L1 loss for phase fidelity
        ri_loss = torch.mean(torch.abs(est_r - target_r) + torch.abs(est_i - target_i))

        return mag_loss + ri_loss


class CompositeTSELoss(nn.Module):
    """Composite multi-task loss combining Scale-Dependent SNR, Complex Spectral, and pVAD losses."""

    def __init__(
        self,
        lambda_snr: float = 1.0,
        lambda_spec: float = 15.0,
        lambda_vad: float = 0.5,
    ):
        super().__init__()
        self.lambda_snr = lambda_snr
        self.lambda_spec = lambda_spec
        self.lambda_vad = lambda_vad

        self.snr_loss = SNRLoss()
        self.spec_loss = AsymmetricComplexSpectralLoss(under_est_penalty=1.5)
        self.bce_loss = nn.BCELoss()

    def forward(
        self,
        est_wav: torch.Tensor,
        target_wav: torch.Tensor,
        est_r: torch.Tensor,
        est_i: torch.Tensor,
        target_r: torch.Tensor,
        target_i: torch.Tensor,
        pred_vad: torch.Tensor,
        target_vad: torch.Tensor,
    ) -> dict:
        l_snr = self.snr_loss(est_wav, target_wav)
        l_spec = self.spec_loss(est_r, est_i, target_r, target_i)
        l_vad = self.bce_loss(pred_vad, target_vad)

        total_loss = (
            self.lambda_snr * l_snr
            + self.lambda_spec * l_spec
            + self.lambda_vad * l_vad
        )

        return {
            "total_loss": total_loss,
            "snr_loss": l_snr,
            "spec_loss": l_spec,
            "vad_loss": l_vad,
            "snr_db": -l_snr.item(),
        }
