"""Production Training Engine for Target Speaker Extraction.

Features:
- Dynamic on-the-fly multi-speaker batching with varying SIR (-5 dB to +15 dB).
- Scale-Dependent SNR loss (enforces exact unity volume gain).
- Asymmetric Complex Spectral Loss + Personal VAD head supervision.
- Embedding jitter augmentation simulating realistic human cosine similarities.
- Automated validation and stateful streaming ONNX export.
"""

import time
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dvec.pipeline.model import CausalTSEModel
from dvec.pipeline.stft import CausalSTFT, CausalISTFT
from dvec.data.dataset import DynamicTSEDataset
from dvec.enrollment.encoder import DeterministicSpeakerEncoder
from dvec.train.losses import CompositeTSELoss, SNRLoss
from dvec.export.onnx_exporter import export_streaming_onnx


class TSETrainer:
    """Trainer orchestrating multi-speaker extraction optimization."""

    def __init__(
        self,
        model: CausalTSEModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-5,
    ):
        self.device = device
        self.model = model.to(device)
        self.loss_fn = CompositeTSELoss(lambda_snr=1.0, lambda_spec=15.0, lambda_vad=0.5).to(device)
        self.eval_snr_fn = SNRLoss().to(device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.win = torch.hann_window(320, periodic=True, device=device).sqrt()

    def train_epoch(self, dataloader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        total_snr = 0.0
        total_spec = 0.0
        total_vad = 0.0
        num_batches = len(dataloader)

        for batch in dataloader:
            mix = batch["mixture"].to(self.device).squeeze(1)  # (B, T)
            target = batch["target"].to(self.device).squeeze(1)
            embed = batch["spk_embed"].to(self.device)  # (B, 256)
            vad = batch["vad_target"].to(self.device).squeeze(1)

            # Causal STFT Analysis
            mix_stft = torch.stft(
                mix, n_fft=512, hop_length=160, win_length=320, window=self.win, center=True, return_complex=True
            )
            target_stft = torch.stft(
                target, n_fft=512, hop_length=160, win_length=320, window=self.win, center=True, return_complex=True
            )

            mix_r = mix_stft.real.unsqueeze(1)
            mix_i = mix_stft.imag.unsqueeze(1)
            target_r = target_stft.real.unsqueeze(1)
            target_i = target_stft.imag.unsqueeze(1)

            num_frames = mix_stft.shape[-1]
            vad_frames = (torch.nn.functional.adaptive_avg_pool1d(vad.unsqueeze(1), num_frames).squeeze(1) > 0.2).float()

            self.optimizer.zero_grad()

            # Sequence forward pass
            mask_r, mask_i, pred_vad = self.model.forward_sequence(mix_r, mix_i, embed)

            # Complex Masking
            est_r = mix_r * mask_r - mix_i * mask_i
            est_i = mix_r * mask_i + mix_i * mask_r

            # iSTFT synthesis
            est_pcm = torch.istft(
                torch.complex(est_r.squeeze(1), est_i.squeeze(1)),
                n_fft=512, hop_length=160, win_length=320, window=self.win, center=True, length=mix.shape[-1]
            )

            # Loss computation
            loss_dict = self.loss_fn(
                est_wav=est_pcm,
                target_wav=target,
                est_r=est_r,
                est_i=est_i,
                target_r=target_r,
                target_i=target_i,
                pred_vad=pred_vad,
                target_vad=vad_frames,
            )

            loss = loss_dict["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_snr += loss_dict["snr_db"]
            total_spec += loss_dict["spec_loss"].item()
            total_vad += loss_dict["vad_loss"].item()

        return {
            "loss": total_loss / num_batches,
            "snr_db": total_snr / num_batches,
            "spec_loss": total_spec / num_batches,
            "vad_loss": total_vad / num_batches,
        }

    def evaluate(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        total_snr = 0.0
        total_base_snr = 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in dataloader:
                mix = batch["mixture"].to(self.device).squeeze(1)
                target = batch["target"].to(self.device).squeeze(1)
                embed = batch["spk_embed"].to(self.device)

                base_snr = -self.eval_snr_fn(mix, target).item()
                total_base_snr += base_snr

                mix_stft = torch.stft(
                    mix, n_fft=512, hop_length=160, win_length=320, window=self.win, center=True, return_complex=True
                )
                mix_r = mix_stft.real.unsqueeze(1)
                mix_i = mix_stft.imag.unsqueeze(1)

                mask_r, mask_i, _ = self.model.forward_sequence(mix_r, mix_i, embed)

                est_r = mix_r * mask_r - mix_i * mask_i
                est_i = mix_r * mask_i + mix_i * mask_r

                est_pcm = torch.istft(
                    torch.complex(est_r.squeeze(1), est_i.squeeze(1)),
                    n_fft=512, hop_length=160, win_length=320, window=self.win, center=True, length=mix.shape[-1]
                )

                est_snr = -self.eval_snr_fn(est_pcm, target).item()
                total_snr += est_snr

        return {
            "snr_db": total_snr / num_batches,
            "base_snr_db": total_base_snr / num_batches,
            "improvement_db": (total_snr - total_base_snr) / num_batches,
        }


def run_training_pipeline(
    num_epochs: int = 10,
    batch_size: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_onnx_path: str = "dvec_v1.onnx",
):
    print("==================================================")
    print("      DVec Minimal Path A Training Pipeline       ")
    print("==================================================")
    print(f"Device: {device.upper()} | Epochs: {num_epochs} | Batch Size: {batch_size}")

    encoder = DeterministicSpeakerEncoder().to(device).eval()

    # Training and validation datasets
    train_dataset = DynamicTSEDataset(
        segment_duration=4.0,
        samples_per_epoch=40,
        sir_range=(-3.0, 10.0),
        embedding_jitter_std=0.04,
        encoder_func=lambda w: encoder(w.to(device)).cpu(),
    )
    val_dataset = DynamicTSEDataset(
        segment_duration=4.0,
        samples_per_epoch=10,
        sir_range=(0.0, 6.0),
        embedding_jitter_std=0.00,
        encoder_func=lambda w: encoder(w.to(device)).cpu(),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CausalTSEModel(num_freq_bins=257, embed_dim=256, hidden_dim=128, num_lstm_layers=2)
    trainer = TSETrainer(model=model, device=device, learning_rate=2e-3)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=num_epochs, eta_min=1e-4)

    best_val_snr = -999.0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        t_metrics = trainer.train_epoch(train_loader)
        v_metrics = trainer.evaluate(val_loader)
        scheduler.step()

        val_snr = v_metrics["snr_db"]
        if val_snr > best_val_snr:
            best_val_snr = val_snr
            # Export best model to ONNX
            export_streaming_onnx(model.cpu(), output_onnx_path)
            model.to(device)

        print(
            f"Epoch [{epoch:2d}/{num_epochs:2d}] | "
            f"Train Loss: {t_metrics['loss']:6.2f} | "
            f"Train SNR: {t_metrics['snr_db']:5.2f} dB | "
            f"Spec Loss: {t_metrics['spec_loss']:.3f} | "
            f"Val SNR: {v_metrics['snr_db']:5.2f} dB (dSNR: {v_metrics['improvement_db']:+5.2f} dB)"
        )

    print(f"\nTraining completed in {time.time() - start_time:.1f}s.")
    print(f"Best Validation Scale-Dependent SNR: {best_val_snr:.2f} dB")
    print(f"Locked ground-truth ONNX model exported to: {output_onnx_path}")


if __name__ == "__main__":
    run_training_pipeline()
