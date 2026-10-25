"""Audio I/O and preprocessing utilities."""

from pathlib import Path
from typing import Tuple
import torch
import torchaudio


TARGET_SAMPLE_RATE = 16000


def load_audio_mono(path: str | Path, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[torch.Tensor, int]:
    """Load an audio file using soundfile, convert to mono float32, and resample if needed.
    
    Args:
        path: Path to audio file.
        target_sr: Required sample rate (default: 16000).
    Returns:
        (waveform, sample_rate) where waveform has shape (1, num_samples).
    """
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32")
    waveform = torch.from_numpy(data)

    # If 1D, shape -> (1, T)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.shape[0] > 1 and waveform.shape[1] > 1:
        # Multi-channel: average channels across dim=0 or dim=1
        if waveform.shape[1] < waveform.shape[0]:  # (T, C)
            waveform = torch.mean(waveform, dim=1, keepdim=True).t()
        else:  # (C, T)
            waveform = torch.mean(waveform, dim=0, keepdim=True)
    elif waveform.shape[0] > 1 and waveform.shape[1] == 1:
        waveform = waveform.t()

    # Resample if sample rate differs
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    return waveform, sr


def compute_rms_dbfs(waveform: torch.Tensor) -> float:
    """Compute Root-Mean-Square (RMS) energy in decibels relative to full scale (dBFS)."""
    rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-12)
    return float(20.0 * torch.log10(rms))


def compute_peak_dbfs(waveform: torch.Tensor) -> float:
    """Compute peak amplitude in dBFS."""
    peak = torch.max(torch.abs(waveform)) + 1e-12
    return float(20.0 * torch.log10(peak))
