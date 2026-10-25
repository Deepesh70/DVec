"""DeepFilterNet3 Production Speech Enhancement Engine.

Wraps the pretrained DeepFilterNet3 causal architecture for zero-latency
fullband (48 kHz / 16 kHz) noise, reverberation, and competing voice suppression.
"""

import sys
import types
from pathlib import Path
from typing import Optional, Union, Tuple
import torch
import torchaudio
import soundfile as sf
import numpy as np


def _apply_torchaudio_compat_shim():
    """Shim torchaudio 2.x for DeepFilterNet backward compatibility."""
    if "torchaudio.backend.common" not in sys.modules:
        backend = types.ModuleType("torchaudio.backend")
        common = types.ModuleType("torchaudio.backend.common")
        common.AudioMetaData = getattr(torchaudio, "AudioMetaData", None)
        backend.common = common
        sys.modules["torchaudio.backend"] = backend
        sys.modules["torchaudio.backend.common"] = common


class DeepFilterEngine:
    """Production wrapper around DeepFilterNet3 causal acoustic model."""

    def __init__(self, post_filter: bool = True):
        _apply_torchaudio_compat_shim()
        from df.enhance import init_df
        self.model, self.df_state, _ = init_df()
        self.target_sr = self.df_state.sr()
        self.post_filter = post_filter

    def enhance_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
    ) -> Tuple[torch.Tensor, int]:
        """Enhance a mono/stereo waveform tensor.
        
        Args:
            waveform: Tensor of shape (1, num_samples) or (channels, num_samples)
            sample_rate: Input audio sample rate
        Returns:
            (enhanced_tensor, output_sample_rate)
        """
        _apply_torchaudio_compat_shim()
        from df.enhance import enhance

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Resample to DeepFilterNet target sample rate (48 kHz) if needed
        orig_sr = sample_rate
        if orig_sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, self.target_sr)

        with torch.no_grad():
            enhanced = enhance(self.model, self.df_state, waveform)

        return enhanced, self.target_sr

    def enhance_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Path:
        """Enhance an audio file directly and write output to disk."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data, sr = sf.read(str(input_path))
        t_data = torch.from_numpy(data).float()
        if t_data.dim() == 1:
            t_data = t_data.unsqueeze(0)
        elif t_data.shape[0] > 1 and t_data.shape[0] < t_data.shape[1]:
            pass
        else:
            t_data = t_data.t()

        enhanced, out_sr = self.enhance_waveform(t_data, sample_rate=sr)
        out_np = enhanced.squeeze().cpu().numpy()
        sf.write(str(output_path), out_np, out_sr)
        return output_path
