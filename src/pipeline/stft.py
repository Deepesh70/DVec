"""Causal Short-Time Fourier Transform (STFT) and Overlap-Add (OLA) Synthesis.

Designed for low-latency streaming audio processing with zero future lookahead.
Standardized on 16 kHz Mono PCM audio.
"""

import torch
import torch.nn as nn


class CausalSTFT(nn.Module):
    """Causal STFT analysis module.
    
    Frame: 320 samples (20 ms at 16 kHz).
    Hop: 160 samples (10 ms at 16 kHz).
    FFT size: 512 points.
    Strictly causal: analysis window looks only at current and past samples.
    """

    def __init__(self, frame_size: int = 320, hop_size: int = 160, fft_size: int = 512):
        super().__init__()
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.fft_size = fft_size

        # Square-root Hann window
        window = torch.hann_window(frame_size, periodic=True).sqrt()
        self.register_buffer("window", window)

    def analyze_frame(self, in_buffer: torch.Tensor):
        """Analyze a single pre-buffered time window of length `frame_size`.
        
        Args:
            in_buffer: (batch, frame_size)
        Returns:
            real: (batch, 1, freq_bins, 1)
            imag: (batch, 1, freq_bins, 1)
        """
        windowed = in_buffer * self.window
        spec = torch.fft.rfft(windowed, n=self.fft_size, dim=-1)
        # Reshape to (B, 1, F, 1) for 2D Conv processing
        real = spec.real.unsqueeze(1).unsqueeze(-1)
        imag = spec.imag.unsqueeze(1).unsqueeze(-1)
        return real, imag


class CausalISTFT(nn.Module):
    """Causal iSTFT synthesis module with overlap-add."""

    def __init__(self, frame_size: int = 320, hop_size: int = 160, fft_size: int = 512):
        super().__init__()
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.fft_size = fft_size

        window = torch.hann_window(frame_size, periodic=True).sqrt()
        self.register_buffer("window", window)

    def synthesize_frame(self, real: torch.Tensor, imag: torch.Tensor):
        """Synthesize a single complex frame back to time domain.
        
        Args:
            real: (batch, 1, freq_bins, 1)
            imag: (batch, 1, freq_bins, 1)
        Returns:
            Windowed time-domain frame: (batch, frame_size)
        """
        spec = torch.complex(real.squeeze(-1).squeeze(1), imag.squeeze(-1).squeeze(1))
        time_frame = torch.fft.irfft(spec, n=self.fft_size, dim=-1)[..., :self.frame_size]
        return time_frame * self.window


class StreamingAudioProcessor:
    """Stateful streaming processor executing frame-by-frame causal STFT/iSTFT.
    
    Incurs an algorithmic delay of exactly 1 hop (160 samples / 10 ms).
    """

    def __init__(self, frame_size: int = 320, hop_size: int = 160, fft_size: int = 512):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.fft_size = fft_size

        self.stft = CausalSTFT(frame_size, hop_size, fft_size)
        self.istft = CausalISTFT(frame_size, hop_size, fft_size)

        # Buffers for single stream
        self.in_buffer = torch.zeros(1, frame_size)
        self.ola_buffer = torch.zeros(1, frame_size)

    def process_hop(self, hop_pcm: torch.Tensor, mask_real: torch.Tensor = None, mask_imag: torch.Tensor = None):
        """Processes an incoming 10ms hop (160 samples).
        
        Args:
            hop_pcm: (1, hop_size)
            mask_real: Optional cIRM real component (1, 1, freq_bins, 1)
            mask_imag: Optional cIRM imag component (1, 1, freq_bins, 1)
        Returns:
            out_pcm: (1, hop_size) cleaned audio frame
        """
        # 1. Slide input FIFO buffer
        self.in_buffer = torch.cat([self.in_buffer[:, self.hop_size:], hop_pcm], dim=-1)

        # 2. Causal STFT Analysis
        spec_r, spec_i = self.stft.analyze_frame(self.in_buffer)

        # 3. Apply Complex Masking if provided
        if mask_real is not None and mask_imag is not None:
            # Complex multiplication: (Xr + jXi) * (Mr + jMi) = (XrMr - XiMi) + j(XrMi + XiMr)
            out_r = spec_r * mask_real - spec_i * mask_imag
            out_i = spec_r * mask_imag + spec_i * mask_real
        else:
            out_r, out_i = spec_r, spec_i

        # 4. Causal iSTFT Synthesis
        time_frame = self.istft.synthesize_frame(out_r, out_i)

        # 5. Overlap-Add accumulation
        self.ola_buffer += time_frame
        out_hop = self.ola_buffer[:, :self.hop_size].clone()

        # Shift overlap buffer left
        self.ola_buffer = torch.cat([self.ola_buffer[:, self.hop_size:], torch.zeros(1, self.hop_size)], dim=-1)

        return out_hop
