"""Pytest suite for Causal STFT/iSTFT and CausalTSEModel."""

import pytest
import torch
from dvec.pipeline.stft import StreamingAudioProcessor, CausalSTFT, CausalISTFT
from dvec.pipeline.model import CausalTSEModel


def test_causal_stft_istft_pass_through():
    """Verify that frame-by-frame streaming achieves >100 dB reconstruction SNR."""
    torch.manual_seed(42)
    sample_rate = 16000
    duration_sec = 0.5
    num_samples = int(sample_rate * duration_sec)
    hop_size = 160
    frame_size = 320

    t = torch.linspace(0, duration_sec, num_samples)
    signal = 0.5 * torch.sin(2 * torch.pi * 440 * t) + 0.2 * torch.randn(num_samples)
    signal = signal.unsqueeze(0)

    processor = StreamingAudioProcessor(frame_size=frame_size, hop_size=hop_size, fft_size=512)

    out_hops = []
    for i in range(0, num_samples, hop_size):
        chunk = signal[:, i : i + hop_size]
        if chunk.shape[-1] < hop_size:
            break
        out_hops.append(processor.process_hop(chunk))

    reconstructed = torch.cat(out_hops, dim=-1)

    delay = hop_size
    s_clean = signal[:, frame_size : reconstructed.shape[-1] - delay]
    s_recon = reconstructed[:, frame_size + delay :]

    noise = s_clean - s_recon
    snr = 10 * torch.log10(torch.sum(s_clean ** 2) / torch.sum(noise ** 2))
    assert snr.item() > 100.0


def test_tse_model_streaming_contract():
    """Verify stateful model execution conforms to shape and value contracts."""
    model = CausalTSEModel(num_freq_bins=257, embed_dim=256, hidden_dim=128, num_lstm_layers=2)
    model.eval()

    h = torch.zeros(2, 1, 128)
    c = torch.zeros(2, 1, 128)
    spk_embed = torch.randn(1, 256)
    spk_embed = spk_embed / torch.norm(spk_embed, p=2, dim=-1, keepdim=True)

    for _ in range(5):
        frame_r = torch.randn(1, 1, 257, 1)
        frame_i = torch.randn(1, 1, 257, 1)

        mask_r, mask_i, vad_prob, h, c = model.forward_streaming_frame(
            frame_r, frame_i, spk_embed, h, c
        )

        assert mask_r.shape == (1, 1, 257, 1)
        assert mask_i.shape == (1, 1, 257, 1)
        assert 0.0 <= vad_prob.item() <= 1.0

        # Verify spectral floor clamp
        mag = torch.sqrt(mask_r ** 2 + mask_i ** 2)
        assert torch.all(mag >= 0.02 - 1e-5)
