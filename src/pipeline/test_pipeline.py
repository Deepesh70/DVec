"""Verification test for Causal STFT/iSTFT and Stateful Causal TSE Model.
"""

import torch
from stft import StreamingAudioProcessor
from model import CausalTSEModel


def test_stft_streaming_reconstruction():
    """Verify that frame-by-frame pass-through yields >100 dB SNR (perfect reconstruction)."""
    torch.manual_seed(42)
    sample_rate = 16000
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)
    hop_size = 160
    frame_size = 320

    # Synthetic signal: dual sinusoids + noise
    t = torch.linspace(0, duration_sec, num_samples)
    signal = 0.6 * torch.sin(2 * torch.pi * 300 * t) + 0.3 * torch.sin(2 * torch.pi * 1200 * t) + 0.05 * torch.randn(num_samples)
    signal = signal.unsqueeze(0)  # (1, T)

    processor = StreamingAudioProcessor(frame_size=frame_size, hop_size=hop_size, fft_size=512)

    out_hops = []
    for i in range(0, num_samples, hop_size):
        chunk = signal[:, i : i + hop_size]
        if chunk.shape[-1] < hop_size:
            break
        out_hop = processor.process_hop(chunk)
        out_hops.append(out_hop)

    reconstructed = torch.cat(out_hops, dim=-1)

    # 1 hop (160 sample) delay compensation
    delay = hop_size
    s_clean = signal[:, frame_size : reconstructed.shape[-1] - delay]
    s_recon = reconstructed[:, frame_size + delay :]

    noise = s_clean - s_recon
    snr = 10 * torch.log10(torch.sum(s_clean ** 2) / torch.sum(noise ** 2))
    print(f"[Streaming STFT/iSTFT Pass-Through] SNR: {snr.item():.2f} dB")
    assert snr.item() > 100.0, f"Streaming SNR below threshold: {snr.item():.2f} dB"


def test_end_to_end_streaming_pipeline():
    """Verify end-to-end streaming coupling: PCM -> STFT -> Model (FiLM+LSTM) -> cIRM -> iSTFT -> PCM."""
    model = CausalTSEModel(num_freq_bins=257, embed_dim=256, hidden_dim=128, num_lstm_layers=2)
    model.eval()

    processor = StreamingAudioProcessor(frame_size=320, hop_size=160, fft_size=512)

    # Enrolled speaker embedding
    spk_embed = torch.randn(1, 256)
    spk_embed = spk_embed / torch.norm(spk_embed, p=2, dim=1, keepdim=True)

    # Stateful LSTM buffers
    h = torch.zeros(2, 1, 128)
    c = torch.zeros(2, 1, 128)

    print("[End-to-End Test] Streaming 20 consecutive 10ms hops:")
    for step in range(20):
        # 160 random samples representing 10ms chunk
        input_pcm = 0.1 * torch.randn(1, 160)

        # 1. Slide input FIFO
        processor.in_buffer = torch.cat([processor.in_buffer[:, 160:], input_pcm], dim=-1)

        # 2. Causal STFT
        spec_r, spec_i = processor.stft.analyze_frame(processor.in_buffer)

        # 3. Model forward step with FiLM & state persistence
        mask_r, mask_i, vad_prob, h, c = model.forward_streaming_frame(
            spec_r, spec_i, spk_embed, h, c
        )

        # 4. Complex Masking & iSTFT Synthesis
        out_pcm = processor.process_hop(input_pcm, mask_r, mask_i)

        assert out_pcm.shape == (1, 160)
        assert not torch.isnan(out_pcm).any()
        assert 0.0 <= vad_prob.item() <= 1.0

    print("  -> Passed: End-to-end streaming audio loop verified without NaN or shape mismatch.")


if __name__ == "__main__":
    test_stft_streaming_reconstruction()
    test_end_to_end_streaming_pipeline()
    print("[ALL PIPELINE TESTS GREEN]")
