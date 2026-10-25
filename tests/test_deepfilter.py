"""Unit tests for DeepFilterNet3 integration."""

import pytest
import torch
from dvec.pipeline.deepfilter import DeepFilterEngine


def test_deepfilter_initialization_and_forward():
    """Verify DeepFilterEngine initializes and enhances synthetic speech."""
    engine = DeepFilterEngine()
    assert engine.target_sr == 48000
    assert engine.model is not None

    # 0.25 seconds of synthetic audio @ 16 kHz
    sr = 16000
    t = torch.linspace(0, 0.25, int(sr * 0.25))
    dummy_audio = 0.3 * torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)

    enhanced, out_sr = engine.enhance_waveform(dummy_audio, sample_rate=sr)
    assert out_sr == 48000
    assert enhanced.shape[-1] == int(0.25 * 48000)
    assert torch.isfinite(enhanced).all()
