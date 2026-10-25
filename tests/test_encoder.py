"""Pytest suite for SpeakerEncoder and Binary Serialization."""

import pytest
import torch
import numpy as np
import soundfile as sf
from dvec.enrollment.encoder import (
    SpeakerEncoder,
    DeterministicSpeakerEncoder,
    TARGET_EMBED_DIM,
)
from dvec.enrollment.pipeline import enroll_user, EnrollmentError


def test_deterministic_encoder_shape_and_norm():
    """Verify deterministic encoder produces normalized 256-d embeddings."""
    encoder = DeterministicSpeakerEncoder(embed_dim=256)
    waveform = torch.randn(1, 160000)

    embedding = encoder(waveform)
    assert embedding.shape == (1, 256)

    norm = torch.norm(embedding, p=2, dim=-1).item()
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_binary_serialization_round_trip(tmp_path):
    """Verify binary save and load produces exact 1024 bytes and preserves float32 values."""
    raw_vector = torch.randn(1, 256)
    raw_vector = raw_vector / torch.norm(raw_vector, p=2, dim=-1, keepdim=True)

    bin_path = tmp_path / "test_enrollment.bin"
    saved_path = SpeakerEncoder.save_binary(raw_vector, bin_path)

    assert saved_path.exists()
    assert saved_path.stat().st_size == 1024  # Exactly 256 * 4 bytes

    loaded_vector = SpeakerEncoder.load_binary(bin_path)
    assert isinstance(loaded_vector, np.ndarray)
    assert loaded_vector.shape == (256,)
    assert loaded_vector.dtype == np.float32

    # Verify values match
    np.testing.assert_allclose(
        raw_vector.squeeze().numpy(), loaded_vector, rtol=1e-5, atol=1e-6
    )


def test_binary_loader_rejects_corrupted_size(tmp_path):
    """Verify binary loader rejects files that are not exactly 1024 bytes."""
    corrupt_file = tmp_path / "corrupt.bin"
    with open(corrupt_file, "wb") as f:
        f.write(b"\x00" * 500)  # Incorrect byte count

    with pytest.raises(ValueError, match="Corrupt embedding file"):
        SpeakerEncoder.load_binary(corrupt_file)


def test_end_to_end_enroll_user_workflow(tmp_path):
    """Verify high-level enroll_user validates input and saves enrollment.bin."""
    # Create synthetic 10-second WAV file at 16 kHz
    sr = 16000
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Synthetic speech-like harmonic carrier
    audio = 0.2 * np.sin(2 * np.pi * 180 * t) + 0.1 * np.sin(2 * np.pi * 360 * t)
    
    wav_path = tmp_path / "sample_voice.wav"
    sf.write(str(wav_path), audio, sr)

    out_bin = tmp_path / "output_user.bin"
    saved_path, report = enroll_user(wav_path, out_bin)

    assert report.passed
    assert saved_path.exists()
    assert saved_path.stat().st_size == 1024

    # Verify loaded vector
    vec = SpeakerEncoder.load_binary(out_bin)
    assert vec.shape == (256,)


def test_enroll_user_rejects_silence(tmp_path):
    """Verify enroll_user raises EnrollmentError on silent audio."""
    sr = 16000
    silent_audio = np.zeros(sr * 10, dtype=np.float32)

    wav_path = tmp_path / "silent.wav"
    sf.write(str(wav_path), silent_audio, sr)

    out_bin = tmp_path / "should_not_exist.bin"
    with pytest.raises(EnrollmentError):
        enroll_user(wav_path, out_bin)
