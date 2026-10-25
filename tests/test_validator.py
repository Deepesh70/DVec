"""Pytest suite for EnrollmentValidator."""

import pytest
import torch
from dvec.enrollment.validator import EnrollmentValidator


def test_validator_rejects_quiet_signal():
    """Verify validator fails when audio RMS is below -35 dBFS."""
    validator = EnrollmentValidator(min_rms_dbfs=-35.0)
    # Extremely quiet audio
    quiet_audio = 0.0001 * torch.randn(1, 160000)

    report = validator.validate(quiet_audio, sample_rate=16000)
    assert not report.passed
    assert any("too quiet" in r.lower() for r in report.failure_reasons)


def test_validator_rejects_clipping():
    """Verify validator flags clipped digital signals."""
    validator = EnrollmentValidator(max_peak_dbfs=-1.0)
    # Hard clipped audio reaching +/- 1.0
    clipped_audio = torch.clamp(2.0 * torch.randn(1, 160000), min=-1.0, max=1.0)

    report = validator.validate(clipped_audio, sample_rate=16000)
    assert not report.passed
    assert any("clipping" in r.lower() for r in report.failure_reasons)


def test_validator_passes_valid_audio():
    """Verify clean synthesized voice-like audio passes sanity checks."""
    validator = EnrollmentValidator(min_rms_dbfs=-35.0, min_voicing_ratio=0.5)

    # 10 seconds of synthetic speech-like signal
    t = torch.linspace(0, 10.0, 160000)
    clean_audio = 0.15 * torch.sin(2 * torch.pi * 200 * t) + 0.08 * torch.sin(2 * torch.pi * 500 * t)
    clean_audio = clean_audio.unsqueeze(0)

    report = validator.validate(clean_audio, sample_rate=16000)
    assert report.passed, f"Validation failed unexpectedly: {report.failure_reasons}"
