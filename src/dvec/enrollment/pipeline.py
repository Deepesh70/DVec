"""End-to-End User Voice Enrollment Workflow."""

from pathlib import Path
from typing import Tuple, Optional
import torch

from dvec.utils.audio import load_audio_mono
from dvec.enrollment.validator import EnrollmentValidator, ValidationReport
from dvec.enrollment.encoder import SpeakerEncoder


class EnrollmentError(Exception):
    """Raised when an enrollment recording fails acoustic sanity checks."""
    pass


def enroll_user(
    audio_path: str | Path,
    output_bin_path: str | Path,
    validator: Optional[EnrollmentValidator] = None,
    encoder: Optional[SpeakerEncoder] = None,
) -> Tuple[Path, ValidationReport]:
    """Execute end-to-end speaker enrollment from a recorded audio file.
    
    Args:
        audio_path: Path to user's 10-second reference voice clip (.wav, .flac, etc.).
        output_bin_path: Destination for the serialized 1024-byte binary vector.
        validator: Optional custom EnrollmentValidator instance.
        encoder: Optional custom SpeakerEncoder instance.
    Returns:
        (saved_path, validation_report)
    Raises:
        EnrollmentError: If the recording fails any verification gate.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, sr = load_audio_mono(audio_path, target_sr=16000)

    if encoder is None:
        encoder = SpeakerEncoder(use_fallback=False)

    if validator is None:
        validator = EnrollmentValidator()

    # Run verification gates
    report = validator.validate(
        waveform=waveform,
        sample_rate=sr,
        encoder_func=encoder.encode,
    )

    if not report.passed:
        reasons_str = "\n  - " + "\n  - ".join(report.failure_reasons)
        raise EnrollmentError(
            f"Enrollment audio rejected by sanity gates:{reasons_str}"
        )

    # Extract final normalized 256-d embedding
    embedding = encoder.encode(waveform)

    # Save to 1024-byte binary file
    saved_path = SpeakerEncoder.save_binary(embedding, output_bin_path)

    return saved_path, report
