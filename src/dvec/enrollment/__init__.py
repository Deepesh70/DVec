"""Enrollment sanity checks, feature extraction, and speaker profiling."""

from dvec.enrollment.validator import EnrollmentValidator, ValidationReport
from dvec.enrollment.encoder import (
    SpeakerEncoder,
    DeterministicSpeakerEncoder,
    TARGET_EMBED_DIM,
)
from dvec.enrollment.pipeline import enroll_user, EnrollmentError

__all__ = [
    "EnrollmentValidator",
    "ValidationReport",
    "SpeakerEncoder",
    "DeterministicSpeakerEncoder",
    "TARGET_EMBED_DIM",
    "enroll_user",
    "EnrollmentError",
]
