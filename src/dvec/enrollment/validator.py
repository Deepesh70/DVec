"""Enrollment Sanity Check Engine.

Enforces three sequential verification gates on 10-second enrollment recordings:
1. RMS dynamic range and clipping check.
2. Voicing density check (sufficient continuous speech).
3. Acoustic purity / co-talker detection via sub-segment similarity.
"""

from dataclasses import dataclass
from typing import Tuple, List
import torch
import numpy as np
from dvec.utils.audio import compute_rms_dbfs, compute_peak_dbfs


@dataclass
class ValidationReport:
    passed: bool
    rms_dbfs: float
    peak_dbfs: float
    voicing_ratio: float
    min_subsegment_similarity: float
    failure_reasons: List[str]


class EnrollmentValidator:
    """Multi-gate validator preventing corrupt or contaminated voice enrollments."""

    def __init__(
        self,
        min_rms_dbfs: float = -35.0,
        max_peak_dbfs: float = -1.0,
        min_voicing_ratio: float = 0.65,
        min_co_talker_similarity: float = 0.45,
    ):
        self.min_rms_dbfs = min_rms_dbfs
        self.max_peak_dbfs = max_peak_dbfs
        self.min_voicing_ratio = min_voicing_ratio
        self.min_co_talker_similarity = min_co_talker_similarity

    def validate(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        encoder_func=None,
    ) -> ValidationReport:
        """Run all sanity checks on the waveform.
        
        Args:
            waveform: Tensor of shape (1, num_samples)
            sample_rate: Sample rate (must be 16000)
            encoder_func: Optional callable (waveform_chunk) -> (1, 256) embedding
        """
        assert sample_rate == 16000, "Enrollment validator requires 16 kHz audio."
        reasons = []

        # Gate 1: Energy & Clipping Check
        rms = compute_rms_dbfs(waveform)
        peak = compute_peak_dbfs(waveform)

        if rms < self.min_rms_dbfs:
            reasons.append(
                f"Signal too quiet (RMS: {rms:.1f} dBFS < {self.min_rms_dbfs:.1f} dBFS). Move closer to mic."
            )
        if peak > self.max_peak_dbfs:
            reasons.append(
                f"Digital clipping detected (Peak: {peak:.1f} dBFS > {self.max_peak_dbfs:.1f} dBFS). Lower mic gain."
            )

        # Gate 2: Voicing Density Check (Energy-based high-res thresholding fallback)
        # Compute energy across 30ms frames (480 samples, hop 160 samples)
        frame_len = 480
        hop_len = 160
        num_frames = (waveform.shape[-1] - frame_len) // hop_len + 1

        voiced_frames = 0
        if num_frames > 0:
            for i in range(num_frames):
                frame = waveform[:, i * hop_len : i * hop_len + frame_len]
                frame_rms = compute_rms_dbfs(frame)
                # Voiced threshold relative to overall RMS
                if frame_rms > (rms - 18.0) and frame_rms > -45.0:
                    voiced_frames += 1
            voicing_ratio = voiced_frames / num_frames
        else:
            voicing_ratio = 0.0

        if voicing_ratio < self.min_voicing_ratio:
            reasons.append(
                f"Insufficient active speech (Voicing: {voicing_ratio*100:.1f}% < {self.min_voicing_ratio*100:.1f}%). Speak continuously without long pauses."
            )

        # Gate 3: Acoustic Purity & Co-Talker Detection
        min_sim = 1.0
        if encoder_func is not None and len(reasons) == 0:
            full_emb = encoder_func(waveform)
            full_emb = full_emb / torch.norm(full_emb, p=2, dim=-1, keepdim=True)

            # Divide into 2-second subsegments (32,000 samples each)
            seg_len = 32000
            total_samples = waveform.shape[-1]
            segments = []
            for start in range(0, total_samples - seg_len + 1, seg_len):
                seg = waveform[:, start : start + seg_len]
                # Only analyze subsegments with active voiced energy
                if compute_rms_dbfs(seg) > -35.0:
                    segments.append(seg)

            if len(segments) >= 2:
                with torch.no_grad():
                    for seg in segments:
                        emb = encoder_func(seg)
                        emb = emb / torch.norm(emb, p=2, dim=-1, keepdim=True)
                        sim = float(torch.sum(full_emb * emb).item())
                        if sim < min_sim:
                            min_sim = sim

                if min_sim < self.min_co_talker_similarity:
                    reasons.append(
                        f"Inconsistent acoustic profile / secondary voice detected (Subsegment similarity: {min_sim:.2f} < {self.min_co_talker_similarity:.2f})."
                    )

        passed = len(reasons) == 0
        return ValidationReport(
            passed=passed,
            rms_dbfs=rms,
            peak_dbfs=peak,
            voicing_ratio=voicing_ratio,
            min_subsegment_similarity=min_sim,
            failure_reasons=reasons,
        )
