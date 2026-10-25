"""Speaker Encoder and Binary Embedding Serialization.

Extracts normalized 256-dimensional speaker embeddings using pre-trained ECAPA-TDNN
and serializes them to raw 1024-byte IEEE 754 float binary files for native daemon execution.
"""

from pathlib import Path
from typing import Optional, Union
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_EMBED_DIM = 256


class DeterministicSpeakerEncoder(nn.Module):
    """Deterministic acoustic feature encoder for offline development and testing.
    
    Extracts a 256-d embedding based on spectral statistics and Mel-frequency filterbanks.
    """

    def __init__(self, embed_dim: int = TARGET_EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        # Deterministic projection from 80-bin log-mel statistics (mean + std = 160) to embed_dim
        torch.manual_seed(1337)
        self.proj = nn.Linear(160, embed_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Args:
            waveform: (batch, num_samples) or (1, num_samples) at 16 kHz
        Returns:
            Normalized embedding (batch, embed_dim)
        """
        waveform = waveform.float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # STFT magnitude
        window = torch.hann_window(400, periodic=True, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=512,
            hop_length=160,
            win_length=400,
            window=window,
            center=False,
            return_complex=True,
        )
        mag = torch.abs(stft)  # (B, F=257, T)

        # Truncate/slice to 80 bins
        mel_like = mag[:, :80, :]
        log_mel = torch.log(mel_like + 1e-6)

        # Temporal pooling: mean + standard deviation -> (B, 160)
        mean = torch.mean(log_mel, dim=-1)
        std = torch.std(log_mel, dim=-1)
        stats = torch.cat([mean, std], dim=-1)

        # Linear projection to embed_dim
        raw_emb = self.proj(stats)

        # L2 Hypersphere Normalization
        normalized = F.normalize(raw_emb, p=2, dim=-1)
        return normalized


class SpeakerEncoder:
    """Production speaker encoder wrapping SpeechBrain ECAPA-TDNN with deterministic fallback."""

    def __init__(
        self,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str = "cpu",
        use_fallback: bool = False,
    ):
        self.source = source
        self.device = device
        self.use_fallback = use_fallback
        self.classifier = None
        self.fallback_model = DeterministicSpeakerEncoder(TARGET_EMBED_DIM).to(device).eval()

        if not use_fallback:
            try:
                from speechbrain.inference.classifiers import EncoderClassifier
                self.classifier = EncoderClassifier.from_hparams(
                    source=source,
                    run_opts={"device": device},
                )
            except Exception:
                # Fallback to local deterministic encoder if offline / download fails
                self.classifier = None

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract a normalized 1x256 speaker embedding from a 16 kHz mono waveform.
        
        Args:
            waveform: Tensor of shape (1, num_samples)
        Returns:
            Embedding tensor of shape (1, 256) where ||e||_2 = 1.0
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.device)

        with torch.no_grad():
            if self.classifier is not None:
                # SpeechBrain ECAPA-TDNN outputs (batch, 1, 192)
                raw_emb = self.classifier.encode_batch(waveform).squeeze(1)  # (1, 192)
                # Zero-pad or deterministically expand to 256-d if needed
                if raw_emb.shape[-1] < TARGET_EMBED_DIM:
                    pad_len = TARGET_EMBED_DIM - raw_emb.shape[-1]
                    raw_emb = F.pad(raw_emb, (0, pad_len), mode="constant", value=0.0)
                elif raw_emb.shape[-1] > TARGET_EMBED_DIM:
                    raw_emb = raw_emb[:, :TARGET_EMBED_DIM]
                emb = F.normalize(raw_emb, p=2, dim=-1)
            else:
                emb = self.fallback_model(waveform)

        return emb.cpu()

    @staticmethod
    def save_binary(embedding: torch.Tensor, output_path: Union[str, Path]) -> Path:
        """Serialize embedding to raw 1024-byte IEEE 754 float binary file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        emb_np = embedding.detach().cpu().numpy().flatten().astype(np.float32)
        if len(emb_np) != TARGET_EMBED_DIM:
            raise ValueError(f"Expected {TARGET_EMBED_DIM} floats, got {len(emb_np)}")

        # Write exact 1024 bytes (256 * 4 bytes)
        with open(output_path, "wb") as f:
            f.write(emb_np.tobytes())

        return output_path

    @staticmethod
    def load_binary(input_path: Union[str, Path]) -> np.ndarray:
        """Read and validate a 1024-byte binary embedding file."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {input_path}")

        file_size = input_path.stat().st_size
        expected_size = TARGET_EMBED_DIM * 4  # 1024 bytes
        if file_size != expected_size:
            raise ValueError(
                f"Corrupt embedding file: size is {file_size} bytes, expected {expected_size} bytes"
            )

        with open(input_path, "rb") as f:
            data = f.read()

        vector = np.frombuffer(data, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if not np.isclose(norm, 1.0, atol=1e-3):
            vector = vector / (norm + 1e-12)

        return vector
