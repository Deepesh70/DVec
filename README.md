# DVec: Low-Latency Target Speaker Extraction Daemon

[![CI Pipeline](https://github.com/Aadit2004/DVec/actions/workflows/ci.yml/badge.svg)](https://github.com/Aadit2004/DVec/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

A real-time, local audio daemon that conditions a neural extraction network on a pre-computed mathematical vector of the user's voice (a d-vector / speaker embedding), masks out all frequencies belonging to interfering talkers and ambient noise, and routes clean audio to a virtual microphone for Discord, Zoom, and in-game chat.

---

## The Gap: Speech vs. Interfering Speech

Standard consumer solutions (Krisp, RTX Voice, RNNoise) solve the **Speech vs. Non-Speech** problem. They suppress stationary noise (fans, HVAC) and transient noises (mechanical keyboards, dogs), but preserve all human speech. In shared environments (dorms, LAN cafes, open offices), roommates talking in the background pass directly through Krisp.

**DVec solves the Cocktail Party Problem in real time:** isolating only the enrolled speaker's vocal acoustics while attenuating all background talkers under a strict $\le 15\text{ ms}$ latency budget.

```
Incoming Mic (16 kHz PCM)
  │ (10ms frame / 160 samples)
  ▼
[Causal STFT] ──► Complex Spectrum (Real, Imag)
                        │
                        ▼
Target Embedding (e ∈ R^256) ──► [FiLM Layer] ──► [Complex CRN + Stateful LSTM]
                                                        │
                                                        ├──► cIRM Mask (Bounded Tanh + Floor)
                                                        └──► Personal VAD Gate (Soft Mute)
                                                        │
[Complex Masking] ◄─────────────────────────────────────┘
  │
  ▼
[Causal iSTFT Overlap-Add] ──► Cleaned 10ms PCM ──► VB-Audio Virtual Cable (Discord / Zoom)
```

---

## Key Technical Specifications

- **Acoustic Standard:** 16 kHz Mono, 16-bit PCM (captures up to 8 kHz, covering fundamental formants $F_0\text{--}F_4$).
- **Causal Framing:** 20ms frame size (320 samples), 10ms hop size (160 samples), 512-point FFT. Algorithmic delay is exactly 1 hop (10 ms) with $>138\text{ dB}$ reconstruction SNR.
- **FiLM Bottleneck Modulation:** Injects zero-shot 256-d speaker embeddings derived from frozen ECAPA-TDNN without requiring runtime training.
- **Artifact Protection:** Complex Ideal Ratio Masking (cIRM) bounded by an explicit spectral floor ($\alpha_{\min} = 0.02$) prevents musical chirps; an auxiliary Personal VAD (pVAD) head eliminates silence hallucinations / ghost whispers.
- **Stateful ONNX Streaming:** Exposes recurrent hidden/cell states $(h_{in}, c_{in} \to h_{out}, c_{out})$ directly in the computational graph for $O(1)$ constant-time inference in ONNX Runtime.
- **Enrollment Sanity Check:** Enforces 3 verification gates (RMS energy, voicing ratio VAD, and sub-segment co-talker purity) before storing user profiles.

---

## Directory Layout

```
DVec/
├── pyproject.toml                         # Standard PEP 517 build metadata
├── requirements.txt                       # Locked production dependencies
├── docs/                                  # Modular engineering specifications
│   ├── 01_mathematical_pipeline.md        # Causal STFT, FiLM, cIRM math & pVAD
│   ├── 02_synthetic_training_pipeline.md  # Dynamic mixing, RIRs, emotional speech
│   ├── 03_streaming_onnx_export.md        # Stateful recurrent ONNX graph contracts
│   ├── 04_audio_engine_and_daemon.md      # Lock-free SPSC ring buffers & WASAPI
│   ├── 05_enrollment_engine.md            # Multi-gate enrollment calibration
│   └── architecture.md                    # Master engineering specification
├── src/
│   └── dvec/
│       ├── pipeline/                      # Causal STFT, iSTFT & Complex CRN
│       ├── enrollment/                    # RMS, voicing ratio & purity validation
│       ├── export/                        # Streaming ONNX model serialization
│       └── utils/                         # Metering & audio format utilities
└── tests/                                 # Unit & parity verification suite
```

---

## Quickstart

### 1. Environment Setup

```bash
# Create Conda environment targeting Python 3.11
conda create -n dvec-env python=3.11 -y
conda activate dvec-env

# Install dependencies and local package in editable mode
pip install -r requirements.txt
pip install -e .
```

### 2. Run Test Suite

```bash
pytest tests/ -v
```

### 3. Export Stateful Streaming ONNX Model

```python
from dvec.pipeline.model import CausalTSEModel
from dvec.export.onnx_exporter import export_streaming_onnx

model = CausalTSEModel()
export_streaming_onnx(model, output_path="model_streaming.onnx")
```

---

## Documentation

For full mathematical derivations, training pipelines, and native audio engine designs, refer to the [docs/](docs/README.md) directory:
- [01. Mathematical Pipeline](docs/01_mathematical_pipeline.md)
- [02. Synthetic Training & RIR Pipeline](docs/02_synthetic_training_pipeline.md)
- [03. Streaming ONNX Export Contract](docs/03_streaming_onnx_export.md)
- [04. Real-Time Audio Engine & Daemon](docs/04_audio_engine_and_daemon.md)
- [05. Enrollment Sanity-Check Engine](docs/05_enrollment_engine.md)

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
