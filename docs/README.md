# Target Speaker Extraction (TSE) Daemon Documentation

Welcome to the technical documentation for the real-time, low-latency Target Speaker Extraction (TSE) daemon. This documentation provides complete engineering specifications, mathematical formulations, training recipes, and runtime architecture for isolating an enrolled user's voice from background chatter and ambient noise.

---

## Documentation Index

| Document | Topic | Description |
| :--- | :--- | :--- |
| [01. Mathematical Pipeline](file:///c:/Users/aadit/Desktop/DVec/docs/01_mathematical_pipeline.md) | Signal Processing & Masking | Causal STFT/iSTFT, FiLM conditioning, Complex Ideal Ratio Masks (cIRM), spectral floor clamping, and Personal VAD (pVAD). |
| [02. Synthetic Training Pipeline](file:///c:/Users/aadit/Desktop/DVec/docs/02_synthetic_training_pipeline.md) | Datasets & Training | Synthetic scene generation, RIR spatial convolution, emotional speech ("gamer scream") robustness, and SI-SDR loss functions. |
| [03. Streaming ONNX Export](file:///c:/Users/aadit/Desktop/DVec/docs/03_streaming_onnx_export.md) | Model Deployment | Stateful LSTM recurrent export, ONNX tensor contracts, INT8 quantization, and hardware acceleration (DirectML/TensorRT). |
| [04. Audio Engine & Daemon](file:///c:/Users/aadit/Desktop/DVec/docs/04_audio_engine_and_daemon.md) | Native Runtime Loop | Dual-thread lock-free SPSC ring buffers, WASAPI callback rules, zero-allocation real-time audio safety, and VB-Cable routing. |
| [05. Enrollment Engine](file:///c:/Users/aadit/Desktop/DVec/docs/05_enrollment_engine.md) | Speaker Verification | 10-second voice calibration, RMS metering, voicing density gates, co-talker purity checks, and frozen ECAPA-TDNN embedding generation. |

---

## Architectural Invariants (What NOT to Build)

1. **Do NOT train speaker encoders from scratch:** Use frozen pre-trained weights from ECAPA-TDNN (SpeechBrain) or WeSpeaker.
2. **Do NOT fine-tune models per user at runtime:** Condition the network via static FiLM embeddings derived in $\le 50\text{ ms}$.
3. **Do NOT use non-causal or lookahead architectures:** Algorithmic future lookahead must remain strictly $0\text{ ms}$.
4. **Do NOT develop custom kernel-mode audio drivers:** Route output into VB-Audio Virtual Cable via user-space WASAPI.
