# Contributing to DVec

We welcome contributions to the DVec real-time Target Speaker Extraction engine. Please follow these guidelines when submitting pull requests.

---

## Architectural Rules (Do Not Violate)

1. **No Non-Causal Lookahead:** All digital signal processing (STFT/iSTFT) and neural network layers must remain strictly causal. Future padding or non-causal convolutions will be rejected.
2. **Fixed 16 kHz Audio Standard:** All audio inputs, STFT frames, and tensor definitions must conform to 16 kHz Mono PCM.
3. **No Per-User Retraining:** Enrolled speakers must be conditioned solely via zero-shot FiLM modulation from static embeddings.
4. **Stateful Recurrent Serialization:** Any recurrent layers (LSTM/GRU) exported to ONNX must explicitly expose hidden and cell states as input and output graph tensors.

---

## Development Workflow

### 1. Set Up Environment

```bash
conda activate dvec-env
pip install -r requirements.txt
pip install -e .
```

### 2. Run Test Suite

All contributions must pass the test suite before submitting a pull request:

```bash
pytest tests/ -v --cov=dvec
```

### 3. Code Quality

Ensure code is formatted cleanly and adheres to PEP 8 standards:
- Type annotations on all public functions.
- Descriptive docstrings explaining algorithmic bounds and tensor shapes.
- No debug print statements or ad-hoc file writing inside the audio loop.
