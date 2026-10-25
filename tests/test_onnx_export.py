"""Pytest suite for ONNX streaming export and ONNX Runtime parity."""

import pytest
import torch
import numpy as np
import onnxruntime as ort
from dvec.pipeline.model import CausalTSEModel
from dvec.export.onnx_exporter import export_streaming_onnx


def test_onnx_streaming_export_and_inference(tmp_path):
    """Verify that exported ONNX model loads in ONNX Runtime and matches PyTorch output."""
    model = CausalTSEModel(num_freq_bins=257, embed_dim=256, hidden_dim=128, num_lstm_layers=2)
    model.eval()

    onnx_path = tmp_path / "test_model_streaming.onnx"
    export_streaming_onnx(model, onnx_path)

    assert onnx_path.exists(), "ONNX model file was not generated."

    # Load with ONNX Runtime
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Prepare inputs
    real = np.random.randn(1, 1, 257, 1).astype(np.float32)
    imag = np.random.randn(1, 1, 257, 1).astype(np.float32)
    embed = np.random.randn(1, 256).astype(np.float32)
    embed = embed / np.linalg.norm(embed, axis=-1, keepdims=True)
    h_in = np.zeros((2, 1, 128), dtype=np.float32)
    c_in = np.zeros((2, 1, 128), dtype=np.float32)

    # PyTorch reference
    with torch.no_grad():
        py_mr, py_mi, py_vad, py_h, py_c = model.forward_streaming_frame(
            torch.from_numpy(real),
            torch.from_numpy(imag),
            torch.from_numpy(embed),
            torch.from_numpy(h_in),
            torch.from_numpy(c_in),
        )

    # ONNX Runtime forward
    ort_inputs = {
        "input_real": real,
        "input_imag": imag,
        "spk_embed": embed,
        "h_in": h_in,
        "c_in": c_in,
    }
    ort_mr, ort_mi, ort_vad, ort_h, ort_c = session.run(None, ort_inputs)

    # Assert numerical consistency
    np.testing.assert_allclose(py_mr.numpy(), ort_mr, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(py_mi.numpy(), ort_mi, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(py_vad.numpy(), ort_vad, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(py_h.numpy(), ort_h, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(py_c.numpy(), ort_c, rtol=1e-3, atol=1e-4)
