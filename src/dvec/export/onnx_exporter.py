"""Streaming ONNX Graph Exporter.

Exposes recurrent hidden states (h_in, c_in) and outputs (h_out, c_out)
to allow O(1) stateful streaming execution in ONNX Runtime.
"""

from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
from dvec.pipeline.model import CausalTSEModel


class StreamingONNXWrapper(nn.Module):
    """Wraps CausalTSEModel for stateful ONNX streaming inference."""

    def __init__(self, core_model: CausalTSEModel):
        super().__init__()
        self.core = core_model

    def forward(
        self,
        input_real: torch.Tensor,
        input_imag: torch.Tensor,
        spk_embed: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.core.forward_streaming_frame(
            input_real, input_imag, spk_embed, h_in, c_in
        )


def export_streaming_onnx(
    model: CausalTSEModel,
    output_path: str | Path,
    opset_version: int = 17,
) -> Path:
    """Export the causal TSE model to stateful ONNX format.
    
    Args:
        model: CausalTSEModel instance.
        output_path: Output .onnx file path.
        opset_version: ONNX opset version (default 17).
    Returns:
        Path to generated .onnx model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = StreamingONNXWrapper(model).eval()

    # Dummy tensors representing 1 frame (10ms, 257 positive frequency bins)
    dummy_real = torch.randn(1, 1, 257, 1)
    dummy_imag = torch.randn(1, 1, 257, 1)
    dummy_embed = torch.randn(1, 256)
    dummy_embed = dummy_embed / torch.norm(dummy_embed, p=2, dim=-1, keepdim=True)
    dummy_h = torch.zeros(model.num_lstm_layers, 1, model.hidden_dim)
    dummy_c = torch.zeros(model.num_lstm_layers, 1, model.hidden_dim)

    input_names = ["input_real", "input_imag", "spk_embed", "h_in", "c_in"]
    output_names = ["mask_real", "mask_imag", "vad_prob", "h_out", "c_out"]

    torch.onnx.export(
        wrapper,
        (dummy_real, dummy_imag, dummy_embed, dummy_h, dummy_c),
        str(output_path),
        dynamo=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    return output_path
