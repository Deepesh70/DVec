"""Offline ONNX Streaming Inference Runner.

Feeds an audio file frame-by-frame (10ms / 160 samples) through the exported stateful
ONNX graph and produces cleaned audio using ONNX Runtime.
"""

from pathlib import Path
import numpy as np
import soundfile as sf
import onnxruntime as ort
import torch

from dvec.utils.audio import load_audio_mono
from dvec.pipeline.stft import StreamingAudioProcessor
from dvec.enrollment.encoder import SpeakerEncoder


def run_offline_onnx_extraction(
    input_wav: str | Path,
    embedding_path: str | Path,
    onnx_model_path: str | Path,
    output_wav: str | Path,
) -> Path:
    input_wav = Path(input_wav)
    embedding_path = Path(embedding_path)
    onnx_model_path = Path(onnx_model_path)
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Audio
    waveform, sr = load_audio_mono(input_wav, target_sr=16000)
    samples = waveform.squeeze().numpy()

    # 2. Load 256-d Embedding Vector
    if embedding_path.suffix == ".bin":
        spk_embed = SpeakerEncoder.load_binary(embedding_path).reshape(1, 256)
    else:
        spk_embed = np.load(embedding_path).reshape(1, 256).astype(np.float32)

    # 3. Initialize ONNX Runtime Session
    session = ort.InferenceSession(
        str(onnx_model_path),
        providers=["CPUExecutionProvider"],
    )

    # 4. Initialize Streaming Audio Processor & Recurrent States
    processor = StreamingAudioProcessor(frame_size=320, hop_size=160, fft_size=512)
    h = np.zeros((2, 1, 128), dtype=np.float32)
    c = np.zeros((2, 1, 128), dtype=np.float32)

    hop_size = 160
    cleaned_chunks = []

    # 5. Process frame-by-frame (simulating real-time 10ms hops)
    num_samples = len(samples)
    for i in range(0, num_samples, hop_size):
        chunk = samples[i : i + hop_size]
        if len(chunk) < hop_size:
            chunk = np.pad(chunk, (0, hop_size - len(chunk)))

        hop_tensor = torch.from_numpy(chunk).unsqueeze(0)

        # 1. Slide input FIFO buffer
        processor.in_buffer = torch.cat(
            [processor.in_buffer[:, hop_size:], hop_tensor], dim=-1
        )

        # 2. Causal STFT Analysis on 320-sample sliding buffer
        spec_r, spec_i = processor.stft.analyze_frame(processor.in_buffer)

        # 3. ONNX Runtime Forward Step
        ort_inputs = {
            "input_real": spec_r.numpy(),
            "input_imag": spec_i.numpy(),
            "spk_embed": spk_embed,
            "h_in": h,
            "c_in": c,
        }
        mask_r, mask_i, vad_prob, h, c = session.run(None, ort_inputs)

        # 4. Complex Masking
        mr = torch.from_numpy(mask_r)
        mi = torch.from_numpy(mask_i)
        out_r = spec_r * mr - spec_i * mi
        out_i = spec_r * mi + spec_i * mr

        # 5. Causal iSTFT & Overlap-Add
        time_frame = processor.istft.synthesize_frame(out_r, out_i)
        processor.ola_buffer += time_frame
        out_hop = processor.ola_buffer[:, :hop_size].clone()
        processor.ola_buffer = torch.cat(
            [processor.ola_buffer[:, hop_size:], torch.zeros(1, hop_size)], dim=-1
        )
        cleaned_chunks.append(out_hop.squeeze().numpy())

    cleaned_audio = np.concatenate(cleaned_chunks)[:num_samples]
    sf.write(str(output_wav), cleaned_audio, 16000)
    return output_wav


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean a WAV file using stateful ONNX TSE model.")
    parser.add_argument("-i", "--input", required=True, help="Input noisy mixture WAV.")
    parser.add_argument("-e", "--embedding", required=True, help="Path to enrollment.bin.")
    parser.add_argument("-m", "--model", default="dvec_v1.onnx", help="Path to stateful ONNX model.")
    parser.add_argument("-o", "--output", default="cleaned_output.wav", help="Output destination WAV.")
    args = parser.parse_args()

    out = run_offline_onnx_extraction(args.input, args.embedding, args.model, args.output)
    print(f"Cleaned audio written to: {out}")
