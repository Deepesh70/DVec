"""Overfit Proof on User's Real Voice.

Synthesizes a phase-aligned test mixture using the user's real clean voice (my_enrollment_norm.wav)
and an authentic human interfering speaker from LibriSpeech mixed at 0 dB SIR (equal loudness).
Trains CausalTSEModel for 400 steps to prove isolation of the user's actual voice profile.
"""

import time
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import soundfile as sf

from dvec.pipeline.model import CausalTSEModel
from dvec.enrollment.encoder import SpeakerEncoder
from dvec.train.losses import CompositeTSELoss, SNRLoss
from dvec.export.onnx_exporter import export_streaming_onnx
from dvec.export.inference import run_offline_onnx_extraction
from dvec.utils.audio import load_audio_mono


def main():
    scratch_dir = Path("scratch")
    scratch_dir.mkdir(exist_ok=True)

    # 1. Load User's Real Clean Audio
    user_wav_path = Path("my_enrollment_norm.wav")
    s_target, sr = load_audio_mono(user_wav_path, target_sr=16000)
    num_samples = s_target.shape[-1]
    print(f"Loaded target user audio: {user_wav_path} ({num_samples / sr:.1f}s)")

    # 2. Load Real Human Interferer from LibriSpeech
    interferer_files = list(Path("data/LibriSpeech/dev-clean").rglob("*.flac"))
    if not interferer_files:
        raise FileNotFoundError("LibriSpeech flac files not found in data/LibriSpeech/dev-clean")
    
    interferer_path = interferer_files[10]  # Take a distinct speaker
    s_interferer_raw, _ = load_audio_mono(interferer_path, target_sr=16000)
    
    # Tile or pad interferer to match exact sample length
    if s_interferer_raw.shape[-1] < num_samples:
        repeats = (num_samples // s_interferer_raw.shape[-1]) + 1
        s_interferer = s_interferer_raw.repeat(1, repeats)[:, :num_samples]
    else:
        s_interferer = s_interferer_raw[:, :num_samples]

    # 3. Create Equal Loudness (0 dB SIR) Phase-Aligned Mixture
    pow_target = torch.sum(s_target ** 2) + 1e-8
    pow_interferer = torch.sum(s_interferer ** 2) + 1e-8
    # Scale interferer so pow_target == pow_interferer (0 dB SIR)
    scale = torch.sqrt(pow_target / pow_interferer)
    s_interferer = s_interferer * scale
    mixture = s_target + s_interferer

    # Save reference WAVs for human ear verification
    mixture_path = scratch_dir / "user_overfit_mixture.wav"
    target_path = scratch_dir / "user_overfit_target.wav"
    interferer_out_path = scratch_dir / "user_overfit_interferer.wav"

    sf.write(str(mixture_path), mixture.squeeze().numpy(), sr)
    sf.write(str(target_path), s_target.squeeze().numpy(), sr)
    sf.write(str(interferer_out_path), s_interferer.squeeze().numpy(), sr)

    # 4. Load User's 256-d ECAPA-TDNN Embedding
    emb_path = Path("my_enrollment.bin")
    spk_embed = torch.from_numpy(SpeakerEncoder.load_binary(emb_path)).unsqueeze(0).float()
    print(f"Loaded enrollment vector: {emb_path} (Norm: {torch.norm(spk_embed):.2f})")

    # 5. Model, Loss, Optimizer
    device = "cpu"
    model = CausalTSEModel(num_freq_bins=257, embed_dim=256, hidden_dim=128, num_lstm_layers=2)
    loss_fn = CompositeTSELoss(lambda_snr=1.0, lambda_spec=15.0, lambda_vad=0.5)
    eval_snr = SNRLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=2e-4)

    win = torch.hann_window(320, periodic=True).sqrt()

    from dvec.pipeline.stft import CausalSTFT, CausalISTFT
    c_stft = CausalSTFT(frame_size=320, hop_size=160, fft_size=512)
    c_istft = CausalISTFT(frame_size=320, hop_size=160, fft_size=512)

    # Pre-compute STFTs using strictly causal sequence analysis matching streaming bit-for-bit
    mix_r, mix_i = c_stft.analyze_sequence(mixture)
    target_r, target_i = c_stft.analyze_sequence(s_target)

    # Pre-compute target PCM through causal filterbank for exact delay matching
    target_pcm = c_istft.synthesize_sequence(target_r, target_i, length=num_samples)

    # VAD target: active speech when target > -35 dBFS
    target_vad = (torch.abs(s_target) > 0.02).float()
    vad_frames = (torch.nn.functional.adaptive_avg_pool1d(target_vad.unsqueeze(1), mix_r.shape[-1]).squeeze(1) > 0.2).float()

    init_snr = -eval_snr(mixture, target_pcm).item()
    print(f"\nInitial Mixture SNR: {init_snr:.2f} dB (Equal loudness mixture)")
    print("Beginning 400-step overfit on user's real voice (Strict Causal Phase Parity)...\n" + "=" * 65)

    start_time = time.time()
    for step in range(1, 401):
        model.train()
        optimizer.zero_grad()

        mask_r, mask_i, pred_vad = model.forward_sequence(mix_r, mix_i, spk_embed)

        est_r = mix_r * mask_r - mix_i * mask_i
        est_i = mix_r * mask_i + mix_i * mask_r

        # Vectorized causal iSTFT synthesis matching StreamingAudioProcessor bit-for-bit
        est_pcm = c_istft.synthesize_sequence(est_r, est_i, length=num_samples)

        loss_dict = loss_fn(
            est_wav=est_pcm,
            target_wav=target_pcm,
            est_r=est_r,
            est_i=est_i,
            target_r=target_r,
            target_i=target_i,
            pred_vad=pred_vad,
            target_vad=vad_frames,
        )

        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0 or step == 1 or step == 400:
            with torch.no_grad():
                current_snr = -eval_snr(est_pcm, target_pcm).item()
                gain = current_snr - init_snr
                print(
                    f"Step [{step:3d}/400] | Loss: {loss_dict['total_loss'].item():6.2f} | "
                    f"Scale-Dep SNR: {current_snr:6.2f} dB (Gain: {gain:+6.2f} dB) | "
                    f"Spec: {loss_dict['spec_loss'].item():.3f} | VAD: {loss_dict['vad_loss'].item():.3f}"
                )

    print("=" * 65)
    print(f"Overfit completed in {time.time() - start_time:.1f}s.")

    # 6. Export to ONNX
    onnx_path = "dvec_v1.onnx"
    export_streaming_onnx(model, onnx_path)
    print(f"Exported overfit ONNX model to: {onnx_path}")

    # 7. Run Frame-by-Frame Streaming Inference through ONNX
    cleaned_out = scratch_dir / "user_overfit_cleaned.wav"
    run_offline_onnx_extraction(
        input_wav=mixture_path,
        embedding_path=emb_path,
        onnx_model_path=onnx_path,
        output_wav=cleaned_out,
    )
    print(f"\n[DONE] Cleaned audio written to: {cleaned_out}")
    print(f"Listen to:")
    print(f"  1. Mixture (both voices loud):     {mixture_path}")
    print(f"  2. Target (your real voice alone): {target_path}")
    print(f"  3. Extracted (cleaned output):     {cleaned_out}")


if __name__ == "__main__":
    main()
