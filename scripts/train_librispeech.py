"""Multi-Speaker Training on LibriSpeech dev-clean Corpus.

Trains the compressed CausalTSEModel (569K parameters, 2.33 MB) on genuine human
speech mixtures, with disjoint speaker partitions (32 train speakers, 8 val speakers)
and SpeechBrain ECAPA-TDNN conditioning.
"""

import time
from pathlib import Path
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dvec.pipeline.model import CausalTSEModel
from dvec.data.dataset import DynamicTSEDataset
from dvec.enrollment.encoder import SpeakerEncoder
from dvec.train.trainer import TSETrainer
from dvec.export.onnx_exporter import export_streaming_onnx
from dvec.utils.audio import load_audio_mono


def precompute_speaker_embeddings(speaker_files: dict, encoder: SpeakerEncoder) -> dict:
    print(f"Precomputing reference embeddings for {len(speaker_files)} speakers with ECAPA-TDNN...")
    embeddings = {}
    for spk_id, files in speaker_files.items():
        # Encode first reference utterance for this speaker
        wav, sr = load_audio_mono(files[0], target_sr=16000)
        emb = encoder.encode(wav).squeeze(0)  # (256,)
        embeddings[spk_id] = emb
    print("Precomputation complete.")
    return embeddings


def main():
    data_dir = Path("data/LibriSpeech/dev-clean")
    if not data_dir.exists():
        raise FileNotFoundError(f"LibriSpeech data directory not found: {data_dir}")

    # Deterministic speaker partition: 32 train, 8 validation
    all_speakers = sorted(list(set(
        p.name.split("-")[0] for p in data_dir.rglob("*.flac") if "-" in p.name
    )))
    print(f"Found {len(all_speakers)} total speakers in {data_dir}.")
    
    random.seed(42)
    shuffled = list(all_speakers)
    random.shuffle(shuffled)
    train_speakers = shuffled[:32]
    val_speakers = shuffled[32:]
    print(f"Partition: {len(train_speakers)} train speakers | {len(val_speakers)} validation speakers (disjoint).")

    # Initialize SpeechBrain ECAPA-TDNN encoder
    encoder = SpeakerEncoder(use_fallback=False, device="cpu")

    # Index files
    full_ds = DynamicTSEDataset(audio_dir=data_dir)
    speaker_embeddings = precompute_speaker_embeddings(full_ds.speaker_files, encoder)

    # Datasets
    train_ds = DynamicTSEDataset(
        audio_dir=data_dir,
        segment_duration=3.0,
        samples_per_epoch=100,  # 25 batches of 4
        sir_range=(-3.0, 10.0),
        embedding_jitter_std=0.03,
        speaker_whitelist=train_speakers,
        speaker_embeddings=speaker_embeddings,
    )

    val_ds = DynamicTSEDataset(
        audio_dir=data_dir,
        segment_duration=3.0,
        samples_per_epoch=40,  # 10 batches of 4
        sir_range=(0.0, 6.0),
        embedding_jitter_std=0.00,
        speaker_whitelist=val_speakers,
        speaker_embeddings=speaker_embeddings,
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    # Initialize compressed CausalTSEModel (569K parameters)
    model = CausalTSEModel(num_freq_bins=257, embed_dim=256, hidden_dim=128, num_lstm_layers=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized: {total_params:,} parameters ({total_params * 4 / (1024*1024):.2f} MB in FP32).")

    trainer = TSETrainer(model=model, device="cpu", learning_rate=1.5e-3)
    num_epochs = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=num_epochs, eta_min=1e-4)

    best_val_snr = -999.0
    best_epoch = 0
    output_onnx = "dvec_v1.onnx"
    start_time = time.time()

    print("\nStarting Human Multi-Speaker Training on CPU...")
    print("=" * 78)

    for epoch in range(1, num_epochs + 1):
        ep_t0 = time.time()
        t_metrics = trainer.train_epoch(train_loader)
        v_metrics = trainer.evaluate(val_loader)
        scheduler.step()
        ep_duration = time.time() - ep_t0

        val_snr = v_metrics["snr_db"]
        improved = ""
        if val_snr > best_val_snr:
            best_val_snr = val_snr
            best_epoch = epoch
            export_streaming_onnx(model, output_onnx)
            improved = " [BEST MODEL SAVED]"

        print(
            f"Epoch [{epoch:2d}/{num_epochs:2d}] ({ep_duration:4.1f}s) | "
            f"Train SNR: {t_metrics['snr_db']:5.2f} dB | "
            f"Spec Loss: {t_metrics['spec_loss']:.3f} | "
            f"Val SNR: {v_metrics['snr_db']:5.2f} dB (dSNR: {v_metrics['improvement_db']:+5.2f} dB){improved}"
        )

    total_time = time.time() - start_time
    print("=" * 78)
    print(f"Training completed in {total_time/60:.1f} minutes.")
    print(f"Best Epoch: {best_epoch} with Validation Scale-Dependent SNR: {best_val_snr:.2f} dB")
    print(f"Exported locked ground-truth ONNX to: {output_onnx} ({Path(output_onnx).stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
