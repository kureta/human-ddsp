import time

import torch
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from .config import Config
from .dataset import CsvAudioDataset
from .model import VoiceDDSP_Training, AudioFeatureEncoder, MultiScaleMelLoss


def save_checkpoint(model, encoder, optimizer, step, cfg, path):
    torch.save({
        'global_step': step,
        'model_state': model.state_dict(),
        'encoder_state': encoder.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'config': cfg
    }, path)


def train_voice_ddsp(cfg: Config):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 1. Models
    encoder = AudioFeatureEncoder(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_mels, z_dim=cfg.z_dim, gru_units=cfg.gru_units
    ).to(device)

    model = VoiceDDSP_Training(
        sample_rate=cfg.sample_rate, hop_length=cfg.hop_length,
        rnn_channels=cfg.controller_hidden_dim, n_formants=cfg.n_formants,
        n_age_groups=len(cfg.age_labels), k_z_dim=cfg.z_dim,
        min_formant_width=cfg.min_formant_width, reverb_length=cfg.reverb_length,
    ).to(device)

    # 2. Optimization
    optimizer = optim.AdamW(
        list(model.parameters()) + list(encoder.parameters()),
        lr=cfg.learning_rate
    )

    # Multi-Scale Mel Loss
    criterion = MultiScaleMelLoss(
        sample_rate=cfg.sample_rate,
        n_ffts=cfg.mel_fft_sizes,
        hop_lengths=[1024, 512, 256, 128],
        win_lengths=[4096, 2048, 1024, 512],
        n_mels=80
    ).to(device)

    # 3. Resume / Load State
    global_step = 0
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint to resume
    checkpoints = sorted(list(cfg.checkpoint_dir.glob("step_*.pt")), key=lambda p: int(p.stem.split('_')[1]))
    if checkpoints:
        latest = checkpoints[-1]
        print(f"Resuming from {latest}...")
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        encoder.load_state_dict(ckpt['encoder_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        global_step = ckpt['global_step']

    # 4. Data
    dataset = CsvAudioDataset(
        csv_path=cfg.csv_path, clips_root=cfg.clips_dir, sample_rate=cfg.sample_rate,
        chunk_size=cfg.chunk_samples, age_labels=cfg.age_labels, limit=cfg.limit
    )

    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )

    model.train()
    encoder.train()

    # 5. Infinite-ish Loop
    # We loop over epochs but logic is driven by global_step
    print(f"Starting Training at step {global_step}...")

    running_loss = 0.0
    start_time = time.time()

    for epoch in range(cfg.n_epochs):
        # Using a simple iterator wrapper to avoid TQDM printing a new line every single batch
        # We will update the description manually.
        pbar = tqdm.tqdm(dataloader, initial=global_step, total=len(dataloader) * cfg.n_epochs)

        for audio, gender, age in pbar:
            # Move Data
            audio = audio.to(device).unsqueeze(1)  # [B, 1, T]
            gender = gender.to(device)
            age = age.to(device)

            # --- Forward ---
            f0, loudness, z = encoder(audio)

            # Detach F0/Loudness (Don't train feature extraction on reconstruction loss)
            f0 = f0.detach()
            loudness = loudness.detach()

            # Expand Static Attrs
            num_frames = f0.shape[1]
            gender_expanded = gender.unsqueeze(1).repeat(1, num_frames, 1)
            age_expanded = age.unsqueeze(1).repeat(1, num_frames, 1)

            # Synthesize
            audio_hat, _ = model(f0, loudness, gender_expanded, age_expanded, z)

            # --- Loss Calculation ---
            # Trim to matching length
            output_samples = audio_hat.shape[1]
            target_audio = audio.transpose(1, 2)  # [B, T, 1]

            min_len = min(output_samples, target_audio.shape[1])
            audio_hat_trim = audio_hat[:, :min_len, :]
            target_audio_trim = target_audio[:, :min_len, :]

            loss = criterion(audio_hat_trim, target_audio_trim)

            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.grad_clip_norm)

            optimizer.step()

            # --- Updates ---
            global_step += 1
            running_loss += loss.item()

            # --- Per-Batch Logging ---
            if global_step % cfg.n_log == 0:
                avg_loss = running_loss / cfg.n_log
                elapsed = time.time() - start_time
                pbar.set_description(f"Step {global_step} | Loss: {avg_loss:.4f}")

                # Optional: Add Tensorboard/WandB logging here
                # writer.add_scalar("Loss/train", avg_loss, global_step)

                running_loss = 0.0
                start_time = time.time()

            # --- Per-Batch Checkpointing ---
            if global_step % cfg.n_checkpoints == 0:
                ckpt_path = cfg.checkpoint_dir / f"step_{global_step}.pt"
                save_checkpoint(model, encoder, optimizer, global_step, cfg, ckpt_path)
                # print(f" Saved: {ckpt_path.name}") # Optional, can clutter output
