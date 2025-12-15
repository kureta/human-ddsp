import torch
import torchaudio
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader

from .config import Config
from .dataset import CsvAudioDataset
from .model import VoiceDDSPTraining, AudioFeatureEncoder, MultiScaleMelLoss


def save_checkpoint(model, encoder, optimizer, step, config, path):
    torch.save({
        'global_step': step,
        'model_state': model.state_dict(),
        'encoder_state': encoder.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'config': config
    }, path)


def train_voice_ddsp(config: Config):
    device = config.device
    print(f"Training on: {device}")

    # 1. Models
    encoder = AudioFeatureEncoder(
        sample_rate=config.sample_rate, fft_size=config.fft_size, hop_length=config.hop_length,
        num_mels=config.num_mels, latent_dim=config.latent_dim, gru_units=config.gru_units,
        f0_min=config.f0_min, f0_max=config.f0_max, f0_threshold=config.f0_threshold
    ).to(device)

    model = VoiceDDSPTraining(
        sample_rate=config.sample_rate, hop_length=config.hop_length,
        rnn_channels=config.controller_hidden_dim, n_formants=config.n_formants,
        n_age_groups=len(config.age_labels), latent_dim=config.latent_dim,
        min_formant_width=config.min_formant_width, reverb_length=config.reverb_length,
        freq_offset=config.formant_freq_offset,
        bw_offset=config.formant_bw_offset
    ).to(device)

    # 2. Optimization
    optimizer = optim.AdamW(
        list(model.parameters()) + list(encoder.parameters()),
        lr=config.learning_rate
    )

    # Multi-Scale Mel Loss
    criterion = MultiScaleMelLoss(
        sample_rate=config.sample_rate,
        n_ffts=config.mel_fft_sizes,
        hop_lengths=config.mel_hop_lengths,
        win_lengths=config.mel_win_lengths,
        num_mels=config.num_mels
    ).to(device)

    # 3. Resume / Load State
    global_step = 0
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint to resume
    checkpoints = sorted(list(config.checkpoint_dir.glob("step_*.pt")), key=lambda p: int(p.stem.split('_')[1]))
    if checkpoints:
        latest = checkpoints[-1]
        print(f"Resuming from {latest}...")
        checkpoint = torch.load(latest, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        encoder.load_state_dict(checkpoint['encoder_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        global_step = checkpoint['global_step']

    # 4. Data
    dataset = CsvAudioDataset(
        csv_path=config.csv_path, clips_root=config.clips_dir, sample_rate=config.sample_rate,
        chunk_size=config.chunk_samples, age_labels=config.age_labels, limit=config.limit,
        epoch_factor=config.dataset_epoch_factor
    )

    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True
    )

    model.train()
    encoder.train()

    # 5. Infinite-ish Loop
    # We loop over epochs, but logic is driven by global_step
    print(f"Starting Training at step {global_step}...")

    running_loss = 0.0

    for epoch in range(config.num_epochs):
        # Using a simple iterator wrapper to avoid TQDM printing a new line every single batch.
        # We will update the description manually.
        progress_bar = tqdm.tqdm(dataloader, initial=global_step, total=len(dataloader) * config.num_epochs)

        for audio, gender, age in progress_bar:
            # Move Data
            audio = audio.to(device).unsqueeze(1)  # [B, 1, T]
            gender = gender.to(device)
            age = age.to(device)

            # --- Forward ---
            f0, loudness, latent_vector = encoder(audio)

            # Detach F0/Loudness (Don't train feature extraction on reconstruction loss)
            f0 = f0.detach()
            loudness = loudness.detach()

            # Expand Static Attrs
            num_frames = f0.shape[1]
            gender_expanded = gender.unsqueeze(1).repeat(1, num_frames, 1)
            age_expanded = age.unsqueeze(1).repeat(1, num_frames, 1)

            # Synthesize
            synthesized_audio, _ = model(f0, loudness, gender_expanded, age_expanded, latent_vector)

            # --- Loss Calculation ---
            # Trim to matching length
            output_samples = synthesized_audio.shape[1]
            target_audio = audio.transpose(1, 2)  # [B, T, 1]

            min_len = min(output_samples, target_audio.shape[1])
            synthesized_audio_trim = synthesized_audio[:, :min_len, :]
            target_audio_trim = target_audio[:, :min_len, :]

            loss = criterion(synthesized_audio_trim, target_audio_trim)

            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.grad_clip_norm)

            optimizer.step()

            # --- Updates ---
            global_step += 1
            running_loss += loss.item()

            # --- Per-Batch Logging ---
            if global_step % config.log_interval == 0:
                avg_loss = running_loss / config.log_interval
                progress_bar.set_description(f"Step {global_step} | Loss: {avg_loss:.4f}")

                # Optional: Add Tensorboard/WandB logging here
                # writer.add_scalar("Loss/train", avg_loss, global_step)

                running_loss = 0.0

            # --- Per-Batch Checkpointing ---
            if global_step % config.checkpoint_interval == 0:
                checkpoint_path = config.checkpoint_dir / f"step_{global_step}.pt"
                save_checkpoint(model, encoder, optimizer, global_step, config, checkpoint_path)

                # Save example output
                example_out_path = config.checkpoint_dir / f"step_{global_step}.wav"
                example_audio = synthesized_audio_trim[0].transpose(0, 1).detach().cpu()
                torchaudio.save(example_out_path, example_audio, config.sample_rate)


def main():
    config = Config()
    config.limit = 4
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    train_voice_ddsp(config)


if __name__ == '__main__':
    main()
