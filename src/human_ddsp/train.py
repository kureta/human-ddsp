import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import random

from human_ddsp.config import AudioConfig
from human_ddsp.data import create_dataloader
from human_ddsp.model import VoiceSynth, AGE_LABELS


class MultiScaleMelLoss(nn.Module):
    """
    Computes multi-scale mel-spectrogram loss.
    """
    def __init__(self, config: AudioConfig, fft_sizes=None, n_mels=None):
        super().__init__()
        self.config = config
        if fft_sizes is None:
            fft_sizes = [2048, 1024, 512]
        if n_mels is None:
            n_mels = [128, 64, 32]

        self.transforms = nn.ModuleList()
        for n_fft, n_mel in zip(fft_sizes, n_mels):
            hop_length = n_fft // 4
            self.transforms.append(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=config.sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mel,
                    f_min=config.f_min,
                    f_max=config.f_max,
                )
            )

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        eps = 1e-7
        for t in self.transforms:
            mel_pred = t(x_pred)
            mel_target = t(x_target)
            
            loss += nn.functional.l1_loss(mel_pred, mel_target)
            loss += nn.functional.l1_loss(
                torch.log(mel_pred + eps), torch.log(mel_target + eps)
            )
        return loss


def train(
    csv_path: str,
    clips_root: str,
    checkpoint_dir: str,
    n_epochs: int = 100,
    limit: int = 0,
    n_log: int = 100,
    n_checkpoints: int = 1000,
):
    """
    Main training loop.
    """
    config = AudioConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Setup ---
    model = VoiceSynth(config).to(device)
    dataloader = create_dataloader(csv_path, clips_root, config, limit)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = MultiScaleMelLoss(config).to(device)

    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Starting Training on {device}...")
    
    num_batches = 0
    for epoch in range(1, n_epochs + 1):
        batch_loss_accum = 0.0
        for i, (audio, gender, age) in enumerate(dataloader):
            num_batches += 1
            
            audio = audio.to(device)
            gender = gender.to(device)
            age = age.to(device)

            optimizer.zero_grad()

            # Forward pass
            wet_audio, _, _ = model(audio, gender, age)

            # Loss calculation
            loss = criterion(wet_audio, audio)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_loss_accum += loss.item()

            # --- Batch-Level Logging ---
            if num_batches % n_log == 0:
                avg_loss = batch_loss_accum / n_log
                print(f"Epoch {epoch} | Batch {num_batches} | Loss: {avg_loss:.4f}")
                batch_loss_accum = 0.0

            # --- Batch-Level Checkpointing ---
            if num_batches % n_checkpoints == 0:
                # 1. Save Model
                save_path = f"{checkpoint_dir}/model_step{num_batches}.pth"
                torch.save(model.state_dict(), save_path)
                save_path_last = f"{checkpoint_dir}/model_last.pth"
                torch.save(model.state_dict(), save_path_last)

                # 2. Save Random Reconstruction
                model.eval()
                with torch.no_grad():
                    ridx = random.randint(0, audio.shape[0] - 1)
                    tgt_audio = audio[ridx].unsqueeze(0)
                    
                    # Get the corresponding gender and age
                    tgt_gender = gender[ridx].unsqueeze(0)
                    tgt_age = age[ridx].unsqueeze(0)
                    
                    recon_audio, _, _ = model(tgt_audio, tgt_gender, tgt_age)

                    g_str = "Female" if torch.argmax(tgt_gender).item() == 1 else "Male"
                    age_str = AGE_LABELS[torch.argmax(tgt_age).item()]

                    fname_base = f"{checkpoint_dir}/step{num_batches}_{g_str}_{age_str}"
                    
                    # Save target and reconstruction
                    torchaudio.save(f"{fname_base}_target.wav", tgt_audio.cpu(), config.sample_rate)
                    torchaudio.save(f"{fname_base}_recon.wav", recon_audio.cpu(), config.sample_rate)
                
                model.train()
                print(f"--> Saved Checkpoint and Reconstruction: {save_path}")
