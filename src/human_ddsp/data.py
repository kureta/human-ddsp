import os
import random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import polars as pl
import torchaudio.transforms as T

from human_ddsp.config import AudioConfig
from human_ddsp.model import AGE_LABELS, NUM_AGE_CLASSES


class CsvAudioDataset(Dataset):
    """
    Dataset for loading audio clips from a CSV file.
    """
    def __init__(self, csv_path: str, clips_root: str, config: AudioConfig, limit: int = 0):
        self.config = config
        self.clips_root = clips_root
        self.chunk_size = int(config.clip_duration * config.sample_rate)

        df = pl.read_csv(csv_path)
        if limit > 0:
            self.data = df.sample(n=limit, shuffle=True)
        else:
            self.data = df

        self.age_to_idx = {label: i for i, label in enumerate(AGE_LABELS)}
        self.gender_map = {"male": [1.0, 0.0], "female": [0.0, 1.0]}

    def __len__(self) -> int:
        # We can oversample to make epochs shorter if the dataset is huge
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.data.row(idx, named=True)
        full_path = os.path.join(self.clips_root, row["path"])

        try:
            audio, sr = torchaudio.load(full_path)
        except Exception as e:
            print(f"Warning: Could not load {full_path}. Skipping. Error: {e}")
            # Return a dummy sample
            return torch.zeros(self.chunk_size), torch.tensor([1.0, 0.0]), F.one_hot(torch.tensor(2), num_classes=NUM_AGE_CLASSES).float()

        # Resample and convert to mono
        if sr != self.config.sample_rate:
            resampler = T.Resample(sr, self.config.sample_rate)
            audio = resampler(audio)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Normalize and chunk
        audio = audio.squeeze(0)
        audio = audio / (torch.abs(audio).max() + 1e-6)
        
        if audio.shape[-1] > self.chunk_size:
            start = random.randint(0, audio.shape[-1] - self.chunk_size)
            chunk = audio[start : start + self.chunk_size]
        else:
            chunk = F.pad(audio, (0, self.chunk_size - audio.shape[-1]))

        # --- Age Processing ---
        age_label = row["age"]
        age_idx = self.age_to_idx.get(age_label, self.age_to_idx["thirties"]) # Default
        age_vec = F.one_hot(torch.tensor(age_idx), num_classes=NUM_AGE_CLASSES).float()

        # --- Gender Processing ---
        gender_label = row.get("gender", "male") # Default to male if missing
        gender_vec = torch.tensor(self.gender_map.get(gender_label, [1.0, 0.0])).float()
        
        return chunk, gender_vec, age_vec


def create_dataloader(
    csv_path: str,
    clips_root: str,
    config: AudioConfig,
    limit: int = 0,
    num_workers: int = 4,
) -> DataLoader:
    """
    Creates a DataLoader for the audio dataset.
    """
    dataset = CsvAudioDataset(csv_path, clips_root, config, limit)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
