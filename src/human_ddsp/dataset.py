import random
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import load_audio


class CsvAudioDataset(Dataset):
    def __init__(
            self, csv_path: Path, clips_root: Path, sample_rate: int, chunk_size: int, age_labels: tuple[str, ...], limit: int = 4
    ):
        self.sr = sample_rate
        self.chunk_size = chunk_size
        self.clips_root = clips_root
        self.num_age_classes = len(age_labels)

        df = pl.read_csv(csv_path, separator=",")
        if limit > 0:
            self.data = df.sample(n=limit, with_replacement=False)
        else:
            self.data = df

        self.age_to_idx = {label: i for i, label in enumerate(age_labels)}

    def __len__(self):
        return len(self.data) * 50

    def __getitem__(self, idx):
        row = self.data.row(idx % len(self.data), named=True)
        full_path = self.clips_root / row["path"]

        audio, sr = load_audio(self.sr, full_path)

        if audio.shape[-1] > self.chunk_size:
            start = random.randint(0, audio.shape[-1] - self.chunk_size)
            chunk = audio[0, start: start + self.chunk_size]
        else:
            chunk = F.pad(audio[0], (0, self.chunk_size - audio.shape[-1]))

        # chunk: [1, Time]
        age_idx = self.age_to_idx.get(row["age"], self.age_to_idx["thirties"])
        age_vec = F.one_hot(torch.tensor(age_idx), num_classes=self.num_age_classes).float()

        gender_vec = (
            torch.tensor([0.0, 1.0])
            if "female" in row["gender"]
            else torch.tensor([1.0, 0.0])
        )
        # Return static vectors [Dim]
        return chunk, gender_vec, age_vec
