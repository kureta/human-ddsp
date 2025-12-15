import random
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import load_audio


class CsvAudioDataset(Dataset):
    def __init__(
            self, csv_path: Path, clips_root: Path, sample_rate: int, chunk_size: int,
            age_labels: tuple[str, ...], limit: int = 4, epoch_factor: int = 50
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.clips_root = clips_root
        self.num_age_classes = len(age_labels)
        self.epoch_factor = epoch_factor

        dataframe = pl.read_csv(csv_path, separator=",")
        if limit > 0:
            self.data = dataframe.sample(n=limit, with_replacement=False)
        else:
            self.data = dataframe

        self.age_to_idx = {label: i for i, label in enumerate(age_labels)}

    def __len__(self):
        return len(self.data) * self.epoch_factor

    def __getitem__(self, index):
        row = self.data.row(index % len(self.data), named=True)
        full_path = self.clips_root / row["path"]

        audio = load_audio(self.sample_rate, full_path)

        if audio.shape[-1] > self.chunk_size:
            start_sample = random.randint(0, audio.shape[-1] - self.chunk_size)
            chunk = audio[0, start_sample: start_sample + self.chunk_size]
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
