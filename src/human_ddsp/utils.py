import torch
import torchaudio
from torch import Tensor
from torchaudio import transforms as T


def load_audio(sample_rate: int, file_path: str) -> Tensor:
    audio, sr = torchaudio.load(file_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != sample_rate:
        audio = T.Resample(sr, sample_rate)(audio)
    return audio
