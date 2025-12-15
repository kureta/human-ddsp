import torch
import torchaudio
from torch import Tensor
from torchaudio import transforms as T


def load_audio(sample_rate: int, file_path: str) -> Tensor:
    audio, source_sample_rate = torchaudio.load(file_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if source_sample_rate != sample_rate:
        audio = T.Resample(source_sample_rate, sample_rate)(audio)
    return audio
