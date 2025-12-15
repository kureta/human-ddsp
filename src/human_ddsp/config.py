from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Age configuration
    age_labels: tuple[str, ...] = (
        "teens",
        "twenties",
        "thirties",
        "fourties",
        "fifties",
        "sixties",
        "seventies",
        "eighties",
    )

    # Audio processing parameters
    sample_rate: int = 48000
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 80
    n_filter_bins: int = 65

    # Model architecture parameters
    z_dim: int = 16
    gru_units: int = 256
    controller_hidden_dim: int = 512
    controller_n_layers: int = 3
    reverb_duration: float = 0.5

    # Loss configuration
    mel_fft_sizes: tuple[int, ...] = (4096, 2048, 1024, 512)
    mel_n_mels: tuple[int, ...] = (80, 80, 64, 40)
    mel_weight: float = 1.0
    log_mel_weight: float = 1.0
    mel_hop_ratio: float = 0.25
    mel_power: float = 1.0
    mel_f_min: float = 0.0
    mel_f_max: float | None = None
    mel_norm: str | None = "slaney"

    # Training parameters
    n_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    chunk_duration: float = 2.0  # seconds
    grad_clip_norm: float = 1.0

    # Logging and checkpointing
    n_checkpoints: int = 1000
    n_log: int = 100

    # Data paths
    csv_path: Path = Path("data/filtered_dataset.csv")
    clips_dir: Path = Path("/mnt/data/ai/cv-corpus-23.0-2025-09-05/tr/clips")
    checkpoint_dir: Path = Path("checkpoints")
    input_wav: Path = Path("/mnt/Data/Audio/misc/haiku.mp3")

    # Dataset parameters
    limit: int = 0  # 0 means no limit
    num_workers: int = 4

    # Device
    device: str = "cuda"  # Will be determined at runtime

    @property
    def num_age_classes(self) -> int:
        return len(self.age_labels)

    @property
    def chunk_samples(self) -> int:
        return int(self.chunk_duration * self.sample_rate)
