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
    fft_size: int = 2048
    hop_length: int = 512
    num_mels: int = 80

    # F0 estimation
    f0_min: float = 50.0
    f0_max: float = 1000.0
    f0_threshold: float = 0.3

    # Model architecture parameters
    latent_dim: int = 16
    gru_units: int = 256
    controller_hidden_dim: int = 512
    n_formants: int = 4
    min_formant_width: float = 50.0
    reverb_length: float = 1.0

    # Synthesis parameters
    formant_freq_offset: float = 200.0
    formant_bw_offset: float = 50.0

    # Loss configuration
    mel_fft_sizes: tuple[int, ...] = (4096, 2048, 1024, 512)
    mel_win_lengths: tuple[int, ...] = (4096, 2048, 1024, 512)
    mel_hop_lengths: tuple[int, ...] = (1024, 512, 256, 128)

    # Training parameters
    num_epochs: int = 100
    dataset_epoch_factor: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    chunk_duration: float = 2.0  # seconds
    grad_clip_norm: float = 1.0

    # Logging and checkpointing
    checkpoint_interval: int = 1000
    log_interval: int = 100

    # Data paths
    csv_path: Path = Path("data/filtered_dataset.csv")
    clips_dir: Path = Path("data/cv-corpus-23.0-2025-09-05/tr/clips")
    checkpoint_dir: Path = Path("checkpoints")

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
