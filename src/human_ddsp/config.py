from dataclasses import dataclass

@dataclass
class AudioConfig:
    sample_rate: int = 48000
    n_fft: int = 2048
    hop_length: int = 240  # 5ms at 48kHz (high temporal resolution for control)
    win_length: int = 2048
    
    # Feature Extraction
    n_mfcc: int = 30
    mfcc_keep_start: int = 1  # Drop 0th (energy)
    mfcc_keep_end: int = 16   # Drop higher (pitch info)
    n_mels: int = 80
    f_min: float = 20.0
    f_max: float = 8000.0
    
    # Content Encoder
    content_dim: int = 128  # Increased from 64
    content_kernel_size: int = 3
    
    # Controller
    hidden_size: int = 512  # Increased from 256
    num_layers: int = 2
    
    # Synth
    n_formants: int = 6
    n_filter_bins: int = 65 # For general spectral shaping if needed, or formant resolution
    
    # Training
    clip_duration: float = 2.0
    batch_size: int = 16
    learning_rate: float = 1e-4
