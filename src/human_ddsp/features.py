import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from human_ddsp.config import AudioConfig


def _create_a_weighting(n_fft: int, sr: int) -> torch.Tensor:
    """Creates the A-weighting curve for Perceptual Loudness."""
    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    f_sq = freqs**2
    
    term1 = 12194.217**2 * f_sq**2
    term2 = (f_sq + 20.6**2) * torch.sqrt((f_sq + 107.7**2) * (f_sq + 737.9**2)) * (f_sq + 12194.217**2)
    gain = term1 / (term2 + 1e-8)
    
    return gain


class PitchDetector(nn.Module):
    """
    Detects fundamental frequency (f0) using a simplified autocorrelation method.
    """
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.sample_rate = config.sample_rate
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.min_f0 = 50
        self.max_f0 = 1000

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Estimates f0 for each frame.
        
        Args:
            audio (torch.Tensor): Input audio, shape [Batch, Length].
        
        Returns:
            torch.Tensor: Fundamental frequency in Hz, shape [Batch, Time, 1].
        """
        # Pad for framing
        pad = self.win_length // 2
        audio_pad = F.pad(audio.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
        
        # Create frames
        frames = audio_pad.unfold(dimension=-1, size=self.win_length, step=self.hop_length)
        
        # Autocorrelation via FFT
        n_fft_corr = 2 * self.win_length
        spec = torch.fft.rfft(frames, n=n_fft_corr, dim=-1)
        power_spec = spec.abs().pow(2)
        autocorr = torch.fft.irfft(power_spec, n=n_fft_corr, dim=-1)
        autocorr = autocorr[..., :self.win_length]

        # Normalize autocorrelation
        norm_factor = autocorr[..., 0:1] + 1e-8
        norm_autocorr = autocorr / norm_factor

        # Define search range for f0 in terms of lag
        min_lag = int(self.sample_rate / self.max_f0)
        max_lag = int(self.sample_rate / self.min_f0)

        # Find the peak in the search region
        search_region = norm_autocorr[..., min_lag:max_lag]
        max_val, max_idx = torch.max(search_region, dim=-1)
        
        # Convert lag index to f0
        true_lag = max_idx + min_lag
        f0 = self.sample_rate / (true_lag.float() + 1e-8)

        # Unvoice threshold
        f0[max_val < 0.3] = 0.0
        
        return f0.unsqueeze(-1)


class LoudnessDetector(nn.Module):
    """
    Calculates perceptual loudness in dB using A-weighting.
    """
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.spectrogram = T.Spectrogram(n_fft=config.n_fft, hop_length=config.hop_length, power=2.0)
        self.register_buffer("a_weighting", _create_a_weighting(config.n_fft, config.sample_rate))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Calculates loudness for each frame.
        
        Args:
            audio (torch.Tensor): Input audio, shape [Batch, Length].
        
        Returns:
            torch.Tensor: Loudness in dB, shape [Batch, Time, 1].
        """
        spec = self.spectrogram(audio) + 1e-8
        
        # Apply A-weighting
        weighted_spec = spec * self.a_weighting.view(1, -1, 1)
        
        # Calculate mean power and convert to dB
        mean_power = torch.mean(weighted_spec, dim=1, keepdim=True)
        loudness_db = 10 * torch.log10(mean_power)
        
        return loudness_db.transpose(1, 2)


class MelContentExtractor(nn.Module):
    """
    Extracts content features using Log-Mel Spectrograms with Instance Normalization.
    """
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.f_min,
            f_max=config.f_max,
            power=2.0
        )
        
        # Instance Norm to remove static timbre/channel effects
        # We normalize over the time dimension for each frequency bin
        self.instance_norm = nn.InstanceNorm1d(config.n_mels, affine=False)
        
        # Simple projection to match content_dim
        self.projection = nn.Linear(config.n_mels, config.content_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extracts content features from audio.
        
        Args:
            audio (torch.Tensor): Input audio, shape [Batch, Length].
        
        Returns:
            torch.Tensor: Content embedding, shape [Batch, Time, content_dim].
        """
        # 1. Mel Spectrogram
        mels = self.mel_transform(audio) # [B, n_mels, T]
        
        # 2. Log Compression
        log_mels = torch.log(mels + 1e-5)
        
        # 3. Instance Normalization
        # InstanceNorm1d expects [B, Channels, Length]
        # It normalizes across Length (Time) for each Channel (Freq Bin)
        norm_mels = self.instance_norm(log_mels)
        
        # 4. Transpose for Linear Projection
        # [B, n_mels, T] -> [B, T, n_mels]
        x = norm_mels.transpose(1, 2)
        
        # 5. Linear Projection
        content_embedding = self.projection(x) # [B, T, content_dim]
        
        return content_embedding
