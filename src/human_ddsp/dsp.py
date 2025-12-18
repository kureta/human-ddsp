import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from human_ddsp.config import AudioConfig


class GlottalPulseSynth(nn.Module):
    """
    Generates a Rosenberg glottal pulse train.
    """
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.sample_rate = config.sample_rate

    def forward(
        self, 
        f0: torch.Tensor, 
        open_quotient: torch.Tensor,
        steepness: torch.Tensor,
        tilt: torch.Tensor
    ) -> torch.Tensor:
        """
        Generates a glottal pulse waveform based on f0 and open_quotient.
        
        Args:
            f0 (torch.Tensor): Fundamental frequency in Hz, shape [Batch, Time].
            open_quotient (torch.Tensor): Open quotient (0.1 to 0.9), shape [Batch, Time].
            steepness (torch.Tensor): Steepness of the closing phase, shape [Batch, Time].
            tilt (torch.Tensor): Spectral tilt (pre-emphasis coefficient), shape [Batch, Time].
                                 0.0 = flat (raw pulse), 1.0 = full derivative.
        
        Returns:
            torch.Tensor: Glottal pulse waveform, shape [Batch, Time].
        """
        # f0, OQ: [Batch, Time]
        # Calculate phase from f0
        # The cumulative sum of f0 / sample_rate gives the phase in cycles
        phase = torch.cumsum(f0 / self.sample_rate, dim=-1)
        p = phase - torch.floor(phase) # Normalize phase to [0, 1)

        # Clamp open_quotient to a reasonable range
        oq = torch.clamp(open_quotient, 0.1, 0.9)
        
        # Scale phase by open quotient for the rising part of the pulse
        scaled_phase = p / (oq + 1e-8) # Add epsilon to prevent division by zero

        # Rosenberg pulse generation
        # The pulse rises from 0 to 1 during the open phase
        pulse = 0.5 * (1.0 - torch.cos(np.pi * scaled_phase))
        
        # Create a mask to zero out the pulse during the closed phase
        # The sigmoid creates a smooth transition
        # Use the provided steepness parameter
        mask = torch.sigmoid((oq - p) * steepness)
        
        glottal_wave = pulse * mask

        # Variable Spectral Tilt (Leaky Differentiator)
        # y[n] = x[n] - tilt[n] * x[n-1]
        # This allows varying between the raw pulse (-12dB/oct) and the derivative (-6dB/oct)
        
        # We need to align tilt with the shifted version of glottal_wave
        # glottal_wave[..., :-1] is x[n-1]
        # tilt[..., 1:] corresponds to the coefficient at time n
        
        # Ensure tilt is in [0, 1] range just in case, though caller should handle it
        a = torch.clamp(tilt, 0.0, 1.0)
        
        diff_wave = glottal_wave[..., 1:] - a[..., 1:] * glottal_wave[..., :-1]
        
        # Pad the beginning with a zero to maintain the original length
        return F.pad(diff_wave, (1, 0))


class FormantFilter(nn.Module):
    """
    Applies formant filtering by constructing the frequency response in the FFT domain.
    """
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.sample_rate = config.sample_rate
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.n_formants = config.n_formants
        self.register_buffer("window", torch.hann_window(config.n_fft))

    def _get_formant_response(
        self,
        formant_freqs: torch.Tensor,  # [B, T, n_formants]
        formant_bandwidths: torch.Tensor, # [B, T, n_formants]
        formant_amplitudes: torch.Tensor, # [B, T, n_formants]
        n_fft: int,
        sample_rate: int,
    ) -> torch.Tensor: # [B, T, n_fft // 2 + 1]
        """
        Calculates the frequency response of a set of formants.
        """
        # Frequencies for the FFT bins
        freq_bins = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, device=formant_freqs.device)
        
        # Reshape for broadcasting: [1, 1, 1, n_fft // 2 + 1]
        freq_bins = freq_bins.view(1, 1, 1, -1)

        # Reshape formant parameters for broadcasting: [B, T, n_formants, 1]
        formant_freqs = formant_freqs.unsqueeze(-1)
        formant_bandwidths = formant_bandwidths.unsqueeze(-1)
        formant_amplitudes = formant_amplitudes.unsqueeze(-1)

        # [B, T, n_formants, n_fft // 2 + 1]
        diff_freq = (freq_bins - formant_freqs)
        bandwidth_half = formant_bandwidths / 2.0
        
        # Ensure bandwidth_half is not too small to avoid sharp peaks
        bandwidth_half = torch.clamp(bandwidth_half, min=10.0) # Minimum bandwidth of 10 Hz
        
        # Calculate the magnitude response for each formant
        magnitude_response_per_formant = formant_amplitudes * (bandwidth_half**2) / (diff_freq**2 + bandwidth_half**2 + 1e-8)
        
        # Sum up the contributions from all formants
        # [B, T, n_formants, n_fft // 2 + 1] -> [B, T, n_fft // 2 + 1]
        total_magnitude_response = torch.sum(magnitude_response_per_formant, dim=2)
        
        # Add a base gain to ensure it's not zero everywhere
        total_magnitude_response = total_magnitude_response + 1e-5
        
        return total_magnitude_response


    def forward(
        self,
        excitation: torch.Tensor, # [B, L]
        formant_freqs: torch.Tensor, # [B, T, n_formants]
        formant_bandwidths: torch.Tensor, # [B, T, n_formants]
        formant_amplitudes: torch.Tensor, # [B, T, n_formants]
    ) -> torch.Tensor: # [B, L]
        """
        Applies formant filtering to the excitation signal.
        
        Args:
            excitation (torch.Tensor): Input audio excitation, shape [Batch, Length].
            formant_freqs (torch.Tensor): Formant frequencies, shape [Batch, Time, n_formants].
            formant_bandwidths (torch.Tensor): Formant bandwidths, shape [Batch, Time, n_formants].
            formant_amplitudes (torch.Tensor): Formant amplitudes, shape [Batch, Time, n_formants].
        
        Returns:
            torch.Tensor: Filtered audio, shape [Batch, Length].
        """
        # 1. STFT of Excitation
        ex_stft = torch.stft(
            excitation,
            self.n_fft,
            self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        ) # [Batch, FreqBins, TimeSteps]

        # Ensure time steps match between STFT and formant parameters
        # The number of time steps in STFT is ex_stft.shape[2]
        # The number of time steps in formant parameters is formant_freqs.shape[1]
        # We need to interpolate the formant parameters if they don't match.
        
        # Calculate the target number of time steps for formant parameters
        target_time_steps = ex_stft.shape[2]
        
        # Interpolate formant parameters to match STFT time steps
        # [B, T_formant, n_formants] -> [B, T_stft, n_formants]
        
        # Reshape to [B * n_formants, 1, T_formant] for interpolation along time axis
        B, T_formant, N_formants = formant_freqs.shape
        
        formant_freqs_interp = F.interpolate(
            formant_freqs.view(B * N_formants, 1, T_formant),
            size=target_time_steps,
            mode="linear",
            align_corners=False,
        ).view(B, N_formants, target_time_steps).transpose(1, 2) # [B, T_stft, n_formants]
        
        formant_bandwidths_interp = F.interpolate(
            formant_bandwidths.view(B * N_formants, 1, T_formant),
            size=target_time_steps,
            mode="linear",
            align_corners=False,
        ).view(B, N_formants, target_time_steps).transpose(1, 2) # [B, T_stft, n_formants]
        
        formant_amplitudes_interp = F.interpolate(
            formant_amplitudes.view(B * N_formants, 1, T_formant),
            size=target_time_steps,
            mode="linear",
            align_corners=False,
        ).view(B, N_formants, target_time_steps).transpose(1, 2) # [B, T_stft, n_formants]

        # 2. Get Formant Frequency Response
        # [Batch, TimeSteps, FreqBins]
        formant_response = self._get_formant_response(
            formant_freqs_interp,
            formant_bandwidths_interp,
            formant_amplitudes_interp,
            self.n_fft,
            self.sample_rate,
        ).transpose(1, 2) # [Batch, FreqBins, TimeSteps] to match STFT

        # 3. Apply Filter in Frequency Domain
        output_stft = ex_stft * formant_response

        # 4. iSTFT
        output_audio = torch.istft(
            output_stft,
            self.n_fft,
            self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            length=excitation.shape[-1],
        )
        return output_audio


class LearnableReverb(nn.Module):
    """
    Applies learnable convolution reverb.
    """
    def __init__(self, config: AudioConfig, reverb_duration: float = 0.5):
        super().__init__()
        self.sample_rate = config.sample_rate
        n_samples = int(self.sample_rate * reverb_duration)
        
        # Initialize with a decaying noise impulse response
        decay = torch.exp(-torch.linspace(0, 5, n_samples))
        noise = torch.randn(n_samples) * 0.1
        initial_ir = noise * decay
        
        # Register as a learnable parameter
        self.impulse_response = nn.Parameter(initial_ir.view(1, 1, -1))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Applies reverb to the input audio.
        
        Args:
            audio (torch.Tensor): Input audio, shape [Batch, Length].
        
        Returns:
            torch.Tensor: Reverb-applied audio, shape [Batch, Length].
        """
        # Ensure audio is 3D: [Batch, Channels, Length]
        if audio.dim() == 2:
            x = audio.unsqueeze(1)
        else:
            x = audio

        batch, channels, dry_len = x.shape
        
        # Calculate FFT size for convolution
        fft_size = dry_len + self.impulse_response.shape[-1] - 1
        
        # Find the next power of 2 for efficient FFT
        n_fft_pow = torch.ceil(torch.log2(torch.tensor(float(fft_size))))
        n_fft = int(2 ** n_fft_pow.item())

        # Perform FFT-based convolution
        dry_fft = torch.fft.rfft(x, n=n_fft, dim=-1)
        ir_fft = torch.fft.rfft(self.impulse_response, n=n_fft, dim=-1)
        
        # Element-wise multiplication in frequency domain
        wet_audio = torch.fft.irfft(dry_fft * ir_fft, n=n_fft, dim=-1)

        # Trim to original dry_len and remove channel dimension
        return wet_audio.squeeze(1)[..., :dry_len]
