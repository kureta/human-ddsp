import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


def nearest_power_of_two(value):
    power = int(np.round(np.log2(value)))
    return 2 ** power


def create_a_weighting(fft_size: int, sample_rate: int) -> torch.Tensor:
    """
    Creates the A-weighting curve for perceptual loudness following IEC 61672-1.
    """
    freqs = torch.linspace(0, sample_rate / 2, fft_size // 2 + 1)
    freq_squared = freqs ** 2

    # Constants from IEC 61672-1
    cutoff_1 = 20.6
    cutoff_2 = 107.7
    cutoff_3 = 737.9
    cutoff_4 = 12194.217

    numerator = (cutoff_4 ** 2) * (freq_squared ** 2)
    denominator = (
            (freq_squared + cutoff_1 ** 2)
            * torch.sqrt(freq_squared + cutoff_2 ** 2)
            * torch.sqrt(freq_squared + cutoff_3 ** 2)
            * (freq_squared + cutoff_4 ** 2)
    )

    gain = numerator / (denominator + 1e-10)
    return gain


# ==============================================================================
# 1. SHARED CONTROLLER (The Brain)
# ==============================================================================

class DDSPController(nn.Module):
    """
    Neural Controller. Shared by both Training and Real-Time models.
    """

    def __init__(self,
                 rnn_channels: int = 512,
                 n_formants: int = 4,
                 n_age_groups: int = 8,
                 latent_dim: int = 16,
                 freq_offset: float = 200.0,
                 bw_offset: float = 50.0):
        super().__init__()
        self.num_formants = n_formants
        self.rnn_channels = rnn_channels
        self.freq_offset = freq_offset
        self.bw_offset = bw_offset

        # Input: f0 (1) + loudness (1) + gender (2) + age (N) + z (K)
        input_dim = 1 + 1 + 2 + n_age_groups + latent_dim

        self.input_layer = nn.Linear(input_dim, rnn_channels)
        self.gru = nn.GRU(rnn_channels, rnn_channels, batch_first=True)

        self.head_oq = nn.Linear(rnn_channels, 1)
        self.head_rise_ratio = nn.Linear(rnn_channels, 1)
        self.head_v_gain = nn.Linear(rnn_channels, 1)
        self.head_u_gain = nn.Linear(rnn_channels, 1)
        self.head_formant_freqs = nn.Linear(rnn_channels, n_formants)
        self.head_formant_bws = nn.Linear(rnn_channels, n_formants)

    def forward(self,
                f0_centered: torch.Tensor,
                loudness: torch.Tensor,
                gender: torch.Tensor,
                age: torch.Tensor,
                latent_vector: torch.Tensor,
                hidden_state: torch.Tensor = None):
        inputs = torch.cat([f0_centered, loudness, gender, age, latent_vector], dim=-1)

        hidden = F.relu(self.input_layer(inputs))
        hidden, new_hidden = self.gru(hidden, hidden_state)

        open_quotient = torch.sigmoid(self.head_oq(hidden))
        rise_ratio = torch.sigmoid(self.head_rise_ratio(hidden))
        voiced_gain = F.softplus(self.head_v_gain(hidden))
        unvoiced_gain = F.softplus(self.head_u_gain(hidden))

        freq_deltas = F.softplus(self.head_formant_freqs(hidden))
        formant_freqs = torch.cumsum(freq_deltas, dim=-1) + self.freq_offset
        formant_bandwidths = F.softplus(self.head_formant_bws(hidden)) + self.bw_offset

        controls = {
            'open_quotient': open_quotient,
            'rise_ratio': rise_ratio,
            'voiced_gain': voiced_gain,
            'unvoiced_gain': unvoiced_gain,
            'formant_freqs': formant_freqs,
            'formant_widths': formant_bandwidths
        }
        return controls, new_hidden


# ==============================================================================
# 2. TRAINING COMPONENTS (Parallel / FIR)
# ==============================================================================

class RosenbergSource(nn.Module):
    """Vectorized Glottal Pulse for Training"""

    def __init__(self, sample_rate: int, hop_length: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.register_buffer('local_ramp', torch.arange(0, hop_length, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer('pi', torch.tensor(math.pi))

    def forward(self, f0: torch.Tensor, open_quotient: torch.Tensor, rise_ratio: torch.Tensor) -> torch.Tensor:
        # Phase Integration
        cycles_per_frame = (f0 * self.hop_length) / self.sample_rate
        cumulative_cycles = torch.cumsum(cycles_per_frame, dim=1)
        frame_start_phase = cumulative_cycles - cycles_per_frame
        frame_start_phase = frame_start_phase - torch.floor(frame_start_phase)
        local_phase_growth = f0 * (self.local_ramp / self.sample_rate)

        total_phase = frame_start_phase + local_phase_growth
        instantaneous_phase = total_phase - torch.floor(total_phase)

        # Rosenberg C Logic
        epsilon = 1e-6
        peak_phase = open_quotient * rise_ratio

        mask_rising = instantaneous_phase < peak_phase
        mask_falling = (instantaneous_phase >= peak_phase) & (instantaneous_phase < open_quotient)

        waveform = torch.zeros_like(instantaneous_phase)

        # Optimization: Remove if statements by calculating math everywhere and using where.
        # 1. Rising Phase Calculation
        rise_val = 0.5 * (1.0 - torch.cos(self.pi * instantaneous_phase / (peak_phase + epsilon)))
        waveform = torch.where(mask_rising, rise_val, waveform)

        # 2. Falling Phase Calculation
        fall_dur = open_quotient - peak_phase
        phase_rel = instantaneous_phase - peak_phase
        fall_val = torch.cos((self.pi * phase_rel) / (2.0 * fall_dur + epsilon))
        waveform = torch.where(mask_falling, fall_val, waveform)

        return waveform


class FormantFilterFIR(nn.Module):
    """Parallel FIR Convolution for Training"""

    def __init__(self, sample_rate: int, min_width: float = 50.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.kernel_size = nearest_power_of_two((2.2 / min_width) * sample_rate)

        self.register_buffer('pi', torch.tensor(math.pi))
        self.register_buffer('two_pi', torch.tensor(2.0 * math.pi))
        self.register_buffer('ir_time', torch.arange(0, self.kernel_size, dtype=torch.float32).view(1, 1, -1))

    def generate_impulse_responses(self, center_freqs, bandwidths):
        batch, frames, num_formants = center_freqs.shape
        cf_flat = center_freqs.view(-1, num_formants, 1)
        bw_flat = bandwidths.view(-1, num_formants, 1)

        time_grid = self.ir_time / self.sample_rate
        impulse_response = torch.exp(-self.pi * bw_flat * time_grid) * torch.sin(self.two_pi * cf_flat * time_grid)

        combined = impulse_response.sum(dim=1, keepdim=True)
        return combined

    def forward(self, excitation: torch.Tensor, center_freqs: torch.Tensor, bandwidths: torch.Tensor):
        batch, frames, hop = excitation.shape
        filters = self.generate_impulse_responses(center_freqs, bandwidths).squeeze(2)

        n_fft = nearest_power_of_two(hop + self.kernel_size - 1)

        exc_flat = excitation.reshape(batch * frames, hop)
        filt_flat = filters.reshape(batch * frames, -1)

        exc_fft = torch.fft.rfft(exc_flat, n=n_fft)
        filt_fft = torch.fft.rfft(filt_flat, n=n_fft)

        out_fft = exc_fft * filt_fft
        out_time = torch.fft.irfft(out_fft, n=n_fft)

        output_len = out_time.shape[-1]
        out_time = out_time.view(batch, frames, output_len).transpose(1, 2)

        total_len = (frames - 1) * hop + output_len
        reconstructed = F.fold(
            out_time,
            output_size=(1, total_len),
            kernel_size=(1, output_len),
            stride=(1, hop)
        )
        return reconstructed[..., :frames * hop].transpose(1, 2)


class LearnableReverb(nn.Module):
    """FFT Convolution Reverb (Training Only)"""

    def __init__(self, sample_rate: int, reverb_length: float = 1.0):
        super().__init__()
        self.reverb_len = int(sample_rate * reverb_length)

        decay = torch.exp(-torch.linspace(0, 5, self.reverb_len))
        noise = torch.randn(self.reverb_len) * decay
        self.impulse_response = nn.Parameter(noise.view(1, 1, -1))
        self.wet_mix_logits = nn.Parameter(torch.tensor(-1.5))

    def forward(self, input_audio):
        input_transposed = input_audio.transpose(1, 2)
        impulse_response = self.impulse_response

        fft_size = input_transposed.shape[-1] + self.reverb_len - 1
        input_spectrum = torch.fft.rfft(input_transposed, n=fft_size)
        ir_spectrum = torch.fft.rfft(impulse_response, n=fft_size)
        wet_audio = torch.fft.irfft(input_spectrum * ir_spectrum, n=fft_size)[..., :input_transposed.shape[-1]]

        wet = torch.sigmoid(self.wet_mix_logits)
        out = (1.0 - wet) * input_transposed + wet * wet_audio
        return out.transpose(1, 2)


# ==============================================================================
# 3. REAL-TIME COMPONENTS (Stateful / IIR)
# ==============================================================================

class StatefulRosenberg(nn.Module):
    """Sample-based Oscillator for Real-Time"""

    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.register_buffer('pi', torch.tensor(math.pi))

    def forward(self, f0: torch.Tensor, open_quotient: torch.Tensor, rise_ratio: torch.Tensor, phase_state: torch.Tensor):
        phase_step = f0 / self.sample_rate
        curr_phase = torch.cumsum(phase_step, dim=1) + phase_state
        next_state = curr_phase[:, -1:, :].detach()
        phase = curr_phase - torch.floor(curr_phase)

        time_peak = open_quotient * rise_ratio
        mask_rise = phase < time_peak
        mask_fall = (phase >= time_peak) & (phase < open_quotient)

        waveform = torch.zeros_like(phase)
        waveform = torch.where(mask_rise, 0.5 * (1.0 - torch.cos(self.pi * phase / (time_peak + 1e-6))), waveform)

        time_fall = open_quotient - time_peak
        waveform = torch.where(mask_fall, torch.cos(self.pi * (phase - time_peak) / (2.0 * time_fall + 1e-6)), waveform)

        return waveform, next_state


class StatefulFormantFilter(nn.Module):
    """IIR Filter Loop for Real-Time"""

    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.register_buffer('two_pi', torch.tensor(2.0 * math.pi))

    def forward(self, excitation, center_freqs, bandwidths, states):
        pole_radius = torch.exp(-torch.pi * bandwidths / self.sample_rate)
        theta = self.two_pi * center_freqs / self.sample_rate
        coeff_1 = -2.0 * pole_radius * torch.cos(theta)
        coeff_2 = pole_radius * pole_radius
        gain = torch.sin(theta)

        x_scaled = excitation.repeat(1, 1, center_freqs.shape[-1]) * gain
        return self.iir_loop(x_scaled, coeff_1, coeff_2, states)

    @torch.jit.export
    def iir_loop(self, input_signal, coeff_1, coeff_2, states):
        state_1 = states[:, :, 0]
        state_2 = states[:, :, 1]
        outputs = torch.jit.annotate(torch.Tensor, torch.empty_like(input_signal))

        for t in range(input_signal.shape[1]):
            current_output = input_signal[:, t] - (coeff_1[:, t] * state_1) - (coeff_2[:, t] * state_2)
            outputs[:, t] = current_output
            state_2 = state_1
            state_1 = current_output

        return outputs.sum(dim=-1, keepdim=True), torch.stack([state_1, state_2], dim=2)


# ==============================================================================
# 4. WRAPPERS (The Top-Level Classes)
# ==============================================================================

class VoiceDDSPTraining(nn.Module):
    """
    Use THIS class for Training.
    Expects Separate Arguments.
    """

    def __init__(
            self,
            sample_rate=24000,
            hop_length=256,
            rnn_channels=512,
            n_formants=4,
            n_age_groups=8,
            latent_dim=16,
            min_formant_width=50.0,
            reverb_length=1.0,
            freq_offset=200.0,
            bw_offset=50.0
    ):
        super().__init__()
        self.hop_length = hop_length

        self.controller = DDSPController(
            rnn_channels=rnn_channels,
            n_formants=n_formants,
            n_age_groups=n_age_groups,
            latent_dim=latent_dim,
            freq_offset=freq_offset,
            bw_offset=bw_offset
        )

        self.source = RosenbergSource(sample_rate, hop_length)
        self.filter = FormantFilterFIR(sample_rate, min_width=min_formant_width)
        self.reverb = LearnableReverb(sample_rate, reverb_length=reverb_length)

    def forward(self,
                f0: torch.Tensor,
                loudness: torch.Tensor,
                gender: torch.Tensor,
                age: torch.Tensor,
                latent_vector: torch.Tensor):
        f0_centered = f0 - f0.mean(dim=1, keepdim=True)

        controls, _ = self.controller(f0_centered, loudness, gender, age, latent_vector)

        voiced = self.source(f0, controls['open_quotient'], controls['rise_ratio'])
        noise = torch.randn_like(voiced)
        excitation = (voiced * controls['voiced_gain']) + (noise * controls['unvoiced_gain'])

        dry_audio = self.filter(excitation, controls['formant_freqs'], controls['formant_widths'])

        wet_audio = self.reverb(dry_audio)
        return wet_audio, controls


class VoiceDDSPRealtime(nn.Module):
    """
    Export THIS class to TorchScript for Max/MSP (nn~).
    """

    def __init__(self, trained_controller, sample_rate=24000):
        super().__init__()
        self.controller = trained_controller

        self.source = StatefulRosenberg(sample_rate)
        self.filter = StatefulFormantFilter(sample_rate)

    def forward(self,
                f0: torch.Tensor,
                loudness: torch.Tensor,
                gender: torch.Tensor,
                age: torch.Tensor,
                latent_vector: torch.Tensor,
                states: torch.Tensor):
        rnn_dim = self.controller.rnn_channels
        num_fmt = self.controller.num_formants

        idx_rnn = rnn_dim
        idx_phase = idx_rnn + 1
        idx_filter = idx_phase + (num_fmt * 2)

        rnn_hidden = states[:, :idx_rnn].unsqueeze(0)
        phase = states[:, idx_rnn:idx_phase].unsqueeze(-1)
        filter_states = states[:, idx_phase:idx_filter].view(-1, num_fmt, 2)

        f0_centered = f0 - f0.mean(dim=1, keepdim=True)

        controls, next_rnn = self.controller(f0_centered, loudness, gender, age, latent_vector, rnn_hidden)

        pulse, next_phase = self.source(f0, controls['open_quotient'], controls['rise_ratio'], phase)
        noise = torch.randn_like(pulse)
        excitation = (pulse * controls['voiced_gain']) + (noise * controls['unvoiced_gain'])

        audio, next_filter_states = self.filter(excitation, controls['formant_freqs'], controls['formant_widths'], filter_states)

        next_states = torch.cat([
            next_rnn.squeeze(0),
            next_phase.view(1, 1),
            next_filter_states.view(1, num_fmt * 2)
        ], dim=1)

        return audio, next_states

    @torch.jit.export
    def get_initial_state(self):
        total_state_dim = self.controller.rnn_channels + 1 + (self.controller.num_formants * 2)
        return torch.zeros(1, total_state_dim)


class AudioFeatureEncoder(nn.Module):
    def __init__(
            self,
            sample_rate,
            fft_size,
            hop_length,
            num_mels,
            latent_dim=16,
            gru_units=256,
            f0_min=50.0,
            f0_max=1000.0,
            f0_threshold=0.3
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fft_size = fft_size
        self.f0_min, self.f0_max, self.f0_threshold = f0_min, f0_max, f0_threshold

        self.spectrogram = T.Spectrogram(n_fft=fft_size, hop_length=hop_length, power=2.0)
        self.register_buffer("a_weighting", create_a_weighting(fft_size, sample_rate))

        self.mel_scale = T.MelScale(
            sample_rate=sample_rate, n_mels=num_mels, n_stft=fft_size // 2 + 1
        )
        self.norm = nn.InstanceNorm1d(num_mels)
        self.rnn = nn.GRU(num_mels, gru_units, batch_first=True)
        self.projection = nn.Linear(gru_units, latent_dim)

    def get_pitch(self, audio):
        pad = self.fft_size // 2
        audio_pad = F.pad(audio, (pad, pad), mode="reflect").squeeze(1)
        frames = audio_pad.unfold(dimension=-1, size=self.fft_size, step=self.hop_length)

        n_fft_corr = 2 * self.fft_size
        spec = torch.fft.rfft(frames, n=n_fft_corr, dim=-1)
        power_spec = spec.abs().pow(2)
        autocorr = torch.fft.irfft(power_spec, n=n_fft_corr, dim=-1)
        autocorr = autocorr[..., : self.fft_size]

        norm_factor = autocorr[..., 0:1] + 1e-8
        norm_autocorr = autocorr / norm_factor

        min_f0, max_f0 = self.f0_min, self.f0_max
        min_lag = int(self.sample_rate / max_f0)
        max_lag = int(self.sample_rate / min_f0)

        search_region = norm_autocorr[..., min_lag:max_lag]
        max_val, max_idx = torch.max(search_region, dim=-1)
        true_lag = max_idx + min_lag
        f0 = self.sample_rate / (true_lag.float() + 1e-8)

        f0[max_val < self.f0_threshold] = 0.0
        return f0.unsqueeze(-1)

    def get_loudness(self, spec):
        spec = spec + 1e-8
        weighted_spec = spec * self.a_weighting.view(1, -1, 1)
        mean_power = torch.mean(weighted_spec, dim=1, keepdim=True)
        loudness_db = 10 * torch.log10(mean_power)
        return loudness_db.transpose(1, 2)

    def get_mels(self, spec):
        mels = self.mel_scale(spec)
        mels = torch.log(mels + 1e-5)
        mels = self.norm(mels)
        mels = mels.transpose(1, 2)
        return mels

    def get_content_from_mels(self, mels):
        rnn_out, _ = self.rnn(mels)
        latent_vector = self.projection(rnn_out)
        return latent_vector

    def forward(self, audio):
        f0 = self.get_pitch(audio)
        spec = self.spectrogram(audio.squeeze(1))
        loudness = self.get_loudness(spec)
        mels = self.get_mels(spec)
        latent_vector = self.get_content_from_mels(mels)

        min_len = min(f0.shape[1], loudness.shape[1], latent_vector.shape[1])
        return f0[:, :min_len, :], loudness[:, :min_len, :], latent_vector[:, :min_len, :]


class MultiScaleMelLoss(nn.Module):
    def __init__(self,
                 sample_rate: int = 48000,
                 n_ffts=None,
                 hop_lengths=None,
                 win_lengths=None,
                 num_mels: int = 80,
                 f_min: float = 0.0,
                 f_max: float = None):
        super().__init__()
        if win_lengths is None:
            win_lengths = [4096, 2048, 1024, 512]
        if hop_lengths is None:
            hop_lengths = [1024, 512, 256, 128]
        if n_ffts is None:
            n_ffts = [4096, 2048, 1024, 512]
        self.losses = nn.ModuleList()

        for n_fft, hop, win in zip(n_ffts, hop_lengths, win_lengths):
            self.losses.append(
                T.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop,
                    win_length=win,
                    n_mels=num_mels,
                    f_min=f_min,
                    f_max=f_max,
                    normalized=True
                )
            )

    def forward(self, predicted_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """
        Expects inputs as [Batch, Samples, 1].
        Unconditionally transposes assuming channel last.
        """
        # Removed shape checking if statements.
        # Strictly enforce [B, T, 1] -> [B, 1, T] for spectral processing
        predicted_audio = predicted_audio.transpose(1, 2)
        target_audio = target_audio.transpose(1, 2)

        # Squeeze to [Batch, Time] as MelSpectrogram expects
        predicted_audio = predicted_audio.squeeze(1)
        target_audio = target_audio.squeeze(1)

        total_loss = torch.zeros(1, device=predicted_audio.device)

        for mel_transform in self.losses:
            mel_transform = mel_transform.to(predicted_audio.device)
            mel_hat = torch.log(mel_transform(predicted_audio) + 1e-6)
            mel_true = torch.log(mel_transform(target_audio) + 1e-6)
            total_loss += torch.nn.functional.l1_loss(mel_hat, mel_true)

        return total_loss


class MultiScaleSTFTLoss(nn.Module):
    def __init__(self,
                 fft_sizes=(2048, 1024, 512, 256, 128, 64),
                 hop_sizes=(512, 256, 128, 64, 32, 16),
                 win_lengths=(2048, 1024, 512, 256, 128, 64)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    @staticmethod
    def stft(x, fft_size, hop_size, win_length):
        x = x.squeeze(1)
        window = torch.hann_window(win_length, device=x.device)
        return torch.stft(x, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window,
                          return_complex=True)

    def forward(self, x_fake, x_real):
        """
        Expects inputs as [Batch, Samples, 1].
        Unconditionally transposes assuming channel last.
        """
        # Removed shape checking if statements.
        x_fake = x_fake.transpose(1, 2)
        x_real = x_real.transpose(1, 2)

        loss = 0.0

        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            x_fake_stft = self.stft(x_fake, fft_size, hop_size, win_length)
            x_real_stft = self.stft(x_real, fft_size, hop_size, win_length)

            mag_fake = torch.abs(x_fake_stft) + 1e-7
            mag_real = torch.abs(x_real_stft) + 1e-7

            sc_loss = torch.norm(mag_real - mag_fake, p="fro") / torch.norm(mag_real, p="fro")
            log_mag_loss = F.l1_loss(torch.log(mag_real), torch.log(mag_fake))

            loss += sc_loss + log_mag_loss

        return loss / len(self.fft_sizes)