import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


def nearest_power_of_two(x):
    power = int(np.round(np.log2(x)))
    return 2 ** power


def create_a_weighting(n_fft: int, sr: int) -> torch.Tensor:
    """
    Creates the A-weighting curve for perceptual loudness following IEC 61672-1.
    """
    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    f_sq = freqs ** 2

    # Constants from IEC 61672-1
    f1 = 20.6
    f2 = 107.7
    f3 = 737.9
    f4 = 12194.217

    numerator = (f4 ** 2) * (f_sq ** 2)
    denominator = (
            (f_sq + f1 ** 2)
            * torch.sqrt(f_sq + f2 ** 2)
            * torch.sqrt(f_sq + f3 ** 2)
            * (f_sq + f4 ** 2)
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
                 k_z_dim: int = 16):
        super().__init__()
        self.num_formants = n_formants
        self.rnn_channels = rnn_channels

        # Input: f0 (1) + loudness (1) + gender (2) + age (N) + z (K)
        input_dim = 1 + 1 + 2 + n_age_groups + k_z_dim

        self.input_layer = nn.Linear(input_dim, rnn_channels)
        self.gru = nn.GRU(rnn_channels, rnn_channels, batch_first=True)

        self.head_oq = nn.Linear(rnn_channels, 1)
        self.head_v_gain = nn.Linear(rnn_channels, 1)
        self.head_u_gain = nn.Linear(rnn_channels, 1)
        self.head_formant_freqs = nn.Linear(rnn_channels, n_formants)
        self.head_formant_bws = nn.Linear(rnn_channels, n_formants)

    def forward(self,
                f0_centered: torch.Tensor,
                loudness: torch.Tensor,
                gender: torch.Tensor,
                age: torch.Tensor,
                z: torch.Tensor,
                hidden_state: torch.Tensor = None):
        inputs = torch.cat([f0_centered, loudness, gender, age, z], dim=-1)

        x = F.relu(self.input_layer(inputs))
        x, new_hidden = self.gru(x, hidden_state)

        oq = torch.sigmoid(self.head_oq(x))
        v_gain = F.softplus(self.head_v_gain(x))
        u_gain = F.softplus(self.head_u_gain(x))

        freq_deltas = F.softplus(self.head_formant_freqs(x))
        ff = torch.cumsum(freq_deltas, dim=-1) + 200.0
        fb = F.softplus(self.head_formant_bws(x)) + 50.0

        controls = {
            'open_quotient': oq,
            'voiced_gain': v_gain,
            'unvoiced_gain': u_gain,
            'formant_freqs': ff,
            'formant_widths': fb
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

    def forward(self, f0: torch.Tensor, open_quotient: torch.Tensor) -> torch.Tensor:
        # Phase Integration
        cycles_per_frame = (f0 * self.hop_length) / self.sample_rate
        cumulative_cycles = torch.cumsum(cycles_per_frame, dim=1)
        frame_start_phase = cumulative_cycles - cycles_per_frame
        frame_start_phase = frame_start_phase - torch.floor(frame_start_phase)
        local_phase_growth = f0 * (self.local_ramp / self.sample_rate)

        total_phase = frame_start_phase + local_phase_growth
        instantaneous_phase = total_phase - torch.floor(total_phase)

        # Rosenberg C Logic
        rise_ratio = 0.66
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

    def generate_impulse_responses(self, cf, bw):
        batch, frames, num_formants = cf.shape
        cf_flat = cf.view(-1, num_formants, 1)
        bw_flat = bw.view(-1, num_formants, 1)

        t = self.ir_time / self.sample_rate
        ir = torch.exp(-self.pi * bw_flat * t) * torch.sin(self.two_pi * cf_flat * t)

        combined = ir.sum(dim=1, keepdim=True)
        scale = combined.abs().max(dim=2, keepdim=True)[0] + 1e-8
        return combined / scale

    def forward(self, excitation: torch.Tensor, cf: torch.Tensor, bw: torch.Tensor, hop_length: int):
        batch, frames, hop = excitation.shape
        filters = self.generate_impulse_responses(cf, bw)

        padding = self.kernel_size - 1
        conv_in = excitation.view(1, batch * frames, hop)

        conv_out = F.conv1d(conv_in, filters, groups=batch * frames, padding=padding)
        output_len = conv_out.shape[-1]

        conv_out = conv_out.view(batch, frames, output_len).transpose(1, 2)

        total_len = (frames - 1) * hop + output_len
        reconstructed = F.fold(
            conv_out,
            output_size=(1, total_len),
            kernel_size=(1, output_len),
            stride=(1, hop)
        )
        return reconstructed.squeeze(2).transpose(1, 2)


class LearnableReverb(nn.Module):
    """FFT Convolution Reverb (Training Only)"""

    def __init__(self, sample_rate: int, reverb_length: float = 1.0):
        super().__init__()
        self.reverb_len = int(sample_rate * reverb_length)

        decay = torch.exp(-torch.linspace(0, 5, self.reverb_len))
        noise = torch.randn(self.reverb_len) * decay
        self.impulse_response = nn.Parameter(noise.view(1, 1, -1))
        self.wet_mix_logits = nn.Parameter(torch.tensor(-1.5))

    def forward(self, audio):
        x = audio.transpose(1, 2)
        ir = self.impulse_response

        fft_size = x.shape[-1] + self.reverb_len - 1
        X = torch.fft.rfft(x, n=fft_size)
        H = torch.fft.rfft(ir, n=fft_size)
        y = torch.fft.irfft(X * H, n=fft_size)[..., :x.shape[-1]]

        wet = torch.sigmoid(self.wet_mix_logits)
        out = (1.0 - wet) * x + wet * y
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

    def forward(self, f0: torch.Tensor, oq: torch.Tensor, phase_state: torch.Tensor):
        phase_step = f0 / self.sample_rate
        curr_phase = torch.cumsum(phase_step, dim=1) + phase_state
        next_state = curr_phase[:, -1:, :].detach()
        phase = curr_phase - torch.floor(curr_phase)

        rise_ratio = 0.66
        tp = oq * rise_ratio
        mask_rise = phase < tp
        mask_fall = (phase >= tp) & (phase < oq)

        wav = torch.zeros_like(phase)
        wav = torch.where(mask_rise, 0.5 * (1.0 - torch.cos(self.pi * phase / (tp + 1e-6))), wav)

        tn = oq - tp
        wav = torch.where(mask_fall, torch.cos(self.pi * (phase - tp) / (2.0 * tn + 1e-6)), wav)

        return wav, next_state


class StatefulFormantFilter(nn.Module):
    """IIR Filter Loop for Real-Time"""

    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.register_buffer('two_pi', torch.tensor(2.0 * math.pi))

    def forward(self, excitation, cf, bw, states):
        r = torch.exp(-torch.pi * bw / self.sample_rate)
        theta = self.two_pi * cf / self.sample_rate
        a1 = -2.0 * r * torch.cos(theta)
        a2 = r * r
        gain = (1.0 - r * r) * torch.sin(theta)

        x_scaled = excitation.repeat(1, 1, cf.shape[-1]) * gain
        return self.iir_loop(x_scaled, a1, a2, states)

    @torch.jit.export
    def iir_loop(self, x, a1, a2, states):
        y1 = states[:, :, 0]
        y2 = states[:, :, 1]
        outputs = torch.jit.annotate(torch.Tensor, torch.empty_like(x))

        for t in range(x.shape[1]):
            yt = x[:, t] - (a1[:, t] * y1) - (a2[:, t] * y2)
            outputs[:, t] = yt
            y2 = y1
            y1 = yt

        return outputs.sum(dim=-1, keepdim=True), torch.stack([y1, y2], dim=2)


# ==============================================================================
# 4. WRAPPERS (The Top-Level Classes)
# ==============================================================================

class VoiceDDSP_Training(nn.Module):
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
            k_z_dim=16,
            min_formant_width=50.0,
            reverb_length=1.0
    ):
        super().__init__()
        self.hop_length = hop_length

        self.controller = DDSPController(
            rnn_channels=rnn_channels,
            n_formants=n_formants,
            n_age_groups=n_age_groups,
            k_z_dim=k_z_dim
        )

        self.source = RosenbergSource(sample_rate, hop_length)
        self.filter = FormantFilterFIR(sample_rate, min_width=min_formant_width)
        self.reverb = LearnableReverb(sample_rate, reverb_length=reverb_length)

    def forward(self,
                f0: torch.Tensor,
                loudness: torch.Tensor,
                gender: torch.Tensor,
                age: torch.Tensor,
                z: torch.Tensor):
        f0_centered = f0 - f0.mean(dim=1, keepdim=True)

        controls, _ = self.controller(f0_centered, loudness, gender, age, z)

        voiced = self.source(f0, controls['open_quotient'])
        noise = torch.randn_like(voiced)
        excitation = (voiced * controls['voiced_gain']) + (noise * controls['unvoiced_gain'])

        B, T, _ = excitation.shape
        frames = T // self.hop_length
        excitation_frames = excitation.view(B, frames, self.hop_length)

        dry_audio = self.filter(excitation_frames, controls['formant_freqs'], controls['formant_widths'],
                                self.hop_length)

        wet_audio = self.reverb(dry_audio)
        return wet_audio, controls


class VoiceDDSP_RealTime(nn.Module):
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
                z: torch.Tensor,
                states: torch.Tensor):
        rnn_dim = self.controller.rnn_channels
        num_fmt = self.controller.num_formants

        i_rnn = rnn_dim
        i_pha = i_rnn + 1
        i_fil = i_pha + (num_fmt * 2)

        rnn_h = states[:, :i_rnn].unsqueeze(0)
        phase = states[:, i_rnn:i_pha].unsqueeze(-1)
        fil_s = states[:, i_pha:i_fil].view(-1, num_fmt, 2)

        f0_centered = f0 - f0.mean(dim=1, keepdim=True)

        controls, next_rnn = self.controller(f0_centered, loudness, gender, age, z, rnn_h)

        pulse, next_phase = self.source(f0, controls['open_quotient'], phase)
        noise = torch.randn_like(pulse)
        excitation = (pulse * controls['voiced_gain']) + (noise * controls['unvoiced_gain'])

        audio, next_fil = self.filter(excitation, controls['formant_freqs'], controls['formant_widths'], fil_s)

        next_states = torch.cat([
            next_rnn.squeeze(0),
            next_phase.view(1, 1),
            next_fil.view(1, num_fmt * 2)
        ], dim=1)

        return audio, next_states

    @torch.jit.export
    def get_initial_state(self):
        d = self.controller.rnn_channels + 1 + (self.controller.num_formants * 2)
        return torch.zeros(1, d)


class AudioFeatureEncoder(nn.Module):
    def __init__(
            self,
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
            z_dim=16,
            gru_units=256,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft

        self.spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
        self.register_buffer("a_weighting", create_a_weighting(n_fft, sample_rate))

        self.melspec = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.norm = nn.InstanceNorm1d(n_mels)
        self.rnn = nn.GRU(n_mels, gru_units, batch_first=True)
        self.projection = nn.Linear(gru_units, z_dim)

    def get_pitch(self, audio):
        pad = self.n_fft // 2
        audio_pad = F.pad(audio, (pad, pad), mode="reflect").squeeze(1)
        frames = audio_pad.unfold(dimension=-1, size=self.n_fft, step=self.hop_length)

        n_fft_corr = 2 * self.n_fft
        spec = torch.fft.rfft(frames, n=n_fft_corr, dim=-1)
        power_spec = spec.abs().pow(2)
        autocorr = torch.fft.irfft(power_spec, n=n_fft_corr, dim=-1)
        autocorr = autocorr[..., : self.n_fft]

        norm_factor = autocorr[..., 0:1] + 1e-8
        norm_autocorr = autocorr / norm_factor

        min_f0, max_f0 = 50, 1000
        min_lag = int(self.sample_rate / max_f0)
        max_lag = int(self.sample_rate / min_f0)

        search_region = norm_autocorr[..., min_lag:max_lag]
        max_val, max_idx = torch.max(search_region, dim=-1)
        true_lag = max_idx + min_lag
        f0 = self.sample_rate / (true_lag.float() + 1e-8)

        f0[max_val < 0.3] = 0.0
        return f0.unsqueeze(-1)

    def get_loudness(self, audio):
        spec = self.spectrogram(audio.squeeze(1)) + 1e-8
        weighted_spec = spec * self.a_weighting.view(1, -1, 1)
        mean_power = torch.mean(weighted_spec, dim=1, keepdim=True)
        loudness_db = 10 * torch.log10(mean_power)
        return loudness_db.transpose(1, 2)

    def get_mels(self, audio):
        mels = self.melspec(audio.squeeze(1))
        mels = torch.log(mels + 1e-5)
        mels = self.norm(mels)
        mels = mels.transpose(1, 2)
        return mels

    def get_content_from_mels(self, mels):
        rnn_out, _ = self.rnn(mels)
        z = self.projection(rnn_out)
        return z

    def forward(self, audio):
        f0 = self.get_pitch(audio)
        loudness = self.get_loudness(audio)
        mels = self.get_mels(audio)
        z = self.get_content_from_mels(mels)

        min_len = min(f0.shape[1], loudness.shape[1], z.shape[1])
        return f0[:, :min_len, :], loudness[:, :min_len, :], z[:, :min_len, :]


class MultiScaleMelLoss(nn.Module):
    def __init__(self,
                 sample_rate: int = 48000,
                 n_ffts=None,
                 hop_lengths=None,
                 win_lengths=None,
                 n_mels: int = 80,
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
                    n_mels=n_mels,
                    f_min=f_min,
                    f_max=f_max,
                    normalized=True
                )
            )

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Expects inputs as [Batch, Samples, 1].
        Unconditionally transposes assuming channel last.
        """
        # Removed shape checking if statements.
        # Strictly enforce [B, T, 1] -> [B, 1, T] for spectral processing
        y_hat = y_hat.transpose(1, 2)
        y = y.transpose(1, 2)

        # Squeeze to [Batch, Time] as MelSpectrogram expects
        y_hat = y_hat.squeeze(1)
        y = y.squeeze(1)

        total_loss = 0.0

        for mel_transform in self.losses:
            mel_transform = mel_transform.to(y_hat.device)
            mel_hat = torch.log(mel_transform(y_hat) + 1e-6)
            mel_true = torch.log(mel_transform(y) + 1e-6)
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

    def stft(self, x, fft_size, hop_size, win_length):
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