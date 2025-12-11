import numpy as np
import scipy.io.wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F


def rosenberg_pulse_soft(phase, open_quotient=0.6, hardness=200.0):
    """
    Differentiable Rosenberg C pulse with Soft Sigmoid Masking.

    Args:
        phase: Tensor [batch, time], values in cycles (0.0 to 1.0).
        open_quotient (OQ): Float or Tensor, 0.0 to 1.0.
                            Portion of cycle where glottis is open.
        hardness: Float. Controls the steepness of the closure.
                  Higher = sharper close (more high frequency buzz).
                  Lower = smoother close (softer sound).
                  Values between 100.0 and 1000.0 are standard.
    """
    # 1. Wrap phase to 0..1
    p = phase - torch.floor(phase)

    # 3. The Pulse Curve (The "Hill")
    # Formula: 0.5 * (1 - cos(pi * p / OQ))
    # We apply this to the whole phase, trusting the mask to kill the invalid part later.
    scaled_phase = p / (open_quotient + 1e-8)
    pulse_curve = 0.5 * (1.0 - torch.cos(np.pi * scaled_phase))

    # 4. The Soft Mask (The "Gate")
    # We want Mask ~ 1 when p < OQ
    # We want Mask ~ 0 when p > OQ
    # Logic: sigmoid( (Target - Current) * Sharpness )

    # If p is slightly less than OQ, (safe_oq - p) is small positive -> sigmoid -> ~1.0
    # If p is slightly more than OQ, (safe_oq - p) is small negative -> sigmoid -> ~0.0
    mask = torch.sigmoid((open_quotient - p) * hardness)

    # 5. Apply Mask
    out = pulse_curve * mask

    return out


class DifferentiableFormantSynthesizer(nn.Module):
    def __init__(self, sample_rate=44100, n_fft=1024, hop_length=256):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(n_fft)

        # Pre-compute frequency bins for the filter response calculation
        # Shape: [1, 1, n_fft // 2 + 1] to broadcast over batch and time
        freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        self.register_buffer('freq_bins', freqs.view(1, 1, -1))

    def generate_source(self, f0, voiced_amp, noise_amp):
        """
        Generates the raw excitation signal: a mix of Sawtooth (Glottal) and Noise.
        We assume inputs are upsampled to audio-rate or linear interpolated.
        """
        # 1. Harmonic Source (Approximated Glottal Pulse)
        # Integrate frequency to get phase: phi[t] = phi[t-1] + 2*pi*f0[t]/sr
        phase = torch.cumsum(f0 / self.sample_rate, dim=-1)

        # Using Rosenberg Pulse
        glottal_wave = rosenberg_pulse_soft(phase)
        # A simple difference filter to simulate radiation at the lips:
        glottal_wave_diff = glottal_wave[..., 1:] - glottal_wave[..., :-1]
        # Pad back to original length
        glottal_wave = F.pad(glottal_wave_diff, (1, 0))
        glottal_wave -= glottal_wave.mean(dim=-1, keepdim=True)

        # Only allow noise when the glottis is OPEN
        noise_envelope = glottal_wave.clamp(min=0)
        noise_envelope /= noise_envelope.abs().max()
        pulsed_noise = torch.randn_like(glottal_wave) * noise_envelope

        # 3. Mix
        # voiced_amp and noise_amp should control the mix
        source = (glottal_wave * voiced_amp) + (pulsed_noise * noise_amp)
        return source

    def get_formant_response(self, center_freqs, bandwidths):
        """
        Calculates the frequency response of the vocal tract.
        Modeled as a cascade (product) of 2-pole resonators.

        center_freqs: [batch, frames, num_formants]
        bandwidths:   [batch, frames, num_formants]
        """
        # Expand freq_bins to match input shape: [batch, frames, n_bins]
        f = self.freq_bins  # [1, 1, n_bins]

        # Expand formants to broadcast against freq bins
        # fc shape: [batch, frames, num_formants, 1]
        fc = center_freqs.unsqueeze(-1)
        bw = bandwidths.unsqueeze(-1)

        # The resonant filter curve equation (squared magnitude of a 2-pole system):
        # H(f) = 1 / sqrt( (f^2 - fc^2)^2 + (f * bw)^2 )
        # We add a small epsilon to avoid division by zero
        denominator = torch.sqrt((f ** 2 - fc ** 2) ** 2 + (f * bw) ** 2 + 1e-8)

        # Response for each formant
        response_per_formant = 1.0 / denominator

        # Normalize to keep unity gain at resonance peak approx
        # (This is optional but helps stability)
        peak_gain = 1.0 / (fc * bw + 1e-8)
        response_per_formant = response_per_formant / peak_gain

        # Cascade: Multiply the responses of all formants together
        # Shape result: [batch, frames, n_bins]
        total_response = torch.prod(response_per_formant, dim=2)

        return total_response

    def forward(self, f0, voiced_amp, noise_amp, formant_freqs, formant_bws):
        """
        f0: [batch, time_steps] (Audio rate or upsampled before passing)
        voiced_amp: [batch, time_steps]
        noise_amp: [batch, time_steps]
        formant_freqs: [batch, n_frames, n_formants] (Control rate, e.g., 1 per 10ms)
        formant_bws:   [batch, n_frames, n_formants]
        """
        batch_size, n_samples = f0.shape

        # 1. Generate Source Signal (Time Domain)
        excitation = self.generate_source(f0, voiced_amp, noise_amp)

        # 2. Move Source to Frequency Domain (STFT)
        # We pad to center the STFT windows
        excitation_stft = torch.stft(excitation, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft,
                                     window=self.window.to(excitation.device), return_complex=True, center=True)
        # excitation_stft shape: [batch, freq_bins, n_frames]
        # We need to transpose to [batch, n_frames, freq_bins] to match our filter
        excitation_stft = excitation_stft.transpose(1, 2)

        # 3. Construct Filter (Frequency Domain)
        # We assume formant inputs are already at the correct 'frame' resolution matching the STFT.
        # If not, you would interpolate `formant_freqs` here to match excitation_stft.shape[1].
        filter_response = self.get_formant_response(formant_freqs, formant_bws)

        # Cast filter to complex for multiplication
        # filter_response_c = filter_response.unsqueeze(-1)  # make it [..., 1] for complex broadcasting if needed
        # or just multiply by magnitude if phase is zero
        # Since we are modeling magnitude response only (zero phase filter),
        # we just multiply the complex STFT by the real magnitude curve.

        output_stft = excitation_stft * filter_response.type_as(excitation_stft)

        # 4. Inverse STFT to get Time Domain Audio
        # Transpose back
        output_stft = output_stft.transpose(1, 2)

        output_audio = torch.istft(output_stft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft,
                                   window=self.window.to(excitation.device), length=n_samples)

        return output_audio


class NeuralVoiceDecoder(nn.Module):
    def __init__(self, sample_rate=44100, n_fft=1024, hop_length=256, n_filter_bins=65):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_filter_bins = n_filter_bins

        # Window for STFT operations
        self.register_buffer('window', torch.hann_window(n_fft))

    def rosenberg_source(self, f0, open_quotient):
        """
        Generates the Glottal Pulse excitation.
        """
        # 1. Integrate Phase
        phase = torch.cumsum(f0 / self.sample_rate, dim=-1)
        p = phase - torch.floor(phase)

        # 2. Glottal Pulse Shape (Rosenberg C)
        # Avoid div/0 and keep OQ in reasonable range
        oq = torch.clamp(open_quotient, 0.1, 0.9)

        # The Pulse
        scaled_phase = p / (oq + 1e-8)
        pulse = 0.5 * (1.0 - torch.cos(np.pi * scaled_phase))

        # Soft Masking (differentiable gating)
        mask = torch.sigmoid((oq - p) * 100.0)
        glottal_wave = pulse * mask

        # 3. Spectral Tilt (Differentiation)
        # Simulates lip radiation
        diff_wave = glottal_wave[..., 1:] - glottal_wave[..., :-1]
        return F.pad(diff_wave, (1, 0))

    def apply_filter(self, excitation, filter_magnitudes):
        """
        Applies a time-varying spectral envelope to the excitation.
        Uses Frequency Sampling (STFT Multiplication).

        excitation: [Batch, Time]
        filter_magnitudes: [Batch, Frames, n_filter_bins]
                           (NN outputs these curves)
        """
        # 1. STFT of Excitation
        ex_stft = torch.stft(excitation, self.n_fft, self.hop_length, win_length=self.n_fft, window=self.window,
                             center=True, return_complex=True)
        # Shape: [Batch, Freqs, Frames]

        # 2. Interpolate Filter Curves to Match STFT
        # The NN might output 65 bins, but STFT has n_fft/2+1 (e.g. 513) bins.
        # We assume filter_magnitudes is sampled linearly or mel-scale.
        # Here we assume linear for simplicity.

        # Transpose to [Batch, n_bins, Frames] for interpolation
        filter_mod = filter_magnitudes.transpose(1, 2)

        # Resize to match STFT frequency bins
        target_bins = ex_stft.shape[1]
        filter_resized = F.interpolate(filter_mod, size=ex_stft.shape[2],  # Match Time Frames
                                       mode='linear', align_corners=False)

        # Now resize Frequency axis (if NN output != STFT bins)
        # Usually NN outputs fewer bands (e.g. 65) than FFT (513).
        # We need to stretch the 65 bands to cover the spectrum.
        filter_final = F.interpolate(filter_resized.transpose(1, 2),  # Back to [Batch, Frames, Bins]
                                     size=target_bins, mode='linear', align_corners=False).transpose(1,
                                                                                                     2)  # [Batch, Bins, Frames]

        # 3. Apply Filter (Complex Multiplication)
        # We filter the magnitudes, preserving the phase of the source
        output_stft = ex_stft * filter_final.type_as(ex_stft)

        # 4. iSTFT
        output_audio = torch.istft(output_stft, self.n_fft, self.hop_length, win_length=self.n_fft, window=self.window,
                                   length=excitation.shape[-1])
        return output_audio

    def forward(self, f0, amplitude, open_quotient, vocal_tract_curve, noise_filter_curve):
        """
        Full DSP Pass.

        f0: [B, T]
        amplitude: [B, T] (Global volume)
        open_quotient: [B, T] (Breathiness control)
        vocal_tract_curve: [B, Frames, n_bins] (The Vowel/Formants)
        noise_filter_curve: [B, Frames, n_bins] (The Consonant/Fricative shape)
        """

        # 1. Generate Harmonic Source (Glottis)
        # Upsample controls to audio rate if necessary
        # (Assuming f0/amp/oq are already audio-rate [B, T] for this snippet)
        glottal_source = self.rosenberg_source(f0, open_quotient)

        # 2. Generate Noise Source
        noise_source = torch.randn_like(glottal_source)

        # OPTIONAL: Pulsed Noise (Modulate noise by glottis for realism)
        # noise_source = noise_source * torch.relu(glottal_source)

        # 3. Apply Vocal Tract Filter to Glottal Source
        # (The NN gives us the frequency curve for the current vowel)
        voiced_part = self.apply_filter(glottal_source, vocal_tract_curve)

        # 4. Apply Noise Filter to Noise Source
        # (The NN gives us the frequency curve for fricatives like "sh", "s")
        unvoiced_part = self.apply_filter(noise_source, noise_filter_curve)

        # 5. Mix and Scale
        # In this architecture, vocal_tract_curve handles the gain of the voice,
        # and noise_filter_curve handles the gain of the noise.
        # 'amplitude' is a global scaler.

        mix = (voiced_part + unvoiced_part) * amplitude

        return mix


def generate_spectral_curve(n_bins, peaks):
    """
    Helper to draw a spectral envelope (Magnitude Response).
    Mimics what a Neural Network decoder would output.

    n_bins: Number of freq bins (e.g., 65)
    peaks: List of tuples [(freq_norm, amplitude, bandwidth_norm), ...]
           freq_norm: 0.0 to 1.0 (Nyquist)
    """
    x = torch.linspace(0, 1, n_bins)
    curve = torch.zeros(n_bins)

    for f, amp, bw in peaks:
        # Gaussian curve for each formant/resonance
        # exp( - (x - f)^2 / (2 * bw^2) )
        blob = torch.exp(-0.5 * ((x - f) / bw) ** 2)
        curve += blob * amp

    return curve


# --- Reuse Helper Functions ---
# (Assuming NeuralVoiceDecoder and generate_spectral_curve are defined/imported)

def get_params_ah_ee(sr=44100):
    duration = 1.0
    n_samples = int(sr * duration)
    hop = 256
    n_frames = n_samples // hop + 1
    n_bins = 65

    # 1. Global Controls
    t = torch.linspace(0, duration, n_samples).unsqueeze(0)

    # Pitch: Slight vibrato around 130Hz
    f0 = torch.linspace(130, 110, n_samples).unsqueeze(0)
    f0 = f0 + 2.0 * torch.sin(2 * np.pi * 5.0 * t)
    f0 = f0 + 130 * 0.1 * torch.randn_like(t)

    # Amp: Fade in/out
    amp_env = torch.clamp(torch.sin(np.pi * t / duration), min=0.0) ** 0.5

    # Open Quotient: 0.5 (Normal Tense Voice)
    oq_traj = torch.ones(1, n_samples) * 0.5

    # --- 2. Define Spectral Shapes (The Vowels) ---

    # "Ah" (Father): High F1, Low F2
    # F1~750Hz (0.034 norm), F2~1100Hz (0.05 norm)
    vt_ah = generate_spectral_curve(n_bins, [
        (0.034, 1.0, 0.02),  # F1 (Strong)
        (0.050, 0.8, 0.02),  # F2 (Close to F1)
        (0.120, 0.4, 0.03)  # F3 (Static high ring)
    ])

    # "Ee" (Beet): Low F1, High F2
    # F1~300Hz (0.013 norm), F2~2200Hz (0.10 norm)
    vt_ee = generate_spectral_curve(n_bins, [
        (0.013, 1.0, 0.01),  # F1 (Boomy/Low)
        (0.100, 0.9, 0.02),  # F2 (Very High/Bright)
        (0.130, 0.5, 0.03)  # F3
    ])

    # --- 3. Interpolate (Morph) ---

    vt_traj = torch.zeros(1, n_frames, n_bins)
    nf_traj = torch.zeros(1, n_frames, n_bins)  # Stays zero

    # Morph over the middle 50% of the clip
    for i in range(n_frames):
        pos = i / n_frames

        # Simple Linear Interpolation (Morphing)
        # 0.0-0.2: Hold Ah
        # 0.2-0.8: Morph Ah->Ee
        # 0.8-1.0: Hold Ee

        if pos < 0.2:
            ratio = 0.0
        elif pos > 0.8:
            ratio = 1.0
        else:
            ratio = (pos - 0.2) / 0.6

        # Linear mix of the spectral vectors
        # This simulates the NN traversing the latent space
        current_shape = vt_ah * (1 - ratio) + vt_ee * ratio
        vt_traj[0, i] = current_shape

    return f0, amp_env, oq_traj, vt_traj, nf_traj


def run_ah_ee_neural():
    sr = 44100
    decoder = NeuralVoiceDecoder(sample_rate=sr, n_filter_bins=65)
    f0, amp, oq, vt, nf = get_params_ah_ee(sr)

    with torch.no_grad():
        audio = decoder(f0, amp, oq, vt, nf)

    wav = audio.squeeze().cpu().numpy()
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav))

    scipy.io.wavfile.write("ddsp_ah_ee.wav", sr, wav.astype(np.float32))
    print("Saved 'ddsp_ah_ee.wav'. Check the smooth spectral morph.")


if __name__ == "__main__":
    run_ah_ee_neural()
