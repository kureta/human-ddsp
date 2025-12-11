import os

import numpy as np
import polars as pl
import scipy.io.wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader


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


class NeuralVoiceDecoder(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_filter_bins=65):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_filter_bins = n_filter_bins
        self.register_buffer('window', torch.hann_window(n_fft))

    def rosenberg_source(self, f0, open_quotient):
        # f0 and OQ are now Audio Rate [Batch, Time]
        phase = torch.cumsum(f0 / self.sample_rate, dim=-1)
        p = phase - torch.floor(phase)

        oq = torch.clamp(open_quotient, 0.1, 0.9)
        scaled_phase = p / (oq + 1e-8)

        pulse = 0.5 * (1.0 - torch.cos(np.pi * scaled_phase))
        mask = torch.sigmoid((oq - p) * 100.0)
        glottal_wave = pulse * mask

        diff_wave = glottal_wave[..., 1:] - glottal_wave[..., :-1]
        return F.pad(diff_wave, (1, 0))

    def apply_filter(self, excitation, filter_magnitudes):
        # excitation: [Batch, Time]
        # filter_magnitudes: [Batch, Frames, Bins]

        # 1. STFT
        ex_stft = torch.stft(
            excitation,
            self.n_fft,
            self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True
        )

        # 2. Interpolate Filter to match STFT Frames
        # [Batch, Frames, Bins] -> [Batch, Bins, Frames]
        filter_mod = filter_magnitudes.transpose(1, 2)

        # Resize Time axis to match STFT
        filter_resized = F.interpolate(
            filter_mod,
            size=ex_stft.shape[2],
            mode='linear', align_corners=False
        )

        # Resize Freq axis to match STFT Bins
        target_bins = ex_stft.shape[1]
        filter_final = F.interpolate(
            filter_resized.transpose(1, 2),  # [B, T, Bins]
            size=target_bins,
            mode='linear', align_corners=False
        ).transpose(1, 2)  # [B, Bins, T]

        # 3. Filter
        output_stft = ex_stft * filter_final.type_as(ex_stft)

        # 4. iSTFT
        output_audio = torch.istft(
            output_stft,
            self.n_fft,
            self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            length=excitation.shape[-1]
        )
        return output_audio

    def forward(self, f0, amplitude, open_quotient, vocal_tract_curve, noise_filter_curve, target_length=None):
        """
        f0, amplitude, open_quotient: [Batch, Frames, 1] (Control Rate)
        target_length: int (Desired Audio Samples)
        """
        # Determine Output Length
        if target_length is None:
            # Fallback: estimate based on frames
            target_length = f0.shape[1] * self.hop_length

        # --- UPSAMPLING (Fixes the error) ---
        def upsample(x):
            # Input: [Batch, Frames, 1]
            x = x.transpose(1, 2)  # [Batch, 1, Frames]
            x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
            return x.squeeze(1)  # [Batch, Time]

        # Convert Controls to Audio Rate
        f0_up = upsample(f0)
        amp_up = upsample(amplitude)
        oq_up = upsample(open_quotient)

        # 1. Source
        glottal_source = self.rosenberg_source(f0_up, oq_up)
        noise_source = torch.randn_like(glottal_source)

        # 2. Filter (Decoder handles Frame->STFT interpolation internally)
        voiced_part = self.apply_filter(glottal_source, vocal_tract_curve)
        unvoiced_part = self.apply_filter(noise_source, noise_filter_curve)

        # 3. Mix
        mix = (voiced_part + unvoiced_part) * amp_up

        return mix


def _create_a_weighting(n_fft, sr):
    """Creates the A-weighting curve for Perceptual Loudness."""
    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    # Standard A-weighting equation
    f_sq = freqs ** 2
    term1 = 12194.217 ** 2 * f_sq ** 2
    term2 = (f_sq + 20.6 ** 2) * (f_sq + 107.7 ** 2) * \
            (f_sq + 737.9 ** 2) * (f_sq + 12194.217 ** 2) ** 0.5
    gain = term1 / (term2 + 1e-8)
    return gain


class AudioFeatureEncoder(nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=1024,
                 hop_length=256,
                 n_mels=80,
                 z_dim=128,
                 gru_units=512,
                 gru_layers=2,
                 ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft

        # --- 1. Loudness Tools ---
        self.spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
        self.register_buffer('a_weighting', _create_a_weighting(n_fft, sample_rate))

        # --- 2. Real-Time Timbre Encoder ---
        self.melspec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.norm = nn.InstanceNorm1d(n_mels)
        self.rnn = nn.GRU(n_mels, gru_units, gru_layers, batch_first=True, bidirectional=False)
        self.projection = nn.Linear(gru_units, z_dim)

    def get_pitch(self, audio):
        """
        Pure PyTorch Pitch Detector using Autocorrelation (ACF).
        Differentiable and dependency-free.
        """
        # 1. Frame the audio [Batch, Frames, Window]
        # Use Unfold to create sliding windows matching STFT logic
        # Pad audio to match STFT centering if possible, or just standard framing
        pad = self.n_fft // 2
        audio_pad = F.pad(audio.unsqueeze(1), (pad, pad), mode='reflect').squeeze(1)

        # Unfold: [Batch, Window_Size, Frames]
        frames = audio_pad.unfold(dimension=-1, size=self.n_fft, step=self.hop_length)

        # 2. Compute Autocorrelation via FFT
        # R(t) = iFFT( |FFT(x)|^2 )
        # FFT size should be >= 2*window to avoid circular convolution aliasing,
        # but for pitch estimation, standard size often suffices if we ignore edge effects.
        # We'll use 2*n_fft for cleaner correlation.
        n_fft_corr = 2 * self.n_fft
        spec = torch.fft.rfft(frames, n=n_fft_corr, dim=-1)
        power_spec = spec.abs().pow(2)
        autocorr = torch.fft.irfft(power_spec, n=n_fft_corr, dim=-1)

        # We only care about the first half (lags 0 to window size)
        autocorr = autocorr[..., :self.n_fft]

        # 3. Peak Picking
        # We normalize by lag 0 to get NCCF (Normalized Cross Correlation) roughly
        # This helps ignore amplitude variations.
        norm_factor = autocorr[..., 0:1] + 1e-8
        norm_autocorr = autocorr / norm_factor

        # Define Pitch Search Range (e.g., 50Hz to 1000Hz)
        min_f0 = 50
        max_f0 = 1000

        # Convert Hz to Lag indices
        # lag = sr / freq
        min_lag = int(self.sample_rate / max_f0)
        max_lag = int(self.sample_rate / min_f0)

        # Restrict search area
        # We slice the autocorrelation buffer to the valid lags
        search_region = norm_autocorr[..., min_lag:max_lag]

        # Find the index of the max correlation in the search region
        max_val, max_idx = torch.max(search_region, dim=-1)

        # Correct the index offset
        true_lag = max_idx + min_lag

        # Convert Lag -> Frequency
        # f0 = sr / lag
        f0 = self.sample_rate / (true_lag.float() + 1e-8)

        # 4. Unvoiced Detection (Thresholding)
        # If correlation is weak (< 0.3), treat as unvoiced (0 Hz)
        # max_val is the correlation coefficient (0.0 to 1.0)
        unvoiced_mask = (max_val < 0.3)
        f0[unvoiced_mask] = 0.0

        return f0.unsqueeze(-1)  # [Batch, Frames, 1]

    def get_loudness(self, audio):
        spec = self.spectrogram(audio) + 1e-8
        weighted_spec = spec * self.a_weighting.view(1, -1, 1)
        mean_power = torch.mean(weighted_spec, dim=1, keepdim=True)
        loudness_db = 10 * torch.log10(mean_power)
        return loudness_db.transpose(1, 2)

    def encode_timbre(self, audio):
        mels = self.melspec(audio)
        mels = torch.log(mels + 1e-5)
        mels = self.norm(mels)

        mels = mels.transpose(1, 2)
        rnn_out, _ = self.rnn(mels)

        z = self.projection(rnn_out)
        return z

    def forward(self, audio):
        f0 = self.get_pitch(audio)
        loudness = self.get_loudness(audio)
        z = self.encode_timbre(audio)

        # Align lengths (Autocorrelation framing vs STFT framing)
        min_len = min(f0.shape[1], loudness.shape[1], z.shape[1])
        return f0[:, :min_len, :], loudness[:, :min_len, :], z[:, :min_len, :]


class LearnableReverb(nn.Module):
    def __init__(self, sample_rate=16000, reverb_duration=1.0):
        super().__init__()
        self.sample_rate = sample_rate

        # Length of the impulse response in samples
        n_samples = int(sample_rate * reverb_duration)

        # 1. Initialize with a realistic "decaying noise" shape
        # This acts as a good prior so the model starts with a "room" sound
        decay = torch.exp(-torch.linspace(0, 5, n_samples))  # Decays to ~0.006 over duration
        noise = torch.randn(n_samples) * 0.1
        initial_ir = noise * decay

        # 2. Register as a Trainable Parameter
        # Shape: [1, 1, Time] to broadcast over batch
        self.impulse_response = nn.Parameter(initial_ir.view(1, 1, -1))

    def forward(self, audio):
        """
        Convolves the input audio with the learned Impulse Response.
        Args:
            audio: [Batch, Time] (Dry Signal)
        Returns:
            wet_audio: [Batch, Time] (Reverberated Signal)
        """
        # Ensure input has channel dim [Batch, 1, Time] for consistency,
        # though we'll squeeze it later.
        if audio.dim() == 2:
            x = audio.unsqueeze(1)
        else:
            x = audio

        # Get dimensions
        batch, channels, dry_len = x.shape
        ir_len = self.impulse_response.shape[-1]

        # --- FFT Convolution ---
        # We must pad to (dry_len + ir_len - 1) to avoid circular aliasing
        fft_size = dry_len + ir_len - 1

        # Next power of 2 is faster for FFT
        import math
        n_fft = 2 ** math.ceil(math.log2(fft_size))

        # 1. FFT
        dry_fft = torch.fft.rfft(x, n=n_fft, dim=-1)
        ir_fft = torch.fft.rfft(self.impulse_response, n=n_fft, dim=-1)

        # 2. Multiply (Convolution in time = Multiplication in Freq)
        wet_fft = dry_fft * ir_fft

        # 3. iFFT
        wet_audio = torch.fft.irfft(wet_fft, n=n_fft, dim=-1)

        # 4. Crop
        # Convolution makes the signal longer (tail).
        # For training, we usually crop back to the input length
        # so the loss function compares aligned sizes.
        return wet_audio.squeeze(1)[..., :dry_len]


class VoiceController(nn.Module):
    def __init__(self,
                 z_dim=128,
                 n_filter_bins=65,
                 hidden_dim=512,
                 n_layers=3):
        super().__init__()
        self.n_filter_bins = n_filter_bins

        # 1. Input Preprocessing
        # We concatenate z (128) + f0 (1) + loudness (1)
        input_dim = z_dim + 1 + 1

        # 2. The Main "Brain" (MLP)
        # Using GRU here is optional.
        # Since the Encoder handled the time-context, an MLP is often enough
        # and has zero latency for the decoding step.
        layers = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU(0.1))

        self.mlp = nn.Sequential(*layers)

        # 3. Output Projections (The "Knobs")

        # A. Open Quotient (Breathiness) -> Scalar (0.0 to 1.0)
        self.proj_oq = nn.Linear(hidden_dim, 1)

        # B. Vocal Tract Curve (Vowels) -> 65 frequency bins (0.0 to 1.0)
        self.proj_vt = nn.Linear(hidden_dim, n_filter_bins)

        # C. Noise Filter Curve (Consonants) -> 65 frequency bins (0.0 to 1.0)
        self.proj_nf = nn.Linear(hidden_dim, n_filter_bins)

    def forward(self, f0, loudness_db, z):
        """
        Args:
            f0: [Batch, Frames, 1] (Hz)
            loudness_db: [Batch, Frames, 1] (dB)
            z: [Batch, Frames, z_dim]

        Returns:
            Dictionary of controls, ready for NeuralVoiceDecoder
        """
        # 1. Normalize Physical Inputs for the Neural Net
        # Neural Nets hate raw Hz (100-1000) and dB (-100 to 0).
        # We scale them roughly to [0, 1] or [-1, 1] range.

        # Log-scale F0 is better for pitch perception
        # map 50Hz..1000Hz -> approx 0..1
        f0_norm = (torch.log(f0 + 1e-5) - 4.0) / 4.0

        # Loudness: map -100dB..0dB -> 0..1
        loudness_norm = (loudness_db / 100.0) + 1.0
        loudness_norm = torch.clamp(loudness_norm, 0.0, 1.0)

        # Concatenate: [Batch, Frames, z_dim + 2]
        decoder_input = torch.cat([z, f0_norm, loudness_norm], dim=-1)

        # 2. Pass through MLP
        hidden = self.mlp(decoder_input)

        # 3. Project to Controls

        # Open Quotient: Sigmoid (0 to 1)
        # Bias the initialization so it starts near 0.5 (normal voice)
        oq = torch.sigmoid(self.proj_oq(hidden))

        # Vocal Tract (The Formants)
        # Modified Sigmoid allows it to be perfectly zero if needed
        # We want smooth curves.
        vt_curve = torch.sigmoid(self.proj_vt(hidden))

        # Noise Filter (The Hiss)
        # Usually mostly zero, so standard sigmoid is fine
        nf_curve = torch.sigmoid(self.proj_nf(hidden))

        # 4. Convert Loudness dB to Linear Amplitude for the Synth
        # amp = 10 ^ (db / 20)
        amplitude_linear = torch.pow(10.0, loudness_db / 20.0)

        # Return everything needed by NeuralVoiceDecoder
        return {
            "f0": f0,  # Pass through raw F0
            "amplitude": amplitude_linear,  # Pass through linear amp
            "open_quotient": oq,
            "vocal_tract_curve": vt_curve,
            "noise_filter_curve": nf_curve
        }


class VoiceAutoEncoder(nn.Module):
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.encoder = AudioFeatureEncoder(sample_rate=sample_rate)
        self.controller = VoiceController(z_dim=128, n_filter_bins=65)
        self.decoder = NeuralVoiceDecoder(sample_rate=sample_rate, n_filter_bins=65)
        self.reverb = LearnableReverb(sample_rate=sample_rate, reverb_duration=0.5)

    def forward(self, audio):
        # 1. Features
        f0, loud_db, z = self.encoder(audio)

        # 2. Controls
        controls = self.controller(f0, loud_db, z)

        # 3. Synthesis
        # PASS THE TARGET LENGTH HERE
        dry_audio = self.decoder(
            f0=controls['f0'],
            amplitude=controls['amplitude'],
            open_quotient=controls['open_quotient'],
            vocal_tract_curve=controls['vocal_tract_curve'],
            noise_filter_curve=controls['noise_filter_curve'],
            target_length=audio.shape[-1]  # <--- Fix
        )

        # 4. Reverb
        wet_audio = self.reverb(dry_audio)

        return wet_audio, dry_audio, controls


class MultiScaleSpectralLoss(nn.Module):
    def __init__(self,
                 fft_sizes=None,
                 overlap_ratio=0.75,
                 mag_weight=1.0,
                 log_mag_weight=1.0):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [2048, 1024, 512, 256, 128, 64]
        self.fft_sizes = fft_sizes
        self.overlap_ratio = overlap_ratio
        self.mag_weight = mag_weight
        self.log_mag_weight = log_mag_weight

    def spectrogram(self, x, n_fft):
        # Calculate hop length based on overlap
        hop_length = int(n_fft * (1 - self.overlap_ratio))

        # Apply Hann Window
        window = torch.hann_window(n_fft).to(x.device)

        # STFT
        # Input x: [Batch, Time]
        x_stft = torch.stft(x,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            win_length=n_fft,
                            window=window,
                            return_complex=True,
                            center=True)

        # Magnitude (ignore phase)
        mag = torch.abs(x_stft)

        # Add small epsilon for log stability
        mag = torch.clamp(mag, min=1e-7)
        return mag

    def forward(self, x_pred, x_target):
        """
        Calculate loss between predicted and target audio.
        x_pred: [Batch, Time]
        x_target: [Batch, Time]
        """
        loss = 0.0

        for n_fft in self.fft_sizes:
            mag_pred = self.spectrogram(x_pred, n_fft)
            mag_target = self.spectrogram(x_target, n_fft)

            # 1. Linear Magnitude Loss (L1)
            # Good for high energy components (Formants, Fundamental)
            lin_loss = F.l1_loss(mag_pred, mag_target)

            # 2. Log Magnitude Loss (L1)
            # Good for low energy components (High freq noise, reverb tails)
            # We compress dynamic range so quiet sounds matter too
            log_loss = F.l1_loss(torch.log(mag_pred), torch.log(mag_target))

            # Combine
            loss += (self.mag_weight * lin_loss) + (self.log_mag_weight * log_loss)

        return loss


# --- Assume you save your model classes in 'model.py' ---
# from model import VoiceAutoEncoder, MultiScaleSpectralLoss
# For this script to be standalone, I assume the classes exist in the namespace.
# (Paste the VoiceAutoEncoder and MultiScaleSpectralLoss classes here if running as one file)

# ==========================================
# 1. The Dataset (Chunker)
# ==========================================
class SingleAudioDataset(Dataset):
    def __init__(self, audio_path, sample_rate=16000, chunk_size=32000, overlap=0.5):
        """
        Loads one file and creates a dataset of overlapping chunks.
        chunk_size: 32000 samples = 2.0 seconds @ 16kHz
        overlap: 0.5 = 50% overlap between chunks
        """
        super().__init__()
        self.chunk_size = chunk_size

        # 1. Load Audio
        # Load and mix to mono
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # 2. Resample if necessary
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            audio = resampler(audio)

        # 3. Normalize (-1.0 to 1.0)
        max_val = torch.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        self.data = audio.squeeze()  # [Total_Samples]

        # 4. Create Slice Indices
        self.starts = []
        stride = int(chunk_size * (1 - overlap))
        total_len = self.data.shape[0]

        # Generate start points
        for i in range(0, total_len - chunk_size + 1, stride):
            self.starts.append(i)

        print(f"Dataset created: {len(self.starts)} chunks from {total_len / sample_rate:.2f}s of audio.")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        end = start + self.chunk_size
        return self.data[start:end]


# ==========================================
# 2. The Training Loop
# ==========================================
def train_overfit(wav_path, epochs=1000, batch_size=8, lr=1e-4, save_interval=50):
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 1. Prepare Data
    # 2 seconds (32000 samples) is a standard DDSP training window
    dataset = SingleAudioDataset(wav_path, sample_rate=16000, chunk_size=32000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 2. Init Model
    model = VoiceAutoEncoder(sample_rate=16000).to(device)

    # 3. Init Loss & Optimizer
    # We define the loss locally if not imported
    criterion = MultiScaleSpectralLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Output dir
    os.makedirs("checkpoints", exist_ok=True)

    # --- Loop ---
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for batch_idx, audio_target in enumerate(dataloader):
            audio_target = audio_target.to(device)

            # Zero grad
            optimizer.zero_grad()

            # Forward
            # wet: The final output (with reverb)
            # dry: The raw synth output
            # controls: The physical params (f0, loudness, etc.)
            wet_pred, dry_pred, controls = model(audio_target)

            # Loss
            # We compare the WET prediction to the original target
            loss = criterion(wet_pred, audio_target)

            # Backward
            loss.backward()

            # Gradient Clipping (Important for RNNs/Synths to prevent explosions)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{epochs}] Loss: {avg_loss:.4f}")

        # --- Checkpointing & Inspection ---
        if epoch % save_interval == 0:
            # Save Model
            torch.save(model.state_dict(), f"checkpoints/model_ep{epoch}.pth")

            # Save Audio Sample (Reconstruction of the first item in the last batch)
            # We save both 'Wet' (Final) and 'Dry' (Anechoic) to hear what the model learned.
            with torch.no_grad():
                # Normalize for wav file
                target_wav = audio_target[0].cpu().numpy()
                wet_wav = wet_pred[0].cpu().numpy()
                dry_wav = dry_pred[0].cpu().numpy()

                scipy.io.wavfile.write(f"checkpoints/ep{epoch}_target.wav", 16000, target_wav)
                scipy.io.wavfile.write(f"checkpoints/ep{epoch}_recon_wet.wav", 16000, wet_wav)
                scipy.io.wavfile.write(f"checkpoints/ep{epoch}_recon_dry.wav", 16000, dry_wav)
            print(f"--> Saved checkpoint and audio samples to 'checkpoints/'")


def process_tsv():
    # Replace with your actual file path
    file_path = "/Users/kureta/Music/cv-corpus-23.0-2025-09-05/tr/validated.tsv"
    # 1. Load the TSV file
    # Polars uses read_csv with a separator argument for TSV
    df = pl.read_csv(file_path, separator="\t", ignore_errors=True, encoding="utf-8", quote_char="")

    # 2. Apply Filters
    # - path is not null
    # - age is not null
    # - gender is exactly 'male' or 'female'
    filtered_df = df.filter(
        pl.col("path").is_not_null() &
        pl.col("age").is_not_null() &
        pl.col("gender").is_not_null()
    ).select(["path", "age", "gender"])  # Discard everything else

    # 3. Inspect results
    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(filtered_df)}")
    print(filtered_df.head())

    # Optional: Save back to TSV
    filtered_df.write_csv("filtered_dataset.csv")

    # Extract unique values
    unique_genders = filtered_df.select(pl.col("gender").unique()).to_series().to_list()
    unique_ages = filtered_df.select(pl.col("age").unique()).to_series().to_list()

    print("Unique Genders:", unique_genders)
    print("Unique Ages:", unique_ages)


def load_csv():
    file_path = "filtered_dataset.csv"
    df = pl.read_csv(file_path, ignore_errors=True, encoding="utf-8", quote_char="")

    return df

def get_info(df):
    print(df.head())
    print(f"Rows: {len(df)}")

    # Extract unique values
    unique_genders = df.select(pl.col("gender").unique()).to_series().to_list()
    unique_ages = df.select(pl.col("age").unique()).to_series().to_list()

    print("Unique Genders:", unique_genders)
    print("Unique Ages:", unique_ages)

    # Set config to display all rows
    pl.Config.set_tbl_rows(-1)

    # 2. Group by both columns and count
    age_gender_counts = (
        df.group_by(["age", "gender"])
        .len()  # Counts rows in each group (aliased as 'len' or 'count')
        .sort(["age", "gender"])  # Sort for readability
    )

    print(age_gender_counts)

age_map = {
    'teens': 16.0,
    'twenties': 25.0,
    'thirties': 35.0,
    'fourties': 45.0,
    'fifties': 55.0,
    'sixties': 65.0,
    'seventies': 75.0,
    'eighties': 85.0
}


def encode_age(age_str):
    if age_str not in age_map:
        return 0.3  # Default to ~30 if unknown

    raw_age = age_map[age_str]
    norm_age = raw_age / 100.0  # e.g., 'twenties' -> 0.25
    return norm_age


# --- Test ---
def main():
    # df = load_csv()
    # get_info(df)
    # return
    # Replace with your actual wav file
    wav_file = "/Users/kureta/Music/cv-corpus-23.0-2025-09-05/tr/clips/common_voice_tr_43561583.mp3"

    # Create dummy file if it doesn't exist for testing
    if not os.path.exists(wav_file):
        print("Creating dummy input file...")
        dummy = torch.sin(2 * 3.1415 * 440 * torch.linspace(0, 5, 16000 * 5))
        scipy.io.wavfile.write(wav_file, 16000, dummy.numpy())

    train_overfit(wav_file, epochs=10000, save_interval=100)


if __name__ == "__main__":
    main()
