# pyright: basic

import os
import random

import numpy as np
import scipy.io.wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

# Optional: Polars for data loading
try:
    import polars as pl
except ImportError:
    pl = None


# ==========================================
# Age Config & Utilities
# ==========================================

AGE_LABELS = [
    "teens",
    "twenties",
    "thirties",
    "fourties",
    "fifties",
    "sixties",
    "seventies",
    "eighties",
]
NUM_AGE_CLASSES = len(AGE_LABELS)


def float_to_weighted_age(age_float: float, device="cpu"):
    """
    Maps a float (0.0 to 1.0) to a weighted one-hot vector across AGE_LABELS.
    """
    val = max(0.0, min(1.0, age_float))
    max_idx = NUM_AGE_CLASSES - 1
    continuous_idx = val * max_idx

    idx_lower = int(continuous_idx)
    idx_upper = min(idx_lower + 1, max_idx)

    alpha = continuous_idx - idx_lower

    vector = torch.zeros(NUM_AGE_CLASSES, device=device)

    if idx_lower == idx_upper:
        vector[idx_lower] = 1.0
    else:
        vector[idx_lower] = 1.0 - alpha
        vector[idx_upper] = alpha

    return vector


# ==========================================
# 1. DSP & Helper Modules
# ==========================================


def _create_a_weighting(n_fft, sr):
    """Creates the A-weighting curve for Perceptual Loudness."""
    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    f_sq = freqs**2
    term1 = 12194.217**2 * f_sq**2
    term2 = (
        (f_sq + 20.6**2)
        * (f_sq + 107.7**2)
        * (f_sq + 737.9**2)
        * (f_sq + 12194.217**2) ** 0.5
    )
    gain = term1 / (term2 + 1e-8)
    return gain


class NeuralVoiceDecoder(nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, n_filter_bins=65):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_filter_bins = n_filter_bins
        self.register_buffer("window", torch.hann_window(n_fft))

    def rosenberg_source(self, f0, open_quotient):
        # f0, OQ: [Batch, Time]
        phase = torch.cumsum(f0 / self.sample_rate, dim=-1)
        p = phase - torch.floor(phase)

        oq = torch.clamp(open_quotient, 0.1, 0.9)
        scaled_phase = p / (oq + 1e-8)

        pulse = 0.5 * (1.0 - torch.cos(np.pi * scaled_phase))
        mask = torch.sigmoid((oq - p) * 100.0)
        glottal_wave = pulse * mask

        # Spectral tilt (derivative)
        diff_wave = glottal_wave[..., 1:] - glottal_wave[..., :-1]
        return F.pad(diff_wave, (1, 0))

    def apply_filter(self, excitation, filter_magnitudes):
        # excitation: [Batch, Length]
        # filter_magnitudes: [Batch, Time, n_filter_bins]

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

        # 2. Interpolate Filter (Frequency-axis only)
        B, T, F_ctrl = filter_magnitudes.shape
        F_target = ex_stft.shape[1] # FreqBins, e.g., 1025

        # Reshape to [B * T, 1, F_ctrl] to interpolate along the last (frequency) dimension
        filter_reshaped = filter_magnitudes.reshape(B * T, 1, F_ctrl) 
        
        filter_resized = F.interpolate(
            filter_reshaped, size=F_target, mode="linear", align_corners=False
        ) # [B * T, 1, F_target]

        # Reshape and transpose back to match STFT shape: [B, F_target, T]
        filter_final = filter_resized.squeeze(1).reshape(B, T, F_target).transpose(1, 2)
        
        # 3. Filter
        # Note: No need to explicitly call .type_as() if the complex-valued STFT
        # is correctly handled by broadcasting with the real-valued filter.
        output_stft = ex_stft * filter_final

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

    @staticmethod
    def _upsample(x, target_length: int):
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_length, mode="linear", align_corners=False)
        return x.squeeze(1)

    def forward(
        self,
        f0,
        amplitude,
        open_quotient,
        vocal_tract_curve,
        noise_filter_curve,
        target_length: int,
    ):
        f0_up = self._upsample(f0, target_length)
        amp_up = self._upsample(amplitude, target_length)
        oq_up = self._upsample(open_quotient, target_length)

        glottal_source = self.rosenberg_source(f0_up, oq_up)
        noise_source = torch.randn_like(glottal_source)

        voiced_part = self.apply_filter(glottal_source, vocal_tract_curve)
        unvoiced_part = self.apply_filter(noise_source, noise_filter_curve)

        mix = (voiced_part + unvoiced_part) * amp_up
        return mix


class LearnableReverb(nn.Module):
    def __init__(self, sample_rate, reverb_duration=1.0):
        super().__init__()
        self.sample_rate = sample_rate
        n_samples = int(sample_rate * reverb_duration)
        decay = torch.exp(-torch.linspace(0, 5, n_samples))
        noise = torch.randn(n_samples) * 0.1
        initial_ir = noise * decay
        self.impulse_response = nn.Parameter(initial_ir.view(1, 1, -1))

    def forward(self, audio):
        if audio.dim() == 2:
            x = audio.unsqueeze(1)
        else:
            x = audio

        batch, channels, dry_len = x.shape
        fft_size = dry_len + self.impulse_response.shape[-1] - 1

        size_tensor = torch.tensor(float(fft_size))
        n_fft_pow = torch.ceil(torch.log2(size_tensor))
        n_fft = int(2 ** n_fft_pow.item())

        dry_fft = torch.fft.rfft(x, n=n_fft, dim=-1)
        ir_fft = torch.fft.rfft(self.impulse_response, n=n_fft, dim=-1)
        wet_audio = torch.fft.irfft(dry_fft * ir_fft, n=n_fft, dim=-1)

        return wet_audio.squeeze(1)[..., :dry_len]


# ==========================================
# 2. Encoder & Controller
# ==========================================


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

        # Loudness
        self.spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
        self.register_buffer("a_weighting", _create_a_weighting(n_fft, sample_rate))

        # Content Encoder
        self.melspec = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.norm = nn.InstanceNorm1d(n_mels)
        self.rnn = nn.GRU(n_mels, gru_units, batch_first=True)
        self.projection = nn.Linear(gru_units, z_dim)

    def get_pitch(self, audio):
        pad = self.n_fft // 2
        audio_pad = F.pad(audio.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
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
        spec = self.spectrogram(audio) + 1e-8
        weighted_spec = spec * self.a_weighting.view(1, -1, 1)
        mean_power = torch.mean(weighted_spec, dim=1, keepdim=True)
        loudness_db = 10 * torch.log10(mean_power)
        return loudness_db.transpose(1, 2)

    def get_mels(self, audio):
        """Extract Mel-Spectrogram (used for pre-computation and content encoding)."""
        mels = self.melspec(audio)
        mels = torch.log(mels + 1e-5)
        mels = self.norm(mels)
        mels = mels.transpose(1, 2)  # [Batch, Time, Mels]
        return mels

    def get_content_from_mels(self, mels):
        """
        Runs RNN on Mels.
        Expected mels shape: [Batch, Time, Mels]
        """
        rnn_out, _ = self.rnn(mels)
        z = self.projection(rnn_out)  # [Batch, Time, z_dim]
        return z

    def forward(self, audio):
        f0 = self.get_pitch(audio)
        loudness = self.get_loudness(audio)
        mels = self.get_mels(audio)
        z = self.get_content_from_mels(mels)

        min_len = min(f0.shape[1], loudness.shape[1], z.shape[1])
        return f0[:, :min_len, :], loudness[:, :min_len, :], z[:, :min_len, :]


class VoiceController(nn.Module):
    def __init__(self, z_dim, n_filter_bins, hidden_dim=512, n_layers=3):
        super().__init__()
        # z(16) + f0(1) + loud(1) + gender(2) + age(NUM_AGE_CLASSES)
        input_dim = z_dim + 4 + NUM_AGE_CLASSES

        layers = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.LeakyReLU(0.1))

        self.mlp = nn.Sequential(*layers)
        self.proj_oq = nn.Linear(hidden_dim, 1)
        self.proj_vt = nn.Linear(hidden_dim, n_filter_bins)
        self.proj_nf = nn.Linear(hidden_dim, n_filter_bins)

    def forward(self, f0, loudness_db, z, gender, age):
        # f0, loudness: [Batch, Time, 1]
        # gender:       [Batch, Time, 2]
        # age:          [Batch, Time, NUM_AGE_CLASSES]

        # --- DECOUPLING LOGIC ---
        # 1. Log Pitch
        f0_log = torch.log(f0 + 1e-5)

        # 2. Instance Normalization (The Fix)
        # We calculate the mean pitch for THIS specific audio clip.
        f0_mean = torch.mean(f0_log, dim=1, keepdim=True)
        f0_relative = f0_log - f0_mean

        # Optional: Normalize Loudness too (removes recording gain differences)
        loudness_mean = torch.mean(loudness_db, dim=1, keepdim=True)
        loudness_relative = loudness_db - loudness_mean

        # ------------------------

        # Use RELATIVE pitch/loudness for the controller inputs
        # (We multiply by constants just to keep values roughly in -1 to 1 range)
        decoder_input = torch.cat(
            [z, f0_relative * 2.0, loudness_relative / 20.0, gender, age],
            dim=-1,
        )

        hidden = self.mlp(decoder_input)

        return {
            "f0": f0,
            "amplitude": torch.pow(10.0, loudness_db / 20.0),
            "open_quotient": torch.sigmoid(self.proj_oq(hidden)),
            "vocal_tract_curve": torch.sigmoid(self.proj_vt(hidden)),
            "noise_filter_curve": torch.sigmoid(self.proj_nf(hidden)),
        }


class VoiceAutoEncoder(nn.Module):
    def __init__(
        self,
        sample_rate,
        n_fft,
        hop_length,
        n_mels,
        z_dim=16,
        n_filter_bins=65,
    ):
        super().__init__()
        self.encoder = AudioFeatureEncoder(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            z_dim=z_dim,
        )
        self.controller = VoiceController(z_dim=z_dim, n_filter_bins=n_filter_bins)
        self.decoder = NeuralVoiceDecoder(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_filter_bins=n_filter_bins,
        )
        self.reverb = LearnableReverb(sample_rate=sample_rate, reverb_duration=0.5)

    def forward(self, audio, gender, age):
        # age shape expected: [Batch, NUM_AGE_CLASSES]
        f0, loud_db, z = self.encoder(audio)

        frames = f0.shape[1]

        # Broadcast static attributes to time axis
        # gender: [Batch, 2] -> [Batch, Time, 2]
        gender_bc = gender.unsqueeze(1).expand(-1, frames, -1)

        # age: [Batch, Classes] -> [Batch, Time, Classes]
        age_bc = age.unsqueeze(1).expand(-1, frames, -1)

        controls = self.controller(f0, loud_db, z, gender_bc, age_bc)
        dry_audio = self.decoder(
            f0=f0,
            amplitude=controls["amplitude"],
            open_quotient=controls["open_quotient"],
            vocal_tract_curve=controls["vocal_tract_curve"],
            noise_filter_curve=controls["noise_filter_curve"],
            target_length=audio.shape[-1],
        )
        wet_audio = self.reverb(dry_audio)
        return wet_audio, dry_audio, controls


class MultiScaleMelLoss(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        fft_sizes=None,
        n_mels=80,
        mel_weight: float = 1.0,
        log_mel_weight: float = 1.0,
        hop_ratio: float = 0.25,
        power: float = 1.0,
        f_min: float = 0.0,
        f_max: float | None = None,
        norm: str | None = "slaney",
    ):
        super().__init__()

        if fft_sizes is None:
            # Default fallback if not provided
            fft_sizes = [2048, 1024, 512, 256]

        if f_max is None:
            f_max = sample_rate / 2.0

        if not (0.0 <= f_min < f_max):
            raise ValueError(
                f"Expected 0 <= f_min < f_max, got f_min={f_min}, f_max={f_max}"
            )

        # Allow a single int or a list per scale
        if isinstance(n_mels, int):
            n_mels_list = [n_mels for _ in range(len(fft_sizes))]
        else:
            n_mels_list = n_mels
            assert len(n_mels_list) == len(
                fft_sizes
            ), "n_mels list must match fft_sizes length"

        self.mel_weight = mel_weight
        self.log_mel_weight = log_mel_weight

        # Build per‑scale MelSpectrogram transforms
        self.transforms = nn.ModuleList()
        for n_fft, nm in zip(fft_sizes, n_mels_list):
            hop_length = max(1, int(n_fft * hop_ratio))

            # Clamp n_mels to something the FFT grid can actually support
            n_freqs = n_fft // 2 + 1
            nm_safe = int(min(nm, n_freqs - 1))

            self.transforms.append(
                T.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=n_fft,
                    window_fn=torch.hann_window,
                    n_mels=nm_safe,
                    f_min=f_min,
                    f_max=f_max,
                    power=power,
                    normalized=False,
                    norm=norm,
                    center=True,
                    pad_mode="reflect",
                )
            )

    def forward(self, x_pred: torch.Tensor, x_target: torch.Tensor):
        # Ensure shape is (B, T)
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)
        if x_target.dim() == 1:
            x_target = x_target.unsqueeze(0)

        loss = 0.0
        eps = 1e-7
        for mel in self.transforms:
            mel_pred = mel(x_pred)
            mel_tgt = mel(x_target)

            loss = loss + self.mel_weight * F.l1_loss(mel_pred, mel_tgt)
            loss = loss + self.log_mel_weight * F.l1_loss(
                torch.log(mel_pred.clamp_min(eps)), torch.log(mel_tgt.clamp_min(eps))
            )

        return loss


# ==========================================
# 3. Data Loading
# ==========================================


class CsvAudioDataset(Dataset):
    def __init__(
        self, csv_path, clips_root, sample_rate, chunk_size, limit=4
    ):
        self.sr = sample_rate
        self.chunk_size = chunk_size
        self.clips_root = clips_root
        if pl is None:
            raise ImportError("Polars required for CsvAudioDataset")

        df = pl.read_csv(csv_path, separator=",")
        if limit > 0:
            self.data = df.sample(n=limit, with_replacement=False)
        else:
            self.data = df

        self.age_to_idx = {label: i for i, label in enumerate(AGE_LABELS)}

    def __len__(self):
        return len(self.data) * 50

    def __getitem__(self, idx):
        row = self.data.row(idx % len(self.data), named=True)
        full_path = os.path.join(self.clips_root, row["path"])

        audio, sr = torchaudio.load(full_path)

        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != self.sr:
            audio = T.Resample(sr, self.sr)(audio)

        if audio.shape[-1] > self.chunk_size:
            start = random.randint(0, audio.shape[-1] - self.chunk_size)
            chunk = audio[0, start : start + self.chunk_size]
        else:
            chunk = F.pad(audio[0], (0, self.chunk_size - audio.shape[-1]))

        # --- Age Processing ---
        age_label = row["age"]
        age_idx = self.age_to_idx.get(
            age_label, self.age_to_idx["thirties"]
        )  # Default to 30s
        age_vec = F.one_hot(torch.tensor(age_idx), num_classes=NUM_AGE_CLASSES).float()

        # --- Gender Processing ---
        gender_vec = (
            torch.tensor([0.0, 1.0])
            if "female" in row["gender"]
            else torch.tensor([1.0, 0.0])
        )
        return chunk, gender_vec, age_vec


# ==========================================
# 4. Inference Helper
# ==========================================


def convert_voice(
    model,
    input_wav,
    output_wav,
    target_gender_balance,
    target_age_float,
    pitch_shift=0.0,
    device="cpu",
    sample_rate=16000,
):
    audio, sr = torchaudio.load(input_wav)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != sample_rate:
        audio = T.Resample(sr, sample_rate)(audio)

    audio = audio / (torch.abs(audio).max() + 1e-6)
    audio = audio.to(device)

    model.eval()
    with torch.no_grad():
        # 1. Extract Features
        f0, loud_db, z = model.encoder(audio)

        # 2. Pitch Shift
        if pitch_shift != 0.0:
            f0 = f0 * (2 ** (pitch_shift / 12.0))

        # 3. Create Condition Tensors
        frames = f0.shape[1]

        # Gender
        gm = 1.0 - target_gender_balance
        gf = target_gender_balance
        g_vec = torch.tensor([gm, gf], device=device).float()
        g_tensor = g_vec.view(1, 1, 2).expand(1, frames, -1)

        # Age
        a_vec = float_to_weighted_age(target_age_float, device=device)
        a_tensor = a_vec.view(1, 1, -1).expand(1, frames, -1)

        # 4. Controller
        controls = model.controller(f0, loud_db, z, g_tensor, a_tensor)

        # 5. Decode
        dry_audio = model.decoder(
            f0=controls["f0"],
            amplitude=controls["amplitude"],
            open_quotient=controls["open_quotient"],
            vocal_tract_curve=controls["vocal_tract_curve"],
            noise_filter_curve=controls["noise_filter_curve"],
            target_length=audio.shape[-1],
        )
        wet_audio = model.reverb(dry_audio)

    wet_audio = wet_audio.cpu()
    wet_audio = wet_audio / (torch.abs(wet_audio).max() + 1e-6)
    torchaudio.save(output_wav, wet_audio, sample_rate)
    print(f"Saved: {output_wav}")


# ==========================================
# 5. Main Execution
# ==========================================

# --- Configuration ---
# PATHS
CSV_PATH = "data/filtered_dataset.csv"
TSV_PATH = "/mnt/data/ai/cv-corpus-23.0-2025-09-05/tr/validated.tsv"
CLIPS_DIR = "/mnt/data/ai/cv-corpus-23.0-2025-09-05/tr/clips"
CHECKPOINT_DIR = "checkpoints"
INPUT_WAV = "/mnt/Data/Audio/misc/haiku.mp3"
CHECKPOINT = f"{CHECKPOINT_DIR}/model_last.pth"

# AUDIO & DSP CONSTANTS
SAMPLE_RATE = 48000
N_FFT = 2048         # Increased for 48k (approx 42ms window)
HOP_LENGTH = 512     # Approx 10ms hop
N_MELS = 80
N_FILTER_BINS = 65   # Resolution of the vocal tract controller output
Z_DIM = 16

# TRAINING HYPERPARAMETERS
N_EPOCHS = 100
BATCH_SIZE = 16
N_CHECKPOINTS = 1000
N_LOG = 100
LIMIT = 0            # 0 = Use Full Dataset

# LOSS CONFIG
# Multiscale FFT sizes appropriate for 48kHz
LOSS_FFT_SIZES = [4096, 2048, 1024, 512]
LOSS_N_MELS = [80, 80, 64, 40]


DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"


def training():
    # --- Setup ---
    # Pass all configuration constants explicitly
    model = VoiceAutoEncoder(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        z_dim=Z_DIM,
        n_filter_bins=N_FILTER_BINS,
    ).to(DEVICE)

    # Load Data
    # Use Global SAMPLE_RATE and a Chunk Size dependent on SR
    chunk_samples = 2 * SAMPLE_RATE  # 2 seconds per chunk
    dataset = CsvAudioDataset(
        CSV_PATH,
        CLIPS_DIR,
        limit=LIMIT,
        sample_rate=SAMPLE_RATE,
        chunk_size=chunk_samples,
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    criterion = MultiScaleMelLoss(
        sample_rate=SAMPLE_RATE,
        fft_sizes=LOSS_FFT_SIZES,
        n_mels=LOSS_N_MELS,
    ).to(DEVICE)

    print(f"Starting Training on {DEVICE}...")
    print(f"SR: {SAMPLE_RATE}, FFT: {N_FFT}, Hop: {HOP_LENGTH}")
    print(f"Dataset Size: {len(dataset)} | Batch Size: {BATCH_SIZE}")

    num_batches = 0

    for epoch in range(1, N_EPOCHS + 1):
        batch_loss_accum = 0.0

        for i, (audio, gender, age) in enumerate(dataloader):
            num_batches += 1

            audio = audio.to(DEVICE)
            gender = gender.to(DEVICE)
            age = age.to(DEVICE)

            optimizer.zero_grad()

            # Forward
            wet, dry, _ = model(audio, gender, age)

            # Loss
            loss = criterion(wet, audio)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_loss_accum += loss.item()

            # --- Batch-Level Logging ---
            if num_batches % N_LOG == 0:
                avg_loss = batch_loss_accum / N_LOG
                print(f"Epoch {epoch} | Batch {num_batches} | Loss: {avg_loss:.4f}")
                batch_loss_accum = 0.0

            # --- Batch-Level Checkpointing ---
            if num_batches % N_CHECKPOINTS == 0:
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)

                # 1. Save Model
                save_path = f"{CHECKPOINT_DIR}/model_step{num_batches}.pth"
                torch.save(model.state_dict(), save_path)
                save_path_last = f"{CHECKPOINT_DIR}/model_last.pth"
                torch.save(model.state_dict(), save_path_last)

                # 2. Save Random Reconstruction
                with torch.no_grad():
                    ridx = random.randint(0, audio.shape[0] - 1)
                    tgt = audio[ridx].cpu().numpy()
                    out = wet[ridx].cpu().numpy()

                    g_str = "Fem" if gender[ridx, 0].item() < 0.5 else "Male"
                    age_idx = torch.argmax(age[ridx]).item()
                    age_str = AGE_LABELS[age_idx]

                    fname_base = (
                        f"{CHECKPOINT_DIR}/step{num_batches}_{g_str}_{age_str}"
                    )
                    scipy.io.wavfile.write(f"{fname_base}_target.wav", SAMPLE_RATE, tgt)
                    scipy.io.wavfile.write(f"{fname_base}_recon.wav", SAMPLE_RATE, out)

                print(f"--> Saved Checkpoint: {save_path}")


def inference():
    # --- Inference Demo ---
    model = VoiceAutoEncoder(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        z_dim=Z_DIM,
        n_filter_bins=N_FILTER_BINS,
    ).to(DEVICE)

    if os.path.exists(CHECKPOINT) and os.path.exists(INPUT_WAV):
        print(f"Loading {CHECKPOINT}...")
        model.load_state_dict(
            torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
        )

        convert_voice(
            model,
            INPUT_WAV,
            "generated/output_female.wav",
            1.0,
            0.4,
            pitch_shift=7.0,
            device=DEVICE,
            sample_rate=SAMPLE_RATE,
        )
        convert_voice(
            model,
            INPUT_WAV,
            "generated/output_male.wav",
            0.0,
            0.4,
            pitch_shift=-7.0,
            device=DEVICE,
            sample_rate=SAMPLE_RATE,
        )
    else:
        print("Checkpoint or Input Wav not found.")


if __name__ == "__main__":
    training()
