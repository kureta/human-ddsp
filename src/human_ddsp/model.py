import torch
import torch.nn as nn
import torch.nn.functional as F

from human_ddsp.config import AudioConfig
from human_ddsp.dsp import GlottalPulseSynth, FormantFilter, LearnableReverb
from human_ddsp.features import PitchDetector, LoudnessDetector, MfcContentExtractor

# Define AGE_LABELS and NUM_AGE_CLASSES at the module level
AGE_LABELS = [
    "teens", "twenties", "thirties", "fourties", "fifties", 
    "sixties", "seventies", "eighties", "nineties"
]
NUM_AGE_CLASSES = len(AGE_LABELS)


class VoiceController(nn.Module):
    """
    Predicts synthesis parameters from features and speaker identity.
    """
    def __init__(self, config: AudioConfig):
        super().__init__()
        
        # Input: f0 (1), loudness (1), content_embedding (content_dim), gender (2), age (NUM_AGE_CLASSES)
        input_dim = 1 + 1 + config.content_dim + 2 + NUM_AGE_CLASSES
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=0.2 if config.num_layers > 1 else 0,
        )
        
        # Shared MLP block after GRU
        self.shared_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.LeakyReLU(0.1),
        )
        
        # Projection layers to output DSP parameters
        self.proj_voiced_amp = nn.Linear(config.hidden_size, 1)
        self.proj_unvoiced_amp = nn.Linear(config.hidden_size, 1)
        self.proj_open_quotient = nn.Linear(config.hidden_size, 1)
        self.proj_steepness = nn.Linear(config.hidden_size, 1)
        self.proj_spectral_tilt = nn.Linear(config.hidden_size, 1)
        
        self.proj_formant_freqs = nn.Linear(config.hidden_size, config.n_formants)
        self.proj_formant_bandwidths = nn.Linear(config.hidden_size, config.n_formants)
        self.proj_formant_amplitudes = nn.Linear(config.hidden_size, config.n_formants)

    def forward(
        self,
        f0: torch.Tensor,          # [B, T, 1]
        loudness: torch.Tensor,    # [B, T, 1]
        content: torch.Tensor,     # [B, T, content_dim]
        gender: torch.Tensor,      # [B, T, 2]
        age: torch.Tensor,         # [B, T, NUM_AGE_CLASSES]
    ) -> dict[str, torch.Tensor]:
        
        # --- Decoupling via Normalization ---
        # Normalize f0 and loudness per utterance
        f0_mean = torch.mean(f0, dim=1, keepdim=True)
        f0_relative = f0 - f0_mean
        
        loudness_mean = torch.mean(loudness, dim=1, keepdim=True)
        loudness_relative = loudness - loudness_mean
        
        # Concatenate all inputs
        controller_input = torch.cat([
            f0_relative, loudness_relative, content, gender, age
        ], dim=-1)
        
        # Process through GRU
        gru_out, _ = self.gru(controller_input)
        
        # Process through shared MLP
        hidden = self.shared_mlp(gru_out)
        
        # Project to synthesis parameters
        voiced_amp = torch.sigmoid(self.proj_voiced_amp(hidden))
        unvoiced_amp = torch.sigmoid(self.proj_unvoiced_amp(hidden))
        open_quotient = torch.sigmoid(self.proj_open_quotient(hidden)) * 0.8 + 0.1 # Range [0.1, 0.9]
        # Scale steepness to a reasonable range, e.g., [50, 250]
        steepness = torch.sigmoid(self.proj_steepness(hidden)) * 200 + 50
        # Spectral tilt as pre-emphasis coeff, [0, 1]
        spectral_tilt = torch.sigmoid(self.proj_spectral_tilt(hidden))
        
        formant_freqs = torch.sigmoid(self.proj_formant_freqs(hidden)) # Normalized [0, 1]
        formant_bandwidths = torch.sigmoid(self.proj_formant_bandwidths(hidden)) # Normalized [0, 1]
        formant_amplitudes = torch.sigmoid(self.proj_formant_amplitudes(hidden)) # Normalized [0, 1]
        
        return {
            "f0": f0, # Pass through original f0 for synthesis
            "voiced_amp": voiced_amp,
            "unvoiced_amp": unvoiced_amp,
            "open_quotient": open_quotient,
            "steepness": steepness,
            "spectral_tilt": spectral_tilt,
            "formant_freqs": formant_freqs,
            "formant_bandwidths": formant_bandwidths,
            "formant_amplitudes": formant_amplitudes,
        }


class VoiceSynth(nn.Module):
    """
    The main model that combines feature extraction, controller, and DSP.
    """
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        
        # Feature Extractors
        self.pitch_detector = PitchDetector(config)
        self.loudness_detector = LoudnessDetector(config)
        self.content_extractor = MfcContentExtractor(config)
        
        # Controller
        self.controller = VoiceController(config)
        
        # DSP Modules
        self.glottal_synth = GlottalPulseSynth(config)
        self.formant_filter = FormantFilter(config)
        self.reverb = LearnableReverb(config)

    def _upsample(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        """Upsamples a control signal to audio rate."""
        # x shape: [B, T, C] -> [B, C, T] for interpolation
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_length, mode='linear', align_corners=False)
        # -> [B, T_audio, C]
        return x.transpose(1, 2)

    def forward(
        self,
        audio: torch.Tensor,    # [B, L_audio]
        gender: torch.Tensor,   # [B, 2]
        age: torch.Tensor,      # [B, NUM_AGE_CLASSES]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        
        # 1. Extract Features
        f0 = self.pitch_detector(audio)
        loudness = self.loudness_detector(audio)
        content = self.content_extractor(audio)
        
        # Ensure all features have the same time dimension
        min_len = min(f0.shape[1], loudness.shape[1], content.shape[1])
        f0 = f0[:, :min_len, :]
        loudness = loudness[:, :min_len, :]
        content = content[:, :min_len, :]
        
        # 2. Broadcast speaker identity to match time dimension
        gender_bc = gender.unsqueeze(1).expand(-1, min_len, -1)
        age_bc = age.unsqueeze(1).expand(-1, min_len, -1)
        
        # 3. Get synthesis parameters from controller
        controls = self.controller(f0, loudness, content, gender_bc, age_bc)
        
        # 4. Upsample control signals to audio rate
        target_len = audio.shape[-1]
        
        f0_up = self._upsample(controls["f0"], target_len).squeeze(-1)
        oq_up = self._upsample(controls["open_quotient"], target_len).squeeze(-1)
        steepness_up = self._upsample(controls["steepness"], target_len).squeeze(-1)
        tilt_up = self._upsample(controls["spectral_tilt"], target_len).squeeze(-1)
        voiced_amp_up = self._upsample(controls["voiced_amp"], target_len).squeeze(-1)
        unvoiced_amp_up = self._upsample(controls["unvoiced_amp"], target_len).squeeze(-1)
        
        # 5. Synthesize Voiced Part
        glottal_source = self.glottal_synth(f0_up, oq_up, steepness_up, tilt_up)
        
        # Formant freqs need to be scaled from [0, 1] to a reasonable Hz range
        # Let's say 50 Hz to 7000 Hz
        formant_freqs_scaled = controls["formant_freqs"] * (self.config.f_max - 50.0) + 50.0
        
        # Bandwidths from [0, 1] to a range, e.g., 10 Hz to 1000 Hz
        formant_bws_scaled = controls["formant_bandwidths"] * 990.0 + 10.0
        
        voiced_signal = self.formant_filter(
            glottal_source,
            formant_freqs_scaled,
            formant_bws_scaled,
            controls["formant_amplitudes"]
        )
        
        # 6. Synthesize Unvoiced Part
        noise_source = torch.randn_like(glottal_source)
        unvoiced_signal = self.formant_filter(
            noise_source,
            formant_freqs_scaled,
            formant_bws_scaled,
            controls["formant_amplitudes"]
        )
        
        # 7. Mix and apply reverb
        dry_audio = (voiced_signal * voiced_amp_up) + (unvoiced_signal * unvoiced_amp_up)
        wet_audio = self.reverb(dry_audio)
        
        return wet_audio, dry_audio, controls
