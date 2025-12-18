import os
import torch
import torchaudio
import argparse

from human_ddsp.config import AudioConfig
from human_ddsp.model import VoiceSynth, AGE_LABELS, NUM_AGE_CLASSES
from human_ddsp.train import train as run_training


# --- Configuration ---
CSV_PATH = "data/filtered_dataset.csv"
CLIPS_DIR = "data/cv-corpus-23.0-2025-09-05/tr/clips"
CHECKPOINT_DIR = "checkpoints"
INPUT_WAV = "/Users/kureta/Music/Random Samples/rumi.wav"
CHECKPOINT_FILE = f"{CHECKPOINT_DIR}/model_last.pth"


def float_to_weighted_age(age_float: float, device="cpu") -> torch.Tensor:
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


def inference(
    input_wav: str,
    output_wav: str,
    checkpoint_path: str,
    target_gender_balance: float, # 0.0 for male, 1.0 for female
    target_age_float: float,      # 0.0 to 1.0
    pitch_shift: float = 0.0,
):
    """
    Runs voice conversion on an input audio file.
    """
    config = AudioConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Model ---
    model = VoiceSynth(config).to(device)
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Aborting.")
        return
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    # --- Load and Prepare Audio ---
    audio, sr = torchaudio.load(input_wav)
    if sr != config.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, config.sample_rate)
        audio = resampler(audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    audio = audio.to(device)
    audio = audio / (torch.abs(audio).max() + 1e-6)

    # --- Prepare Control Tensors ---
    # Gender
    g_male = 1.0 - target_gender_balance
    g_female = target_gender_balance
    gender_vec = torch.tensor([g_male, g_female], device=device).float().unsqueeze(0)

    # Age
    age_vec = float_to_weighted_age(target_age_float, device=device).unsqueeze(0)

    # --- Run Inference ---
    with torch.no_grad():
        # 1. Extract features from the original audio
        f0 = model.pitch_detector(audio)
        loudness = model.loudness_detector(audio)
        content = model.content_extractor(audio)

        # Optional: Pitch Shift
        if pitch_shift != 0.0:
            f0 = f0 * (2 ** (pitch_shift / 12.0))

        # 2. Broadcast speaker identity
        min_len = f0.shape[1]
        gender_bc = gender_vec.unsqueeze(1).expand(-1, min_len, -1)
        age_bc = age_vec.unsqueeze(1).expand(-1, min_len, -1)

        # 3. Get controls from the controller using new identity
        controls = model.controller(f0, loudness, content, gender_bc, age_bc)

        # 4. Synthesize audio using the new controls
        target_len = audio.shape[-1]
        f0_up = model._upsample(controls["f0"], target_len).squeeze(-1)
        oq_up = model._upsample(controls["open_quotient"], target_len).squeeze(-1)
        steepness_up = model._upsample(controls["steepness"], target_len).squeeze(-1)
        tilt_up = model._upsample(controls["spectral_tilt"], target_len).squeeze(-1)
        voiced_amp_up = model._upsample(controls["voiced_amp"], target_len).squeeze(-1)
        unvoiced_amp_up = model._upsample(controls["unvoiced_amp"], target_len).squeeze(-1)
        
        glottal_source = model.glottal_synth(f0_up, oq_up, steepness_up, tilt_up)
        
        formant_freqs_scaled = controls["formant_freqs"] * (config.f_max - 50.0) + 50.0
        formant_bws_scaled = controls["formant_bandwidths"] * 990.0 + 10.0
        
        voiced_signal = model.formant_filter(
            glottal_source,
            formant_freqs_scaled,
            formant_bws_scaled,
            controls["formant_amplitudes"]
        )
        
        noise_source = torch.randn_like(glottal_source)
        unvoiced_signal = model.formant_filter(
            noise_source,
            formant_freqs_scaled,
            formant_bws_scaled,
            controls["formant_amplitudes"]
        )
        
        dry_audio = (voiced_signal * voiced_amp_up) + (unvoiced_signal * unvoiced_amp_up)
        wet_audio = model.reverb(dry_audio)

    # --- Save Output ---
    os.makedirs(os.path.dirname(output_wav), exist_ok=True)
    torchaudio.save(output_wav, wet_audio.cpu(), config.sample_rate)
    print(f"Saved converted audio to {output_wav}")


def main():
    parser = argparse.ArgumentParser(description="Human DDSP: Train or Run Inference")
    parser.add_argument('mode', choices=['train', 'inference'], help="Mode to run: 'train' or 'inference'")
    
    # Inference arguments
    parser.add_argument('--input', type=str, default=INPUT_WAV, help="Input WAV file for inference")
    parser.add_argument('--output', type=str, default="generated/output.wav", help="Output WAV file for inference")
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_FILE, help="Path to model checkpoint")
    parser.add_argument('--gender', type=float, default=0.5, help="Target gender balance (0.0 male, 1.0 female)")
    parser.add_argument('--age', type=float, default=0.5, help="Target age (0.0 to 1.0)")
    parser.add_argument('--pitch_shift', type=float, default=0.0, help="Pitch shift in semitones")

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--limit', type=int, default=0, help="Limit dataset size (0 for full dataset)")

    args = parser.parse_args()

    if args.mode == 'train':
        run_training(
            csv_path=CSV_PATH,
            clips_root=CLIPS_DIR,
            checkpoint_dir=CHECKPOINT_DIR,
            n_epochs=args.epochs,
            limit=args.limit,
        )
    elif args.mode == 'inference':
        inference(
            input_wav=args.input,
            output_wav=args.output,
            checkpoint_path=args.checkpoint,
            target_gender_balance=args.gender,
            target_age_float=args.age,
            pitch_shift=args.pitch_shift,
        )

if __name__ == "__main__":
    main()
