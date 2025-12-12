# human-ddsp

Lightweight differentiable digital signal processing (DDSP) voice autoencoder with a streaming-friendly TorchScript export for nn~ (nn_tilde).

## Project layout
- `src/human_ddsp/main.py` – core model pieces:
  - `AudioFeatureEncoder` extracts pitch, loudness, and a bottlenecked content embedding.
  - `VoiceController` maps content + conditioning (gender/age) to synthesis controls.
  - `NeuralVoiceDecoder` renders voiced/unvoiced excitation and filter curves; `LearnableReverb` adds room response.
  - `VoiceAutoEncoder` stitches encoder/controller/decoder together.
  - `CsvAudioDataset` reads audio metadata and produces normalized chunks for training.
  - `convert_voice` performs offline conversion; `process_tsv` filters TSV metadata.
- `src/human_ddsp/export.py` – `ScriptedDDSP` wraps a pretrained autoencoder for nn_tilde / TorchScript, with streaming state (phase accumulator + RNN hidden state) and registered attributes for pitch shift / gender / age.
- `src/human_ddsp/__init__.py` – package entry.

## Improvements in this change set
- Made `LearnableReverb` use a device-agnostic next-power-of-two calculation, simplifying TorchScript export and avoiding unnecessary tensor allocations.
- Added device-safe state handling in `ScriptedDDSP.forward` so RNN and phase buffers follow the incoming audio device even when batch size is constant.
- Cached the block Hann window as a registered buffer to cut per-block allocations during streaming inference.
- Wrapped streaming inference in `torch.no_grad()` and added type hints for exported methods and helper utilities to improve readability and TorchScript clarity.

## Usage notes
- Training entrypoint is `training()` in `main.py`; it expects a CSV listing audio files (see `CSV_PATH`, `CLIPS_DIR`).
- Offline conversion: call `convert_voice` with a trained checkpoint and target attributes.
- nn_tilde export: run `python -m human_ddsp.export` to script the model; the exported module consumes `[audio, pitch_shift, gender, age]` channels per block.

## Future suggestions
- Add unit tests around streaming export (state resets, device transfers, pitch-shift correctness).
- Provide config-driven training (learning rate, loss weights, FFT sizes) and a small sample dataset for smoke tests.
- Integrate lightweight evaluation metrics (PESQ/STOI surrogate, loudness/pitch RMSE) and logging (TensorBoard/W&B).
- Consider replacing the simple Rosenberg source with a neural source model and experiment with higher sample rate decoders.
