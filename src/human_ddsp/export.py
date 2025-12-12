# pyright: basic

import math
import os

import torch
import torch.nn.functional as F
from nn_tilde import Module

from human_ddsp import main


class ScriptedDDSP(Module):
    def __init__(self, pretrained: main.VoiceAutoEncoder, block_size: int = 256):
        super().__init__()

        self.model = pretrained
        self.model.eval()

        # 1. Configuration
        self.sr = 48000
        self.block_size = block_size
        self.n_fft = 1024

        # 2. Register Attributes
        self._pitch_shift = 0.0
        self._gender = 0.0
        self._age = 0.3

        self.register_attribute("pitch_shift", 0.0)
        self.register_attribute("gender", 0.0)
        self.register_attribute("age", 0.3)

        # 3. STATE BUFFERS
        self.register_buffer("rnn_state", torch.zeros(1, 1, 256))
        self.register_buffer("phase_accum", torch.zeros(1, 1, 1))
        self.register_buffer(
            "block_window", torch.hann_window(self.block_size, device=torch.device("cpu"))
        )

        # 4. Register Method
        self.register_method(
            "forward",
            in_channels=4,
            in_ratio=1,
            out_channels=1,
            out_ratio=1,
            input_labels=["Audio", "Pitch", "Gender", "Age"],
            output_labels=["Audio Out"],
        )

    # --- GETTERS & SETTERS ---
    @torch.jit.export
    def get_pitch_shift(self) -> float:
        return self._pitch_shift

    @torch.jit.export
    def set_pitch_shift(self, value: float) -> int:
        self._pitch_shift = value
        return 0

    @torch.jit.export
    def get_gender(self) -> float:
        return self._gender

    @torch.jit.export
    def set_gender(self, value: float) -> int:
        self._gender = value
        return 0

    @torch.jit.export
    def get_age(self) -> float:
        return self._age

    @torch.jit.export
    def set_age(self, value: float) -> int:
        self._age = value
        return 0

    # -------------------------

    def stateful_rosenberg(
        self, f0: torch.Tensor, open_quotient: torch.Tensor
    ) -> torch.Tensor:
        """Streaming Rosenberg source with phase tracking."""
        phase_step = f0 / self.sr
        phase_block = torch.cumsum(phase_step, dim=1)
        current_phase = self.phase_accum + phase_block
        p = current_phase - torch.floor(current_phase)

        # Update accumulator for next block
        last_val = current_phase[:, -1:, :]
        self.phase_accum.copy_(last_val - torch.floor(last_val))

        oq = torch.clamp(open_quotient, 0.1, 0.9)
        scaled_phase = p / (oq + 1e-8)

        pulse = 0.5 * (1.0 - torch.cos(math.pi * scaled_phase))
        mask = torch.sigmoid((oq - p) * 100.0)
        glottal_wave = pulse * mask

        diff_wave = torch.zeros_like(glottal_wave)
        diff_wave[:, 1:, :] = glottal_wave[:, 1:, :] - glottal_wave[:, :-1, :]
        diff_wave[:, 0, :] = glottal_wave[:, 0, :]

        return diff_wave

    def apply_filter_single_frame(
        self, excitation: torch.Tensor, filter_curve: torch.Tensor
    ) -> torch.Tensor:
        """Applies spectral filter to a block_size-sample block."""
        # excitation: [Batch, block_size, 1]
        ex_t = excitation.transpose(1, 2)

        # Standard RFFT (n_fft -> n_fft/2+1 bins)
        spec = torch.fft.rfft(ex_t, n=self.n_fft)

        curve_interp = F.interpolate(
            filter_curve, size=spec.shape[-1], mode="linear", align_corners=False
        )

        filtered_spec = spec * curve_interp
        audio = torch.fft.irfft(filtered_spec, n=self.n_fft)

        # Windowing to smooth block boundaries
        window = self.block_window.to(audio.device)
        audio = audio * window

        return audio.transpose(1, 2)

    def process_block(self, x_chunk: torch.Tensor) -> torch.Tensor:
        """
        Processes exactly one block of `block_size` samples.
        x_chunk: [Batch, 4, block_size]
        """
        # 1. Unpack
        audio = x_chunk[:, 0:1, :]
        p_shift = x_chunk[:, 1, :].mean(dim=-1)
        gen_val = x_chunk[:, 2, :].mean(dim=-1)
        age_val = x_chunk[:, 3, :].mean(dim=-1)

        # 2. Encoder Analysis (On block_size samples)
        window = audio.squeeze(1)

        # Take LAST frame features
        f0_enc = self.model.encoder.get_pitch(window)[:, -1:, :]
        loud_enc = self.model.encoder.get_loudness(window)[:, -1:, :]
        mels = self.model.encoder.get_mels(window)[:, -1:, :]

        # 3. RNN (State Update)
        rnn_out, new_state = self.model.encoder.rnn(mels, self.rnn_state)
        self.rnn_state.copy_(new_state)
        z = self.model.encoder.projection(rnn_out)

        # 4. Conditions
        shift_factor = torch.pow(2.0, p_shift / 12.0).view(-1, 1, 1)
        f0_shifted = f0_enc * shift_factor

        gm, gf = 1.0 - gen_val, gen_val
        g_tensor = torch.stack([gm, gf], dim=-1).unsqueeze(1)
        a_tensor = age_val.view(-1, 1, 1)

        # 5. Controller
        controls = self.model.controller(f0_shifted, loud_enc, z, g_tensor, a_tensor)

        # 6. Synthesis
        # Expand to block_size (self.block_size)
        f0_up = f0_shifted.expand(-1, self.block_size, -1)
        amp_up = controls["amplitude"].expand(-1, self.block_size, -1)
        oq_up = controls["open_quotient"].expand(-1, self.block_size, -1)

        # Phase Accumulator updates inside here
        glottal = self.stateful_rosenberg(f0_up, oq_up)
        noise = torch.randn_like(glottal)

        voiced_audio = self.apply_filter_single_frame(
            glottal, controls["vocal_tract_curve"]
        )
        unvoiced_audio = self.apply_filter_single_frame(
            noise, controls["noise_filter_curve"]
        )

        mix = (voiced_audio + unvoiced_audio) * amp_up
        return mix.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: [Batch, 4, Time]
        Handles arbitrary buffer sizes by splitting into block_size chunks.
        Inference-only: gradients are disabled for streaming export.
        """
        with torch.no_grad():
            # Check Batch Size & Resize States
            current_batch = x.shape[0]
            target_device = x.device
            if self.rnn_state.device != target_device:
                self.rnn_state = self.rnn_state.to(target_device)
            if self.rnn_state.shape[1] != current_batch:
                self.rnn_state.data.resize_(1, current_batch, 256)
            self.rnn_state.data.zero_()

            if self.phase_accum.device != target_device:
                self.phase_accum = self.phase_accum.to(target_device)
            if self.phase_accum.shape[0] != current_batch:
                self.phase_accum.data.resize_(current_batch, 1, 1)
            self.phase_accum.data.zero_()

            # Split into chunks of block_size
            chunks = torch.split(x, self.block_size, dim=-1)

            outputs = []
            for chunk in chunks:
                # Only process if chunk is full size (block_size)
                # nn_tilde should deliver exact multiples of block_size.
                if chunk.shape[-1] == self.block_size:
                    out_chunk = self.process_block(chunk)
                    outputs.append(out_chunk)
                else:
                    # Fallback for partial chunks (silence or just skip)
                    # Max/MSP should be configured to block_size, so this shouldn't happen often.
                    outputs.append(
                        torch.zeros(current_batch, 1, chunk.shape[-1], device=target_device)
                    )

            # Concatenate back
            return torch.cat(outputs, dim=-1)


def main_export():
    checkpoint_path = "checkpoints/model_last.pth"
    base_model = main.VoiceAutoEncoder(sample_rate=48000)

    if os.path.exists(checkpoint_path):
        print(f"Loading {checkpoint_path}...")
        base_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    scripted_model = ScriptedDDSP(base_model)
    scripted_model.export_to_ts("ddsp_voice.ts")
    print("Done.")


if __name__ == "__main__":
    main_export()
