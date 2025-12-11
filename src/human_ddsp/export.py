# pyright: basic

import os

import torch
import torch.nn as nn

import human_ddsp.main as main  # Your model file


class StreamingVoiceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # CONFIGURATION
        self.block_size = 256  # Matches nn~ buffer size
        self.n_fft = 1024      # Required by STFT
        
        # INTERNAL STATE 1: Audio History
        # We need (n_fft - block_size) of history + block_size new samples = n_fft total
        # 1024 - 256 = 768 samples of history
        self.history_len = self.n_fft - self.block_size
        self.register_buffer('audio_history', torch.zeros(1, self.history_len))
        
        # INTERNAL STATE 2: RNN State
        # [1 layer, 1 batch, 256 hidden]
        self.register_buffer('rnn_state', torch.zeros(1, 1, 256))

    def forward(self, x):
        """
        Input: x [Batch, 4, 256]
        Output: y [Batch, 1, 256]
        """
        # 1. Unpack Inputs (Control rate is sampled at end of block)
        audio_in = x[:, 0:1, :]    # [B, 1, 256]
        gender = x[:, 1, -1]       # [B]
        age = x[:, 2, -1]          # [B]
        pitch = x[:, 3, -1]        # [B]

        # 2. Construct Analysis Window (History + New Block)
        # [1, 768] + [1, 256] = [1, 1024]
        window = torch.cat([self.audio_history, audio_in.squeeze(1)], dim=1)
        
        # 3. Update History (Slide window left)
        # Keep the last 768 samples of the CURRENT window
        self.audio_history.copy_(window[:, -self.history_len:])
        
        # 4. Feature Extraction (On 1024 samples)
        # We only need the features for the *last* frame (the current block)
        # encoder methods return [B, Frames, Feat]. We take index -1.
        f0 = self.model.encoder.get_pitch(window)[:, -1:, :]
        loud = self.model.encoder.get_loudness(window)[:, -1:, :]
        mels = self.model.encoder.get_mels(window)[:, -1:, :]
        
        # 5. RNN Step
        rnn_out, new_state = self.model.encoder.rnn(mels, self.rnn_state)
        self.rnn_state.copy_(new_state)
        z = self.model.encoder.projection(rnn_out)
        
        # 6. Conditions & Decoding
        shift_factor = torch.pow(2.0, pitch / 12.0).view(-1, 1, 1)
        f0 = f0 * shift_factor
        
        gm = 1.0 - gender
        gf = gender
        g_tensor = torch.stack([gm, gf], dim=-1).unsqueeze(1)
        a_tensor = age.view(-1, 1, 1)

        controls = self.model.controller(f0, loud, z, g_tensor, a_tensor)
        
        # 7. Synthesis
        # Generate exactly 256 samples
        dry_audio = self.model.decoder(
            f0=controls['f0'],
            amplitude=controls['amplitude'],
            open_quotient=controls['open_quotient'],
            vocal_tract_curve=controls['vocal_tract_curve'],
            noise_filter_curve=controls['noise_filter_curve'],
            target_length=self.block_size
        )
        
        return dry_audio.unsqueeze(1)

def export_for_nn(checkpoint_path, output_path):
    # Load Weights
    model = main.VoiceAutoEncoder(sample_rate=16000)
    if os.path.exists(checkpoint_path):
        print(f"Loading weights: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    # Wrap
    wrapper = StreamingVoiceWrapper(model)
    wrapper.eval()

    # Trace with 256 block size (The intended nn~ buffer size)
    example_input = torch.zeros(1, 4, 256)
    
    print("Tracing model...")
    traced_model = torch.jit.trace(wrapper, example_input)
    
    # Metadata for nn~
    extra_files = {'attributes.json': '{"description": "DDSP Voice: Audio, Gender, Age, Pitch"}'}
    
    traced_model.save(output_path, _extra_files=extra_files)
    print(f"Success! Exported to {output_path}")

if __name__ == "__main__":
    export_for_nn("checkpoints/model_last.pth", "ddsp_rt.ts")
