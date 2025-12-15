import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from matplotlib.figure import Figure
    import librosa
    import human_ddsp.main as hd
    import torch

    SR = 48000
    HOP = 512
    BATCH = 1
    FRAMES = 300
    NUM_FORMANTS = 4
    F0 = 110.
    OQ = 0.4
    V_GAIN = 0.6
    U_GAIN = 0.3
    N_AGE = 5
    K_Z = 16


@app.cell
def _():
    model = hd.GlottalFormantSynthesizer(SR, HOP).eval()

    # Dummy Inputs
    f0 = torch.linspace(F0, 2.0 * F0, FRAMES).unsqueeze(0).unsqueeze(-1)
    oq = 0.6 * torch.ones(BATCH, FRAMES, 1)
    vg = 0.8 * torch.ones(BATCH, FRAMES, 1)
    ug = 0.05 * torch.ones(BATCH, FRAMES, 1)

    ff = torch.tensor([800., 1200., 2500., 3500., 4400.]).view(1, 1, -1).expand(BATCH, FRAMES, -1)
    ff = ff + torch.linspace(0, -400, FRAMES).unsqueeze(0).unsqueeze(-1)
    fw = torch.tensor([50., 100., 200., 300., 10.]).view(1, 1, -1).expand(BATCH, FRAMES, -1)
    fw = fw + torch.linspace(0, -25, FRAMES).unsqueeze(0).unsqueeze(-1)

    with torch.no_grad():
        audio = model(f0, oq, vg, ug, ff, fw)

    print("Output Shape:", audio.shape)  # Expect [2, 25600, 1]
    mo.audio(src=audio.squeeze().numpy(), rate=48000)
    return f0, model


@app.cell
def _(f0, model):
    # Instantiate
    controller = hd.DDSPController(
        sample_rate=SR,
        hop_length=HOP,
        n_age_groups=N_AGE,
        k_z_dim=K_Z
    ).eval()

    # Dummy Inputs
    # Normalized F0 (0 to 1)
    norm_f0 = f0 - f0.mean()
    loudness = torch.rand(BATCH, FRAMES, 1)
    gender = torch.randn(BATCH, FRAMES, 2)  # Continuous embedding or one-hot
    age = torch.randn(BATCH, FRAMES, N_AGE)
    z = torch.randn(BATCH, FRAMES, K_Z)

    # Forward Pass
    with torch.no_grad():
        controls_out = controller(norm_f0, loudness, gender, age, z)

    print(f"Formant Freqs Sample: {controls_out['formant_freqs'][0, 0, :]}")
    # Check if Freqs are sorted (increasing)
    with torch.no_grad():
        audio_out = model(**controls_out)
    mo.audio(src=audio_out.squeeze().numpy(), rate=48000)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
