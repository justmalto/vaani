import io, base64
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torchaudio
from scipy.ndimage import gaussian_filter1d  # very small dependency

SAMPLE_RATE = 16000
N_MELS = 64

EMOTION_COLORS = {
    "Angry": (255, 0, 0),
    "Disgust": (34, 139, 34),
    "Fear": (138, 43, 226),
    "Happy": (255, 215, 0),
    "Neutral": (200, 200, 200),
    "Pleasant_surprise": (255, 140, 0),
    "Sad": (70, 130, 180),
}


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# -----------------------------------------------------------------------------
# MEL SPECTROGRAM (NO LIBROSA VERSION)
# -----------------------------------------------------------------------------

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=N_MELS
)

_amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")


def mel_spectrogram_raw_db(wav_path):
    """Load WAV → MelSpectrogram → dB → numpy array"""
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(
            torch.tensor(audio, dtype=torch.float32),
            sr, SAMPLE_RATE
        )
    else:
        audio = torch.tensor(audio, dtype=torch.float32)

    audio = audio.unsqueeze(0)  # (1, T)

    mel = _mel_transform(audio)           # (1, M, T)
    mel_db = _amp_to_db(mel)[0].numpy()   # (M, T)
    duration = len(audio[0]) / SAMPLE_RATE
    return mel_db, duration


def mel_db_to_image_array(mel_db):
    """Render mel spectrogram to RGB numpy array (PNG buffer)."""
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks([]); ax.set_yticks([])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = plt.imread(buf)
    if img.shape[-1] == 4:
        img = img[..., :3]

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    return img


# -----------------------------------------------------------------------------
# RAW SPECTROGRAM (BASE64)
# -----------------------------------------------------------------------------

def generate_raw_spectrogram_base64(wav_path):
    mel_db, _ = mel_spectrogram_raw_db(wav_path)
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks([]); ax.set_yticks([])
    return fig_to_base64(fig)


# -----------------------------------------------------------------------------
# EMOTION GRADIENT
# -----------------------------------------------------------------------------

def build_time_gradient_windows(windows, width_px, duration):
    centers = [(w["start"] + w["end"]) / 2 for w in windows]
    emotions = [
        w["emotion"] if "emotion" in w else max(w["probs"], key=lambda k: float(w["probs"][k]))
        for w in windows
    ]

    timeline = []
    for x in range(width_px):
        t = (x / width_px) * duration
        idx = np.argmin([abs(t - c) for c in centers])
        timeline.append(emotions[idx])

    rgb = np.array([EMOTION_COLORS[e] for e in timeline], dtype=float)

    for ch in range(3):
        rgb[:, ch] = gaussian_filter1d(rgb[:, ch], sigma=10)

    return rgb.astype(np.uint8)


def blend_emotion_gradient(base_img, gradient_rgb, alpha=0.55):
    H, W, _ = base_img.shape
    out = base_img.astype(float)
    for x in range(W):
        out[:, x, :] = (1 - alpha) * out[:, x, :] + alpha * gradient_rgb[x]
    return np.clip(out, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# FINAL EMOTION SPECTROGRAM (BASE64)
# -----------------------------------------------------------------------------

def generate_emotion_gradient_spectrogram_base64(wav_path, windows, shap_map=None):
    mel_db, duration = mel_spectrogram_raw_db(wav_path)
    img = mel_db_to_image_array(mel_db)

    H, W, _ = img.shape
    gradient = build_time_gradient_windows(windows, W, duration)

    blended = blend_emotion_gradient(img, gradient)

    fig = plt.figure(figsize=(6, 2))
    plt.imshow(blended)
    plt.axis("off")
    return fig_to_base64(fig)
