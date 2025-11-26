# spectrogram_utils.py

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import json, io, base64
from pathlib import Path

SAMPLE_RATE = 16000
N_MELS = 64

# --- EMOTION COLORS (fixed, nice palette) ---
EMOTION_COLORS = {
    "Angry": (255, 0, 0),
    "Disgust": (34, 139, 34),
    "Fear": (138, 43, 226),
    "Happy": (255, 215, 0),
    "Neutral": (200, 200, 200),
    "Pleasant_surprise": (255, 140, 0),
    "Sad": (70, 130, 180),
}


# =======================
# HELPERS
# =======================

def fig_to_base64(fig):
    """Convert a matplotlib figure to base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def mel_spectrogram_raw_db(wav_path):
    """Return mel spectrogram dB array + duration."""
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    dur = librosa.get_duration(y=y, sr=sr)
    return mel_db, dur


# def mel_db_to_image_array(mel_db):
#     """Render mel dB to RGB numpy array."""
#     fig, ax = plt.subplots(figsize=(12, 4))
#     librosa.display.specshow(mel_db, cmap="magma", ax=ax)
#     ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
#     fig.canvas.draw()
#     img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     plt.close(fig)
#     return img

def mel_db_to_image_array(mel_db):
    """
    Render mel dB to an RGB uint8 numpy array safely using PNG buffer.
    Avoids backend-specific canvas.tostring_rgb() errors.
    """
    fig, ax = plt.subplots(figsize=(6, 2))
    librosa.display.specshow(mel_db, cmap="magma", ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # load PNG buffer as numpy array
    img = plt.imread(buf)

    # drop alpha if present
    if img.shape[-1] == 4:
        img = img[..., :3]

    # ensure uint8 format
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    return img



# ================================
# 1) RAW SPECTROGRAM (no colors)
# ================================

def generate_raw_spectrogram_base64(wav_path):
    mel_db, _ = mel_spectrogram_raw_db(wav_path)
    fig, ax = plt.subplots(figsize=(6, 2))
    librosa.display.specshow(mel_db, cmap="magma", ax=ax)
    ax.set_xticks([]); ax.set_yticks([])
    return fig_to_base64(fig)


# ======================================
# 2) EMOTION GRADIENT SPECTROGRAM
# ======================================

def build_time_gradient_windows(windows, width_px, duration):
    """
    Returns an array (width_px, 3) of RGB colors for each time pixel.
    Smooths with Gaussian blur for gradient effect.
    """
    # Compute window centers
    centers = [(w["start"] + w["end"]) / 2 for w in windows]

    # Determine emotion of each window
    emotions = []
    for w in windows:
        if "emotion" in w:
            emotions.append(w["emotion"])
        elif "probs" in w:
            emotions.append(max(w["probs"], key=lambda k: float(w["probs"][k])))

    # For each pixel x (time), find nearest window
    timeline = []
    for x in range(width_px):
        t = (x / width_px) * duration
        idx = np.argmin([abs(t - c) for c in centers])
        timeline.append(emotions[idx])

    # Convert to RGB
    rgb = np.array([EMOTION_COLORS[e] for e in timeline], dtype=float)

    # Smooth for gradient
    for ch in range(3):
        rgb[:, ch] = gaussian_filter1d(rgb[:, ch], sigma=10)

    return rgb.astype(np.uint8)   # (W, 3)


def blend_emotion_gradient(base_img, gradient_rgb, alpha=0.55):
    """Blend gradient colors into spectrogram image."""
    H, W, _ = base_img.shape
    out = base_img.astype(float)
    for x in range(W):
        out[:, x, :] = (1 - alpha) * out[:, x, :] + alpha * gradient_rgb[x]
    return np.clip(out, 0, 255).astype(np.uint8)


# ======================================
# 3) OPTIONAL: SHAP OVERLAY
# ======================================

def blend_shap_overlay(img, shap_map):
    """
    shap_map: 2D numpy array (n_mels, T) with values 0..1
    Converted to grayscale heat overlay @ 30% intensity
    """
    H, W, _ = img.shape
    shap_1d = shap_map.max(axis=0)                # reduce freq axis
    xs = np.linspace(0, len(shap_1d) - 1, W)
    shap_resampled = np.interp(xs, np.arange(len(shap_1d)), shap_1d)
    shap_img = np.tile((shap_resampled * 255).astype(np.uint8), (H, 1))
    shap_img = np.stack([shap_img] * 3, axis=-1)
    return (0.7 * img + 0.3 * shap_img).astype(np.uint8)


# ======================================
# FINAL: EMOTION GRADIENT SPECTROGRAM
# ======================================

def generate_emotion_gradient_spectrogram_base64(wav_path, windows, shap_map=None):
    mel_db, duration = mel_spectrogram_raw_db(wav_path)
    img = mel_db_to_image_array(mel_db)

    # 1. Build smooth gradient
    H, W, _ = img.shape
    gradient = build_time_gradient_windows(windows, W, duration)

    # 2. Blend into spectrogram
    blended = blend_emotion_gradient(img, gradient)

    # 3. Optionally apply SHAP mask
    if shap_map is not None:
        blended = blend_shap_overlay(blended, shap_map)

    # 4. Convert to base64
    fig = plt.figure(figsize=(6, 2))
    plt.imshow(blended)
    plt.axis("off")
    return fig_to_base64(fig)
