# model_inference.py
import torch, torchaudio, librosa, numpy as np
from torch import nn

SAMPLE_RATE = 16000
N_MELS = 64
CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Pleasant_surprise",
    "Sad"
]

# <-- This file must contain ONLY a state_dict -->
MODEL_PATH = "/home/om/Documents/trials/ser_masked_attn_best.pt"


# ============================================================
#                    FEATURE TRANSFORM
# ============================================================
class ToMelLogCMVN:
    def __init__(self, sr=SAMPLE_RATE, n_mels=N_MELS):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=512, n_mels=n_mels
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __call__(self, wav_1d: torch.Tensor):
        mel = self.mel(wav_1d)
        db = self.to_db(mel)
        mu = db.mean(dim=1, keepdim=True)
        std = db.std(dim=1, keepdim=True).clamp_min(1e-5)
        return (db - mu) / std


# ============================================================
#                    EMOTION MODEL
# ============================================================
class CNNLSTM_MaskedAttn(nn.Module):
    def __init__(self, n_mels, n_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.lstm = nn.LSTM(
            input_size=(n_mels // 4) * 64,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        self.attn = nn.Linear(256, 1)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x, mask_time):
        B, _, M, T = x.shape
        h = self.cnn(x)                    # (B,64, M/4, T/4)
        h = h.permute(0, 3, 1, 2)          # (B, T', 64, M/4)
        B, Tp, C, F = h.shape
        h = h.reshape(B, Tp, C * F)        # (B, T', D)

        # Downsample mask
        k = 4
        T_cut = (T // k) * k
        m = mask_time[:, :T_cut].reshape(B, T_cut // k, k).max(dim=2).values

        lstm_out, _ = self.lstm(h)
        attn_logits = self.attn(lstm_out).squeeze(-1)

        neg_inf = torch.finfo(attn_logits.dtype).min
        attn_logits = attn_logits.masked_fill(m < 0.5, neg_inf)

        alpha = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(lstm_out * alpha.unsqueeze(-1), dim=1)
        return self.fc(pooled)


# ============================================================
#                    MODEL LOADING
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None

def load_model():
    """
    Load the model weights using state_dict (portable, correct).
    """
    global _model
    if _model is None:
        model = CNNLSTM_MaskedAttn(N_MELS, len(CLASS_NAMES))

        # Load state_dict only (NOT full pickle model)
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)

        model.to(device).eval()
        _model = model

    return _model


# ============================================================
#                    AUDIO PREPROCESSING
# ============================================================
def preprocess_audio_segment(y, transform):
    y = torch.tensor(y, dtype=torch.float32)
    feat = transform(y)
    feat = feat.unsqueeze(0).unsqueeze(0)    # (1, 1, n_mels, T)
    mask = torch.ones(feat.shape[-1]).unsqueeze(0)
    return feat, mask


# ============================================================
#                    MAIN INFERENCE
# ============================================================
# def predict_emotion_from_wav(wav_path, window_sec=5.0, hop_sec=2.0):
#     model = load_model()
#     transform = ToMelLogCMVN()

#     y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
#     y, _ = librosa.effects.trim(y, top_db=20)

#     win_len = int(sr * window_sec)
#     hop_len = int(sr * hop_sec)
#     probs_all = []

#     for start in range(0, len(y) - win_len + 1, hop_len):
#         seg = y[start:start + win_len]
#         feat, mask = preprocess_audio_segment(seg, transform)
#         feat, mask = feat.to(device), mask.to(device)

#         with torch.no_grad():
#             logits = model(feat, mask)
#             probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

#         probs_all.append(probs)

#     if not probs_all:
#         return None

#     probs_all = np.stack(probs_all)
#     mean_probs = probs_all.mean(axis=0)
#     pred_idx = mean_probs.argmax()

#     return {
#         "emotion": CLASS_NAMES[pred_idx],
#         "probabilities": {
#             cls: round(float(p), 3)
#             for cls, p in zip(CLASS_NAMES, mean_probs)
#         }
#     }

# def predict_emotion_from_wav(wav_path, window_sec=5.0, hop_sec=2.0):
#     model = load_model()
#     transform = ToMelLogCMVN()

#     y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
#     # y, _ = librosa.effects.trim(y, top_db=20)

#     win_len = int(sr * window_sec)
#     hop_len = int(sr * hop_sec)
#     probs_all = []

#     print("\n===== DEBUG: Processing Windows =====")

#     for start in range(0, len(y) - win_len + 1, hop_len):
#         seg = y[start:start + win_len]
#         feat, mask = preprocess_audio_segment(seg, transform)
#         feat, mask = feat.to(device), mask.to(device)

#         with torch.no_grad():
#             logits = model(feat, mask)
#             probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

#         # 🔥 Debug: print probability vector for this segment
#         print(f"Window {start/sr:.2f}–{(start+win_len)/sr:.2f}s:")
#         for cls, val in zip(CLASS_NAMES, probs):
#             print(f"   {cls:>18}: {val:.4f}")

#         probs_all.append(probs)

#     # If no windows were processed
#     if not probs_all:
#         print("\n[DEBUG] No windows processed → Audio too short.")
#         return None

#     print("\n===== DEBUG: Averaged Result =====")
#     probs_all = np.stack(probs_all)
#     mean_probs = probs_all.mean(axis=0)
#     pred_idx = mean_probs.argmax()

#     # Print final mean vector
#     for cls, val in zip(CLASS_NAMES, mean_probs):
#         print(f"   {cls:>18}: {val:.4f}")

#     print(f"\nFinal Predicted Emotion: {CLASS_NAMES[pred_idx]}\n")

#     return [
#         CLASS_NAMES[pred_idx]
#     ]

def predict_emotion_from_wav(wav_path, window_sec=5.0, hop_sec=2.0):
    model = load_model()
    transform = ToMelLogCMVN()

    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

    win_len = int(sr * window_sec)
    hop_len = int(sr * hop_sec)
    probs_all = []

    # NEW: we will store window objects here
    windows = []

    print("\n===== DEBUG: Processing Windows =====")

    for start in range(0, len(y) - win_len + 1, hop_len):
        seg = y[start:start + win_len]
        feat, mask = preprocess_audio_segment(seg, transform)
        feat, mask = feat.to(device), mask.to(device)

        with torch.no_grad():
            logits = model(feat, mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

        # Debug print
        print(f"Window {start/sr:.2f}–{(start+win_len)/sr:.2f}s:")
        for cls, val in zip(CLASS_NAMES, probs):
            print(f"   {cls:>18}: {val:.4f}")

        probs_all.append(probs)

        # ▶ NEW: Build window object
        window_emotion = CLASS_NAMES[np.argmax(probs)]
        windows.append({
            "start": start / sr,
            "end": (start + win_len) / sr,
            "emotion": window_emotion,
            "probs": {cls: float(v) for cls, v in zip(CLASS_NAMES, probs)}
        })

    # Handle no windows
    if not probs_all:
        print("\n[DEBUG] No windows processed → Audio too short.")
        return None

    # Average for final emotion
    print("\n===== DEBUG: Averaged Result =====")
    probs_all = np.stack(probs_all)
    mean_probs = probs_all.mean(axis=0)
    pred_idx = mean_probs.argmax()

    for cls, val in zip(CLASS_NAMES, mean_probs):
        print(f"   {cls:>18}: {val:.4f}")

    print(f"\nFinal Predicted Emotion: {CLASS_NAMES[pred_idx]}\n")

    # 🔥 EXACT RETURN FORMAT REQUIRED
    return [
        CLASS_NAMES[pred_idx],  # backend STILL uses this
        windows                 # NEW (used by emotion spectrogram)
    ]
