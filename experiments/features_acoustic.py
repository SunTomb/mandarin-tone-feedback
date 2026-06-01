from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


FEATURE_NAMES = [
    "f0_start",
    "f0_mid",
    "f0_end",
    "f0_range",
    "f0_slope",
    "duration",
    "voiced_ratio",
]


def load_audio(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int, float]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    duration = float(len(audio) / sr) if sr else 0.0
    return audio, sr, duration


def normalize_f0(f0: np.ndarray) -> np.ndarray:
    voiced = f0[np.isfinite(f0)]
    if voiced.size == 0:
        return np.array([], dtype=float)
    low = float(np.min(voiced))
    high = float(np.max(voiced))
    if high == low:
        return np.full(voiced.shape, 3.0, dtype=float)
    return 1.0 + 4.0 * (voiced - low) / (high - low)


def extract_f0_contour(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    f0, _, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
    )
    if f0 is None:
        return np.array([], dtype=float), 0.0
    voiced_ratio = float(np.isfinite(f0).sum() / len(f0)) if len(f0) else 0.0
    return normalize_f0(f0), voiced_ratio


def summarize_f0_contour(contour: np.ndarray, duration: float, voiced_ratio: float) -> dict[str, float]:
    if contour.size == 0:
        return {
            "f0_start": 0.0,
            "f0_mid": 0.0,
            "f0_end": 0.0,
            "f0_range": 0.0,
            "f0_slope": 0.0,
            "duration": float(duration),
            "voiced_ratio": float(voiced_ratio),
        }
    start = float(contour[0])
    mid = float(contour[len(contour) // 2])
    end = float(contour[-1])
    return {
        "f0_start": start,
        "f0_mid": mid,
        "f0_end": end,
        "f0_range": float(np.max(contour) - np.min(contour)),
        "f0_slope": end - start,
        "duration": float(duration),
        "voiced_ratio": float(voiced_ratio),
    }


def extract_acoustic_feature_row(path: Path) -> list[float]:
    audio, sr, duration = load_audio(path)
    contour, voiced_ratio = extract_f0_contour(audio, sr)
    summary = summarize_f0_contour(contour, duration, voiced_ratio)
    return [summary[name] for name in FEATURE_NAMES]
