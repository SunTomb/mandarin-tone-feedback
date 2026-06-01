import librosa
import numpy as np


def normalize_f0_values(values: np.ndarray) -> np.ndarray:
    voiced = values[np.isfinite(values)]
    if voiced.size == 0:
        return np.array([], dtype=float)
    low = float(np.min(voiced))
    high = float(np.max(voiced))
    if high == low:
        return np.full_like(voiced, 3.0, dtype=float)
    return 1.0 + 4.0 * (voiced - low) / (high - low)


def five_level_value(values: list[float]) -> str:
    if not values:
        return ""
    indices = [0, len(values) // 2, len(values) - 1]
    return "".join(str(int(round(values[index]))) for index in indices)


def summarize_contour(values: list[float], duration: float) -> dict[str, float]:
    if not values:
        return {
            "f0_start": 0.0,
            "f0_mid": 0.0,
            "f0_end": 0.0,
            "f0_range": 0.0,
            "f0_slope": 0.0,
            "duration": duration,
            "voiced_ratio": 0.0,
        }
    start = float(values[0])
    mid = float(values[len(values) // 2])
    end = float(values[-1])
    return {
        "f0_start": start,
        "f0_mid": mid,
        "f0_end": end,
        "f0_range": float(max(values) - min(values)),
        "f0_slope": end - start,
        "duration": duration,
        "voiced_ratio": 1.0,
    }


def extract_normalized_contour(audio: np.ndarray, sr: int) -> list[float]:
    f0, _, _ = librosa.pyin(audio, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr)
    if f0 is None:
        return []
    normalized = normalize_f0_values(f0)
    return [float(value) for value in normalized if np.isfinite(value)]
