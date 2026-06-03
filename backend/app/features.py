import librosa
import numpy as np


def normalize_f0_values(values: np.ndarray) -> np.ndarray:
    voiced = values[np.isfinite(values)]
    if voiced.size == 0:
        return np.array([], dtype=float)
    semitone_range = float(np.max(12.0 * np.log2(voiced / np.median(voiced))) - np.min(12.0 * np.log2(voiced / np.median(voiced))))
    if semitone_range < 2.0:
        return np.full_like(voiced, 3.0, dtype=float)
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


def smooth_contour(values: list[float]) -> list[float]:
    if len(values) < 3:
        return values
    arr = np.array(values, dtype=float)
    padded = np.pad(arr, (1, 1), mode="edge")
    smoothed = np.array([float(np.median(padded[index : index + 3])) for index in range(len(arr))])
    if len(smoothed) >= 5:
        body = smoothed[:-1]
        body_median = float(np.median(body))
        body_spread = max(float(np.percentile(body, 85) - np.percentile(body, 15)), 0.4)
        if smoothed[-1] - body_median > body_spread * 1.4:
            smoothed[-1] = float(np.median(smoothed[-4:-1]))
    return [float(value) for value in smoothed]


def resample_contour(values: list[float], points: int = 12) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [values[0], values[0], values[0]]
    if len(values) >= points:
        return values
    x_old = np.linspace(0.0, 1.0, num=len(values))
    x_new = np.linspace(0.0, 1.0, num=points)
    return [float(value) for value in np.interp(x_new, x_old, np.array(values, dtype=float))]


def extract_normalized_contour(audio: np.ndarray, sr: int) -> list[float]:
    frame_length = 1024
    hop_length = 160
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    if f0 is None:
        return []

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    frame_count = min(len(f0), len(rms), len(voiced_prob))
    f0 = f0[:frame_count]
    rms = rms[:frame_count]
    voiced_prob = voiced_prob[:frame_count]
    voiced_flag = voiced_flag[:frame_count]
    finite = np.isfinite(f0)
    if not np.any(finite):
        return []

    voiced_energy = rms[finite]
    energy_threshold = max(float(np.percentile(voiced_energy, 35)), float(np.max(voiced_energy)) * 0.18)
    mask = finite & voiced_flag & (voiced_prob >= 0.35) & (rms >= energy_threshold)
    if np.count_nonzero(mask) < 3:
        mask = finite & (rms >= energy_threshold)
    if np.count_nonzero(mask) < 2:
        mask = finite

    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    start = int(indices[0])
    end = int(indices[-1]) + 1
    values = f0[start:end]
    values = values[np.isfinite(values)]
    if values.size >= 5:
        values = np.array(smooth_contour([float(value) for value in values]), dtype=float)
    normalized = normalize_f0_values(values)
    return smooth_contour(resample_contour([float(value) for value in normalized if np.isfinite(value)]))
