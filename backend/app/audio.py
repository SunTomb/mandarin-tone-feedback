from io import BytesIO

import librosa
import numpy as np
import soundfile as sf


def load_audio_bytes(content: bytes, target_sr: int = 16000) -> tuple[np.ndarray, int, float]:
    data, sr = sf.read(BytesIO(content), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    duration = float(len(data) / sr) if sr else 0.0
    return data, sr, duration
