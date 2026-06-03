from io import BytesIO

import numpy as np
import soundfile as sf

from app.audio import load_audio_bytes


def test_load_audio_bytes_trims_leading_and_trailing_silence():
    sr = 16000
    silence = np.zeros(int(sr * 0.5), dtype=np.float32)
    time = np.linspace(0.0, 0.3, int(sr * 0.3), endpoint=False)
    voiced = (0.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    audio = np.concatenate([silence, voiced, silence])
    buffer = BytesIO()
    sf.write(buffer, audio, sr, format="WAV")

    trimmed_audio, _, duration = load_audio_bytes(buffer.getvalue(), target_sr=sr)

    assert 0.25 <= duration <= 0.45
    assert len(trimmed_audio) < len(audio) * 0.5
