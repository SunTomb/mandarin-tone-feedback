from __future__ import annotations

import re
from dataclasses import dataclass
from collections import Counter
from pathlib import Path


PINYIN_TONE_RE = re.compile(r"^([a-züv:]+)([1-5])$", re.IGNORECASE)


@dataclass(frozen=True)
class PinyinToneToken:
    original: str
    base: str
    tone: int


def extract_pinyin_tone(token: str) -> PinyinToneToken | None:
    stripped = token.strip()
    match = PINYIN_TONE_RE.match(stripped)
    if match is None:
        return None
    tone = int(match.group(2))
    if tone not in {1, 2, 3, 4}:
        return None
    return PinyinToneToken(original=stripped, base=match.group(1), tone=tone)


def extract_tonal_pinyin_tokens(pinyin_line: str) -> list[PinyinToneToken]:
    return [token for token in pinyin_token_slots(pinyin_line) if token is not None]


def pinyin_token_slots(pinyin_line: str) -> list[PinyinToneToken | None]:
    return [extract_pinyin_tone(raw_token) for raw_token in pinyin_line.split()]


def resolve_trn_path(path: Path) -> Path:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) == 1 and lines[0].startswith("../"):
        return (path.parent / lines[0]).resolve()
    return path.resolve()


def read_thchs_trn(path: Path) -> tuple[str, str, str]:
    resolved = resolve_trn_path(path)
    lines = resolved.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"THCHS transcript must contain at least 3 lines: {resolved}")
    return lines[0].strip(), lines[1].strip(), lines[2].strip()


def token_intervals(token_count: int, duration: float, margin: float = 0.0) -> list[tuple[float, float]]:
    if token_count <= 0 or duration <= 0:
        return []
    step = duration / token_count
    intervals = []
    for index in range(token_count):
        start = index * step
        end = (index + 1) * step
        if index == 0:
            start = min(margin, end)
        if index == token_count - 1:
            end = max(start, duration - margin)
        intervals.append((start, end))
    return intervals


def choose_duration_filtered_interval(
    start_sec: float,
    end_sec: float,
    min_duration: float,
    max_duration: float,
) -> tuple[float, float] | None:
    duration = end_sec - start_sec
    if duration < min_duration or duration > max_duration:
        return None
    return (start_sec, end_sec)


def speaker_id_from_stem(stem: str) -> str:
    return stem.split("_", 1)[0]


def is_clean_slice_row(
    row: dict[str, str],
    min_duration: float = 0.30,
    max_duration: float = 0.90,
) -> bool:
    token = extract_pinyin_tone(row.get("pinyin", ""))
    if token is None or str(token.tone) != str(row.get("tone", "")):
        return False
    text = row.get("text", "").strip()
    if len(text) != 1:
        return False
    try:
        duration = float(row.get("end_sec", "0")) - float(row.get("start_sec", "0"))
    except ValueError:
        return False
    return min_duration <= duration <= max_duration


def filter_metadata_rows(
    rows: list[dict[str, str]],
    quotas: dict[str, int],
    min_duration: float = 0.30,
    max_duration: float = 0.90,
) -> list[dict[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    filtered: list[dict[str, str]] = []
    for row in rows:
        if not is_clean_slice_row(row, min_duration=min_duration, max_duration=max_duration):
            continue
        split = row.get("split", "")
        tone = row.get("tone", "")
        quota = quotas.get(split)
        if quota is not None and counts[(split, tone)] >= quota:
            continue
        filtered.append(row)
        counts[(split, tone)] += 1
    return filtered
