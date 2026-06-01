from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path


URL = "https://openslr.elda.org/resources/18/data_thchs30.tgz"
OUTPUT = Path("data/downloads/data_thchs30.tgz")
TMP = OUTPUT.with_suffix(".tgz.tmp")
LOG = Path("data/downloads/data_thchs30.python.log")
CHUNK_SIZE = 1024 * 1024


def log(message: str) -> None:
    text = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(text, flush=True)
    with LOG.open("a", encoding="utf-8") as file:
        file.write(text + "\n")


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for path in (OUTPUT, TMP):
        if path.exists():
            path.unlink()

    request = urllib.request.Request(URL, headers={"User-Agent": "mandarin-tone-feedback-dataset-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length", "0"))
        log(f"start url={URL} content_length={total}")
        downloaded = 0
        last_report = time.time()
        with TMP.open("wb") as file:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                file.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_report >= 60:
                    log(f"progress bytes={downloaded} mb={downloaded // 1024 // 1024}")
                    last_report = now

    actual = TMP.stat().st_size
    log(f"finished bytes={actual} expected={total}")
    if total and actual != total:
        log("size_mismatch")
        return 2
    TMP.replace(OUTPUT)
    log(f"renamed output={OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
