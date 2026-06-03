import subprocess
import sys
from pathlib import Path


def test_validate_self_recordings_cli_reports_valid_dataset(tmp_path: Path):
    project_root = tmp_path
    batch_dir = project_root / "data" / "self" / "1-1-1"
    batch_dir.mkdir(parents=True)
    rows = ["audio_path,text,pinyin,tone,speaker_id,repetition,split"]
    for index in range(30):
        filename = f"speaker01_item{index}1_01.wav"
        (batch_dir / filename).write_bytes(b"RIFF")
        rows.append(f"{filename},字{index},item{index}1,1,speaker01,1,train")
    (batch_dir / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    script = Path(__file__).parents[1] / "validate_self_recordings.py"
    result = subprocess.run(
        [sys.executable, str(script), "--self-root", str(project_root / "data" / "self"), "--project-root", str(project_root)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "Validated 30 rows" in result.stdout
