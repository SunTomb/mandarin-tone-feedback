import argparse
from pathlib import Path

from self_recording_manifest import build_manifest_rows, validate_manifest_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate self-recorded Mandarin tone metadata and audio paths.")
    parser.add_argument("--self-root", type=Path, default=Path("data/self"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    rows = build_manifest_rows(args.self_root)
    errors = validate_manifest_rows(rows, project_root=args.project_root)
    if errors:
        for error in errors:
            print(error)
        print(f"Found {len(errors)} validation errors in {len(rows)} rows")
        return 1
    print(f"Validated {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
