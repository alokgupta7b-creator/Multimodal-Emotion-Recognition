"""Validate MELD videos and optionally extract frames."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import pandas as pd

from project_config import MELD_MANIFESTS, MELD_MEDIA_ROOT, MELD_PREPARED_ROOT, MELD_SPLITS


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root.resolve() and root.resolve() not in candidate.parents:
        raise ValueError(f"Path escapes configured root: {relative}")
    return candidate


def _probe(path: Path) -> tuple[dict[str, object] | None, str]:
    command = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None, result.stderr.strip() or f"ffprobe exit code {result.returncode}"
    try:
        payload = json.loads(result.stdout)
        video = next(stream for stream in payload["streams"] if stream.get("codec_type") == "video")
        return {
            "duration_seconds": float(payload["format"].get("duration", 0.0)),
            "width": int(video["width"]),
            "height": int(video["height"]),
            "frame_rate": str(video.get("r_frame_rate", "")),
            "codec": str(video.get("codec_name", "")),
            "frame_count": video.get("nb_frames", ""),
        }, ""
    except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, f"invalid ffprobe response: {exc}"


def prepare_split(split: str, media_root: Path, output_root: Path, extract_frames: bool) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    frame = pd.read_csv(MELD_MANIFESTS / f"{split}_manifest.csv")
    valid: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        source = _safe_child(media_root, str(row["video_path"]))
        metadata, error = _probe(source) if source.is_file() else (None, "Video file does not exist")
        record = {**row, "resolved_video_path": str(source), "status": "ok" if metadata else "invalid", "error": error, **(metadata or {})}
        if metadata:
            valid.append(record)
            if extract_frames:
                frame_dir = output_root / "frames" / split / f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(source), "-q:v", "2", str(frame_dir / "%06d.jpg")], check=False)
        else:
            failures.append(record)
    return valid, failures


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("sample_id,status,error\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, default=MELD_MEDIA_ROOT)
    parser.add_argument("--output-root", type=Path, default=MELD_PREPARED_ROOT)
    parser.add_argument("--extract-frames", action="store_true")
    args = parser.parse_args()
    valid_all: list[dict[str, object]] = []
    failed_all: list[dict[str, object]] = []
    for split in MELD_SPLITS:
        valid, failed = prepare_split(split, args.media_root, args.output_root, args.extract_frames)
        _write(args.output_root / "reports" / split / "video_manifest.csv", valid)
        _write(args.output_root / "reports" / split / "video_failures.csv", failed)
        valid_all.extend(valid)
        failed_all.extend(failed)
        print(f"{split}: valid={len(valid)} failed={len(failed)}")
    _write(args.output_root / "reports" / "video_validation_failures.csv", failed_all)
    print(f"total_valid={len(valid_all)} total_failed={len(failed_all)}")
    return 0 if not failed_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
