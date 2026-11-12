"""Extract and validate mono 16 kHz WAV audio from MELD videos."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import pandas as pd
import soundfile as sf

from project_config import MELD_MANIFESTS, MELD_MEDIA_ROOT, MELD_PREPARED_ROOT, MELD_SPLITS


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root.resolve() and root.resolve() not in candidate.parents:
        raise ValueError(f"Path escapes configured root: {relative}")
    return candidate


def _run_ffmpeg(source: Path, target: Path) -> tuple[bool, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_target = target.with_suffix(".tmp.wav")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(temporary_target),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        temporary_target.unlink(missing_ok=True)
        return False, result.stderr.strip() or f"ffmpeg exit code {result.returncode}"
    try:
        info = sf.info(temporary_target)
        if info.samplerate != 16000 or info.channels != 1 or info.duration <= 0:
            raise ValueError(
                f"invalid output format: sample_rate={info.samplerate}, "
                f"channels={info.channels}, duration={info.duration}"
            )
    except (OSError, RuntimeError, ValueError) as exc:
        temporary_target.unlink(missing_ok=True)
        return False, f"invalid ffmpeg output: {exc}"
    if not temporary_target.is_file() or temporary_target.stat().st_size == 0:
        temporary_target.unlink(missing_ok=True)
        return False, "ffmpeg produced no output"
    temporary_target.replace(target)
    return True, ""


def _is_valid_existing_audio(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "output does not exist"
    if path.stat().st_size == 0:
        return False, "output is zero bytes"
    try:
        info = sf.info(path)
        if info.samplerate != 16000:
            return False, f"sample rate is {info.samplerate} Hz"
        if info.channels != 1:
            return False, f"channel count is {info.channels}"
        if info.duration <= 0:
            return False, "duration is not positive"
        return True, ""
    except (OSError, RuntimeError, ValueError) as exc:
        return False, f"audio is unreadable: {exc}"


def extract_split(split: str, media_root: Path, output_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    manifest = MELD_MANIFESTS / f"{split}_manifest.csv"
    frame = pd.read_csv(manifest)
    successes: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        sample_id = str(row["sample_id"])
        video_relative = str(row["video_path"])
        source = _safe_child(media_root, video_relative)
        audio_relative = Path("audio") / split / f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}.wav"
        target = _safe_child(output_root, str(audio_relative))
        existing_valid, existing_error = _is_valid_existing_audio(target)
        if existing_valid:
            successes.append({**row, "source_path": str(source), "output_path": str(target), "status": "reused", "error": ""})
            continue
        if not source.is_file():
            failures.append({**row, "source_path": str(source), "output_path": str(target), "status": "missing_video", "error": "Source video does not exist"})
            continue
        ok, error = _run_ffmpeg(source, target)
        if not ok and existing_error:
            error = f"{error}; previous output: {existing_error}"
        record = {**row, "source_path": str(source), "output_path": str(target), "status": "ok" if ok else "conversion_failed", "error": error}
        (successes if ok else failures).append(record)
    return successes, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, default=MELD_MEDIA_ROOT)
    parser.add_argument("--output-root", type=Path, default=MELD_PREPARED_ROOT)
    args = parser.parse_args()
    report_root = args.output_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    all_failures: list[dict[str, object]] = []
    all_successes: list[dict[str, object]] = []
    for split in MELD_SPLITS:
        successes, failures = extract_split(split, args.media_root, args.output_root)
        all_successes.extend(successes)
        all_failures.extend(failures)
        for name, rows in (("audio_manifest.csv", successes), ("audio_failures.csv", failures)):
            path = report_root / split / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if rows:
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                path.write_text("sample_id,status,error\n", encoding="utf-8")
        print(f"{split}: converted={len(successes)} failed={len(failures)}")
    combined = report_root / "audio_conversion_failures.csv"
    if all_failures:
        with combined.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_failures[0]))
            writer.writeheader()
            writer.writerows(all_failures)
    else:
        combined.write_text("sample_id,status,error\n", encoding="utf-8")
    print(f"total_converted={len(all_successes)} total_failed={len(all_failures)}")
    return 0 if not all_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
