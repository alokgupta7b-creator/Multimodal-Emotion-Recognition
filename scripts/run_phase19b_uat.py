"""Phase 19B — Automated UI & All-7-Modality UAT Validation Runner.

Selects one MELD development WAV + MP4 pair (never held-out test data), launches
Uvicorn backend + built Vite frontend preview, runs `scripts/e2e_phase19b_uat.cjs`
in Chromium, and saves `models/facial_emotion/resnet18/phase19b_uat_validation.json`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("EMOTION_API_DEVICE", "auto")

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import csv
import json
import subprocess
import tempfile
import time
import urllib.request

import cv2
import numpy as np

from project_config import MELD_PREPARED_ROOT, REPOSITORY_ROOT

OUT_JSON = REPOSITORY_ROOT / "models" / "facial_emotion" / "resnet18" / "phase19b_uat_validation.json"


def _find_dev_pair() -> tuple[Path, Path]:
    video_manifest = MELD_PREPARED_ROOT / "reports" / "dev" / "video_manifest.csv"
    audio_manifest = MELD_PREPARED_ROOT / "reports" / "dev" / "audio_manifest.csv"
    v_map: dict[str, Path] = {}
    a_map: dict[str, Path] = {}
    with video_manifest.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("video_status") == "ok":
                p = Path(row.get("resolved_video_path") or row.get("output_path") or "")
                if p.is_file():
                    v_map[str(row["sample_id"])] = p
    with audio_manifest.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("audio_status") == "ok":
                p = Path(row.get("resolved_audio_path") or row.get("output_path") or "")
                if p.is_file():
                    a_map[str(row["sample_id"])] = p

    sid = "dev_dia0_utt0" if "dev_dia0_utt0" in (v_map.keys() & a_map.keys()) else sorted(v_map.keys() & a_map.keys())[0]
    return a_map[sid], v_map[sid]


def _wait_http(url: str, proc: subprocess.Popen[bytes], log_path: Path, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            txt = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            raise RuntimeError(f"Process exited prematurely ({proc.returncode}) waiting for {url}\n{txt}")
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:
            last_err = exc
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {url}: {last_err}")


def main() -> None:
    dev_wav, dev_mp4 = _find_dev_pair()
    frontend_dir = REPOSITORY_ROOT / "frontend"
    backend_env = os.environ.copy()
    backend_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    backend_env["EMOTION_API_DEVICE"] = "auto"

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        no_face_mp4 = tmp / "no_face.mp4"
        writer = cv2.VideoWriter(str(no_face_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 12, (160, 120))
        for _ in range(12):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        writer.release()

        blog_path = tmp / "backend.log"
        flog_path = tmp / "frontend.log"
        with blog_path.open("wb") as blog, flog_path.open("wb") as flog:
            backend_proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "api.app:app", "--host", "127.0.0.1", "--port", "8000"],
                cwd=str(REPOSITORY_ROOT),
                env=backend_env,
                stdout=blog,
                stderr=subprocess.STDOUT,
            )
            vite_bin = frontend_dir / "node_modules" / ".bin" / "vite.cmd"
            frontend_proc = subprocess.Popen(
                ["cmd.exe", "/c", str(vite_bin), "preview", "--host", "localhost", "--port", "5173", "--strictPort"],
                cwd=str(frontend_dir),
                env=os.environ.copy(),
                stdout=flog,
                stderr=subprocess.STDOUT,
            )
            try:
                _wait_http("http://127.0.0.1:8000/models/status", backend_proc, blog_path, timeout_s=90.0)
                _wait_http("http://localhost:5173", frontend_proc, flog_path, timeout_s=30.0)
                node_env = os.environ.copy()
                node_env["FRONTEND_URL"] = "http://localhost:5173"
                node_env["DEV_WAV_PATH"] = str(dev_wav)
                node_env["DEV_MP4_PATH"] = str(dev_mp4)
                node_env["NOFACE_MP4_PATH"] = str(no_face_mp4)
                node_env["OUT_JSON"] = str(OUT_JSON)
                res = subprocess.run(
                    ["node", str(REPOSITORY_ROOT / "scripts" / "e2e_phase19b_uat.cjs")],
                    cwd=str(REPOSITORY_ROOT),
                    env=node_env,
                    check=False,
                )
                if res.returncode != 0:
                    raise SystemExit(res.returncode)
            finally:
                for proc in (frontend_proc, backend_proc):
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    elif proc.poll() is None:
                        proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                time.sleep(0.4)

    data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    if not data.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

