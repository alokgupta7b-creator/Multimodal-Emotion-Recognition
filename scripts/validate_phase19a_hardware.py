"""Phase 19A — Automated Live-Capture Format, Scenario, Cleanup & Browser UI Validator.

Uses MELD development clips strictly (zero held-out `test` split access), synthetic invalid/silent/no-face
inputs, temporary FFmpeg VP9 WebM encoding (when available), and live Chrome UI fault-injection tests
without claiming or fabricating physical webcam/microphone human interaction results.
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
import ctypes
import ctypes.wintypes
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
import wave
from typing import Any

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

import api.app as api_app
from project_config import MELD_PREPARED_ROOT, REPOSITORY_ROOT

OUT_JSON = REPOSITORY_ROOT / "models" / "facial_emotion" / "resnet18" / "phase19a_hardware_validation.json"
BROWSER_JSON = REPOSITORY_ROOT / "models" / "facial_emotion" / "resnet18" / "phase19a_browser_validation.json"


def _get_process_rss_mb() -> float:
    class _PMC(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.wintypes.DWORD),
            ("PageFaultCount", ctypes.wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), ctypes.wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL
    if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return round(pmc.WorkingSetSize / (1024 * 1024), 2)
    return 0.0


def _temp_media_files() -> set[Path]:
    root = Path(tempfile.gettempdir())
    found: set[Path] = set()
    for pat in ("*.wav", "*.mp4", "*.webm", "emotion*"):
        found.update(p for p in root.glob(pat) if p.is_file())
    return found


def _load_dev_candidates() -> dict[str, Path]:
    video_manifest = MELD_PREPARED_ROOT / "reports" / "dev" / "video_manifest.csv"
    audio_manifest = MELD_PREPARED_ROOT / "reports" / "dev" / "audio_manifest.csv"

    video_map: dict[str, Path] = {}
    with video_manifest.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("video_status") == "ok":
                p = Path(row.get("resolved_video_path") or row.get("output_path") or "")
                if p.is_file():
                    video_map[str(row["sample_id"])] = p

    audio_map: dict[str, Path] = {}
    with audio_manifest.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("audio_status") == "ok":
                p = Path(row.get("resolved_audio_path") or row.get("output_path") or "")
                if p.is_file():
                    audio_map[str(row["sample_id"])] = p

    selected: dict[str, Path] = {}
    for sid in sorted(video_map.keys() & audio_map.keys()):
        if sid == "dev_dia0_utt0":
            selected["normal_wav"] = audio_map[sid]
            selected["multi_face_mp4"] = video_map[sid]
        elif sid == "dev_dia99_utt3":
            selected["short_mp4"] = video_map[sid]
        elif sid == "dev_dia74_utt19":
            selected["long_mp4"] = video_map[sid]
        elif sid == "dev_dia10_utt3":
            selected["low_light_mp4"] = video_map[sid]

    if "normal_wav" not in selected and (video_map.keys() & audio_map.keys()):
        first_sid = sorted(video_map.keys() & audio_map.keys())[0]
        selected["normal_wav"] = audio_map[first_sid]
        selected["multi_face_mp4"] = video_map[first_sid]
    return selected


def _wait_http(url: str, proc: subprocess.Popen[bytes], log_path: Path, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            txt = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            raise RuntimeError(f"Process exited prematurely ({rc}) waiting for {url}\n{txt}")
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:
            last_err = exc
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {url}: {last_err}")


def run_browser_ui_checks() -> dict[str, Any]:
    frontend_dir = REPOSITORY_ROOT / "frontend"
    backend_env = os.environ.copy()
    backend_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    backend_env["EMOTION_API_DEVICE"] = "auto"

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        blog_path = Path(td) / "backend.log"
        flog_path = Path(td) / "frontend.log"
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
                node_env["OUT_JSON"] = str(BROWSER_JSON)
                res = subprocess.run(
                    ["node", str(REPOSITORY_ROOT / "scripts" / "e2e_phase19a_live_capture.cjs")],
                    cwd=str(REPOSITORY_ROOT),
                    env=node_env,
                    check=False,
                )
                if res.returncode != 0:
                    raise RuntimeError(f"Phase 19A browser validation exited with code {res.returncode}")
                return json.loads(BROWSER_JSON.read_text(encoding="utf-8"))
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


def main() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    before_temps = _temp_media_files()
    dev = _load_dev_candidates()
    assert "normal_wav" in dev and "multi_face_mp4" in dev

    scenario_results: dict[str, Any] = {}

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Prepare synthetic format & stress test payloads
        silent_wav = tmp / "silent.wav"
        with wave.open(str(silent_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 16000)

        ogg_fake = tmp / "unsupported.ogg"
        ogg_fake.write_bytes(b"OggS\x00\x02dummy_unsupported_audio_payload")

        no_face_mp4 = tmp / "no_face.mp4"
        writer = cv2.VideoWriter(str(no_face_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 12, (160, 120))
        for _ in range(12):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
        writer.release()

        corrupt_mp4 = tmp / "corrupt.mp4"
        corrupt_mp4.write_bytes(b"\x00\x00\x00\x18ftypisomcorrupt_body")

        webm_clip: Path | None = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            webm_out = tmp / "live_sample.webm"
            proc = subprocess.run(
                [
                    ffmpeg, "-y", "-i", str(dev["multi_face_mp4"]),
                    "-t", "2.0", "-c:v", "libvpx", "-b:v", "400k", "-an", str(webm_out),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if proc.returncode == 0 and webm_out.is_file():
                webm_clip = webm_out

        with TestClient(api_app.app) as client:
            # 1. Valid WAV -> /predict/audio
            t0 = time.perf_counter()
            r_wav = client.post(
                "/predict/audio",
                files={"file": ("live-speech.wav", dev["normal_wav"].read_bytes(), "audio/wav")},
            )
            scenario_results["wav_audio_200"] = {
                "status": r_wav.status_code,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "predicted": r_wav.json().get("predicted_emotion") if r_wav.status_code == 200 else None,
            }

            # 2. Unsupported audio format (.ogg) -> 415
            r_ogg = client.post(
                "/predict/audio",
                files={"file": ("clip.ogg", ogg_fake.read_bytes(), "audio/ogg")},
            )
            scenario_results["unsupported_ogg_415"] = {"status": r_ogg.status_code}

            # 3. Silent WAV -> 422
            r_silent = client.post(
                "/predict/audio",
                files={"file": ("silent.wav", silent_wav.read_bytes(), "audio/wav")},
            )
            scenario_results["silent_wav_422"] = {"status": r_silent.status_code}

            # 4. MP4 multi-face, short, long, low-light dev scenarios -> /predict/video
            for label, key in (
                ("multi_face_mp4", "multi_face_mp4"),
                ("short_mp4", "short_mp4"),
                ("long_mp4", "long_mp4"),
                ("low_light_mp4", "low_light_mp4"),
            ):
                if key not in dev:
                    continue
                t0 = time.perf_counter()
                r = client.post(
                    "/predict/video",
                    files={"file": (f"{label}.mp4", dev[key].read_bytes(), "video/mp4")},
                )
                body = r.json()
                scenario_results[label] = {
                    "status": r.status_code,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                    "predicted": body.get("predicted_emotion"),
                    "quality": body.get("quality"),
                }

            # 5. WebM Live Recording Compatibility -> /predict/video
            if webm_clip is not None:
                t0 = time.perf_counter()
                r_webm = client.post(
                    "/predict/video",
                    files={"file": ("live-video.webm", webm_clip.read_bytes(), "video/webm")},
                )
                scenario_results["webm_video_200"] = {
                    "status": r_webm.status_code,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                    "predicted": r_webm.json().get("predicted_emotion") if r_webm.status_code == 200 else None,
                }
            else:
                scenario_results["webm_video_200"] = {"skipped": True, "reason": "ffmpeg not installed"}

            # 6. No-face video -> 422 & Corrupt MP4 -> 422
            r_nf = client.post(
                "/predict/video",
                files={"file": ("no_face.mp4", no_face_mp4.read_bytes(), "video/mp4")},
            )
            r_corrupt = client.post(
                "/predict/video",
                files={"file": ("corrupt.mp4", corrupt_mp4.read_bytes(), "video/mp4")},
            )
            scenario_results["no_face_422"] = {"status": r_nf.status_code}
            scenario_results["corrupt_mp4_422"] = {"status": r_corrupt.status_code}

            # 7. Multimodal Audio + Video upload endpoint routing -> /predict/multimodal
            t0 = time.perf_counter()
            r_mm = client.post(
                "/predict/multimodal",
                data={"text": "Wait, really?"},
                files={
                    "file": ("mic.wav", dev["normal_wav"].read_bytes(), "audio/wav"),
                    "video": ("cam.mp4", dev["multi_face_mp4"].read_bytes(), "video/mp4"),
                },
            )
            scenario_results["multimodal_live_style_200"] = {
                "status": r_mm.status_code,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "fusion_method": r_mm.json().get("fusion_method") if r_mm.status_code == 200 else None,
            }

    after_temps = _temp_media_files()
    leaked = sorted(str(p) for p in (after_temps - before_temps))

    browser_report = run_browser_ui_checks()

    resources = {
        "process_rss_mb": _get_process_rss_mb(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        if torch.cuda.is_available()
        else 0.0,
        "cuda_reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
        if torch.cuda.is_available()
        else 0.0,
    }

    checks_ok = (
        scenario_results["wav_audio_200"]["status"] == 200
        and scenario_results["unsupported_ogg_415"]["status"] == 415
        and scenario_results["silent_wav_422"]["status"] == 422
        and scenario_results["multi_face_mp4"]["status"] == 200
        and scenario_results["no_face_422"]["status"] == 422
        and scenario_results["corrupt_mp4_422"]["status"] == 422
        and scenario_results["multimodal_live_style_200"]["status"] == 200
        and len(leaked) == 0
        and browser_report.get("passed") is True
    )

    report = {
        "automated_passed": checks_ok,
        "physical_hardware_status": "AWAITING_USER_VERIFICATION",
        "zero_test_split_access": True,
        "temp_files_leaked": leaked,
        "scenarios": scenario_results,
        "browser_live_capture_ui": browser_report,
        "resources": resources,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not checks_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
