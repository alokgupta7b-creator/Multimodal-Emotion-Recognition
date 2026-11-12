"""Self-contained runner for Phase 18E real Chrome browser UI verification."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGRESSION_JSON = ROOT / "models" / "facial_emotion" / "resnet18" / "phase18e_regression_validation.json"
FRONTEND_DIR = ROOT / "frontend"


def _wait_http(url: str, proc: subprocess.Popen[bytes], log_path: Path, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            output = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            raise RuntimeError(f"Server exited prematurely (code={rc}) waiting for {url}.\nLog:\n{output}")
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:
            last_err = exc
        time.sleep(0.5)
    output = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    raise TimeoutError(f"Timed out waiting for {url}: {last_err}\nLog:\n{output}")


def main() -> None:
    if not REGRESSION_JSON.is_file():
        raise FileNotFoundError(f"Run validate_phase18e_regression.py first: {REGRESSION_JSON}")
    reg = json.loads(REGRESSION_JSON.read_text(encoding="utf-8"))
    wav_path = reg["dev_probe_sample"]["audio_path"]
    mp4_path = reg["dev_probe_sample"]["video_path"]

    backend_env = os.environ.copy()
    backend_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    backend_env["EMOTION_API_DEVICE"] = "auto"

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        backend_log = Path(td) / "backend.log"
        frontend_log = Path(td) / "frontend.log"

        with backend_log.open("wb") as blog, frontend_log.open("wb") as flog:
            backend_proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "api.app:app", "--host", "127.0.0.1", "--port", "8000"],
                cwd=str(ROOT),
                env=backend_env,
                stdout=blog,
                stderr=subprocess.STDOUT,
            )

            vite_bin = FRONTEND_DIR / "node_modules" / ".bin" / "vite.cmd"
            frontend_proc = subprocess.Popen(
                ["cmd.exe", "/c", str(vite_bin), "preview", "--host", "localhost", "--port", "5173", "--strictPort"],
                cwd=str(FRONTEND_DIR),
                env=os.environ.copy(),
                stdout=flog,
                stderr=subprocess.STDOUT,
            )

            try:
                _wait_http("http://127.0.0.1:8000/models/status", backend_proc, backend_log, timeout_s=90.0)
                _wait_http("http://localhost:5173", frontend_proc, frontend_log, timeout_s=30.0)

                with urllib.request.urlopen("http://127.0.0.1:8000/models/status", timeout=5) as resp:
                    status_payload = json.loads(resp.read().decode("utf-8"))
                assert status_payload["status"] == "ready"
                assert status_payload["models"]["video"]["model_identifier"] == "facial-resnet18-ce-v1"

                node_env = os.environ.copy()
                node_env["FRONTEND_URL"] = "http://localhost:5173"
                node_env["TEST_WAV"] = wav_path
                node_env["TEST_MP4"] = mp4_path

                res = subprocess.run(
                    ["node", str(ROOT / "scripts" / "e2e_phase18e_browser.cjs")],
                    cwd=str(ROOT),
                    env=node_env,
                    check=False,
                )
                if res.returncode != 0:
                    print("--- BACKEND LOG ---")
                    print(backend_log.read_text(encoding="utf-8", errors="replace"))
                    print("--- FRONTEND LOG ---")
                    print(frontend_log.read_text(encoding="utf-8", errors="replace"))
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
                time.sleep(0.5)


if __name__ == "__main__":
    main()
