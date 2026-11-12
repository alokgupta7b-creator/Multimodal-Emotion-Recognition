"""Master orchestrator for Phase 15 End-to-End Browser & Application Testing.

Starts:
1. FastAPI backend server on 127.0.0.1:8000 with CUDA acceleration.
2. Vite frontend dev server on http://localhost:5173.
3. Node.js Puppeteer runner executing real Chrome tests and taking screenshots.
4. Gracefully terminates both processes on exit.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from project_config import REPOSITORY_ROOT

PYTHON_EXE = r"C:\Users\Bhavanth D\anaconda3\python.exe"
FRONTEND_DIR = REPOSITORY_ROOT / "frontend"


def wait_for_url(url: str, timeout_sec: float = 30.0, check_str: str | None = None) -> bool:
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HealthCheck/1.0"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    content = resp.read().decode("utf-8", errors="ignore")
                    if check_str is None or check_str in content:
                        return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    print("=" * 70)
    print("STARTING PHASE 15 END-TO-END APPLICATION SERVERS")
    print("=" * 70)

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONPATH"] = str(REPOSITORY_ROOT)
    env["EMOTION_API_DEVICE"] = "cuda"
    env["EMOTION_API_ALLOWED_ORIGINS"] = "http://localhost:5173,http://127.0.0.1:5173"

    backend_proc = None
    frontend_proc = None

    try:
        # 1. Start Backend
        backend_cmd = [
            PYTHON_EXE,
            "-m",
            "uvicorn",
            "api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "info",
        ]
        print(f"Launching backend: {' '.join(backend_cmd)}")
        backend_proc = subprocess.Popen(
            backend_cmd,
            cwd=str(REPOSITORY_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        print("Waiting for backend on http://127.0.0.1:8000/health...")
        if not wait_for_url("http://127.0.0.1:8000/health", timeout_sec=25.0, check_str="ok"):
            stdout, _ = backend_proc.communicate(timeout=2)
            print("Backend failed to start. Logs:\n", stdout)
            sys.exit(1)
        print("Backend is healthy and models loaded!")

        # 2. Start Frontend
        frontend_cmd = ["npx.cmd", "vite", "--port", "5173", "--host", "localhost"]
        print(f"Launching frontend: {' '.join(frontend_cmd)}")
        frontend_proc = subprocess.Popen(
            frontend_cmd,
            cwd=str(FRONTEND_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        print("Waiting for frontend on http://localhost:5173...")
        if not wait_for_url("http://localhost:5173", timeout_sec=20.0):
            stdout, _ = frontend_proc.communicate(timeout=2)
            print("Frontend failed to start. Logs:\n", stdout)
            sys.exit(1)
        print("Frontend Vite dev server is running!")

        # 3. Launch Puppeteer Browser Tests
        print("\nStarting Node.js Puppeteer Chrome Runner...")
        runner_cmd = ["node", str(REPOSITORY_ROOT / "scripts" / "e2e_browser_test.cjs")]
        res = subprocess.run(runner_cmd, cwd=str(FRONTEND_DIR), env=env, check=False)

        if res.returncode != 0:
            print(f"Browser validation failed with exit code {res.returncode}")
            sys.exit(res.returncode)
        else:
            print("ALL BROWSER TESTS AND SCREENSHOTS COMPLETED SUCCESSFULLY!")

    finally:
        print("\nTerminating background server processes...")
        if frontend_proc is not None:
            frontend_proc.terminate()
            try:
                frontend_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                frontend_proc.kill()
            print("Frontend server stopped.")

        if backend_proc is not None:
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                backend_proc.kill()
            print("Backend server stopped.")

    print("=" * 70)


if __name__ == "__main__":
    main()

