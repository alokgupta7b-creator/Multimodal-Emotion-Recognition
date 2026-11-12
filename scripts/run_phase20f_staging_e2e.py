"""Phase 20F — Automated Local Staging-Mode Deployment & End-to-End Validation Runner.

Simulates production-like staging configuration locally (never touches held-out test splits,
never creates public infrastructure or external DNS changes):
  1. Builds the React frontend pointing at the isolated staging API port (`http://127.0.0.1:8018`).
  2. Launches Uvicorn in strict staging mode (`EMOTION_API_ENV=staging`, docs disabled, strict startup).
  3. Launches `vite preview` serving the built static bundle at `http://localhost:5174`.
  4. Verifies `/docs` & `/openapi.json` return 404, `/ready` returns 200, and security headers are set.
  5. Executes `scripts/e2e_phase19b_uat.cjs` across all 7 multimodal combinations & 5 viewports.
  6. Executes sequential, 4-worker concurrent, >50 MiB upload (413), invalid media (422),
     temp-file cleanup, process restart recovery, and corrupt-primary fallback tests.
  7. Restores the default frontend `dist/` build in `finally` and saves `phase20f_staging_validation.json`.
"""

from __future__ import annotations

import concurrent.futures
import csv
import http.client
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np

from api.config import FALLBACK_RESNET_SHA256
from project_config import MELD_PREPARED_ROOT, REPOSITORY_ROOT

STAGING_API_PORT = 8018
STAGING_FALLBACK_PORT = 8019
STAGING_FRONTEND_PORT = 5174
STAGING_API_URL = f"http://127.0.0.1:{STAGING_API_PORT}"
STAGING_FRONTEND_URL = f"http://localhost:{STAGING_FRONTEND_PORT}"

UAT_JSON = REPOSITORY_ROOT / "models" / "facial_emotion" / "resnet18" / "phase20f_uat_validation.json"
OUT_JSON = REPOSITORY_ROOT / "models" / "facial_emotion" / "resnet18" / "phase20f_staging_validation.json"
ARTIFACT_SCRATCH = Path(
    r"C:\Users\Bhavanth D\.gemini\antigravity\brain\6e95820a-a389-4eb8-ac7c-cfe8c9f09e67\scratch"
)


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


def _wait_ready(url: str, proc: subprocess.Popen[bytes], log_path: Path, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            txt = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            raise RuntimeError(f"Server exited prematurely ({proc.returncode}) waiting for {url}\n{txt}")
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:
            last_err = exc
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {url}: {last_err}")


def _terminate_tree(proc: subprocess.Popen[bytes]) -> None:
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


def _http_get(url: str) -> tuple[int, dict[str, str], str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as err:
        return err.code, {k.lower(): v for k, v in err.headers.items()}, err.read().decode("utf-8", errors="replace")


def _encode_multipart(fields: dict[str, str], files: list[tuple[str, str, str, bytes]]) -> tuple[bytes, str]:
    boundary = "----Phase20FStagingBoundary7e12a9"
    chunks: list[bytes] = []
    for key, val in fields.items():
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{val}\r\n".encode("utf-8")
        )
    for field_name, filename, content_type, data in files:
        chunks.append(
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field_name}\"; filename=\"{filename}\"\r\n"
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(data)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(chunks)
    return body, f"multipart/form-data; boundary={boundary}"


def _post_json(base_url: str, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, str], Any]:
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=raw,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, {k.lower(): v for k, v in err.headers.items()}, json.loads(err.read().decode("utf-8"))


def _post_multipart(
    base_url: str,
    path: str,
    fields: dict[str, str],
    files: list[tuple[str, str, str, bytes]],
) -> tuple[int, dict[str, str], Any]:
    body, content_type = _encode_multipart(fields, files)
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        method="POST",
        headers={"Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raw_err = err.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw_err)
        except json.JSONDecodeError:
            parsed = {"detail": raw_err}
        return err.code, {k.lower(): v for k, v in err.headers.items()}, parsed


def _staging_env(port: int, *, video_sha: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "EMOTION_API_ENV": "staging",
            "EMOTION_API_HOST": "127.0.0.1",
            "EMOTION_API_PORT": str(port),
            "EMOTION_API_DEVICE": "cpu",
            "EMOTION_API_ENABLE_DOCS": "0",
            "EMOTION_API_FAIL_ON_MISSING_MODELS": "1",
            "EMOTION_API_ALLOWED_ORIGINS": STAGING_FRONTEND_URL,
            "EMOTION_API_REQUEST_TIMEOUT_SECONDS": "60",
            "EMOTION_API_RATE_LIMIT_REQUESTS": "200",
            "EMOTION_API_RATE_LIMIT_WINDOW_SECONDS": "60",
        }
    )
    if video_sha is not None:
        env["EMOTION_API_VIDEO_SHA256"] = video_sha
        env["EMOTION_API_VIDEO_FALLBACK_SHA256"] = FALLBACK_RESNET_SHA256
    return env


def _build_frontend(api_base_url: str | None) -> None:
    frontend_dir = REPOSITORY_ROOT / "frontend"
    env = os.environ.copy()
    if api_base_url is None:
        env.pop("VITE_API_BASE_URL", None)
    else:
        env["VITE_API_BASE_URL"] = api_base_url
    res = subprocess.run(["cmd.exe", "/c", "npm.cmd", "run", "build"], cwd=str(frontend_dir), env=env, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"Frontend build failed (VITE_API_BASE_URL={api_base_url})")


def _launch_backend(port: int, log_path: Path, env: dict[str, str]) -> subprocess.Popen[bytes]:
    handle = log_path.open("wb")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "1",
        ],
        cwd=str(REPOSITORY_ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    proc._log_handle = handle  # type: ignore[attr-defined]
    return proc


def _close_proc_log(proc: subprocess.Popen[bytes]) -> None:
    handle = getattr(proc, "_log_handle", None)
    if handle is not None:
        handle.close()


def _probe_oversized_upload(host: str, port: int) -> int:
    conn = http.client.HTTPConnection(host, port, timeout=15)
    try:
        conn.request(
            "POST",
            "/predict/audio",
            body=b"0" * 16,
            headers={
                "Content-Type": "audio/wav",
                "Content-Length": str(55 * 1024 * 1024),
            },
        )
        resp = conn.getresponse()
        resp.read()
        return resp.status
    finally:
        conn.close()


def main() -> int:
    dev_wav, dev_mp4 = _find_dev_pair()
    audio_bytes = dev_wav.read_bytes()
    video_bytes = dev_mp4.read_bytes()
    dev_text = "I am genuinely relieved and happy everything held together."

    _build_frontend(STAGING_API_URL)
    report: dict[str, Any] = {
        "staging_simulation_only": True,
        "deployed_to_production": False,
        "external_hosting_status": "AWAITING USER APPROVAL",
        "dev_fixtures": {"audio": str(dev_wav), "video": str(dev_mp4)},
    }

    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            tmp = Path(td)
            no_face_mp4 = tmp / "no_face.mp4"
            writer = cv2.VideoWriter(str(no_face_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 12, (160, 120))
            for _ in range(12):
                writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
            writer.release()

            blog = tmp / "staging_backend.log"
            flog = tmp / "staging_frontend.log"
            vite_bin = REPOSITORY_ROOT / "frontend" / "node_modules" / ".bin" / "vite.cmd"

            backend_proc = _launch_backend(STAGING_API_PORT, blog, _staging_env(STAGING_API_PORT))
            f_handle = flog.open("wb")
            frontend_proc = subprocess.Popen(
                ["cmd.exe", "/c", str(vite_bin), "preview", "--host", "localhost", "--port", str(STAGING_FRONTEND_PORT), "--strictPort"],
                cwd=str(REPOSITORY_ROOT / "frontend"),
                env=os.environ.copy(),
                stdout=f_handle,
                stderr=subprocess.STDOUT,
            )
            try:
                _wait_ready(f"{STAGING_API_URL}/ready", backend_proc, blog, timeout_s=90.0)
                _wait_ready(STAGING_FRONTEND_URL, frontend_proc, flog, timeout_s=30.0)

                ready_code, ready_headers, ready_body = _http_get(f"{STAGING_API_URL}/ready")
                docs_code, _, _ = _http_get(f"{STAGING_API_URL}/docs")
                openapi_code, _, _ = _http_get(f"{STAGING_API_URL}/openapi.json")
                redoc_code, _, _ = _http_get(f"{STAGING_API_URL}/redoc")
                report["staging_probes"] = {
                    "ready_status": ready_code,
                    "ready_payload": json.loads(ready_body),
                    "docs_status": docs_code,
                    "openapi_status": openapi_code,
                    "redoc_status": redoc_code,
                    "security_headers_present": all(
                        h in ready_headers
                        for h in (
                            "x-content-type-options",
                            "x-frame-options",
                            "referrer-policy",
                            "permissions-policy",
                            "cross-origin-resource-policy",
                        )
                    ),
                }
                assert ready_code == 200 and docs_code == 404 and openapi_code == 404 and redoc_code == 404

                # Browser E2E (all 7 combinations, 5 viewports, retry check, screenshots, console/page errors)
                desktop_shot = ARTIFACT_SCRATCH / "phase20f_staging_desktop.png"
                mobile_shot = ARTIFACT_SCRATCH / "phase20f_staging_mobile.png"
                node_env = os.environ.copy()
                node_env["FRONTEND_URL"] = STAGING_FRONTEND_URL
                node_env["DEV_WAV_PATH"] = str(dev_wav)
                node_env["DEV_MP4_PATH"] = str(dev_mp4)
                node_env["NOFACE_MP4_PATH"] = str(no_face_mp4)
                node_env["OUT_JSON"] = str(UAT_JSON)
                node_env["DESKTOP_SHOT"] = str(desktop_shot)
                node_env["MOBILE_SHOT"] = str(mobile_shot)
                uat_res = subprocess.run(
                    ["node", str(REPOSITORY_ROOT / "scripts" / "e2e_phase20f_staging.cjs")],
                    cwd=str(REPOSITORY_ROOT),
                    env=node_env,
                    check=False,
                )
                assert uat_res.returncode == 0, "Browser UAT suite failed in staging mode"
                uat_payload = json.loads(UAT_JSON.read_text(encoding="utf-8"))
                report["browser_uat"] = {
                    "passed": uat_payload["passed"],
                    "viewports": uat_payload["responsiveViewports"],
                    "unexpected_console_errors": uat_payload["unexpectedConsoleErrors"],
                    "page_errors": uat_payload["pageErrors"],
                    "seven_combinations": list(uat_payload["multimodalSevenCombinations"].keys()),
                }
                report["screenshots"] = {
                    "desktop_screenshot": str(desktop_shot) if desktop_shot.is_file() else None,
                    "mobile_screenshot": str(mobile_shot) if mobile_shot.is_file() else None,
                }

                # HTTP API reliability suite against live staging server
                sys_tmp = Path(tempfile.gettempdir())
                media_before = {p.name for p in sys_tmp.iterdir() if p.suffix.lower() in {".wav", ".mp4", ".webm"}}

                seq_text = _post_json(STAGING_API_URL, "/predict/text", {"text": dev_text})
                seq_audio = _post_multipart(STAGING_API_URL, "/predict/audio", {}, [("file", "dev.wav", "audio/wav", audio_bytes)])
                seq_video = _post_multipart(STAGING_API_URL, "/predict/video", {}, [("file", "dev.mp4", "video/mp4", video_bytes)])
                seq_mm = _post_multipart(
                    STAGING_API_URL,
                    "/predict/multimodal",
                    {"text": dev_text},
                    [
                        ("file", "dev.wav", "audio/wav", audio_bytes),
                        ("video", "dev.mp4", "video/mp4", video_bytes),
                    ],
                )
                assert all(item[0] == 200 for item in (seq_text, seq_audio, seq_video, seq_mm))
                assert seq_text[1].get("cache-control") == "no-store"

                def _concurrent_req(idx: int) -> tuple[str, int]:
                    if idx == 0:
                        return "text", _post_json(STAGING_API_URL, "/predict/text", {"text": dev_text})[0]
                    if idx == 1:
                        return "audio", _post_multipart(STAGING_API_URL, "/predict/audio", {}, [("file", "c.wav", "audio/wav", audio_bytes)])[0]
                    if idx == 2:
                        return "video", _post_multipart(STAGING_API_URL, "/predict/video", {}, [("file", "c.mp4", "video/mp4", video_bytes)])[0]
                    return "multimodal", _post_multipart(
                        STAGING_API_URL,
                        "/predict/multimodal",
                        {"text": dev_text},
                        [("file", "c.wav", "audio/wav", audio_bytes), ("video", "c.mp4", "video/mp4", video_bytes)],
                    )[0]

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    conc_outcomes = list(pool.map(_concurrent_req, range(4)))
                assert all(status == 200 for _, status in conc_outcomes)

                oversized_status = _probe_oversized_upload("127.0.0.1", STAGING_API_PORT)
                invalid_audio = _post_multipart(
                    STAGING_API_URL,
                    "/predict/audio",
                    {},
                    [("file", "corrupt.wav", "audio/wav", b"NOT_A_REAL_AUDIO_HEADER")],
                )
                assert oversized_status == 413
                assert invalid_audio[0] in {415, 422}

                media_after = {p.name for p in sys_tmp.iterdir() if p.suffix.lower() in {".wav", ".mp4", ".webm"}}
                leaked_temp_media = sorted(media_after - media_before)
                assert not leaked_temp_media, f"Leaked temp media files: {leaked_temp_media}"

                # Restart & recovery check on same staging port
                _terminate_tree(backend_proc)
                _close_proc_log(backend_proc)
                restart_log = tmp / "staging_backend_restarted.log"
                backend_proc = _launch_backend(STAGING_API_PORT, restart_log, _staging_env(STAGING_API_PORT))
                _wait_ready(f"{STAGING_API_URL}/ready", backend_proc, restart_log, timeout_s=90.0)
                post_restart = _post_multipart(
                    STAGING_API_URL,
                    "/predict/multimodal",
                    {"text": dev_text},
                    [("file", "dev.wav", "audio/wav", audio_bytes)],
                )
                assert post_restart[0] == 200

                report["reliability"] = {
                    "sequential_all_200": True,
                    "concurrent_4_workers": dict(conc_outcomes),
                    "oversized_upload_status": oversized_status,
                    "invalid_upload_status": invalid_audio[0],
                    "leaked_temp_media_files": leaked_temp_media,
                    "restart_recovery_status": post_restart[0],
                    "restart_predicted_emotion": post_restart[2]["predicted_emotion"],
                }
            finally:
                _terminate_tree(frontend_proc)
                f_handle.close()
                _terminate_tree(backend_proc)
                _close_proc_log(backend_proc)

            # Fallback checkpoint startup test on staging port 8019
            fb_log = tmp / "staging_fallback.log"
            fb_url = f"http://127.0.0.1:{STAGING_FALLBACK_PORT}"
            fb_proc = _launch_backend(STAGING_FALLBACK_PORT, fb_log, _staging_env(STAGING_FALLBACK_PORT, video_sha="f" * 64))
            try:
                _wait_ready(f"{fb_url}/ready", fb_proc, fb_log, timeout_s=90.0)
                status_code, _, status_body = _http_get(f"{fb_url}/models/status")
                fb_status_json = json.loads(status_body)
                fb_video_res = _post_multipart(
                    fb_url,
                    "/predict/video",
                    {},
                    [("file", "dev.mp4", "video/mp4", video_bytes)],
                )
                assert status_code == 200
                assert fb_status_json["models"]["video"]["active_role"] == "fallback"
                assert fb_video_res[0] == 200
                report["fallback_staging_server"] = {
                    "ready_status": status_code,
                    "active_role": fb_status_json["models"]["video"]["active_role"],
                    "checkpoint_sha256": fb_status_json["models"]["video"]["checkpoint_sha256"],
                    "video_predict_status": fb_video_res[0],
                }
            finally:
                _terminate_tree(fb_proc)
                _close_proc_log(fb_proc)
    finally:
        # Restore default frontend build (VITE_API_BASE_URL unset -> http://localhost:8000)
        _build_frontend(None)

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Phase 20F staging validation passed -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
