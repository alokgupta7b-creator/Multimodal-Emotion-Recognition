"""Phase 18E — Comprehensive Backend Reliability, Parity, Stress & Integrity Suite.

Validates after Phase 18A-18D integration:
1. Zero MELD held-out `test` split references in Phase 18 integration/validation scripts.
2. Exact SHA-256 integrity of all protected legacy/text/speech/fusion/experimental assets.
3. CPU vs CUDA strict FP32 parity (`max_abs_diff < 1e-4`), fallback checkpoint loading, and
   `EMOTION_API_FACIAL_BACKEND=legacy_cnn` rollback initialization.
4. Full FastAPI endpoint regression, Candidate 4 exact parity, all modality combinations,
   invalid/corrupt/audio-only-MP4/blank-face/oversized rejections, temp-file cleanup,
   multi-threaded concurrent video + multimodal execution, and RAM / VRAM / latency telemetry.
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

import concurrent.futures
import csv
import ctypes
import ctypes.wintypes
import json
import shutil
import subprocess
import tempfile
import time
import wave
from typing import Any

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

import api.app as api_app
from api.config import ApiSettings
from api.fusion import CLASS_NAMES, Candidate4FusionEngine
from api.model_registry import ModelRegistry
from facial_emotion.model import FacialEmotionPredictor
from facial_emotion.resnet_predictor import (
    DEFAULT_FALLBACK_CHECKPOINT,
    FALLBACK_CHECKPOINT_SHA256,
    ResNetFacialEmotionPredictor,
    compute_sha256,
)
from project_config import MELD_PREPARED_ROOT, REPOSITORY_ROOT

OUT_JSON = REPOSITORY_ROOT / "models" / "facial_emotion" / "resnet18" / "phase18e_regression_validation.json"

PROTECTED_HASHES = {
    "facial_emotion/model.py": "08df340cc1fa81fd5420511a248f44149961f8b19d0dbed9023621ab0545061b",
    "facial_emotion/detector.py": "24c183be7c61f336940d2e2fb1fd55cd030a5779044e88dd3f964840752f075d",
    "facial_emotion/remediation.py": "3f2fa952afc890af0bc19a4b5800daa56e5f2e237f6cc5665d3aa64f5a947f07",
    "api/fusion.py": "ffa76bea0a7a305c67e4ebd79790730cdc5e7be488f96d76df7768486efb2f2d",
    "models/facial_emotion/facial_cnn_best.pt": "85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760",
    "models/text_emotion/distilbert/best/model.safetensors": "6fbc700eee6cd6936019be7be3c4dd2526a792fd21b79d5d1aa2f22195021474",
    "models/speech_emotion/wav2vec2/best/model.safetensors": "15256b6625bc0aa7602e53335b49e3c3d01bc0a8def0c71cf5620115aa783d6e",
    "models/facial_emotion/resnet18/exp5_ce_weighted_best.pt": "8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64",
    "models/facial_emotion/resnet18/exp5_focal_g3_wt_best.pt": "4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0",
    "models/facial_experiments/exp5_focal_loss/exp5_ce_weighted_best.pt": "8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64",
    "models/facial_experiments/exp5_focal_loss/exp5_focal_g3_wt_best.pt": "4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0",
}


def _get_process_rss_mb() -> float:
    try:
        import psutil  # type: ignore

        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 2)
    except Exception:
        pass

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
    handle = k32.GetCurrentProcess()
    if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
        return round(pmc.WorkingSetSize / (1024 * 1024), 2)
    return 0.0


def _temp_media_files() -> set[Path]:
    root = Path(tempfile.gettempdir())
    found: set[Path] = set()
    for pat in ("*.wav", "*.mp4", "*.webm", "emotion*"):
        found.update(p for p in root.glob(pat) if p.is_file())
    return found


def _audit_zero_test_split_access() -> dict[str, Any]:
    phase18_files = [
        "facial_emotion/mtcnn_detector.py",
        "facial_emotion/resnet_predictor.py",
        "api/multimodal_extension.py",
        "api/app.py",
        "api/config.py",
        "api/model_registry.py",
        "api/schemas.py",
        "scripts/validate_phase18c_video.py",
        "scripts/validate_phase18d_multimodal.py",
        "scripts/validate_phase18e_regression.py",
    ]
    forbidden_tokens = ("test_sent_emo", "prepared/reports/test", "\\reports\\test", "/meld/test", "\\meld\\test")
    violations: list[dict[str, Any]] = []
    for rel in phase18_files:
        path = REPOSITORY_ROOT / rel
        lines = path.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            low = line.lower().replace("\\", "/")
            # Ignore explicit blocklist guard definitions inside _assert_no_test_path / _audit_zero_test_split_access
            if "test_sent_emo" in low and ("if " in low or "forbidden_tokens" in low):
                continue
            for tok in forbidden_tokens:
                if tok in low:
                    violations.append({"file": rel, "line": lineno, "text": line.strip()})
    return {"checked_files": len(phase18_files), "violations": violations, "passed": len(violations) == 0}


def _verify_protected_hashes() -> dict[str, Any]:
    details: dict[str, Any] = {}
    all_ok = True
    for rel, expected in PROTECTED_HASHES.items():
        path = REPOSITORY_ROOT / rel
        actual = compute_sha256(path)
        matched = actual == expected
        if not matched:
            all_ok = False
        details[rel] = {"expected": expected, "actual": actual, "matched": matched}
    return {"passed": all_ok, "files": details}


def _find_real_dev_samples() -> tuple[Path, Path]:
    video_manifest = MELD_PREPARED_ROOT / "reports" / "dev" / "video_manifest.csv"
    audio_manifest = MELD_PREPARED_ROOT / "reports" / "dev" / "audio_manifest.csv"

    video_map: dict[str, Path] = {}
    with video_manifest.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("video_status") == "ok":
                p = Path(row.get("resolved_video_path") or row.get("output_path") or "")
                if p.is_file():
                    video_map[str(row["sample_id"])] = p

    audio_map: dict[str, Path] = {}
    with audio_manifest.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("audio_status") == "ok":
                p = Path(row.get("resolved_audio_path") or row.get("output_path") or "")
                if p.is_file():
                    audio_map[str(row["sample_id"])] = p

    # Pick first dev clip that has valid face detections and valid audio
    probe_predictor = ResNetFacialEmotionPredictor(device="cuda" if torch.cuda.is_available() else "cpu")
    for sid in sorted(video_map.keys() & audio_map.keys())[:25]:
        vpath = video_map[sid]
        apath = audio_map[sid]
        if "test" in str(vpath).lower().replace("\\", "/").split("/"):
            continue
        try:
            res = probe_predictor.predict_video(vpath)
            if res["num_faces"] > 0:
                return apath, vpath
        except ValueError:
            continue
    raise RuntimeError("Could not locate a valid MELD dev sample with both face video and speech audio.")


def _assert_prob_contract(payload: dict[str, Any]) -> None:
    probs = payload["probabilities"]
    assert tuple(probs.keys()) == CLASS_NAMES
    vals = np.array([float(probs[c]) for c in CLASS_NAMES], dtype=np.float64)
    assert np.isfinite(vals).all()
    assert (vals >= 0.0).all()
    assert abs(float(vals.sum()) - 1.0) < 1e-4


def main() -> None:
    started = time.perf_counter()
    report: dict[str, Any] = {
        "phase": "18E",
        "deployed_to_production": False,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    # 1. Zero test-split leakage audit
    audit = _audit_zero_test_split_access()
    assert audit["passed"], f"Forbidden test split reference found: {audit['violations']}"
    report["zero_test_split_audit"] = audit

    # 2. Protected file & checkpoint SHA-256 verification
    hashes = _verify_protected_hashes()
    assert hashes["passed"], "Protected checkpoint/source SHA-256 mismatch!"
    report["protected_hashes"] = hashes

    # 3. CPU vs CUDA parity + Fallback checkpoint + Legacy rollback check
    rng = np.random.default_rng(1805)
    crops = rng.integers(0, 256, size=(16, 96, 96, 3), dtype=np.uint8)
    cpu_pred = ResNetFacialEmotionPredictor(device="cpu", strict_fp32=True)
    t_c = time.perf_counter()
    cpu_logits, _ = cpu_pred.infer_crop_logits(crops[:1])
    cpu_single_ms = (time.perf_counter() - t_c) * 1000.0
    cpu_batch, _ = cpu_pred.infer_crop_logits(crops)
    cpu_probs = torch.softmax(torch.from_numpy(cpu_batch), dim=-1).numpy()

    parity: dict[str, Any] = {"cpu_single_crop_ms": round(cpu_single_ms, 2)}
    if torch.cuda.is_available():
        cuda_pred = ResNetFacialEmotionPredictor(device="cuda", strict_fp32=True)
        _ = cuda_pred.infer_crop_logits(crops[:1])  # warmup
        t_g = time.perf_counter()
        _, _ = cuda_pred.infer_crop_logits(crops[:1])
        cuda_single_ms = (time.perf_counter() - t_g) * 1000.0
        cuda_batch, _ = cuda_pred.infer_crop_logits(crops)
        cuda_probs = torch.softmax(torch.from_numpy(cuda_batch), dim=-1).numpy()
        max_diff = float(np.max(np.abs(cpu_probs - cuda_probs)))
        argmax_matches = int((np.argmax(cpu_probs, axis=1) == np.argmax(cuda_probs, axis=1)).sum())
        assert max_diff < 1e-4 and argmax_matches == 16
        parity.update(
            {
                "cuda_single_crop_ms": round(cuda_single_ms, 2),
                "max_abs_prob_diff": max_diff,
                "argmax_matches": argmax_matches,
                "total_crops": 16,
            }
        )
    report["cpu_cuda_parity"] = parity

    fb_pred = ResNetFacialEmotionPredictor(
        model_path=DEFAULT_FALLBACK_CHECKPOINT.parent / "__nonexistent_primary_trigger_fallback__.pt",
        expected_sha256="0" * 64,
        fallback_model_path=DEFAULT_FALLBACK_CHECKPOINT,
        fallback_expected_sha256=FALLBACK_CHECKPOINT_SHA256,
        device="cpu",
    )
    assert fb_pred.active_role == "fallback"
    assert fb_pred.model_identifier == "facial-resnet18-focal-v1"
    fb_out = fb_pred.predict_faces([crops[0]])
    _assert_prob_contract(fb_out)
    report["fallback_verification"] = {
        "model_identifier": fb_pred.model_identifier,
        "checkpoint_sha256": fb_pred.checkpoint_sha256,
        "predicted_label": fb_out["predicted_label"],
    }

    legacy_reg = ModelRegistry(ApiSettings(facial_backend="legacy_cnn", device="cpu"))
    legacy_video = legacy_reg._build_video_predictor("cpu")
    assert isinstance(legacy_video, FacialEmotionPredictor)
    report["legacy_rollback_check"] = {"backend": "legacy_cnn", "class": type(legacy_video).__name__, "passed": True}

    # 4. Locate real MELD dev sample + run FastAPI full end-to-end & concurrency tests
    dev_audio, dev_video = _find_real_dev_samples()
    report["dev_probe_sample"] = {
        "audio": dev_audio.name,
        "video": dev_video.name,
        "audio_path": str(dev_audio),
        "video_path": str(dev_video),
    }

    temp_before = _temp_media_files()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        # Prepare blank MP4, silent WAV, corrupt files, audio-only MP4, oversized payload
        blank_mp4 = tdir / "blank.mp4"
        wr = cv2.VideoWriter(str(blank_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 12, (160, 120))
        for _ in range(12):
            wr.write(np.zeros((120, 160, 3), dtype=np.uint8))
        wr.release()

        silent_wav = tdir / "silent.wav"
        with wave.open(str(silent_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(16000, dtype=np.int16).tobytes())

        audio_only_mp4 = tdir / "audio_only.mp4"
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            subprocess.run(
                [ffmpeg, "-y", "-i", str(dev_audio), "-vn", "-c:a", "aac", str(audio_only_mp4)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        with TestClient(api_app.app) as client:
            health = client.get("/health").json()
            status = client.get("/models/status").json()
            assert health["status"] == "ok" and status["status"] == "ready"
            assert status["models"]["video"]["model_identifier"] == "facial-resnet18-ce-v1"
            assert status["models"]["video"]["checkpoint_sha256"] == PROTECTED_HASHES["models/facial_emotion/resnet18/exp5_ce_weighted_best.pt"]

            # Candidate 4 preservation check via API
            txt = "What on earth are you doing here?!"
            r_text = client.post("/predict/text", json={"text": txt})
            assert r_text.status_code == 200
            _assert_prob_contract(r_text.json())

            with dev_audio.open("rb") as af:
                r_audio = client.post("/predict/audio", files={"file": (dev_audio.name, af, "audio/wav")})
            assert r_audio.status_code == 200
            _assert_prob_contract(r_audio.json())

            with dev_video.open("rb") as vf:
                r_video = client.post("/predict/video", files={"file": (dev_video.name, vf, "video/mp4")})
            assert r_video.status_code == 200
            _assert_prob_contract(r_video.json())

            # Candidate 4 direct parity vs /predict/multimodal without video
            c4 = Candidate4FusionEngine()
            c4_direct = c4.fuse(r_text.json()["probabilities"], r_audio.json()["probabilities"])
            with dev_audio.open("rb") as af:
                r_ta = client.post(
                    "/predict/multimodal",
                    data={"text": txt},
                    files={"file": (dev_audio.name, af, "audio/wav")},
                )
            assert r_ta.status_code == 200
            d_ta = r_ta.json()
            assert d_ta["fusion_method"] == c4_direct.fusion_method
            assert d_ta["modality_weights"] == c4_direct.modality_weights
            for c in CLASS_NAMES:
                assert abs(d_ta["probabilities"][c] - c4_direct.probabilities[c]) < 1e-6

            # Extended multimodal combinations
            with dev_video.open("rb") as vf:
                r_tv = client.post("/predict/multimodal", data={"text": txt}, files={"video": (dev_video.name, vf, "video/mp4")})
            assert r_tv.status_code == 200
            _assert_prob_contract(r_tv.json())

            with dev_audio.open("rb") as af, dev_video.open("rb") as vf:
                r_tav = client.post(
                    "/predict/multimodal",
                    data={"text": txt},
                    files={"file": (dev_audio.name, af, "audio/wav"), "video": (dev_video.name, vf, "video/mp4")},
                )
            assert r_tav.status_code == 200
            _assert_prob_contract(r_tav.json())

            with dev_video.open("rb") as vf:
                r_vo = client.post("/predict/multimodal", files={"video": (dev_video.name, vf, "video/mp4")})
            assert r_vo.status_code == 200 and r_vo.json()["modality_weights"]["video"] == 1.0
            _assert_prob_contract(r_vo.json())

            with dev_audio.open("rb") as af, dev_video.open("rb") as vf:
                r_av = client.post(
                    "/predict/multimodal",
                    files={"file": (dev_audio.name, af, "audio/wav"), "video": (dev_video.name, vf, "video/mp4")},
                )
            assert r_av.status_code == 200
            _assert_prob_contract(r_av.json())

            # Negative & edge rejection matrix
            r_blank_txt = client.post("/predict/text", json={"text": "   "})
            assert r_blank_txt.status_code == 422
            with silent_wav.open("rb") as sf:
                r_silent_wav = client.post("/predict/audio", files={"file": ("silent.wav", sf, "audio/wav")})
            assert r_silent_wav.status_code == 422
            with blank_mp4.open("rb") as bf:
                r_blank_mp4 = client.post("/predict/video", files={"file": ("blank.mp4", bf, "video/mp4")})
            assert r_blank_mp4.status_code == 422
            r_bad_sig = client.post("/predict/video", files={"file": ("bad_sig.mp4", b"corrupt", "video/mp4")})
            assert r_bad_sig.status_code == 415
            r_bad_dec = client.post(
                "/predict/video",
                files={"file": ("bad_decode.mp4", b"\x00\x00\x00\x18ftypisom" + b"bad_payload" * 32, "video/mp4")},
            )
            assert r_bad_dec.status_code == 422
            r_ao_code = None
            if ffmpeg:
                with audio_only_mp4.open("rb") as aof:
                    r_ao = client.post("/predict/video", files={"file": ("ao.mp4", aof, "video/mp4")})
                assert r_ao.status_code == 422
                r_ao_code = r_ao.status_code
            # Oversized (> 50 MB)
            oversized = b"\x00\x00\x00\x18ftypisom" + (b"\x00" * (api_app.settings.max_upload_bytes + 1024))
            r_big = client.post("/predict/video", files={"file": ("big.mp4", oversized, "video/mp4")})
            assert r_big.status_code == 413
            report["negative_rejection_matrix"] = {
                "blank_text_422": r_blank_txt.status_code,
                "silent_wav_422": r_silent_wav.status_code,
                "no_face_mp4_422": r_blank_mp4.status_code,
                "invalid_signature_415": r_bad_sig.status_code,
                "corrupt_mp4_decode_422": r_bad_dec.status_code,
                "audio_only_mp4_422": r_ao_code,
                "oversized_video_413": r_big.status_code,
            }

            # Latency benchmark (5 runs each after warm-up)
            latencies: dict[str, list[float]] = {"text": [], "audio": [], "video": [], "multimodal_tav": []}
            audio_bytes = dev_audio.read_bytes()
            video_bytes = dev_video.read_bytes()
            for _ in range(5):
                t0 = time.perf_counter()
                assert client.post("/predict/text", json={"text": txt}).status_code == 200
                latencies["text"].append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                assert client.post("/predict/audio", files={"file": ("a.wav", audio_bytes, "audio/wav")}).status_code == 200
                latencies["audio"].append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                assert client.post("/predict/video", files={"file": ("v.mp4", video_bytes, "video/mp4")}).status_code == 200
                latencies["video"].append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                assert (
                    client.post(
                        "/predict/multimodal",
                        data={"text": txt},
                        files={"file": ("a.wav", audio_bytes, "audio/wav"), "video": ("v.mp4", video_bytes, "video/mp4")},
                    ).status_code
                    == 200
                )
                latencies["multimodal_tav"].append((time.perf_counter() - t0) * 1000.0)

            report["endpoint_latencies_ms"] = {
                k: {
                    "mean": round(float(np.mean(v)), 2),
                    "p95": round(float(np.percentile(v, 95)), 2),
                    "min": round(float(np.min(v)), 2),
                    "max": round(float(np.max(v)), 2),
                }
                for k, v in latencies.items()
            }

            # Concurrent video + multimodal requests
            def _run_concurrent(i: int) -> tuple[int, dict[str, float]]:
                if i % 2 == 0:
                    resp = client.post("/predict/video", files={"file": (f"v{i}.mp4", video_bytes, "video/mp4")})
                else:
                    resp = client.post(
                        "/predict/multimodal",
                        data={"text": f"Concurrent request {i}!"},
                        files={"file": (f"a{i}.wav", audio_bytes, "audio/wav"), "video": (f"v{i}.mp4", video_bytes, "video/mp4")},
                    )
                assert resp.status_code == 200
                data = resp.json()
                _assert_prob_contract(data)
                return resp.status_code, data["probabilities"]

            t_conc = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                conc_results = list(pool.map(_run_concurrent, range(6)))
            conc_ms = (time.perf_counter() - t_conc) * 1000.0
            report["concurrency_test"] = {
                "requests": len(conc_results),
                "workers": 3,
                "all_200": all(code == 200 for code, _ in conc_results),
                "wall_time_ms": round(conc_ms, 2),
            }

            report["api_model_status_snapshot"] = status

    temp_leaked = [p.name for p in (_temp_media_files() - temp_before)]
    assert not temp_leaked, f"Temp files leaked: {temp_leaked}"
    report["temp_file_cleanup"] = {"leaked_files": temp_leaked, "passed": True}

    report["resources"] = {
        "process_rss_mb": _get_process_rss_mb(),
        "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
        "cuda_reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
        "total_script_seconds": round(time.perf_counter() - started, 2),
    }

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Phase 18E regression script PASSED cleanly. Saved: {OUT_JSON}")
    print(json.dumps({"endpoint_latencies_ms": report["endpoint_latencies_ms"], "resources": report["resources"], "concurrency_test": report["concurrency_test"]}, indent=2))


if __name__ == "__main__":
    main()
