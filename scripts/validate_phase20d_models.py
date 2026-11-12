"""Phase 20D runtime model inventory, fallback, CPU/CUDA, memory, and concurrency validation.

Uses only MELD development fixtures and verified runtime checkpoints; never accesses held-out test splits.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import torch
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import (
    FALLBACK_RESNET_SHA256,
    PRIMARY_RESNET_SHA256,
    ApiSettings,
)
from api.model_registry import ModelRegistry
from api.telemetry import get_process_rss_mb

DEV_TEXT_SAMPLE = "I cannot believe you actually pulled that off, I am thrilled!"
DEV_AUDIO_PATH = REPOSITORY_ROOT / "datasets" / "MELD_raw_acquisition" / "prepared" / "audio" / "dev" / "dia0_utt0.wav"
DEV_VIDEO_PATH = REPOSITORY_ROOT / "datasets" / "MELD_raw_acquisition" / "media" / "dev_splits_complete" / "dia0_utt0.mp4"
REPORT_JSON_PATH = REPOSITORY_ROOT / "models" / "facial_emotion" / "resnet18" / "phase20d_model_validation.json"

RUNTIME_ARTIFACT_FILES = {
    "text_distilbert": [
        REPOSITORY_ROOT / "models" / "text_emotion" / "distilbert" / "best" / "config.json",
        REPOSITORY_ROOT / "models" / "text_emotion" / "distilbert" / "best" / "model.safetensors",
        REPOSITORY_ROOT / "models" / "text_emotion" / "distilbert" / "best" / "tokenizer.json",
        REPOSITORY_ROOT / "models" / "text_emotion" / "distilbert" / "best" / "tokenizer_config.json",
    ],
    "speech_wav2vec2": [
        REPOSITORY_ROOT / "models" / "speech_emotion" / "wav2vec2" / "best" / "config.json",
        REPOSITORY_ROOT / "models" / "speech_emotion" / "wav2vec2" / "best" / "model.safetensors",
        REPOSITORY_ROOT / "models" / "speech_emotion" / "wav2vec2" / "best" / "preprocessor_config.json",
    ],
    "facial_resnet18": [
        REPOSITORY_ROOT / "models" / "facial_emotion" / "resnet18" / "exp5_ce_weighted_best.pt",
        REPOSITORY_ROOT / "models" / "facial_emotion" / "resnet18" / "exp5_focal_g3_wt_best.pt",
        REPOSITORY_ROOT / "models" / "facial_emotion" / "resnet18" / "manifest.json",
    ],
    "facial_legacy_cnn": [
        REPOSITORY_ROOT / "models" / "facial_emotion" / "facial_cnn_best.pt",
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    except ImportError:
        return get_process_rss_mb()


def _collect_artifact_manifest() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    total_bytes = 0

    for group, paths in RUNTIME_ARTIFACT_FILES.items():
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing runtime model artifact: {path}")
            size = path.stat().st_size
            total_bytes += size
            entries.append(
                {
                    "group": group,
                    "relative_path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                    "bytes": size,
                    "sha256": _sha256(path),
                }
            )

    return {
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "total_mib": round(total_bytes / (1024 * 1024), 2),
        "files": entries,
    }


def export_runtime_bundle(destination: Path) -> Path:
    """Copy strictly the verified runtime checkpoints/configs into a staging bundle directory."""
    destination.mkdir(parents=True, exist_ok=True)
    for paths in RUNTIME_ARTIFACT_FILES.values():
        for src in paths:
            rel = src.relative_to(REPOSITORY_ROOT / "models")
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    return destination


def _run_device_benchmark(device: str) -> dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        return {"device": "cuda", "available": False}

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    rss_before = _rss_mb()
    registry = ModelRegistry(ApiSettings(device=device))
    registry.load(force=True)
    rss_after_load = _rss_mb()

    if not registry.all_loaded():
        raise RuntimeError(f"Failed loading all models on {device}: {registry.status()}")

    started_text = time.perf_counter()
    text_res = registry.predict("text", DEV_TEXT_SAMPLE)
    text_ms = round((time.perf_counter() - started_text) * 1000.0, 2)

    started_audio = time.perf_counter()
    audio_res = registry.predict("audio", DEV_AUDIO_PATH)
    audio_ms = round((time.perf_counter() - started_audio) * 1000.0, 2)

    started_video = time.perf_counter()
    video_res = registry.predict("video", DEV_VIDEO_PATH, max_video_seconds=60.0, use_sequential_grab=True)
    video_ms = round((time.perf_counter() - started_video) * 1000.0, 2)

    cuda_stats: dict[str, Any] | None = None
    if device == "cuda":
        cuda_stats = {
            "allocated_mib": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
            "reserved_mib": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
        }

    result = {
        "device": device,
        "available": True,
        "load_duration_ms": registry.load_duration_ms,
        "per_modality_load_ms": {k: v.load_ms for k, v in registry.models.items()},
        "rss_before_mb": rss_before,
        "rss_after_load_mb": rss_after_load,
        "cuda_memory": cuda_stats,
        "inference": {
            "text": {
                "label": text_res["predicted_label"],
                "confidence": round(float(text_res["confidence"]), 4),
                "latency_ms": text_ms,
            },
            "audio": {
                "label": audio_res["predicted_label"],
                "confidence": round(float(audio_res["confidence"]), 4),
                "latency_ms": audio_ms,
            },
            "video": {
                "label": video_res["predicted_label"],
                "confidence": round(float(video_res["confidence"]), 4),
                "num_faces": video_res.get("num_faces"),
                "num_sampled_frames": video_res.get("num_sampled_frames"),
                "latency_ms": video_ms,
            },
        },
        "statuses": registry.status(),
    }
    registry.unload()
    return result


def _validate_fallbacks() -> dict[str, Any]:
    # 1. Corrupt/mismatched primary SHA triggers weighted focal-loss fallback ResNet18
    bad_primary_settings = ApiSettings(
        device="cpu",
        video_model_sha256="f" * 64,
        video_fallback_sha256=FALLBACK_RESNET_SHA256,
    )
    fallback_reg = ModelRegistry(bad_primary_settings)
    fallback_reg.load()
    video_status = fallback_reg.status()["video"]
    if not video_status["loaded"] or video_status["checkpoint_sha256"] != FALLBACK_RESNET_SHA256:
        raise RuntimeError(f"Fallback checkpoint did not activate as expected: {video_status}")
    fallback_pred = fallback_reg.predict("video", DEV_VIDEO_PATH, max_video_seconds=60.0, use_sequential_grab=True)
    fallback_reg.unload()

    # 2. Explicit legacy CNN fallback backend
    legacy_settings = ApiSettings(device="cpu", facial_backend="legacy_cnn")
    legacy_reg = ModelRegistry(legacy_settings)
    legacy_reg.load()
    legacy_status = legacy_reg.status()["video"]
    if not legacy_status["loaded"]:
        raise RuntimeError(f"Legacy CNN backend failed to load: {legacy_status}")
    legacy_pred = legacy_reg.predict("video", DEV_VIDEO_PATH)
    legacy_reg.unload()

    return {
        "corrupt_primary_to_focal_fallback": {
            "loaded": video_status["loaded"],
            "active_role": video_status.get("active_role"),
            "checkpoint_sha256": video_status.get("checkpoint_sha256"),
            "predicted_label": fallback_pred["predicted_label"],
            "confidence": round(float(fallback_pred["confidence"]), 4),
        },
        "legacy_cnn_backend": {
            "loaded": legacy_status["loaded"],
            "model_identifier": legacy_status.get("model_identifier"),
            "predicted_label": legacy_pred["predicted_label"],
            "confidence": round(float(legacy_pred["confidence"]), 4),
        },
    }


def _validate_concurrency() -> dict[str, Any]:
    audio_bytes = DEV_AUDIO_PATH.read_bytes()
    video_bytes = DEV_VIDEO_PATH.read_bytes()
    staging_app = create_app(ApiSettings(env="staging", enable_docs=False, device="cpu", rate_limit_requests=60))

    with TestClient(staging_app) as client:
        def _call(idx: int) -> tuple[int, str, float]:
            started = time.perf_counter()
            if idx % 4 == 0:
                resp = client.post("/predict/text", json={"text": DEV_TEXT_SAMPLE})
                kind = "text"
            elif idx % 4 == 1:
                resp = client.post("/predict/audio", files={"file": ("dev.wav", audio_bytes, "audio/wav")})
                kind = "audio"
            elif idx % 4 == 2:
                resp = client.post("/predict/video", files={"file": ("dev.mp4", video_bytes, "video/mp4")})
                kind = "video"
            else:
                resp = client.post(
                    "/predict/multimodal",
                    data={"text": DEV_TEXT_SAMPLE},
                    files=[
                        ("file", ("dev.wav", audio_bytes, "audio/wav")),
                        ("video", ("dev.mp4", video_bytes, "video/mp4")),
                    ],
                )
                kind = "multimodal"
            return resp.status_code, kind, round((time.perf_counter() - started) * 1000.0, 2)

        started_batch = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            outcomes = list(pool.map(_call, range(4)))
        wall_ms = round((time.perf_counter() - started_batch) * 1000.0, 2)

    all_ok = all(code == 200 for code, _, _ in outcomes)
    if not all_ok:
        raise RuntimeError(f"Concurrent staging validation failed: {outcomes}")

    return {
        "workers": 4,
        "wall_ms": wall_ms,
        "all_200": all_ok,
        "requests": [{"kind": kind, "status": code, "latency_ms": ms} for code, kind, ms in outcomes],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 20D model storage and inference.")
    parser.add_argument(
        "--export-bundle",
        type=Path,
        default=None,
        help="Optional directory to export strictly runtime model files for staging transfer.",
    )
    args = parser.parse_args()

    if not DEV_AUDIO_PATH.is_file() or not DEV_VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Expected MELD dev fixtures at {DEV_AUDIO_PATH} and {DEV_VIDEO_PATH}")

    manifest = _collect_artifact_manifest()
    by_path = {item["relative_path"]: item["sha256"] for item in manifest["files"]}
    assert by_path["models/facial_emotion/resnet18/exp5_ce_weighted_best.pt"] == PRIMARY_RESNET_SHA256
    assert by_path["models/facial_emotion/resnet18/exp5_focal_g3_wt_best.pt"] == FALLBACK_RESNET_SHA256

    bundle_exported = None
    if args.export_bundle is not None:
        exported = export_runtime_bundle(args.export_bundle)
        bundle_exported = str(exported)

    cpu_bench = _run_device_benchmark("cpu")
    cuda_bench = _run_device_benchmark("cuda")
    fallbacks = _validate_fallbacks()
    concurrency = _validate_concurrency()

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dev_fixtures_only": True,
        "exported_bundle_dir": bundle_exported,
        "manifest": manifest,
        "benchmarks": {
            "cpu": cpu_bench,
            "cuda": cuda_bench,
        },
        "fallbacks": fallbacks,
        "concurrency": concurrency,
    }
    REPORT_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {REPORT_JSON_PATH} ({manifest['total_mib']} MiB across {manifest['file_count']} runtime files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
