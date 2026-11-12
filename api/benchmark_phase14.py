"""Phase 14 Latency and Integration Benchmark.

Measures end-to-end API latency, device execution (CPU & CUDA), fallback behavior,
and verifies model checkpoint integrity.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import csv
import hashlib
import io
import json
import math
import os
import struct
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from api.app import app
from api.config import ApiSettings, settings
from project_config import MELD_PREPARED_ROOT, REPOSITORY_ROOT


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def get_dev_audio() -> Path:
    manifest = MELD_PREPARED_ROOT / "reports" / "dev" / "audio_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("audio_status") == "ok":
                path = Path(row.get("output_path", ""))
                if path.is_file():
                    return path
    raise RuntimeError("No dev audio file found.")


def generate_synthetic_wav(silent: bool = False, duration_sec: float = 1.0) -> bytes:
    sr = 16_000
    n_samples = int(sr * duration_sec)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        if silent:
            handle.writeframes(b"\x00\x00" * n_samples)
        else:
            data = bytearray()
            for i in range(n_samples):
                val = int(12000 * math.sin(2 * math.pi * 440 * (i / sr)))
                data.extend(struct.pack("<h", val))
            handle.writeframes(bytes(data))
    return buffer.getvalue()


def stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values)
    return {
        "min_ms": float(np.min(arr)),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(np.max(arr)),
        "std_ms": float(np.std(arr)),
    }


def run_benchmark(device: str = "cpu", num_warmup: int = 3, num_runs: int = 20) -> dict[str, Any]:
    print(f"\n--- Running benchmark on device: {device.upper()} ---")
    os.environ["EMOTION_API_DEVICE"] = device

    # Spin up TestClient with fresh lifespan
    bench_settings = ApiSettings(device=device)
    # Patch settings in api.app and api.config
    import api.config
    import api.app

    original_settings = api.config.settings
    api.config.settings = bench_settings
    api.app.settings = bench_settings

    results: dict[str, Any] = {"device": device, "benchmarks": {}}

    dev_audio_path = get_dev_audio()
    dev_audio_bytes = dev_audio_path.read_bytes()
    silent_bytes = generate_synthetic_wav(silent=True)
    corrupt_bytes = b"NOT_A_VALID_WAV_HEADER_DATA"

    with TestClient(app) as client:
        # Check model status
        status_resp = client.get("/models/status")
        results["model_status"] = status_resp.json()
        print(f"Model status: {results['model_status']}")

        scenarios = [
            (
                "multimodal_text_json",
                lambda: client.post("/predict/multimodal", json={"text": "I am so happy and delighted!"}),
            ),
            (
                "multimodal_text_form",
                lambda: client.post("/predict/multimodal", data={"text": "I am so happy and delighted!"}),
            ),
            (
                "multimodal_text_and_audio",
                lambda: client.post(
                    "/predict/multimodal",
                    data={"text": "Get out of here right now!"},
                    files={"file": (dev_audio_path.name, dev_audio_bytes, "audio/wav")},
                ),
            ),
            (
                "multimodal_fallback_silent_audio",
                lambda: client.post(
                    "/predict/multimodal",
                    data={"text": "Just another regular day."},
                    files={"file": ("silent.wav", silent_bytes, "audio/wav")},
                ),
            ),
            (
                "multimodal_fallback_corrupt_upload",
                lambda: client.post(
                    "/predict/multimodal",
                    data={"text": "Just another regular day."},
                    files={"file": ("corrupt.wav", corrupt_bytes, "audio/wav")},
                ),
            ),
        ]

        # Single-modality legacy endpoints for comparison
        if device == "cpu":
            scenarios.extend([
                (
                    "legacy_predict_text",
                    lambda: client.post("/predict/text", json={"text": "I am so happy and delighted!"}),
                ),
                (
                    "legacy_predict_audio",
                    lambda: client.post(
                        "/predict/audio",
                        files={"file": (dev_audio_path.name, dev_audio_bytes, "audio/wav")},
                    ),
                ),
            ])

        for name, run_fn in scenarios:
            print(f"Benchmarking scenario: {name} (warmup={num_warmup}, runs={num_runs})...")
            # Warmup
            for _ in range(num_warmup):
                resp = run_fn()
                assert resp.status_code == 200, f"Warmup failed for {name}: {resp.text}"

            roundtrip_times: list[float] = []
            server_times: list[float] = []
            last_body: dict[str, Any] = {}

            for _ in range(num_runs):
                t0 = time.perf_counter()
                resp = run_fn()
                t1 = time.perf_counter()
                assert resp.status_code == 200, f"Run failed for {name}: {resp.text}"
                roundtrip_times.append((t1 - t0) * 1000.0)
                body = resp.json()
                last_body = body
                if "inference_duration_ms" in body:
                    server_times.append(body["inference_duration_ms"])

            results["benchmarks"][name] = {
                "client_roundtrip": stats(roundtrip_times),
                "server_inference": stats(server_times) if server_times else None,
                "sample_output": {
                    "predicted_emotion": last_body.get("predicted_emotion"),
                    "confidence": last_body.get("confidence"),
                    "fusion_method": last_body.get("fusion_method"),
                    "modalities_available": last_body.get("modalities_available"),
                    "exclusion_reasons": last_body.get("exclusion_reasons"),
                    "rule_applied": last_body.get("rule_applied"),
                    "modality_weights": last_body.get("modality_weights"),
                },
            }
            rt_median = results["benchmarks"][name]["client_roundtrip"]["median_ms"]
            print(f"  -> Roundtrip Median: {rt_median:.2f} ms")

    # Restore settings
    api.config.settings = original_settings
    api.app.settings = original_settings
    return results


def main() -> None:
    output_dir = REPOSITORY_ROOT / "models" / "phase14_integration"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model Checkpoints Integrity
    checkpoints = {
        "text_distilbert": REPOSITORY_ROOT / "models" / "text_emotion" / "distilbert" / "best" / "model.safetensors",
        "speech_wav2vec2": REPOSITORY_ROOT / "models" / "speech_emotion" / "wav2vec2" / "best" / "model.safetensors",
        "facial_cnn": REPOSITORY_ROOT / "models" / "facial_emotion" / "facial_cnn_best.pt",
    }
    checkpoint_integrity = {}
    print("Computing Checkpoint Hashes:")
    for name, path in checkpoints.items():
        assert path.is_file(), f"Checkpoint {path} does not exist!"
        h = compute_sha256(path)
        sz = path.stat().st_size
        checkpoint_integrity[name] = {
            "path": str(path.relative_to(REPOSITORY_ROOT)),
            "sha256": h,
            "bytes": sz,
        }
        print(f"  {name}: {h} ({sz:,} bytes)")

    # 2. Test Split Isolation Check
    test_dirs = [
        REPOSITORY_ROOT / "datasets" / "test_splits_complete",
        REPOSITORY_ROOT / "data" / "meld" / "audio" / "test",
        MELD_PREPARED_ROOT / "reports" / "test",
    ]
    test_isolation = {}
    for td in test_dirs:
        test_isolation[str(td.name)] = {
            "path": str(td),
            "exists": td.exists(),
        }

    # 3. CPU Benchmarks
    cpu_results = run_benchmark(device="cpu", num_warmup=3, num_runs=20)

    # 4. CUDA Benchmarks (if available)
    import torch

    cuda_results = None
    if torch.cuda.is_available():
        cuda_results = run_benchmark(device="cuda", num_warmup=3, num_runs=20)

    final_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint_integrity": checkpoint_integrity,
        "test_isolation": test_isolation,
        "cpu_benchmarks": cpu_results,
        "cuda_benchmarks": cuda_results,
    }

    report_path = output_dir / "benchmark_results.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nPhase 14 benchmark successfully saved to {report_path}")


if __name__ == "__main__":
    main()
