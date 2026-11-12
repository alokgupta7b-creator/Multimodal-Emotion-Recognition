"""Phase 15 End-to-End Backend, Security, and Performance Validation Suite.

Validates:
1. Cold-start model registry loading on CPU and CUDA.
2. Warm text and text+audio model inference latency (CPU vs CUDA).
3. All 6 endpoints (GET /health, GET /models/status, POST /predict/text,
   POST /predict/audio, POST /predict/video, POST /predict/multimodal).
4. Full security matrix (missing Content-Length, streaming oversized rejection,
   invalid signature, malformed multipart, temp file cleanup, CORS, concurrency).
5. Latency breakdown (raw model execution vs total HTTP turnaround).
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import struct
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from fastapi.testclient import TestClient

from api.app import app
from api.config import ApiSettings, settings
from api.model_registry import ModelRegistry
from project_config import MELD_PREPARED_ROOT, REPOSITORY_ROOT
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES


def get_dev_audio() -> Path:
    manifest = MELD_PREPARED_ROOT / "reports" / "dev" / "audio_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("audio_status") == "ok":
                p = Path(row.get("output_path", ""))
                if p.is_file():
                    return p
    raise FileNotFoundError("Dev audio fixture not found.")


def get_dev_video() -> Path:
    manifest = MELD_PREPARED_ROOT / "reports" / "dev" / "video_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("video_status") == "ok":
                key = "resolved_video_path" if "resolved_video_path" in row else "output_path"
                p = Path(row[key])
                if p.is_file():
                    return p
    raise FileNotFoundError("Dev video fixture not found.")


def generate_synthetic_wav(silent: bool = False, duration_sec: float = 1.0) -> bytes:
    sr = 16000
    n = int(sr * duration_sec)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as h:
        h.setnchannels(1)
        h.setsampwidth(2)
        h.setframerate(sr)
        if silent:
            h.writeframes(b"\x00\x00" * n)
        else:
            data = bytearray()
            for i in range(n):
                val = int(12000 * math.sin(2 * math.pi * 440 * (i / sr)))
                data.extend(struct.pack("<h", val))
            h.writeframes(bytes(data))
    return buf.getvalue()


def get_silent_audio() -> Path:
    buf = generate_synthetic_wav(silent=True, duration_sec=1.0)
    temp = Path(tempfile.gettempdir()) / "test_synth_silent.wav"
    temp.write_bytes(buf)
    return temp


def stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values)
    return {
        "count": len(values),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(np.max(arr)),
    }


def main():
    print("=" * 70)
    print("PHASE 15: BACKEND, PERFORMANCE & SECURITY VALIDATION")
    print("=" * 70)

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": {
            "os": os.name,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        },
    }

    # ---------------------------------------------------------
    # 1. Performance: Cold-Start Model Registry Loading
    # ---------------------------------------------------------
    print("\n--- [1] Cold-Start Model Loading Benchmark ---")
    loading_results = {}

    # CPU Cold Start
    print("Measuring CPU cold-start loading...")
    cpu_settings = ApiSettings(device="cpu")
    t0 = time.perf_counter()
    cpu_registry = ModelRegistry(cpu_settings)
    cpu_registry.load()
    cpu_load_time = (time.perf_counter() - t0) * 1000
    loading_results["cpu_load_ms"] = cpu_load_time
    print(f"  CPU Cold Start Load Time: {cpu_load_time:.2f} ms")

    # CUDA Cold Start (if available)
    if torch.cuda.is_available():
        print("Measuring CUDA cold-start loading...")
        cuda_settings = ApiSettings(device="cuda")
        t0 = time.perf_counter()
        cuda_registry = ModelRegistry(cuda_settings)
        cuda_registry.load()
        cuda_load_time = (time.perf_counter() - t0) * 1000
        loading_results["cuda_load_ms"] = cuda_load_time
        print(f"  CUDA Cold Start Load Time: {cuda_load_time:.2f} ms")
    else:
        loading_results["cuda_load_ms"] = None
        print("  CUDA not available.")

    report["cold_start_loading"] = loading_results

    # ---------------------------------------------------------
    # 2. Performance: Warm Inference Benchmarking (CPU vs CUDA)
    # ---------------------------------------------------------
    print("\n--- [2] Warm Inference Benchmarks (100 Iterations) ---")
    dev_audio_path = get_dev_audio()
    sample_text = "I am utterly astonished and delighted by these incredible results!"

    warmup_n = 5
    eval_n = 100

    benchmarks = {}

    for dev_name, reg in [("cpu", cpu_registry)] + ([("cuda", cuda_registry)] if torch.cuda.is_available() else []):
        print(f"\nBenchmarking {dev_name.upper()}...")

        # Warmup Text
        for _ in range(warmup_n):
            reg.predict("text", sample_text)
        # Benchmark Text
        text_times = []
        for _ in range(eval_n):
            t0 = time.perf_counter()
            res = reg.predict("text", sample_text)
            text_times.append((time.perf_counter() - t0) * 1000)

        # Warmup Audio
        for _ in range(warmup_n):
            reg.predict("audio", dev_audio_path)
        # Benchmark Audio
        audio_times = []
        for _ in range(eval_n):
            t0 = time.perf_counter()
            res = reg.predict("audio", dev_audio_path)
            audio_times.append((time.perf_counter() - t0) * 1000)

        # Combined Text + Audio sequentially (pure model time)
        combined_times = []
        for _ in range(eval_n):
            t0 = time.perf_counter()
            t_res = reg.predict("text", sample_text)
            a_res = reg.predict("audio", dev_audio_path)
            combined_times.append((time.perf_counter() - t0) * 1000)

        benchmarks[f"{dev_name}_text"] = stats(text_times)
        benchmarks[f"{dev_name}_audio"] = stats(audio_times)
        benchmarks[f"{dev_name}_text_plus_audio"] = stats(combined_times)

        print(f"  {dev_name.upper()} Text Only: Median = {benchmarks[f'{dev_name}_text']['median_ms']:.2f} ms (p95 = {benchmarks[f'{dev_name}_text']['p95_ms']:.2f} ms)")
        print(f"  {dev_name.upper()} Audio Only: Median = {benchmarks[f'{dev_name}_audio']['median_ms']:.2f} ms (p95 = {benchmarks[f'{dev_name}_audio']['p95_ms']:.2f} ms)")
        print(f"  {dev_name.upper()} Text+Audio: Median = {benchmarks[f'{dev_name}_text_plus_audio']['median_ms']:.2f} ms (p95 = {benchmarks[f'{dev_name}_text_plus_audio']['p95_ms']:.2f} ms)")

    report["model_inference_benchmarks"] = benchmarks

    # ---------------------------------------------------------
    # 3. Backend Endpoints Validation (All 6 Endpoints)
    # ---------------------------------------------------------
    print("\n--- [3] Validating All 6 Endpoints via TestClient ---")
    endpoint_results = {}

    with TestClient(app) as client:
        # GET /health
        r_health = client.get("/health")
        print(f"GET /health: {r_health.status_code}")
        assert r_health.status_code == 200
        health_data = r_health.json()
        assert health_data["status"] == "ok"
        endpoint_results["get_health"] = {"status": r_health.status_code, "body": health_data}

        # GET /models/status
        r_models = client.get("/models/status")
        print(f"GET /models/status: {r_models.status_code}")
        assert r_models.status_code == 200
        models_data = r_models.json()
        assert all(m["loaded"] for m in models_data["models"].values())
        endpoint_results["get_models_status"] = {"status": r_models.status_code, "body": models_data}

        # POST /predict/text (JSON)
        r_text_json = client.post("/predict/text", json={"text": "I am so happy today!"})
        print(f"POST /predict/text (JSON): {r_text_json.status_code}")
        assert r_text_json.status_code == 200
        assert set(r_text_json.json()["probabilities"].keys()) == set(MELD_CLASS_NAMES)
        endpoint_results["post_predict_text_json"] = {"status": r_text_json.status_code, "pred": r_text_json.json()["predicted_emotion"]}

        # POST /predict/audio (multipart form)
        with dev_audio_path.open("rb") as f:
            r_audio = client.post("/predict/audio", files={"file": ("dia0_utt0.wav", f, "audio/wav")})
        print(f"POST /predict/audio: {r_audio.status_code}")
        assert r_audio.status_code == 200
        endpoint_results["post_predict_audio"] = {"status": r_audio.status_code, "pred": r_audio.json()["predicted_emotion"]}

        # POST /predict/video (multipart form)
        dev_video_path = get_dev_video()
        with dev_video_path.open("rb") as f:
            r_video = client.post("/predict/video", files={"file": ("dia0_utt0.mp4", f, "video/mp4")})
        print(f"POST /predict/video: {r_video.status_code}")
        assert r_video.status_code == 200
        endpoint_results["post_predict_video"] = {"status": r_video.status_code, "pred": r_video.json()["predicted_emotion"]}

        # POST /predict/multimodal (JSON text-only)
        r_mm_json = client.post("/predict/multimodal", json={"text": "That was completely shocking!"})
        print(f"POST /predict/multimodal (JSON Text-Only): {r_mm_json.status_code}")
        assert r_mm_json.status_code == 200
        mm_data = r_mm_json.json()
        assert mm_data["modalities_available"]["text"] is True
        assert mm_data["modalities_available"]["audio"] is False
        assert mm_data["exclusion_reasons"]["audio"] == "No audio file provided"
        endpoint_results["post_predict_multimodal_text_json"] = {"status": r_mm_json.status_code, "pred": mm_data["predicted_emotion"], "rule": mm_data.get("rule_applied")}

        # POST /predict/multimodal (Form text-only)
        r_mm_form_text = client.post("/predict/multimodal", data={"text": "Just text in a form"})
        print(f"POST /predict/multimodal (Form Text-Only): {r_mm_form_text.status_code}")
        assert r_mm_form_text.status_code == 200
        endpoint_results["post_predict_multimodal_text_form"] = {"status": r_mm_form_text.status_code, "pred": r_mm_form_text.json()["predicted_emotion"]}

        # POST /predict/multimodal (Form text + audio)
        with dev_audio_path.open("rb") as f:
            r_mm_both = client.post(
                "/predict/multimodal",
                data={"text": "What on earth are you doing here?!"},
                files={"file": ("dia0_utt0.wav", f, "audio/wav")},
            )
        print(f"POST /predict/multimodal (Form Text+Audio): {r_mm_both.status_code}")
        assert r_mm_both.status_code == 200
        mm_both_data = r_mm_both.json()
        assert mm_both_data["modalities_available"]["text"] is True
        assert mm_both_data["modalities_available"]["audio"] is True
        assert mm_both_data["individual_predictions"]["text"] is not None
        assert mm_both_data["individual_predictions"]["audio"] is not None
        assert mm_both_data["individual_predictions"]["video"] is None
        endpoint_results["post_predict_multimodal_text_and_audio"] = {
            "status": r_mm_both.status_code,
            "pred": mm_both_data["predicted_emotion"],
            "rule": mm_both_data.get("rule_applied"),
            "weights": mm_both_data["modality_weights"],
        }

        # POST /predict/multimodal with Silent Audio Fallback
        silent_audio_path = get_silent_audio()
        with silent_audio_path.open("rb") as f:
            r_silent = client.post(
                "/predict/multimodal",
                data={"text": "I feel so quiet and peaceful."},
                files={"file": ("silent.wav", f, "audio/wav")},
            )
        print(f"POST /predict/multimodal (Silent Audio Fallback): {r_silent.status_code}")
        assert r_silent.status_code == 200
        silent_data = r_silent.json()
        assert silent_data["modalities_available"]["audio"] is False
        assert "silent" in str(silent_data["exclusion_reasons"]["audio"]).lower()
        endpoint_results["post_predict_multimodal_silent_fallback"] = {
            "status": r_silent.status_code,
            "exclusion_reason": silent_data["exclusion_reasons"]["audio"],
            "pred": silent_data["predicted_emotion"],
        }

    report["endpoint_validation"] = endpoint_results

    # ---------------------------------------------------------
    # 4. Security & Robustness Matrix
    # ---------------------------------------------------------
    print("\n--- [4] Security & Robustness Verification ---")
    security_results = {}

    with TestClient(app) as client:
        # 4.1 Missing Content-Length with chunked streaming body
        def generate_stream():
            yield b"------Boundary12345\r\n"
            yield b'Content-Disposition: form-data; name="text"\r\n\r\n'
            yield b"chunked text test\r\n"
            yield b"------Boundary12345--\r\n"

        t_res = client.post(
            "/predict/multimodal",
            content=generate_stream(),
            headers={"Content-Type": "multipart/form-data; boundary=----Boundary12345"},
        )
        print(f"Security: Streamed request without Content-Length: {t_res.status_code}")
        assert t_res.status_code == 200
        security_results["streamed_without_content_length"] = {"status": t_res.status_code, "passed": True}

        # 4.2 Streamed oversized upload (middleware truncation check)
        def generate_huge_stream():
            yield b"------BoundaryHuge\r\n"
            yield b'Content-Disposition: form-data; name="file"; filename="huge.wav"\r\n'
            yield b"Content-Type: audio/wav\r\n\r\n"
            # Stream 60 MB in chunks (exceeding 50 MB default)
            chunk = b"X" * (1024 * 1024)
            for _ in range(55):
                yield chunk
            yield b"\r\n------BoundaryHuge--\r\n"

        huge_res = client.post(
            "/predict/audio",
            content=generate_huge_stream(),
            headers={"Content-Type": "multipart/form-data; boundary=----BoundaryHuge"},
        )
        print(f"Security: Oversized streamed body: {huge_res.status_code}")
        assert huge_res.status_code == 413
        security_results["oversized_streamed_rejection"] = {"status": huge_res.status_code, "passed": True}

        # 4.3 Invalid content signatures (fake WAV header)
        fake_wav = b"RIFF\x24\x00\x00\x00NOT_WAVE\x00\x00\x00\x00corrupt_payload_data"
        fake_res = client.post(
            "/predict/audio",
            files={"file": ("fake.wav", fake_wav, "audio/wav")},
        )
        print(f"Security: Invalid WAV signature: {fake_res.status_code}")
        assert fake_res.status_code == 415
        assert "not a valid audio file" in fake_res.json()["detail"].lower()
        security_results["invalid_content_signature"] = {"status": fake_res.status_code, "detail": fake_res.json()["detail"]}

        # 4.4 Malformed multipart requests
        malformed_res = client.post(
            "/predict/multimodal",
            content=b"malformed multipart non-boundary junk",
            headers={"Content-Type": "multipart/form-data; boundary=MissingBoundary"},
        )
        print(f"Security: Malformed multipart body: {malformed_res.status_code}")
        # Handled gracefully without 500 crash
        assert malformed_res.status_code in (400, 422)
        security_results["malformed_multipart"] = {"status": malformed_res.status_code, "passed": True}

        # 4.5 Strict non-string JSON rejection
        for invalid_val in [None, 12345, True, False, ["list"], {"dict": 1}]:
            bad_json_res = client.post("/predict/multimodal", json={"text": invalid_val})
            assert bad_json_res.status_code == 422
        security_results["strict_non_string_rejection"] = {"status": 422, "passed": True}

        # 4.6 CORS configuration
        cors_preflight = client.options(
            "/predict/multimodal",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        print(f"Security: CORS preflight allowed origin: {cors_preflight.status_code}")
        assert cors_preflight.status_code == 200
        assert cors_preflight.headers.get("access-control-allow-origin") == "http://localhost:5173"

        # Disallowed origin
        bad_origin = client.options(
            "/predict/multimodal",
            headers={
                "Origin": "http://evil-attacker.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert bad_origin.headers.get("access-control-allow-origin") is None
        security_results["cors_configuration"] = {"allowed_origin_passed": True, "disallowed_blocked": True}

        # 4.7 Temporary-file cleanup check
        temp_dir = Path(tempfile.gettempdir())
        pre_temp_files = set(temp_dir.glob("emotion_*"))
        with dev_audio_path.open("rb") as f:
            client.post("/predict/audio", files={"file": ("dia0_utt0.wav", f, "audio/wav")})
        post_temp_files = set(temp_dir.glob("emotion_*"))
        new_leaked = post_temp_files - pre_temp_files
        print(f"Security: Temporary files leaked after upload: {len(new_leaked)}")
        assert len(new_leaked) == 0
        security_results["temp_file_cleanup"] = {"leaked_files_count": len(new_leaked), "passed": True}

        # 4.8 Concurrent upload handling
        print("Security: Testing 10 concurrent requests...")
        synth_wav = generate_synthetic_wav(1.0)

        def make_req(idx: int):
            with TestClient(app) as local_client:
                return local_client.post(
                    "/predict/multimodal",
                    data={"text": f"Concurrent test request #{idx}"},
                    files={"file": (f"test_{idx}.wav", io.BytesIO(synth_wav), "audio/wav")},
                ).status_code

        with ThreadPoolExecutor(max_workers=5) as pool:
            codes = list(pool.map(make_req, range(10)))
        print(f"Security: Concurrent upload response codes: {set(codes)}")
        assert all(c == 200 for c in codes)
        security_results["concurrent_uploads"] = {"requests": 10, "status_codes": codes, "passed": True}

    report["security_validation"] = security_results

    # ---------------------------------------------------------
    # 5. HTTP Turnaround Latency vs Model Latency
    # ---------------------------------------------------------
    print("\n--- [5] HTTP Latency vs Model Latency Breakdown (50 runs) ---")
    latency_breakdown = {}

    with TestClient(app) as client:
        # Text-only HTTP roundtrip
        http_text_times = []
        model_text_durations = []
        for _ in range(50):
            t0 = time.perf_counter()
            r = client.post("/predict/multimodal", json={"text": sample_text})
            http_text_times.append((time.perf_counter() - t0) * 1000)
            model_text_durations.append(r.json()["inference_duration_ms"])

        latency_breakdown["http_text_only"] = stats(http_text_times)
        latency_breakdown["model_text_only"] = stats(model_text_durations)

        # Text + Audio HTTP roundtrip
        http_both_times = []
        model_both_durations = []
        for _ in range(50):
            t0 = time.perf_counter()
            r = client.post(
                "/predict/multimodal",
                data={"text": sample_text},
                files={"file": ("dia0_utt0.wav", io.BytesIO(synth_wav), "audio/wav")},
            )
            http_both_times.append((time.perf_counter() - t0) * 1000)
            model_both_durations.append(r.json()["inference_duration_ms"])

        latency_breakdown["http_text_and_audio"] = stats(http_both_times)
        latency_breakdown["model_text_and_audio"] = stats(model_both_durations)

        print(f"  Text Only HTTP Turnaround: Median = {latency_breakdown['http_text_only']['median_ms']:.2f} ms | Model Execution: {latency_breakdown['model_text_only']['median_ms']:.2f} ms")
        print(f"  Text + Audio HTTP Turnaround: Median = {latency_breakdown['http_text_and_audio']['median_ms']:.2f} ms | Model Execution: {latency_breakdown['model_text_and_audio']['median_ms']:.2f} ms")

    report["latency_breakdown"] = latency_breakdown

    # Save validation results
    out_path = REPOSITORY_ROOT / "models" / "phase15_test_experiments"
    out_path.mkdir(parents=True, exist_ok=True)
    results_file = out_path / "backend_validation_results.json"
    results_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved backend validation results to {results_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()

