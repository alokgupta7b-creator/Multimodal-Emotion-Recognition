"""Phase 19D — Performance & Resource Optimization Benchmark (MELD dev split only).

Measures before/after latencies (p50, p95, mean), video sub-stage breakdown (decode, MTCNN detect,
ResNet18 neural), 3-modality multimodal latency, concurrent request throughput, RAM/VRAM peaks,
frontend bundle byte sizes, and strict numerical parity across identical MELD dev inputs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("EMOTION_API_DEVICE", "auto")

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from concurrent.futures import ThreadPoolExecutor
import csv
import ctypes
import ctypes.wintypes
from dataclasses import replace
import json
import statistics
import time
from typing import Any

from fastapi.testclient import TestClient
import numpy as np
import torch

import api.app as api_app
from api.config import ApiSettings
from project_config import MELD_PREPARED_ROOT, REPOSITORY_ROOT

BASELINE_JSON = REPOSITORY_ROOT / "models" / "facial_emotion" / "resnet18" / "phase19d_before_baseline.json"
FINAL_JSON = REPOSITORY_ROOT / "models" / "facial_emotion" / "resnet18" / "phase19d_performance_validation.json"


def _process_rss_mb() -> float:
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


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return round(float(ordered[idx]), 2)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean_ms": round(float(statistics.mean(values)), 2),
        "p50_ms": _pct(values, 50),
        "p95_ms": _pct(values, 95),
        "min_ms": round(float(min(values)), 2),
        "max_ms": round(float(max(values)), 2),
    }


def _load_dev_fixtures(limit: int = 6) -> list[dict[str, Any]]:
    v_manifest = MELD_PREPARED_ROOT / "reports" / "dev" / "video_manifest.csv"
    a_manifest = MELD_PREPARED_ROOT / "reports" / "dev" / "audio_manifest.csv"
    t_manifest = REPOSITORY_ROOT / "datasets" / "MELD" / "manifests" / "dev_manifest.csv"

    text_map: dict[str, str] = {}
    if t_manifest.is_file():
        with t_manifest.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                sid = str(row.get("sample_id") or "").strip()
                txt = str(row.get("text") or "").strip()
                if sid and txt:
                    text_map[sid] = txt

    v_map: dict[str, Path] = {}
    with v_manifest.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("video_status") == "ok":
                p = Path(row.get("resolved_video_path") or row.get("output_path") or "")
                if p.is_file():
                    v_map[str(row["sample_id"])] = p

    a_map: dict[str, Path] = {}
    with a_manifest.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("audio_status") == "ok":
                p = Path(row.get("resolved_audio_path") or row.get("output_path") or "")
                if p.is_file():
                    a_map[str(row["sample_id"])] = p

    common = sorted(v_map.keys() & a_map.keys())
    selected: list[dict[str, Any]] = []
    # Prefer deterministic dev IDs first, then fill up to `limit`
    preferred = ["dev_dia0_utt0", "dev_dia10_utt3", "dev_dia74_utt19", "dev_dia99_utt3", "dev_dia0_utt1", "dev_dia1_utt0"]
    seen: set[str] = set()
    for sid in preferred + common:
        if sid in seen or sid not in v_map or sid not in a_map:
            continue
        seen.add(sid)
        selected.append(
            {
                "sample_id": sid,
                "text": text_map.get(sid, "Wait, did that really just happen right now?"),
                "wav_bytes": a_map[sid].read_bytes(),
                "mp4_bytes": v_map[sid].read_bytes(),
                "mp4_path": str(v_map[sid]),
            }
        )
        if len(selected) >= limit:
            break
    return selected


def _frontend_bundle_stats() -> dict[str, Any]:
    assets_dir = REPOSITORY_ROOT / "frontend" / "dist" / "assets"
    files = []
    total_bytes = 0
    if assets_dir.is_dir():
        for p in sorted(assets_dir.iterdir()):
            if p.is_file():
                sz = p.stat().st_size
                total_bytes += sz
                files.append({"name": p.name, "bytes": sz})
    return {"total_bytes": total_bytes, "files": files}


def run_benchmark(mode: str) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    fixtures = _load_dev_fixtures(limit=6)
    assert len(fixtures) >= 4, "Insufficient MELD dev fixtures"

    with TestClient(api_app.app) as client:
        # Warmup all 3 predictors + fusion
        client.post("/predict/text", json={"text": "Warmup utterance."})
        client.post("/predict/audio", files={"file": ("w.wav", fixtures[0]["wav_bytes"], "audio/wav")})
        client.post("/predict/video", files={"file": ("w.mp4", fixtures[0]["mp4_bytes"], "video/mp4")})
        client.post(
            "/predict/multimodal",
            data={"text": "Warmup multimodal"},
            files={
                "file": ("w.wav", fixtures[0]["wav_bytes"], "audio/wav"),
                "video": ("w.mp4", fixtures[0]["mp4_bytes"], "video/mp4"),
            },
        )

        # 1. Text latency
        text_latencies: list[float] = []
        text_refs: dict[str, Any] = {}
        for fx in fixtures:
            t0 = time.perf_counter()
            r = client.post("/predict/text", json={"text": fx["text"]})
            assert r.status_code == 200
            text_latencies.append((time.perf_counter() - t0) * 1000.0)
            body = r.json()
            text_refs[fx["sample_id"]] = {"predicted": body["predicted_emotion"], "probs": body["probabilities"]}

        # 2. Audio latency
        audio_latencies: list[float] = []
        audio_refs: dict[str, Any] = {}
        for fx in fixtures:
            t0 = time.perf_counter()
            r = client.post("/predict/audio", files={"file": ("s.wav", fx["wav_bytes"], "audio/wav")})
            assert r.status_code == 200
            audio_latencies.append((time.perf_counter() - t0) * 1000.0)
            body = r.json()
            audio_refs[fx["sample_id"]] = {"predicted": body["predicted_emotion"], "probs": body["probabilities"]}

        # 3. Video direct sub-stage & API latency
        video_pred = client.app.state.registry.models["video"].predictor
        assert video_pred is not None
        decode_ms_list: list[float] = []
        detect_ms_list: list[float] = []
        neural_ms_list: list[float] = []
        video_latencies: list[float] = []
        video_refs: dict[str, Any] = {}
        for fx in fixtures:
            direct = video_pred.predict_video(fx["mp4_path"])
            stages = direct["timing_profile"]
            decode_ms_list.append(stages["decoding_ms"])
            detect_ms_list.append(stages["detection_ms"])
            neural_ms_list.append(stages["neural_forward_ms"])

            t0 = time.perf_counter()
            r = client.post("/predict/video", files={"file": ("v.mp4", fx["mp4_bytes"], "video/mp4")})
            assert r.status_code == 200
            video_latencies.append((time.perf_counter() - t0) * 1000.0)
            body = r.json()
            video_refs[fx["sample_id"]] = {
                "predicted": body["predicted_emotion"],
                "probs": body["probabilities"],
                "num_faces": body["quality"].get("num_faces"),
            }

        # 4. 3-modality multimodal latency
        mm_latencies: list[float] = []
        mm_refs: dict[str, Any] = {}
        for fx in fixtures:
            t0 = time.perf_counter()
            r = client.post(
                "/predict/multimodal",
                data={"text": fx["text"]},
                files={
                    "file": ("a.wav", fx["wav_bytes"], "audio/wav"),
                    "video": ("v.mp4", fx["mp4_bytes"], "video/mp4"),
                },
            )
            assert r.status_code == 200
            mm_latencies.append((time.perf_counter() - t0) * 1000.0)
            body = r.json()
            mm_refs[fx["sample_id"]] = {
                "predicted": body["predicted_emotion"],
                "probs": body["probabilities"],
                "fusion_method": body["fusion_method"],
            }

        # 5. Concurrent batch (4 parallel multimodal requests)
        t_conc0 = time.perf_counter()

        def _worker(idx: int) -> int:
            fx = fixtures[idx % len(fixtures)]
            r = client.post(
                "/predict/multimodal",
                data={"text": fx["text"]},
                files={
                    "file": ("c.wav", fx["wav_bytes"], "audio/wav"),
                    "video": ("c.mp4", fx["mp4_bytes"], "video/mp4"),
                },
            )
            return r.status_code

        with ThreadPoolExecutor(max_workers=4) as pool:
            statuses = list(pool.map(_worker, range(4)))
        conc_wall_ms = round((time.perf_counter() - t_conc0) * 1000.0, 2)

    resources = {
        "process_rss_mb": _process_rss_mb(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        if torch.cuda.is_available()
        else 0.0,
        "cuda_reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
        if torch.cuda.is_available()
        else 0.0,
    }

    return {
        "mode": mode,
        "optimized_video_pipeline_env": os.getenv("EMOTION_API_OPTIMIZED_VIDEO_PIPELINE", "1"),
        "sample_ids": [f["sample_id"] for f in fixtures],
        "latencies_ms": {
            "text_api": _stats(text_latencies),
            "audio_api": _stats(audio_latencies),
            "video_api": _stats(video_latencies),
            "video_stages": {
                "decode": _stats(decode_ms_list),
                "mtcnn_detect": _stats(detect_ms_list),
                "resnet_neural": _stats(neural_ms_list),
            },
            "multimodal_3mod_api": _stats(mm_latencies),
            "concurrent_4_multimodal_wall_ms": conc_wall_ms,
            "concurrent_statuses": statuses,
        },
        "resources": resources,
        "frontend_bundle": _frontend_bundle_stats(),
        "reference_outputs": {
            "text": text_refs,
            "audio": audio_refs,
            "video": video_refs,
            "multimodal": mm_refs,
        },
    }


def _compare_parity(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    parity: dict[str, Any] = {}
    for mod in ("text", "audio", "video", "multimodal"):
        b_map = before["reference_outputs"][mod]
        a_map = after["reference_outputs"][mod]
        argmax_matches = 0
        max_abs_diff = 0.0
        for sid, b_entry in b_map.items():
            a_entry = a_map[sid]
            if b_entry["predicted"] == a_entry["predicted"]:
                argmax_matches += 1
            diffs = [abs(float(b_entry["probs"][k]) - float(a_entry["probs"][k])) for k in b_entry["probs"]]
            max_abs_diff = max(max_abs_diff, max(diffs))
        parity[mod] = {
            "samples": len(b_map),
            "argmax_matches": argmax_matches,
            "max_abs_prob_diff": round(max_abs_diff, 8),
        }
    return parity


def run_ab_paired() -> dict[str, Any]:
    """Run thermally paired baseline vs optimized pipeline inside one warmed session."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    fixtures = _load_dev_fixtures(limit=6)
    base_settings: ApiSettings = api_app.settings

    def _measure_video_and_mm(opt: bool, warmup: bool = False) -> dict[str, Any]:
        api_app.settings = replace(base_settings, optimized_video_pipeline=opt)
        with TestClient(api_app.app) as client:
            if warmup:
                client.post("/predict/video", files={"file": ("w.mp4", fixtures[0]["mp4_bytes"], "video/mp4")})
                client.post(
                    "/predict/multimodal",
                    data={"text": "Warmup"},
                    files={"file": ("w.wav", fixtures[0]["wav_bytes"], "audio/wav"), "video": ("w.mp4", fixtures[0]["mp4_bytes"], "video/mp4")},
                )
            video_pred = client.app.state.registry.models["video"].predictor
            assert video_pred is not None
            dec_ms: list[float] = []
            det_ms: list[float] = []
            neu_ms: list[float] = []
            v_api_ms: list[float] = []
            mm_api_ms: list[float] = []
            v_refs: dict[str, Any] = {}
            mm_refs: dict[str, Any] = {}
            for fx in fixtures:
                direct = video_pred.predict_video(fx["mp4_path"], use_sequential_grab=opt)
                st = direct["timing_profile"]
                dec_ms.append(st["decoding_ms"])
                det_ms.append(st["detection_ms"])
                neu_ms.append(st["neural_forward_ms"])

                t0 = time.perf_counter()
                rv = client.post("/predict/video", files={"file": ("v.mp4", fx["mp4_bytes"], "video/mp4")})
                assert rv.status_code == 200
                v_api_ms.append((time.perf_counter() - t0) * 1000.0)
                v_refs[fx["sample_id"]] = {"predicted": rv.json()["predicted_emotion"], "probs": rv.json()["probabilities"]}

                t1 = time.perf_counter()
                rm = client.post(
                    "/predict/multimodal",
                    data={"text": fx["text"]},
                    files={"file": ("a.wav", fx["wav_bytes"], "audio/wav"), "video": ("v.mp4", fx["mp4_bytes"], "video/mp4")},
                )
                assert rm.status_code == 200
                mm_api_ms.append((time.perf_counter() - t1) * 1000.0)
                mm_refs[fx["sample_id"]] = {"predicted": rm.json()["predicted_emotion"], "probs": rm.json()["probabilities"]}

            return {
                "optimized_video_pipeline": opt,
                "decode": _stats(dec_ms),
                "mtcnn_detect": _stats(det_ms),
                "resnet_neural": _stats(neu_ms),
                "video_api": _stats(v_api_ms),
                "multimodal_3mod_api": _stats(mm_api_ms),
                "refs": {"video": v_refs, "multimodal": mm_refs},
            }

    try:
        off_res = _measure_video_and_mm(False, warmup=True)
        on_res = _measure_video_and_mm(True, warmup=True)
    finally:
        api_app.settings = base_settings

    paired_parity: dict[str, Any] = {}
    for mod in ("video", "multimodal"):
        b_map = off_res["refs"][mod]
        a_map = on_res["refs"][mod]
        matches = sum(1 for sid in b_map if b_map[sid]["predicted"] == a_map[sid]["predicted"])
        max_diff = max(
            abs(float(b_map[sid]["probs"][k]) - float(a_map[sid]["probs"][k]))
            for sid in b_map
            for k in b_map[sid]["probs"]
        )
        paired_parity[mod] = {"samples": len(b_map), "argmax_matches": matches, "max_abs_prob_diff": round(max_diff, 8)}

    return {
        "toggle_off_baseline": {k: v for k, v in off_res.items() if k != "refs"},
        "toggle_on_optimized": {k: v for k, v in on_res.items() if k != "refs"},
        "parity": paired_parity,
        "resources": {
            "process_rss_mb": _process_rss_mb(),
            "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["before", "after", "ab"], required=True)
    args = parser.parse_args()

    if args.mode == "ab":
        ab = run_ab_paired()
        existing = json.loads(FINAL_JSON.read_text(encoding="utf-8")) if FINAL_JSON.is_file() else {}
        existing["paired_ab"] = ab
        FINAL_JSON.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        print(json.dumps(ab, indent=2))
        return

    result = run_benchmark(args.mode)
    if args.mode == "before":
        BASELINE_JSON.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "reference_outputs"}, indent=2))
        return

    before = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    parity = _compare_parity(before, result)
    report = {
        "before": {k: v for k, v in before.items() if k != "reference_outputs"},
        "after": {k: v for k, v in result.items() if k != "reference_outputs"},
        "parity": parity,
    }
    FINAL_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
