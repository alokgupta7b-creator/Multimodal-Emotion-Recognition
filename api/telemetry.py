"""Thread-safe, PII-free runtime telemetry and Prometheus v0.0.4 text exporter."""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import os
import threading
from pathlib import Path

KNOWN_ROUTES = frozenset(
    {
        "/health",
        "/ready",
        "/metrics",
        "/models/status",
        "/predict/text",
        "/predict/audio",
        "/predict/video",
        "/predict/multimodal",
    }
)

LATENCY_BUCKETS_SECONDS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)


def normalize_route_label(path: str) -> str:
    """Restrict Prometheus route labels to known API paths to prevent cardinality explosion and path leakage."""
    return path if path in KNOWN_ROUTES else "other"


def get_process_rss_bytes() -> int:
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
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

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
    else:
        statm = Path("/proc/self/statm")
        if statm.is_file():
            try:
                pages_rss = int(statm.read_text(encoding="utf-8").split()[1])
                return pages_rss * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                pass
    return 0


def get_process_rss_mb() -> float | None:
    rss_bytes = get_process_rss_bytes()
    return round(rss_bytes / (1024 * 1024), 2) if rss_bytes > 0 else None


def _cuda_allocated_bytes() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated())
    except (ImportError, OSError, RuntimeError):
        pass
    return 0


class RuntimeTelemetry:
    """In-memory thread-safe counter and latency histogram store without PII."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str, str], int] = {}
        self._errors: dict[str, int] = {}
        self._upload_rejections: dict[str, int] = {}
        self._cleanup_failures = 0
        self._duration_buckets: dict[tuple[str, float | str], int] = {}
        self._duration_sum: dict[str, float] = {}
        self._duration_count: dict[str, int] = {}

    def record_request(self, path: str, method: str, status_code: int, duration_seconds: float) -> None:
        route = normalize_route_label(path)
        clean_method = method.upper() if method.upper() in {"GET", "POST", "OPTIONS"} else "OTHER"
        status_class = f"{status_code // 100}xx"

        with self._lock:
            req_key = (route, clean_method, status_class)
            self._requests[req_key] = self._requests.get(req_key, 0) + 1
            if status_code >= 400:
                self._errors[route] = self._errors.get(route, 0) + 1
            if route.startswith("/predict/"):
                if status_code == 413:
                    self._upload_rejections["size_limit"] = self._upload_rejections.get("size_limit", 0) + 1
                elif status_code in {415, 422} and route != "/predict/text":
                    self._upload_rejections["media_validation"] = self._upload_rejections.get("media_validation", 0) + 1

            self._duration_sum[route] = self._duration_sum.get(route, 0.0) + max(0.0, duration_seconds)
            self._duration_count[route] = self._duration_count.get(route, 0) + 1
            for bound in LATENCY_BUCKETS_SECONDS:
                if duration_seconds <= bound:
                    b_key = (route, bound)
                    self._duration_buckets[b_key] = self._duration_buckets.get(b_key, 0) + 1
            inf_key = (route, "+Inf")
            self._duration_buckets[inf_key] = self._duration_buckets.get(inf_key, 0) + 1

    def record_cleanup_failure(self) -> None:
        with self._lock:
            self._cleanup_failures += 1

    def render_prometheus(self, *, models_ready: bool, load_duration_ms: float | None) -> str:
        lines: list[str] = [
            "# HELP emotion_api_models_ready Whether all required modality models are loaded (1=ready, 0=not ready).",
            "# TYPE emotion_api_models_ready gauge",
            f"emotion_api_models_ready {1 if models_ready else 0}",
            "# HELP emotion_api_model_load_duration_ms Startup model initialization duration in milliseconds.",
            "# TYPE emotion_api_model_load_duration_ms gauge",
            f"emotion_api_model_load_duration_ms {load_duration_ms if load_duration_ms is not None else 0.0:.2f}",
            "# HELP emotion_api_process_rss_bytes Current API worker resident set size in bytes.",
            "# TYPE emotion_api_process_rss_bytes gauge",
            f"emotion_api_process_rss_bytes {get_process_rss_bytes()}",
            "# HELP emotion_api_cuda_allocated_bytes Current CUDA allocated memory in bytes (0 on CPU).",
            "# TYPE emotion_api_cuda_allocated_bytes gauge",
            f"emotion_api_cuda_allocated_bytes {_cuda_allocated_bytes()}",
        ]

        with self._lock:
            lines.extend(
                [
                    "# HELP emotion_api_cleanup_failures_total Count of temporary uploaded files still present after cleanup.",
                    "# TYPE emotion_api_cleanup_failures_total counter",
                    f"emotion_api_cleanup_failures_total {self._cleanup_failures}",
                ]
            )

            lines.append("# HELP emotion_api_requests_total Total HTTP requests handled by route, method, and status class.")
            lines.append("# TYPE emotion_api_requests_total counter")
            for (route, method, status_class), count in sorted(self._requests.items()):
                lines.append(
                    f'emotion_api_requests_total{{route="{route}",method="{method}",status_class="{status_class}"}} {count}'
                )

            lines.append("# HELP emotion_api_request_errors_total Total 4xx/5xx HTTP responses by route.")
            lines.append("# TYPE emotion_api_request_errors_total counter")
            for route, count in sorted(self._errors.items()):
                lines.append(f'emotion_api_request_errors_total{{route="{route}"}} {count}')

            lines.append("# HELP emotion_api_upload_rejections_total Count of rejected media uploads by reason.")
            lines.append("# TYPE emotion_api_upload_rejections_total counter")
            for reason, count in sorted(self._upload_rejections.items()):
                lines.append(f'emotion_api_upload_rejections_total{{reason="{reason}"}} {count}')

            lines.append("# HELP emotion_api_request_duration_seconds HTTP request latency histogram by route.")
            lines.append("# TYPE emotion_api_request_duration_seconds histogram")
            for route in sorted(self._duration_count.keys()):
                for bound in (*LATENCY_BUCKETS_SECONDS, "+Inf"):
                    count = self._duration_buckets.get((route, bound), 0)
                    lines.append(f'emotion_api_request_duration_seconds_bucket{{route="{route}",le="{bound}"}} {count}')
                lines.append(f'emotion_api_request_duration_seconds_sum{{route="{route}"}} {self._duration_sum[route]:.6f}')
                lines.append(f'emotion_api_request_duration_seconds_count{{route="{route}"}} {self._duration_count[route]}')

        return "\n".join(lines) + "\n"
