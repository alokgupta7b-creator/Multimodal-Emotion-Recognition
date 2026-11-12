"""Environment-driven backend configuration for local, testing, and staging runtimes."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from project_config import REPOSITORY_ROOT

PRIMARY_RESNET_SHA256 = "8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64"
PRIMARY_RESNET_V2_SHA256 = "1822ba9dc13a7406c13ad4bd48a7c8be7d309ead286cbe40352525bf85c1211a"
FALLBACK_RESNET_SHA256 = "4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0"
KNOWN_RESNET_SHA256 = {
    "resnet18_v2_best.pt": PRIMARY_RESNET_V2_SHA256,
    "exp5_ce_weighted_best.pt": PRIMARY_RESNET_SHA256,
    "exp5_focal_g3_wt_best.pt": FALLBACK_RESNET_SHA256,
}
VALID_ENVIRONMENTS = {"development", "testing", "staging", "production"}
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _str_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    return stripped or default


def _optional_str_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip().lower()
    return stripped or None


def _optional_path_env(name: str) -> Path | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return Path(stripped) if stripped else None


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _float_env(name: str, default: float, *, allow_zero: bool = False) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if allow_zero:
        if parsed < 0:
            raise ValueError(f"{name} must be non-negative")
    elif parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _path_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return Path(value.strip())


def _origins_env(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(origin.strip() for origin in raw.split(",") if origin.strip())


@dataclass(frozen=True)
class ApiSettings:
    env: str = field(default_factory=lambda: _str_env("EMOTION_API_ENV", "development").lower())
    host: str = field(default_factory=lambda: _str_env("EMOTION_API_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _int_env("EMOTION_API_PORT", 8000))
    log_level: str = field(default_factory=lambda: _str_env("EMOTION_API_LOG_LEVEL", "INFO").upper())
    repository_root: Path = REPOSITORY_ROOT
    model_root: Path = field(default_factory=lambda: _path_env("EMOTION_API_MODEL_ROOT", REPOSITORY_ROOT / "models"))
    text_model_dir: Path | None = None
    audio_model_dir: Path | None = None
    facial_backend: str = field(default_factory=lambda: _str_env("EMOTION_API_FACIAL_BACKEND", "resnet18").lower())
    facial_pooling: str = field(default_factory=lambda: _str_env("EMOTION_API_FACIAL_POOLING", "raw_all_detections_mean_prob"))
    facial_strict_fp32: bool = field(default_factory=lambda: _bool_env("EMOTION_API_FACIAL_STRICT_FP32", True))
    use_speech_calibration: bool = field(default_factory=lambda: _bool_env("EMOTION_API_USE_SPEECH_CALIBRATION", True))
    use_facial_calibration: bool = field(default_factory=lambda: _bool_env("EMOTION_API_USE_FACIAL_CALIBRATION", True))
    speech_calibration_path: Path | None = field(default_factory=lambda: _optional_path_env("EMOTION_API_SPEECH_CALIBRATION_PATH"))
    facial_calibration_path: Path | None = field(default_factory=lambda: _optional_path_env("EMOTION_API_FACIAL_CALIBRATION_PATH"))
    video_model_path: Path | None = None
    video_fallback_model_path: Path | None = None
    video_model_sha256: str | None = field(default_factory=lambda: _optional_str_env("EMOTION_API_VIDEO_SHA256"))
    video_fallback_sha256: str = field(default_factory=lambda: _str_env("EMOTION_API_VIDEO_FALLBACK_SHA256", FALLBACK_RESNET_SHA256).lower())
    legacy_video_model_path: Path | None = None
    device: str = field(default_factory=lambda: _str_env("EMOTION_API_DEVICE", "cpu").lower())
    max_upload_bytes: int = field(default_factory=lambda: _int_env("EMOTION_API_MAX_UPLOAD_BYTES", 50 * 1024 * 1024))
    max_video_seconds: float = field(default_factory=lambda: _float_env("EMOTION_API_MAX_VIDEO_SECONDS", 60.0))
    max_text_chars: int = field(default_factory=lambda: _int_env("EMOTION_API_MAX_TEXT_CHARS", 20_000))
    request_timeout_seconds: float = field(default_factory=lambda: _float_env("EMOTION_API_REQUEST_TIMEOUT_SECONDS", 60.0, allow_zero=True))
    enable_docs: bool = field(default_factory=lambda: _bool_env("EMOTION_API_ENABLE_DOCS", True))
    optimized_video_pipeline: bool = field(default_factory=lambda: _bool_env("EMOTION_API_OPTIMIZED_VIDEO_PIPELINE", True))
    fail_on_missing_models: bool = field(default_factory=lambda: _bool_env("EMOTION_API_FAIL_ON_MISSING_MODELS", False))
    rate_limit_requests: int = field(default_factory=lambda: _int_env("EMOTION_API_RATE_LIMIT_REQUESTS", 120))
    rate_limit_window_seconds: int = field(default_factory=lambda: _int_env("EMOTION_API_RATE_LIMIT_WINDOW_SECONDS", 60))
    allowed_origins: tuple[str, ...] = field(default_factory=lambda: _origins_env("EMOTION_API_ALLOWED_ORIGINS", "http://localhost:5173"))

    def __post_init__(self) -> None:
        normalized_env = self.env.strip().lower()
        object.__setattr__(self, "env", normalized_env)
        if normalized_env not in VALID_ENVIRONMENTS:
            raise ValueError(f"EMOTION_API_ENV must be one of {sorted(VALID_ENVIRONMENTS)}")

        normalized_level = self.log_level.strip().upper()
        object.__setattr__(self, "log_level", normalized_level)
        if normalized_level not in VALID_LOG_LEVELS:
            raise ValueError(f"EMOTION_API_LOG_LEVEL must be one of {sorted(VALID_LOG_LEVELS)}")

        if self.port > 65_535:
            raise ValueError("EMOTION_API_PORT must be <= 65535")

        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("EMOTION_API_DEVICE must be cpu, cuda, or auto")
        if self.facial_backend not in {"resnet18", "legacy_cnn"}:
            raise ValueError("EMOTION_API_FACIAL_BACKEND must be resnet18 or legacy_cnn")
        if self.max_video_seconds <= 0:
            raise ValueError("EMOTION_API_MAX_VIDEO_SECONDS must be positive")
        if "*" in self.allowed_origins:
            raise ValueError("Wildcard '*' in EMOTION_API_ALLOWED_ORIGINS is not permitted.")

        root = Path(self.model_root)
        object.__setattr__(self, "model_root", root)

        if self.text_model_dir is None:
            object.__setattr__(self, "text_model_dir", _path_env("EMOTION_API_TEXT_MODEL_DIR", root / "text_emotion" / "distilbert" / "best"))
        if self.audio_model_dir is None:
            object.__setattr__(self, "audio_model_dir", _path_env("EMOTION_API_AUDIO_MODEL_DIR", root / "speech_emotion" / "wav2vec2_v2" / "best"))
        if self.video_model_path is None:
            object.__setattr__(self, "video_model_path", _path_env("EMOTION_API_VIDEO_MODEL_PATH", root / "facial_emotion" / "resnet18_v2" / "resnet18_v2_best.pt"))
        if self.video_fallback_model_path is None:
            object.__setattr__(self, "video_fallback_model_path", _path_env("EMOTION_API_VIDEO_FALLBACK_MODEL_PATH", root / "facial_emotion" / "resnet18" / "exp5_focal_g3_wt_best.pt"))
        if self.legacy_video_model_path is None:
            object.__setattr__(self, "legacy_video_model_path", _path_env("EMOTION_API_LEGACY_VIDEO_MODEL_PATH", root / "facial_emotion" / "facial_cnn_best.pt"))
        if self.video_model_sha256 is None:
            resolved_sha = KNOWN_RESNET_SHA256.get(Path(self.video_model_path).name)
            object.__setattr__(self, "video_model_sha256", resolved_sha)
        else:
            object.__setattr__(self, "video_model_sha256", self.video_model_sha256.strip().lower())
        if self.speech_calibration_path is not None:
            object.__setattr__(self, "speech_calibration_path", Path(self.speech_calibration_path))
        if self.facial_calibration_path is not None:
            object.__setattr__(self, "facial_calibration_path", Path(self.facial_calibration_path))

    @classmethod
    def from_env(cls) -> ApiSettings:
        """Construct a fresh settings snapshot from current process environment variables."""
        return cls()

    def selected_device(self) -> str:
        if self.device == "auto":
            return "cuda" if _cuda_available() else "cpu"
        if self.device == "cuda" and not _cuda_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return self.device

    def validate_staging_readiness(self) -> list[str]:
        """Return advisory/error notices when inspecting configuration for staging exposure."""
        issues: list[str] = []
        if self.enable_docs:
            issues.append("EMOTION_API_ENABLE_DOCS should be disabled (0) in staging.")
        if not self.allowed_origins:
            issues.append("EMOTION_API_ALLOWED_ORIGINS is empty; browser requests from external staging origins will fail CORS.")
        for origin in self.allowed_origins:
            if "localhost" in origin or "127.0.0.1" in origin:
                if self.env == "staging":
                    issues.append(f"Staging CORS origin contains local address: {origin}")
        return issues


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except (ImportError, OSError):
        return False


settings = ApiSettings()
