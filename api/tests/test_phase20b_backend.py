"""Phase 20B tests for staging config, app factory parity, readiness probes, timeouts, and shutdown."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

import api.app as api_app_module
import api.model_registry as registry_module
from api.config import ApiSettings
from api.model_registry import ModelRegistry
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES


def _valid_probs() -> dict[str, float]:
    probs = {label: 0.05 for label in MELD_CLASS_NAMES}
    probs["neutral"] = 0.70
    return probs


class _FakeTextPredictor:
    model_identifier = "fake-distilbert"

    def __init__(self, model_dir: Path, device: str = "cpu", delay: float = 0.0) -> None:
        self.model_dir = model_dir
        self.device = device
        self.delay = delay

    def predict(self, text: str) -> dict[str, Any]:
        if self.delay:
            time.sleep(self.delay)
        return {
            "predicted_label": "neutral",
            "probabilities": _valid_probs(),
            "confidence": 0.70,
        }


class _FakeAudioPredictor:
    model_identifier = "fake-wav2vec2"

    def __init__(self, model_dir: Path, device: str = "cpu", delay: float = 0.0) -> None:
        self.model_dir = model_dir
        self.device = device
        self.delay = delay

    def predict_file(self, path: Path) -> dict[str, Any]:
        if self.delay:
            time.sleep(self.delay)
        return {
            "predicted_label": "neutral",
            "probabilities": _valid_probs(),
            "confidence": 0.70,
        }


class _FakeVideoPredictor:
    model_identifier = "fake-resnet18"
    architecture = "ResNet18"
    checkpoint_sha256 = "0" * 64
    active_role = "primary"
    default_pooling = "raw_all_detections_mean_prob"

    def __init__(self, *args: Any, delay: float = 0.0, **kwargs: Any) -> None:
        self.delay = delay

    def predict_video(self, path: Path, **kwargs: Any) -> dict[str, Any]:
        if self.delay:
            time.sleep(self.delay)
        return {
            "predicted_label": "neutral",
            "probabilities": _valid_probs(),
            "confidence": 0.70,
            "num_faces": 1,
            "num_sampled_frames": 2,
            "face_detection_coverage": 0.5,
            "prediction_entropy": 0.8,
        }


def _patch_fake_predictors(
    monkeypatch: pytest.MonkeyPatch,
    *,
    text_delay: float = 0.0,
    audio_delay: float = 0.0,
    fail_audio: bool = False,
) -> list[str]:
    events: list[str] = []

    def _make_text(model_dir: Path, device: str = "cpu") -> _FakeTextPredictor:
        events.append("load:text")
        return _FakeTextPredictor(model_dir, device=device, delay=text_delay)

    def _make_audio(model_dir: Path, device: str = "cpu") -> _FakeAudioPredictor:
        events.append("load:audio")
        if fail_audio:
            raise FileNotFoundError("Missing fake audio checkpoint")
        return _FakeAudioPredictor(model_dir, device=device, delay=audio_delay)

    def _make_video(*args: Any, **kwargs: Any) -> _FakeVideoPredictor:
        events.append("load:video")
        return _FakeVideoPredictor(*args, **kwargs)

    monkeypatch.setattr(registry_module, "DistilBertTextPredictor", _make_text)
    monkeypatch.setattr(registry_module, "Wav2Vec2Predictor", _make_audio)
    monkeypatch.setattr(registry_module, "ResNetFacialEmotionPredictor", _make_video)
    return events


def test_env_configuration_and_staging_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("EMOTION_API_ENV", "staging")
    monkeypatch.setenv("EMOTION_API_HOST", "0.0.0.0")
    monkeypatch.setenv("EMOTION_API_PORT", "8088")
    monkeypatch.setenv("EMOTION_API_LOG_LEVEL", "debug")
    monkeypatch.setenv("EMOTION_API_MODEL_ROOT", str(tmp_path / "models_bundle"))
    monkeypatch.setenv("EMOTION_API_REQUEST_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("EMOTION_API_FAIL_ON_MISSING_MODELS", "1")
    monkeypatch.setenv("EMOTION_API_ENABLE_DOCS", "0")
    monkeypatch.setenv("EMOTION_API_ALLOWED_ORIGINS", "https://staging.example.com,https://preview.example.com")

    cfg = ApiSettings.from_env()
    assert cfg.env == "staging"
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8088
    assert cfg.log_level == "DEBUG"
    assert cfg.request_timeout_seconds == 45.0
    assert cfg.fail_on_missing_models is True
    assert cfg.enable_docs is False
    assert cfg.allowed_origins == ("https://staging.example.com", "https://preview.example.com")
    assert cfg.text_model_dir == tmp_path / "models_bundle" / "text_emotion" / "distilbert" / "best"
    assert cfg.audio_model_dir == tmp_path / "models_bundle" / "speech_emotion" / "wav2vec2_v2" / "best"
    assert cfg.video_model_path == tmp_path / "models_bundle" / "facial_emotion" / "resnet18_v2" / "resnet18_v2_best.pt"
    assert cfg.use_speech_calibration is True
    assert cfg.use_facial_calibration is True
    assert cfg.validate_staging_readiness() == []

    monkeypatch.setenv("EMOTION_API_ALLOWED_ORIGINS", "*")
    with pytest.raises(ValueError, match="Wildcard"):
        ApiSettings.from_env()

    monkeypatch.setenv("EMOTION_API_ALLOWED_ORIGINS", "http://localhost:8080")
    monkeypatch.setenv("EMOTION_API_ENABLE_DOCS", "1")
    warnings = ApiSettings.from_env().validate_staging_readiness()
    assert len(warnings) == 2

    monkeypatch.setenv("EMOTION_API_ENABLE_DOCS", "0")
    monkeypatch.setenv("EMOTION_API_ALLOWED_ORIGINS", "https://staging.example.com")
    monkeypatch.setenv("EMOTION_API_PORT", "70000")
    with pytest.raises(ValueError, match="65535"):
        ApiSettings.from_env()


def test_create_app_middleware_parity_docs_cors_headers_and_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fake_predictors(monkeypatch)
    custom = ApiSettings(
        env="staging",
        enable_docs=False,
        allowed_origins=("https://staging.example.com",),
        rate_limit_requests=2,
        rate_limit_window_seconds=60,
        max_upload_bytes=1024,
    )
    staging_app = api_app_module.create_app(custom)

    assert len([m for m in api_app_module.app.user_middleware if m.cls is CORSMiddleware]) == 1
    assert len([m for m in staging_app.user_middleware if m.cls is CORSMiddleware]) == 1

    with TestClient(staging_app) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404

        ready_res = client.get("/ready", headers={"Origin": "https://staging.example.com"})
        assert ready_res.status_code == 200
        assert ready_res.json() == {"status": "ok", "ready": True, "classes": list(MELD_CLASS_NAMES)}
        assert ready_res.headers.get("access-control-allow-origin") == "https://staging.example.com"
        assert ready_res.headers.get("x-content-type-options") == "nosniff"
        assert ready_res.headers.get("x-frame-options") == "DENY"
        assert ready_res.headers.get("referrer-policy") == "no-referrer"

        denied_origin = client.get("/health", headers={"Origin": "https://untrusted.example.com"})
        assert denied_origin.headers.get("access-control-allow-origin") is None

        assert client.post("/predict/text", json={"text": "first"}).status_code == 200
        r2 = client.post("/predict/text", json={"text": "second"})
        assert r2.status_code == 200
        assert r2.headers.get("cache-control") == "no-store"

        r3 = client.post("/predict/text", json={"text": "third"})
        assert r3.status_code == 429
        assert "Retry-After" in r3.headers

    # A second factory instance must isolate rate-limiter buckets and still enforce max_upload_bytes.
    with TestClient(api_app_module.create_app(custom)) as isolated_client:
        oversized = isolated_client.post(
            "/predict/audio",
            files={"file": ("clip.wav", b"RIFF" + b"0" * 4096, "audio/wav")},
        )
        assert oversized.status_code == 413


def test_readiness_degraded_strict_startup_and_idempotent_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = _patch_fake_predictors(monkeypatch, fail_audio=True)

    degraded_settings = ApiSettings(model_root=tmp_path / "missing_models", fail_on_missing_models=False)
    reg = ModelRegistry(degraded_settings)
    missing = reg.missing_model_paths()
    assert set(missing.keys()) == {"text", "audio", "video"}

    reg.load()
    assert reg.load_duration_ms is not None and reg.load_duration_ms >= 0.0
    assert events == ["load:text", "load:audio", "load:video"]
    reg.load()
    assert events == ["load:text", "load:audio", "load:video"]
    assert reg.all_loaded() is False
    reg.unload()
    assert reg.models == {}

    degraded_app = api_app_module.create_app(degraded_settings)
    with TestClient(degraded_app) as client:
        health_res = client.get("/health")
        assert health_res.status_code == 200
        assert health_res.json() == {"status": "degraded", "ready": False, "classes": list(MELD_CLASS_NAMES)}

        ready_res = client.get("/ready")
        assert ready_res.status_code == 503
        assert ready_res.json() == {"status": "degraded", "ready": False, "classes": list(MELD_CLASS_NAMES)}

    strict_app = api_app_module.create_app(
        ApiSettings(model_root=tmp_path / "missing_models", fail_on_missing_models=True)
    )
    with pytest.raises(RuntimeError, match="strict startup mode"):
        with TestClient(strict_app):
            pass


def test_inference_timeout_returns_504_and_cleans_up_temp_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_fake_predictors(monkeypatch, text_delay=0.25, audio_delay=0.25)

    captured_paths: list[Path] = []
    original_store_upload = api_app_module.store_upload

    async def _tracking_store_upload(upload: Any, modality: str, active_settings: ApiSettings) -> Path:
        saved = await original_store_upload(upload, modality, active_settings)
        captured_paths.append(saved)
        return saved

    monkeypatch.setattr(api_app_module, "store_upload", _tracking_store_upload)

    timeout_app = api_app_module.create_app(
        ApiSettings(request_timeout_seconds=0.05, rate_limit_requests=20)
    )
    with TestClient(timeout_app) as client:
        text_res = client.post("/predict/text", json={"text": "please wait"})
        assert text_res.status_code == 504
        assert "timed out" in text_res.json()["detail"].lower()

        audio_res = client.post(
            "/predict/audio",
            files={"file": ("sample.wav", b"RIFF\x24\x00\x00\x00WAVEfmt ", "audio/wav")},
        )
        assert audio_res.status_code == 504
        assert "timed out" in audio_res.json()["detail"].lower()

    assert len(captured_paths) == 1
    assert not captured_paths[0].exists()
    assert timeout_app.state.registry.models == {}

