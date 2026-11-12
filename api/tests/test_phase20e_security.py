"""Phase 20E security verification: staging env validation, Host-header bypass hardening, and header checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import api.app as api_app_module
import api.model_registry as registry_module
from api.config import ApiSettings
from project_config import REPOSITORY_ROOT
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES


def _neutral_probs() -> dict[str, float]:
    probs = {label: 0.05 for label in MELD_CLASS_NAMES}
    probs["neutral"] = 0.70
    return probs


class _DummyPredictor:
    model_identifier = "dummy"
    architecture = "ResNet18"
    checkpoint_sha256 = "0" * 64
    active_role = "primary"
    default_pooling = "raw_all_detections_mean_prob"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def predict(self, _text: str) -> dict[str, Any]:
        return {"predicted_label": "neutral", "probabilities": _neutral_probs(), "confidence": 0.70}

    def predict_file(self, _path: Path) -> dict[str, Any]:
        return {"predicted_label": "neutral", "probabilities": _neutral_probs(), "confidence": 0.70}

    def predict_video(self, _path: Path, **_kwargs: Any) -> dict[str, Any]:
        return {"predicted_label": "neutral", "probabilities": _neutral_probs(), "confidence": 0.70}


def test_env_staging_example_passes_staging_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = REPOSITORY_ROOT / ".env.staging.example"
    assert env_file.is_file()

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        monkeypatch.setenv(key.strip(), val.strip())

    # Validate path construction without requiring Linux `/app/models` on Windows
    cfg = ApiSettings.from_env()
    assert cfg.env == "staging"
    assert cfg.enable_docs is False
    assert cfg.fail_on_missing_models is True
    assert cfg.allowed_origins == ("https://staging.example.invalid",)
    assert cfg.validate_staging_readiness() == []


def test_malformed_host_header_cannot_bypass_rate_limit_or_no_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_module, "DistilBertTextPredictor", _DummyPredictor)
    monkeypatch.setattr(registry_module, "Wav2Vec2Predictor", _DummyPredictor)
    monkeypatch.setattr(registry_module, "ResNetFacialEmotionPredictor", _DummyPredictor)

    app_instance = api_app_module.create_app(
        ApiSettings(env="staging", enable_docs=False, rate_limit_requests=1, rate_limit_window_seconds=60)
    )
    with TestClient(app_instance) as client:
        ok = client.post(
            "/predict/text",
            headers={"Host": "attacker.example:80@trusted.local"},
            json={"text": "hello world"},
        )
        assert ok.status_code == 200
        assert ok.headers.get("cache-control") == "no-store"
        assert ok.headers.get("x-content-type-options") == "nosniff"
        assert ok.headers.get("x-frame-options") == "DENY"

        limited = client.post(
            "/predict/text",
            headers={"Host": "attacker.example:80@trusted.local"},
            json={"text": "second call"},
        )
        assert limited.status_code == 429
        assert limited.headers.get("cache-control") == "no-store"

