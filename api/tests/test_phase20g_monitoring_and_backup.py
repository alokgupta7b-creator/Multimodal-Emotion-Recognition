"""Phase 20G tests for PII-free Prometheus /metrics telemetry and staging backup/restore tooling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import api.app as api_app_module
import api.model_registry as registry_module
import scripts.backup_staging_state as backup_mod
from api.config import ApiSettings
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES


def _valid_probs() -> dict[str, float]:
    probs = {label: 0.05 for label in MELD_CLASS_NAMES}
    probs["neutral"] = 0.70
    return probs


class _FakePredictor:
    model_identifier = "fake"
    architecture = "ResNet18"
    checkpoint_sha256 = "0" * 64
    active_role = "primary"
    default_pooling = "raw_all_detections_mean_prob"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def predict(self, _text: str) -> dict[str, Any]:
        return {"predicted_label": "neutral", "probabilities": _valid_probs(), "confidence": 0.70}

    def predict_file(self, _path: Path) -> dict[str, Any]:
        return {"predicted_label": "neutral", "probabilities": _valid_probs(), "confidence": 0.70}

    def predict_video(self, _path: Path, **_kwargs: Any) -> dict[str, Any]:
        return {"predicted_label": "neutral", "probabilities": _valid_probs(), "confidence": 0.70}


def test_prometheus_metrics_exports_telemetry_without_pii(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry_module, "DistilBertTextPredictor", _FakePredictor)
    monkeypatch.setattr(registry_module, "Wav2Vec2Predictor", _FakePredictor)
    monkeypatch.setattr(registry_module, "ResNetFacialEmotionPredictor", _FakePredictor)

    app_instance = api_app_module.create_app(
        ApiSettings(env="staging", enable_docs=False, rate_limit_requests=3, max_upload_bytes=1024)
    )
    with TestClient(app_instance) as client:
        assert client.post("/predict/text", json={"text": "SecretCustomerName_99182"}).status_code == 200
        assert (
            client.post(
                "/predict/audio",
                files={"file": ("private_user_voice_memo.wav", b"RIFF" + b"0" * 4096, "audio/wav")},
            ).status_code
            == 413
        )
        assert client.get("/some/sensitive/path?token=xyz123").status_code == 404

        metrics_res = client.get("/metrics")
        assert metrics_res.status_code == 200
        assert metrics_res.headers["content-type"].startswith("text/plain; version=0.0.4")
        body = metrics_res.text

        assert "emotion_api_models_ready 1" in body
        assert "emotion_api_model_load_duration_ms" in body
        assert "emotion_api_process_rss_bytes" in body
        assert 'emotion_api_requests_total{route="/predict/text",method="POST",status_class="2xx"} 1' in body
        assert 'emotion_api_requests_total{route="/predict/audio",method="POST",status_class="4xx"} 1' in body
        assert 'emotion_api_requests_total{route="other",method="GET",status_class="4xx"} 1' in body
        assert 'emotion_api_upload_rejections_total{reason="size_limit"} 1' in body
        assert "emotion_api_request_duration_seconds_bucket" in body

        for forbidden in ("SecretCustomerName_99182", "private_user_voice_memo", "sensitive", "xyz123"):
            assert forbidden not in body


def test_cleanup_failure_counter_increments_when_temp_file_persists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(registry_module, "DistilBertTextPredictor", _FakePredictor)
    monkeypatch.setattr(registry_module, "Wav2Vec2Predictor", _FakePredictor)
    monkeypatch.setattr(registry_module, "ResNetFacialEmotionPredictor", _FakePredictor)

    lingering = tmp_path / "persisted.wav"
    lingering.write_bytes(b"temp")
    monkeypatch.setattr(api_app_module, "cleanup_file", lambda _p: None)

    app_instance = api_app_module.create_app(ApiSettings())
    api_app_module._cleanup_tracked(lingering, app_instance)
    metrics_text = app_instance.state.telemetry.render_prometheus(models_ready=True, load_duration_ms=10.0)
    assert "emotion_api_cleanup_failures_total 1" in metrics_text


def test_backup_verify_and_restore_cycle_detects_corruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_repo = tmp_path / "repo"
    (fake_repo / "models").mkdir(parents=True)
    (fake_repo / "models" / "weights.bin").write_bytes(b"model-weights-v1")
    (fake_repo / ".env.staging.example").write_text("EMOTION_API_ENV=staging\n", encoding="utf-8")

    monkeypatch.setattr(backup_mod, "REPOSITORY_ROOT", fake_repo)
    monkeypatch.setattr(backup_mod, "CRITICAL_RELATIVE_PATHS", [".env.staging.example", "models/weights.bin"])

    backups_dir = tmp_path / "backups"
    snap = backup_mod.create_backup(backups_dir, label="unit")
    verify_ok = backup_mod.verify_backup(snap)
    assert verify_ok["ok"] is True
    assert verify_ok["verified_files"] == 2

    restore_target = tmp_path / "restored_target"
    restored = backup_mod.restore_backup(snap, restore_target)
    assert restored["ok"] is True
    assert restored["restored_files"] == 2
    assert (restore_target / "models" / "weights.bin").read_bytes() == b"model-weights-v1"

    # Simulate corruption in snapshot and ensure verify + restore refuse corrupted state
    (snap / "payload" / "models" / "weights.bin").write_bytes(b"corrupted")
    bad_verify = backup_mod.verify_backup(snap)
    assert bad_verify["ok"] is False
    assert bad_verify["mismatches"] == ["models/weights.bin"]
    with pytest.raises(RuntimeError, match="corrupted"):
        backup_mod.restore_backup(snap, tmp_path / "blocked_restore")

