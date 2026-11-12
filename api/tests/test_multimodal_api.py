"""Integration and regression tests for the FastAPI /predict/multimodal endpoint."""

from __future__ import annotations

import csv
import io
import math
import struct
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.app import app
from project_config import MELD_PREPARED_ROOT
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES


def _dev_audio_fixture() -> Path:
    """Find a verified development audio fixture from the prepared manifest."""
    manifest = MELD_PREPARED_ROOT / "reports" / "dev" / "audio_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("audio_status") == "ok":
                path = Path(row.get("output_path", ""))
                if path.is_file():
                    return path
    raise AssertionError("No valid development audio fixture found.")


def _generate_synthetic_wav(silent: bool = False, duration_sec: float = 1.0) -> bytes:
    """Generate a 16 kHz mono 16-bit PCM WAV in memory."""
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
            # 440 Hz tone with audible amplitude (~ -10 dB RMS)
            data = bytearray()
            for i in range(n_samples):
                val = int(12000 * math.sin(2 * math.pi * 440 * (i / sr)))
                data.extend(struct.pack("<h", val))
            handle.writeframes(bytes(data))
    return buffer.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as value:
        yield value


class TestMultimodalApiIntegration:
    def test_predict_multimodal_text_only_json(self, client):
        """JSON payload with text only: should return 200 and text-authoritative prediction."""
        response = client.post("/predict/multimodal", json={"text": "I am absolutely thrilled about this news!"})
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["fusion_method"] == "text_anchored_class_conditional_bounded_fusion_v1"
        assert body["predicted_emotion"] in MELD_CLASS_NAMES
        assert set(body["probabilities"].keys()) == set(MELD_CLASS_NAMES)
        assert math.isclose(sum(body["probabilities"].values()), 1.0, abs_tol=1e-3)
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["inference_duration_ms"] > 0

        # Modality availability
        assert body["modalities_available"]["text"] is True
        assert body["modalities_available"]["audio"] is False
        assert body["modalities_available"]["video"] is False
        assert body["exclusion_reasons"]["audio"] is not None

        # Individual predictions
        assert body["individual_predictions"]["text"] is not None
        assert body["individual_predictions"]["audio"] is None
        assert body["individual_predictions"]["video"] is None

        # Weights
        assert body["modality_weights"]["text"] == 1.0
        assert body["modality_weights"]["audio"] == 0.0

    def test_predict_multimodal_text_only_form(self, client):
        """Form payload with text only: should return 200."""
        response = client.post("/predict/multimodal", data={"text": "I am so sad and heartbroken."})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["modalities_available"]["text"] is True
        assert body["modalities_available"]["audio"] is False
        assert body["predicted_emotion"] in MELD_CLASS_NAMES

    def test_predict_multimodal_text_and_real_dev_audio(self, client):
        """Multimodal payload with text and a valid development WAV audio file."""
        audio_path = _dev_audio_fixture()
        with audio_path.open("rb") as handle:
            response = client.post(
                "/predict/multimodal",
                data={"text": "Get out of here right now!"},
                files={"file": (audio_path.name, handle, "audio/wav")},
            )
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["modalities_available"]["text"] is True
        assert body["modalities_available"]["audio"] is True
        assert body["exclusion_reasons"]["audio"] is None
        assert body["individual_predictions"]["text"] is not None
        assert body["individual_predictions"]["audio"] is not None

        assert set(body["probabilities"].keys()) == set(MELD_CLASS_NAMES)
        assert math.isclose(sum(body["probabilities"].values()), 1.0, abs_tol=1e-3)
        assert body["inference_duration_ms"] > 0

    def test_predict_multimodal_with_silent_audio_graceful_fallback(self, client):
        """When audio is silent (-inf dB), audio is excluded explicitly without crashing the request."""
        silent_wav = _generate_synthetic_wav(silent=True)
        response = client.post(
            "/predict/multimodal",
            data={"text": "This is a normal sentence."},
            files={"file": ("silent.wav", silent_wav, "audio/wav")},
        )
        assert response.status_code == 200, response.text
        body = response.json()

        # Audio should be excluded explicitly
        assert body["modalities_available"]["audio"] is False
        assert "silent_audio" in body["exclusion_reasons"]["audio"]
        assert body["individual_predictions"]["audio"] is None

        # Text prediction should be authoritative
        assert body["modalities_available"]["text"] is True
        assert body["modality_weights"]["text"] == 1.0
        assert body["predicted_emotion"] in MELD_CLASS_NAMES

    def test_predict_multimodal_with_invalid_upload_signature_fallback(self, client):
        """When uploaded file is corrupt or not a WAV, audio is excluded without crashing the request."""
        response = client.post(
            "/predict/multimodal",
            data={"text": "I feel great."},
            files={"file": ("corrupt.wav", b"not a wave header", "audio/wav")},
        )
        assert response.status_code == 200, response.text
        body = response.json()

        assert body["modalities_available"]["audio"] is False
        assert body["exclusion_reasons"]["audio"] is not None
        assert "Upload rejected" in body["exclusion_reasons"]["audio"]
        assert body["individual_predictions"]["audio"] is None
        assert body["modalities_available"]["text"] is True

    def test_predict_multimodal_empty_text_rejected(self, client):
        """Empty or whitespace-only text must be rejected with HTTP 422."""
        response = client.post("/predict/multimodal", json={"text": "   "})
        assert response.status_code == 422

    def test_predict_multimodal_missing_text_rejected(self, client):
        """Request missing text must be rejected with HTTP 422."""
        response = client.post("/predict/multimodal", json={})
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "invalid_text",
        [
            None,
            12345,
            True,
            False,
            3.14159,
            ["hello", "world"],
            {"nested": "text"},
        ],
    )
    def test_predict_multimodal_non_string_text_rejected(self, client, invalid_text):
        """JSON payload with non-string text values must be rejected with HTTP 422 without coercion."""
        response = client.post("/predict/multimodal", json={"text": invalid_text})
        assert response.status_code == 422, f"Failed for {invalid_text} (status {response.status_code})"

    def test_predict_multimodal_oversized_content_length_rejected(self, client):
        """Requests with Content-Length exceeding configured limit are rejected upfront with HTTP 413."""
        response = client.post(
            "/predict/multimodal",
            headers={"Content-Length": "60000000", "Content-Type": "application/json"},
            content=b'{"text": "hello"}',
        )
        assert response.status_code == 413

    def test_predict_audio_oversized_content_length_rejected(self, client):
        """Legacy file endpoints also reject Content-Length exceeding limit with HTTP 413."""
        response = client.post(
            "/predict/audio",
            headers={"Content-Length": "60000000", "Content-Type": "multipart/form-data; boundary=xyz"},
            content=b"test",
        )
        assert response.status_code == 413

    def test_oversized_audio_upload_cleanup(self, client, monkeypatch):
        """Oversized audio stream triggers 413 and unlinks temporary file without disk leak."""
        import tempfile
        from dataclasses import replace
        import api.config
        import api.app

        # Temporarily restrict max upload bytes to 512 bytes for test
        orig_settings = api.config.settings
        restricted = replace(orig_settings, max_upload_bytes=512)
        monkeypatch.setattr(api.config, "settings", restricted)
        monkeypatch.setattr(api.app, "settings", restricted)

        # Generate a 1-second 16kHz audio (~32,000 bytes, well over 512 bytes)
        wav_data = _generate_synthetic_wav(silent=False, duration_sec=1.0)
        assert len(wav_data) > 512

        # 1. Direct audio endpoint
        before_temp = list(Path(tempfile.gettempdir()).glob("emotion_*.wav"))
        resp_audio = client.post(
            "/predict/audio",
            files={"file": ("test.wav", wav_data, "audio/wav")},
        )
        assert resp_audio.status_code == 413
        assert ("exceeds the configured size limit" in resp_audio.text) or ("exceeds maximum allowed size" in resp_audio.text)
        after_temp = list(Path(tempfile.gettempdir()).glob("emotion_*.wav"))
        assert len(after_temp) == len(before_temp), "Temporary file leaked on oversized audio upload!"

        # 2. Multimodal endpoint oversized handling
        resp_mm = client.post(
            "/predict/multimodal",
            data={"text": "I am so happy today!"},
            files={"file": ("test.wav", wav_data, "audio/wav")},
        )
        assert resp_mm.status_code in (200, 413)
        if resp_mm.status_code == 200:
            body = resp_mm.json()
            assert body["modalities_available"]["audio"] is False
            assert "Upload rejected" in body["exclusion_reasons"]["audio"]
        after_temp_mm = list(Path(tempfile.gettempdir()).glob("emotion_*.wav"))
        assert len(after_temp_mm) == len(before_temp), "Temporary file leaked on multimodal oversized audio upload!"

    def test_chunked_stream_without_content_length_rejected(self, client, monkeypatch):
        """Chunked / streaming body without Content-Length exceeding limit is halted and rejected with 413."""
        from dataclasses import replace
        import api.config
        import api.app

        orig_settings = api.config.settings
        restricted = replace(orig_settings, max_upload_bytes=512)
        monkeypatch.setattr(api.config, "settings", restricted)
        monkeypatch.setattr(api.app, "settings", restricted)

        def chunk_generator():
            # 5 chunks of 500 bytes = 2500 bytes (> 1536 max_allowed)
            for _ in range(5):
                yield b"X" * 500

        resp = client.post(
            "/predict/text",
            content=chunk_generator(),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 413
        assert "exceeds maximum allowed size" in resp.json()["detail"]

    def test_chunked_multipart_stream_without_content_length_rejected(self, client, monkeypatch):
        """Chunked multipart stream without Content-Length cannot spool unbounded data or leak files."""
        import tempfile
        from dataclasses import replace
        import api.config
        import api.app

        orig_settings = api.config.settings
        restricted = replace(orig_settings, max_upload_bytes=512)
        monkeypatch.setattr(api.config, "settings", restricted)
        monkeypatch.setattr(api.app, "settings", restricted)

        boundary = "----TestBoundaryChunked123"

        def chunked_multipart_gen():
            yield f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
            for _ in range(5):
                yield b"W" * 500  # 2500 bytes > 1536 limit
            yield f"\r\n--{boundary}--\r\n".encode()

        before_temp = list(Path(tempfile.gettempdir()).glob("emotion_*.wav"))
        resp = client.post(
            "/predict/audio",
            content=chunked_multipart_gen(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        assert resp.status_code == 413
        assert "exceeds maximum allowed size" in resp.json()["detail"]

        after_temp = list(Path(tempfile.gettempdir()).glob("emotion_*.wav"))
        assert len(after_temp) == len(before_temp), "Temporary file leaked during chunked multipart stream abort!"



class TestExistingEndpointsRegression:
    """Ensure existing endpoints remain strictly backward-compatible."""

    def test_predict_text_endpoint_backward_compatibility(self, client):
        response = client.post("/predict/text", json={"text": "Congratulations on your success!"})
        assert response.status_code == 200
        body = response.json()
        assert body["modality"] == "text"
        assert body["predicted_emotion"] in MELD_CLASS_NAMES
        assert set(body["probabilities"].keys()) == set(MELD_CLASS_NAMES)

    def test_predict_audio_endpoint_backward_compatibility(self, client):
        audio_path = _dev_audio_fixture()
        with audio_path.open("rb") as handle:
            response = client.post("/predict/audio", files={"file": (audio_path.name, handle, "audio/wav")})
        assert response.status_code == 200
        body = response.json()
        assert body["modality"] == "audio"
        assert body["predicted_emotion"] in MELD_CLASS_NAMES

    def test_predict_audio_silent_rejection(self, client):
        silent_wav = _generate_synthetic_wav(silent=True)
        response = client.post("/predict/audio", files={"file": ("silent.wav", silent_wav, "audio/wav")})
        assert response.status_code == 422

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["ready"] is True

    def test_models_status_endpoint(self, client):
        response = client.get("/models/status")
        assert response.status_code == 200
        models = response.json()["models"]
        assert set(models.keys()) == {"text", "audio", "video"}
        assert models["text"]["loaded"] is True
        assert models["audio"]["loaded"] is True
        assert models["video"]["loaded"] is True

