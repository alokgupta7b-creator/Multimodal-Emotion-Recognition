import asyncio
import csv
import io
import shutil
import subprocess
import wave
from pathlib import Path

import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

import api.app as app_module
from api.app import app
from api.config import settings
from api.validation import _signature, store_upload
from project_config import MELD_PREPARED_ROOT
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES


def _media_path(split: str, column: str) -> Path:
    manifest = MELD_PREPARED_ROOT / "reports" / split / f"{column}_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            status_key = "video_status" if column == "video" else "audio_status"
            if row[status_key] == "ok":
                key = f"resolved_{column}_path" if f"resolved_{column}_path" in row else "output_path"
                return Path(row[key])
    raise AssertionError(f"No valid {column} fixture found")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as value:
        yield value


def _assert_prediction(response):
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body["probabilities"]) == set(MELD_CLASS_NAMES)
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4
    assert 0 <= body["confidence"] <= 1
    assert body["inference_duration_ms"] >= 0


def test_health_and_model_status(client):
    health = client.get("/health")
    assert health.status_code == 200
    status = client.get("/models/status")
    assert status.status_code == 200
    models = status.json()["models"]
    assert set(models) == {"text", "audio", "video"}
    assert all(value["loaded"] for value in models.values()), status.text


def test_text_prediction_uses_real_distilbert(client):
    _assert_prediction(client.post("/predict/text", json={"text": "I am very happy today."}))


def test_audio_prediction_uses_real_wav2vec2(client):
    path = _media_path("test", "audio")
    with path.open("rb") as handle:
        _assert_prediction(client.post("/predict/audio", files={"file": (path.name, handle, "audio/wav")}))


def test_video_prediction_uses_real_phase8_facial_model(client):
    path = _media_path("test", "video")
    with path.open("rb") as handle:
        response = client.post("/predict/video", files={"file": (path.name, handle, "video/mp4")})
    _assert_prediction(response)
    assert "face_detection_coverage" in response.json()["quality"]


def test_invalid_upload_is_rejected_and_not_retained(client, tmp_path, monkeypatch):
    response = client.post("/predict/audio", files={"file": ("bad.wav", b"not wav", "audio/wav")})
    assert response.status_code == 415


def test_empty_text_is_rejected(client):
    response = client.post("/predict/text", json={"text": "   "})
    assert response.status_code == 422


def test_silent_audio_is_rejected(client):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 16_000)
    response = client.post("/predict/audio", files={"file": ("silent.wav", buffer.getvalue(), "audio/wav")})
    assert response.status_code == 422


def test_missing_face_is_rejected(client, tmp_path):
    video_path = tmp_path / "blank.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 96))
    for _ in range(3):
        writer.write(np.zeros((96, 96, 3), dtype=np.uint8))
    writer.release()
    response = client.post("/predict/video", files={"file": ("blank.mp4", video_path.read_bytes(), "video/mp4")})
    assert response.status_code == 422


def test_temporary_upload_is_cleaned(client, monkeypatch):
    removed = []
    original = app_module.cleanup_file

    def record_and_cleanup(path):
        removed.append(path)
        original(path)

    monkeypatch.setattr(app_module, "cleanup_file", record_and_cleanup)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 16_000)
    response = client.post("/predict/audio", files={"file": ("silent.wav", buffer.getvalue(), "audio/wav")})
    assert response.status_code == 422
    assert len(removed) == 1
    assert not removed[0].exists()


def test_video_signature_supports_mp4_and_webm_only():
    assert _signature(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00", "video") is True
    assert _signature(b"\x1aE\xdf\xa3\x9fB\x86\x81\x01B\x82\x84webm", "video") is True
    assert _signature(b"\x1aE\xdf\xa3\x00\x00\x00\x00invalid", "video") is False
    assert _signature(b"not a video payload", "video") is False


def test_invalid_video_upload_is_rejected(client):
    response = client.post("/predict/video", files={"file": ("bad.webm", b"not webm bytes", "video/webm")})
    assert response.status_code == 415


def test_chunked_small_header_upload_stores_with_webm_suffix():
    webm_bytes = b"\x1aE\xdf\xa3\x9fB\x86\x81\x01B\xf7\x81\x01B\x82\x84webm" + (b"\x00" * 128)

    class SmallChunkUpload(UploadFile):
        def __init__(self, payload: bytes) -> None:
            super().__init__(file=io.BytesIO(payload), filename="clip.webm")
            self._pos = 0
            self._payload = payload

        async def read(self, size: int = -1) -> bytes:
            step = 8 if size <= 0 else min(8, size)
            chunk = self._payload[self._pos : self._pos + step]
            self._pos += len(chunk)
            return chunk

    stored = asyncio.run(store_upload(SmallChunkUpload(webm_bytes), "video", settings))
    try:
        assert stored.suffix == ".webm"
        assert stored.read_bytes() == webm_bytes
    finally:
        stored.unlink(missing_ok=True)


def test_webm_video_prediction_if_ffmpeg_available(client, tmp_path):
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        pytest.skip("ffmpeg binary not found")
    mp4_path = _media_path("dev", "video")
    webm_path = tmp_path / "dev_sample.webm"
    subprocess.run(
        [ffmpeg_bin, "-y", "-i", str(mp4_path), "-t", "1.5", "-c:v", "libvpx", "-an", str(webm_path)],
        check=True,
        capture_output=True,
    )
    with webm_path.open("rb") as handle:
        response = client.post("/predict/video", files={"file": ("live.webm", handle, "video/webm")})
    _assert_prediction(response)
    assert "face_detection_coverage" in response.json()["quality"]


def test_audio_only_mp4_is_rejected_without_hanging(client, tmp_path):
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        pytest.skip("ffmpeg binary not found")
    audio_mp4 = tmp_path / "audio_only.mp4"
    subprocess.run(
        [ffmpeg_bin, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.6", "-c:a", "aac", str(audio_mp4)],
        check=True,
        capture_output=True,
    )
    with audio_mp4.open("rb") as handle:
        response = client.post("/predict/video", files={"file": ("audio_only.mp4", handle, "video/mp4")})
    assert response.status_code == 422
