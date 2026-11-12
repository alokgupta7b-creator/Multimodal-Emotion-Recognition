"""Phase 19C — Security, Privacy & Hardening Verification Tests."""

from __future__ import annotations

from pathlib import Path
import tempfile

from fastapi.testclient import TestClient
import pytest

import api.app as app_module
from api.config import ApiSettings
from api.validation import sanitize_upload_label, store_upload


def test_sanitize_upload_label_strips_traversal_and_specials() -> None:
    assert sanitize_upload_label("../../etc/passwd") == "passwd"
    assert sanitize_upload_label(r"..\..\Windows\System32\evil<script>.wav") == "evil_script_.wav"
    assert sanitize_upload_label("") == "upload"
    assert sanitize_upload_label(None) == "upload"


def test_security_headers_and_no_store_on_predict() -> None:
    with TestClient(app_module.app) as client:
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.headers.get("x-content-type-options") == "nosniff"
        assert r_health.headers.get("x-frame-options") == "DENY"
        assert r_health.headers.get("referrer-policy") == "no-referrer"
        assert r_health.headers.get("cross-origin-resource-policy") == "same-site"
        assert "geolocation=()" in (r_health.headers.get("permissions-policy") or "")

        r_pred = client.post("/predict/text", json={"text": "I feel good today."})
        assert r_pred.status_code == 200
        assert r_pred.headers.get("cache-control") == "no-store"
        assert r_pred.headers.get("x-content-type-options") == "nosniff"


def test_sliding_window_rate_limiter_enforces_429_and_recovers() -> None:
    limiter = app_module.SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
    assert limiter.allow("127.0.0.1", now=100.0) == (True, 0)
    assert limiter.allow("127.0.0.1", now=101.0) == (True, 0)
    allowed, retry_after = limiter.allow("127.0.0.1", now=102.0)
    assert not allowed
    assert retry_after >= 1
    assert limiter.allow("127.0.0.1", now=161.0) == (True, 0)

    with TestClient(app_module.app) as client:
        original_limiter = client.app.state.rate_limiter
        client.app.state.rate_limiter = app_module.SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
        try:
            assert client.post("/predict/text", json={"text": "First"}).status_code == 200
            assert client.post("/predict/text", json={"text": "Second"}).status_code == 200
            r_limited = client.post("/predict/text", json={"text": "Third"})
            assert r_limited.status_code == 429
            assert int(r_limited.headers.get("retry-after", "0")) >= 1
        finally:
            client.app.state.rate_limiter = original_limiter


def test_docs_toggle_disables_openapi_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "settings", ApiSettings(enable_docs=False))
    hardened_app = app_module._create_app()
    assert hardened_app.docs_url is None
    assert hardened_app.redoc_url is None
    assert hardened_app.openapi_url is None


def test_cors_does_not_expose_wildcard_headers_with_credentials() -> None:
    cors = [m for m in app_module.app.user_middleware if "CORSMiddleware" in repr(m.cls)]
    assert len(cors) == 1
    assert "*" not in (cors[0].kwargs.get("allow_headers") or [])


def test_magic_byte_spoofing_and_path_traversal_filename_safe() -> None:
    with TestClient(app_module.app) as client:
        # PE/MZ executable spoofed as .wav with path traversal filename
        mz_payload = b"MZ\x90\x00" + b"\x00" * 128
        res = client.post(
            "/predict/audio",
            files={"file": ("../../evil_payload.wav", mz_payload, "audio/wav")},
        )
        assert res.status_code == 415
        assert not (Path("evil_payload.wav")).exists()


def test_runtime_error_details_are_sanitized_in_multimodal_exclusion() -> None:
    with TestClient(app_module.app) as client:
        registry = client.app.state.registry
        orig_predict = registry.predict

        def _failing_audio_predict(modality: str, payload: object):
            if modality == "audio":
                raise RuntimeError(r"C:\internal\secret\model\crash_trace.pt")
            return orig_predict(modality, payload)

        registry.predict = _failing_audio_predict  # type: ignore[assignment]
        try:
            # Tiny syntactically valid RIFF/WAVE header (non-zero PCM frames would fail model -> hits RuntimeError)
            import io, wave
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"\x10\x00" * 16000)
            res = client.post(
                "/predict/multimodal",
                data={"text": "Hello world"},
                files={"file": ("test.wav", buf.getvalue(), "audio/wav")},
            )
            assert res.status_code == 200
            reason = res.json()["exclusion_reasons"]["audio"]
            assert reason == "Audio model unavailable."
            assert "secret" not in reason
        finally:
            registry.predict = orig_predict  # type: ignore[assignment]


def test_temp_files_strictly_inside_tempdir_and_cleaned() -> None:
    import asyncio, io, wave
    from starlette.datastructures import UploadFile

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x01\x00" * 8000)
    buf.seek(0)
    upload = UploadFile(file=buf, filename=r"..\..\outside.wav")
    saved = asyncio.run(store_upload(upload, "audio", ApiSettings()))
    try:
        assert saved.resolve().parent == Path(tempfile.gettempdir()).resolve()
        assert saved.name.startswith("emotion_")
        assert saved.suffix == ".wav"
    finally:
        saved.unlink(missing_ok=True)

