"""FastAPI application for text, audio, and video emotion inference."""

from __future__ import annotations

from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import inspect
import logging
import threading
import time
from typing import Any

import cv2
from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from api.config import ApiSettings, settings
from api.fusion import Candidate4FusionEngine
from api.model_registry import ModelRegistry
from api.multimodal_extension import ExtendedMultimodalFusionEngine
from api.telemetry import RuntimeTelemetry
from api.schemas import (
    HealthResponse,
    ModelsStatusResponse,
    MultimodalPredictionResponse,
    PredictionResponse,
    TextRequest,
)
from api.validation import (
    StreamTooLargeError,
    cleanup_file,
    store_upload,
    validate_content_length,
)
from facial_emotion.detector import open_video_capture
from facial_emotion.mtcnn_detector import VideoDurationExceededError
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES

logger = logging.getLogger("emotion_api.app")


def _configure_logging(log_level: str) -> None:
    """Configure structured non-PII API logging."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    api_logger = logging.getLogger("emotion_api")
    api_logger.setLevel(numeric_level)
    if not api_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        api_logger.addHandler(handler)
    else:
        for handler in api_logger.handlers:
            handler.setLevel(numeric_level)


class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding-window rate limiter keyed by client IP."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> tuple[bool, int]:
        ts = time.monotonic() if now is None else now
        cutoff = ts - self.window_seconds
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (ts - bucket[0])) + 1)
                return False, retry_after
            bucket.append(ts)
            if len(self._hits) > 4096:
                stale = [k for k, v in self._hits.items() if not v]
                for k in stale:
                    del self._hits[k]
            return True, 0


def _active_settings(app_instance: FastAPI | Any) -> ApiSettings:
    """Resolve effective settings cleanly for both factory instances and monkeypatched module app."""
    custom = getattr(getattr(app_instance, "state", None), "custom_settings", None)
    return custom if custom is not None else settings


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    active_settings = _active_settings(app_instance)
    _configure_logging(active_settings.log_level)
    logger.info(
        "Starting Multimodal Emotion API env=%s device=%s model_root=%s docs=%s",
        active_settings.env,
        active_settings.device,
        active_settings.model_root,
        active_settings.enable_docs,
    )
    registry = ModelRegistry(active_settings)
    missing = registry.missing_model_paths()
    if missing:
        logger.warning("Missing model artifacts before startup load: %s", sorted(missing.keys()))
    registry.load()
    if active_settings.fail_on_missing_models and not registry.all_loaded():
        failed = {k: v["error"] for k, v in registry.status().items() if not v["loaded"]}
        raise RuntimeError(f"Required models failed to load in strict startup mode: {failed}")

    app_instance.state.registry = registry
    app_instance.state.fusion_engine = Candidate4FusionEngine()
    app_instance.state.extended_fusion_engine = ExtendedMultimodalFusionEngine(app_instance.state.fusion_engine)
    app_instance.state.rate_limiter = SlidingWindowRateLimiter(
        max_requests=active_settings.rate_limit_requests,
        window_seconds=active_settings.rate_limit_window_seconds,
    )
    if getattr(app_instance.state, "telemetry", None) is None:
        app_instance.state.telemetry = RuntimeTelemetry()
    try:
        yield
    finally:
        logger.info("Shutting down Multimodal Emotion API gracefully.")
        registry.unload()


router = APIRouter()


def _registry(request: Request) -> ModelRegistry:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry is not initialized.")
    return registry


def _response(modality: str, result: dict[str, Any], elapsed: float) -> PredictionResponse:
    probabilities = result.get("probabilities", {})
    if tuple(probabilities) != MELD_CLASS_NAMES:
        raise HTTPException(status_code=502, detail="Model returned an invalid class mapping.")
    total = sum(float(value) for value in probabilities.values())
    if not all(0 <= float(value) <= 1 for value in probabilities.values()) or abs(total - 1.0) > 1e-4:
        raise HTTPException(status_code=502, detail="Model returned invalid probabilities.")
    quality = {
        key: result[key]
        for key in (
            "num_faces",
            "num_sampled_frames",
            "face_detection_coverage",
            "prediction_entropy",
            "audio_duration_s",
            "processed_duration_s",
            "trimmed_silence_s",
            "audio_rms_db",
            "low_volume_warning",
        )
        if key in result and result[key] is not None
    }
    return PredictionResponse(
        predicted_emotion=str(result["predicted_label"]),
        probabilities={str(key): float(value) for key, value in probabilities.items()},
        confidence=float(result["confidence"]),
        inference_duration_ms=float((time.perf_counter() - elapsed) * 1000),
        modality=modality,
        quality=quality,
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        return HealthResponse(status="starting", ready=False)
    ready = all(item["loaded"] for item in registry.status().values())
    return HealthResponse(status="ok" if ready else "degraded", ready=ready)


@router.get("/ready")
def readiness(request: Request) -> JSONResponse:
    """Strict orchestrator readiness probe returning HTTP 503 unless all models are loaded."""
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        payload = HealthResponse(status="uninitialized", ready=False).model_dump()
        return JSONResponse(status_code=503, content=payload)
    ready = all(item["loaded"] for item in registry.status().values())
    payload = HealthResponse(status="ok" if ready else "degraded", ready=ready).model_dump()
    return JSONResponse(status_code=200 if ready else 503, content=payload)


@router.get("/metrics")
def prometheus_metrics(request: Request) -> PlainTextResponse:
    """Expose Prometheus v0.0.4 runtime metrics with zero PII or raw input labels."""
    telemetry = getattr(request.app.state, "telemetry", None)
    if telemetry is None:
        telemetry = RuntimeTelemetry()
        request.app.state.telemetry = telemetry
    registry = getattr(request.app.state, "registry", None)
    ready = bool(registry is not None and registry.all_loaded())
    load_duration_ms = getattr(registry, "load_duration_ms", None) if registry is not None else None
    return PlainTextResponse(
        content=telemetry.render_prometheus(models_ready=ready, load_duration_ms=load_duration_ms),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/models/status", response_model=ModelsStatusResponse)
def model_status(request: Request) -> ModelsStatusResponse:
    registry = _registry(request)
    statuses = registry.status()
    return ModelsStatusResponse(status="ready" if all(item["loaded"] for item in statuses.values()) else "degraded", models=statuses)


async def _run_inference_with_timeout(fn: Any, *args: Any, timeout_seconds: float = 0.0) -> Any:
    coro = run_in_threadpool(fn, *args)
    if timeout_seconds > 0:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    return await coro


@router.post("/predict/text", response_model=PredictionResponse)
async def predict_text(payload: TextRequest, request: Request) -> PredictionResponse:
    started = time.perf_counter()
    active_settings = _active_settings(request.app)
    try:
        result = await _run_inference_with_timeout(
            _registry(request).predict,
            "text",
            payload.text,
            timeout_seconds=active_settings.request_timeout_seconds,
        )
        return _response("text", result, started)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Text inference timed out.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Text model is unavailable.") from exc


def _video_predict_kwargs_support(registry: Any) -> tuple[bool, bool]:
    """Return whether ``registry.predict`` accepts ``max_video_seconds`` and ``use_sequential_grab``."""
    try:
        params = inspect.signature(registry.predict).parameters
    except (TypeError, ValueError):
        return False, False
    accepts_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    return (
        accepts_var_kw or ("max_video_seconds" in params),
        accepts_var_kw or ("use_sequential_grab" in params),
    )


def _validate_and_predict_file_sync(
    modality: str,
    registry: ModelRegistry,
    path: Path,
    started: float,
    active_settings: ApiSettings,
) -> tuple[dict[str, Any], PredictionResponse]:
    if modality == "video":
        supports_duration_arg, supports_seq_arg = _video_predict_kwargs_support(registry)
        if (
            active_settings.optimized_video_pipeline
            and active_settings.facial_backend != "legacy_cnn"
            and supports_duration_arg
            and supports_seq_arg
        ):
            try:
                result = registry.predict(
                    modality,
                    path,
                    max_video_seconds=active_settings.max_video_seconds,
                    use_sequential_grab=True,
                )
            except VideoDurationExceededError as exc:
                raise HTTPException(status_code=413, detail="Video exceeds the configured duration limit.") from exc
            return result, _response(modality, result, started)

        capture = open_video_capture(path)
        if not capture.isOpened():
            raise HTTPException(status_code=422, detail="Video could not be decoded.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.release()
        if fps > 0 and frames > 0 and frames / fps > active_settings.max_video_seconds:
            raise HTTPException(status_code=413, detail="Video exceeds the configured duration limit.")
        if supports_seq_arg and active_settings.facial_backend != "legacy_cnn":
            result = registry.predict(modality, path, use_sequential_grab=False)
        else:
            result = registry.predict(modality, path)
        return result, _response(modality, result, started)

    result = registry.predict(modality, path)
    return result, _response(modality, result, started)


def _cleanup_tracked(path: Path, app_instance: FastAPI | Any) -> None:
    cleanup_file(path)
    if path.exists():
        telemetry = getattr(getattr(app_instance, "state", None), "telemetry", None)
        if telemetry is not None:
            telemetry.record_cleanup_failure()


async def _predict_file(modality: str, request: Request, upload: UploadFile) -> PredictionResponse:
    active_settings = _active_settings(request.app)
    validate_content_length(request, active_settings)
    path = await store_upload(upload, modality, active_settings)
    started = time.perf_counter()

    def _run_file_inference() -> PredictionResponse:
        _, resp = _validate_and_predict_file_sync(modality, _registry(request), path, started, active_settings)
        return resp

    try:
        return await _run_inference_with_timeout(
            _run_file_inference,
            timeout_seconds=active_settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail=f"{modality.capitalize()} inference timed out.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"{modality.capitalize()} model is unavailable.") from exc
    finally:
        _cleanup_tracked(path, request.app)


@router.post("/predict/audio", response_model=PredictionResponse)
async def predict_audio(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    return await _predict_file("audio", request, file)


@router.post("/predict/video", response_model=PredictionResponse)
async def predict_video(request: Request, file: UploadFile = File(...)) -> PredictionResponse:
    return await _predict_file("video", request, file)


@router.post("/predict/multimodal", response_model=MultimodalPredictionResponse)
async def predict_multimodal(request: Request) -> MultimodalPredictionResponse:
    started_all = time.perf_counter()
    active_settings = _active_settings(request.app)
    validate_content_length(request, active_settings)
    content_type = request.headers.get("content-type", "").lower()
    text_content: str = ""
    audio_upload: UploadFile | None = None
    video_upload: UploadFile | None = None

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=422, detail="Invalid JSON body.") from exc
        if not isinstance(body, dict) or "text" not in body:
            raise HTTPException(status_code=422, detail="JSON payload must contain a 'text' field.")
        raw_text = body["text"]
        if not isinstance(raw_text, str) or isinstance(raw_text, bool):
            raise HTTPException(status_code=422, detail="The 'text' field must be a string.")
        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="Text input must be a non-empty string.")
        if len(raw_text) > active_settings.max_text_chars:
            raise HTTPException(status_code=413, detail="Text exceeds the maximum allowed length.")
        text_content = raw_text
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(status_code=422, detail="Invalid form data.") from exc

        candidate_audio = form.get("audio") or form.get("file")
        if (
            isinstance(candidate_audio, (UploadFile, StarletteUploadFile))
            and getattr(candidate_audio, "filename", None)
        ):
            audio_upload = candidate_audio

        candidate_video = form.get("video")
        if (
            isinstance(candidate_video, (UploadFile, StarletteUploadFile))
            and getattr(candidate_video, "filename", None)
        ):
            video_upload = candidate_video

        raw_text = form.get("text")
        if raw_text is not None:
            if not isinstance(raw_text, str):
                raise HTTPException(status_code=422, detail="The 'text' field must be a string.")
            if raw_text and not raw_text.strip():
                raise HTTPException(status_code=422, detail="Text input must be a non-empty string.")
            if len(raw_text) > active_settings.max_text_chars:
                raise HTTPException(status_code=413, detail="Text exceeds the maximum allowed length.")
            text_content = raw_text.strip()

        if not text_content and audio_upload is None and video_upload is None:
            raise HTTPException(
                status_code=422,
                detail="Request must provide non-empty 'text' or at least one audio/video upload.",
            )
    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported Media Type. Use application/json or multipart/form-data.",
        )

    registry = _registry(request)

    async def _store_optional_upload(
        modality: str,
        upload: UploadFile | None,
    ) -> tuple[Path | None, str | None]:
        if upload is None:
            return None, f"No {modality} file provided"
        try:
            return await store_upload(upload, modality, active_settings), None
        except HTTPException as exc:
            return None, f"Upload rejected: {exc.detail}"

    async def _run_stored_media(
        modality: str,
        media_path: Path | None,
        store_err: str | None,
    ) -> tuple[dict[str, Any] | None, PredictionResponse | None, str | None]:
        if media_path is None:
            return None, None, store_err
        started_mod = time.perf_counter()
        try:
            raw_res, mod_resp = await _run_inference_with_timeout(
                _validate_and_predict_file_sync,
                modality,
                registry,
                media_path,
                started_mod,
                active_settings,
                timeout_seconds=active_settings.request_timeout_seconds,
            )
            return raw_res, mod_resp, None
        except asyncio.TimeoutError:
            return None, None, f"{modality.capitalize()} inference timed out."
        except HTTPException as exc:
            return None, None, f"Upload rejected: {exc.detail}"
        except ValueError as exc:
            return None, None, str(exc)
        except RuntimeError:
            return None, None, f"{modality.capitalize()} model unavailable."
        finally:
            _cleanup_tracked(media_path, request.app)

    (audio_path, audio_store_err), (video_path, video_store_err) = await asyncio.gather(
        _store_optional_upload("audio", audio_upload),
        _store_optional_upload("video", video_upload),
    )

    text_probabilities: dict[str, float] | None = None
    text_response: PredictionResponse | None = None
    text_exclusion_reason: str | None = None
    audio_result: dict[str, Any] | None = None
    audio_response: PredictionResponse | None = None
    audio_exclusion_reason: str | None = audio_store_err
    video_result: dict[str, Any] | None = None
    video_response: PredictionResponse | None = None
    video_exclusion_reason: str | None = video_store_err

    try:
        if text_content:
            started_text = time.perf_counter()
            try:
                text_res = await _run_inference_with_timeout(
                    registry.predict,
                    "text",
                    text_content,
                    timeout_seconds=active_settings.request_timeout_seconds,
                )
                text_response = _response("text", text_res, started_text)
                text_probabilities = text_res["probabilities"]
            except asyncio.TimeoutError as exc:
                raise HTTPException(status_code=504, detail="Text inference timed out.") from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail="Text model is unavailable.") from exc
        else:
            text_exclusion_reason = "No text input provided"

        audio_result, audio_response, audio_exclusion_reason = await _run_stored_media(
            "audio", audio_path, audio_store_err
        )
        audio_path = None
        video_result, video_response, video_exclusion_reason = await _run_stored_media(
            "video", video_path, video_store_err
        )
        video_path = None
    finally:
        if audio_path is not None:
            _cleanup_tracked(audio_path, request.app)
        if video_path is not None:
            _cleanup_tracked(video_path, request.app)

    audio_probabilities = audio_result["probabilities"] if audio_result is not None else None
    video_probabilities = video_result["probabilities"] if video_result is not None else None

    if text_probabilities is None and audio_probabilities is None and video_probabilities is None:
        reasons = "; ".join(
            part for part in (text_exclusion_reason, audio_exclusion_reason, video_exclusion_reason) if part
        )
        raise HTTPException(status_code=422, detail=f"No usable modality available: {reasons}")

    extended_engine = getattr(request.app.state, "extended_fusion_engine", None)
    if extended_engine is None:
        base_engine = getattr(request.app.state, "fusion_engine", None) or Candidate4FusionEngine()
        extended_engine = ExtendedMultimodalFusionEngine(base_engine)
        request.app.state.extended_fusion_engine = extended_engine

    video_quality = video_response.quality if video_response is not None else None
    fusion_result = extended_engine.fuse(
        text_probabilities=text_probabilities,
        audio_probabilities=audio_probabilities,
        video_probabilities=video_probabilities,
        audio_exclusion_reason=audio_exclusion_reason,
        video_exclusion_reason=video_exclusion_reason,
        video_quality=video_quality,
        text_exclusion_reason=text_exclusion_reason,
    )

    quality_payload: dict[str, Any] = {"margin_audio": fusion_result.margin_audio}
    if audio_result is not None:
        quality_payload["audio_rms_db"] = audio_result.get("audio_rms_db")
        quality_payload["audio_duration_s"] = audio_result.get("audio_duration_s")
        quality_payload["audio_trimmed_silence_s"] = audio_result.get("trimmed_silence_s")
        quality_payload["audio_low_volume_warning"] = audio_result.get("low_volume_warning")
        quality_payload["audio_prediction_entropy"] = audio_result.get("prediction_entropy")
    if video_result is not None:
        quality_payload["video_num_faces"] = video_result.get("num_faces")
        quality_payload["video_num_sampled_frames"] = video_result.get("num_sampled_frames")
        quality_payload["video_face_detection_coverage"] = video_result.get("face_detection_coverage")
        quality_payload["video_prediction_entropy"] = video_result.get("prediction_entropy")

    total_latency_ms = float((time.perf_counter() - started_all) * 1000)

    return MultimodalPredictionResponse(
        predicted_emotion=fusion_result.predicted_emotion,
        probabilities=fusion_result.probabilities,
        confidence=fusion_result.confidence,
        fusion_method=fusion_result.fusion_method,
        modalities_available=fusion_result.modalities_available,
        exclusion_reasons=fusion_result.exclusion_reasons,
        modality_weights=fusion_result.modality_weights,
        individual_predictions={
            "text": text_response,
            "audio": audio_response,
            "video": video_response,
        },
        inference_duration_ms=total_latency_ms,
        rule_applied=fusion_result.rule_applied,
        quality=quality_payload,
    )


def create_app(custom_settings: ApiSettings | None = None) -> FastAPI:
    """Construct a FastAPI application with full routing, security middleware, CORS, and rate limiting."""
    init_settings = custom_settings if custom_settings is not None else settings
    docs_url = "/docs" if init_settings.enable_docs else None
    redoc_url = "/redoc" if init_settings.enable_docs else None
    openapi_url = "/openapi.json" if init_settings.enable_docs else None

    instance = FastAPI(
        title="Multimodal Emotion Recognition API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
    )
    instance.state.custom_settings = custom_settings
    instance.state.telemetry = RuntimeTelemetry()
    instance.include_router(router)

    async def limit_upload_size_middleware(request: Request, call_next):
        active_settings = _active_settings(request.app)
        extra_buffer = min(65_536, max(1024, active_settings.max_upload_bytes // 10))
        max_allowed = active_settings.max_upload_bytes + extra_buffer

        raw_length = request.headers.get("content-length")
        if raw_length is not None:
            try:
                length = int(raw_length)
                if length > max_allowed:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Request payload exceeds maximum allowed size ({active_settings.max_upload_bytes} bytes)."},
                    )
            except ValueError:
                pass

        total_received = 0
        stream_exceeded = False
        original_receive = request._receive

        async def bounded_receive():
            nonlocal total_received, stream_exceeded
            message = await original_receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                total_received += len(chunk)
                if total_received > max_allowed:
                    stream_exceeded = True
                    raise StreamTooLargeError(
                        f"Streamed request payload exceeds maximum allowed size ({active_settings.max_upload_bytes} bytes)."
                    )
            return message

        request._receive = bounded_receive
        try:
            response = await call_next(request)
            if stream_exceeded:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"Streamed request payload exceeds maximum allowed size ({active_settings.max_upload_bytes} bytes)."},
                )
            return response
        except StreamTooLargeError as exc:
            return JSONResponse(status_code=413, content={"detail": str(exc)})

    async def security_and_rate_limit_middleware(request: Request, call_next):
        started = time.perf_counter()
        scope_path = str(request.scope.get("path", request.url.path))
        telemetry = getattr(request.app.state, "telemetry", None)
        if scope_path.startswith("/predict/"):
            limiter = getattr(request.app.state, "rate_limiter", None)
            if limiter is not None:
                client_ip = request.client.host if request.client else "unknown"
                allowed, retry_after = limiter.allow(client_ip)
                if not allowed:
                    response = JSONResponse(
                        status_code=429,
                        headers={"Retry-After": str(retry_after), "Cache-Control": "no-store"},
                        content={"detail": "Rate limit exceeded. Wait briefly before retrying."},
                    )
                    if telemetry is not None:
                        telemetry.record_request(scope_path, request.method, 429, time.perf_counter() - started)
                    return response
        response = await call_next(request)
        if telemetry is not None:
            telemetry.record_request(scope_path, request.method, response.status_code, time.perf_counter() - started)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(self), geolocation=()")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-site")
        if scope_path.startswith("/predict/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    # Registration order matters in Starlette (last added runs outermost):
    # limit_upload_size (inner) -> security_and_rate_limit (middle) -> CORS (outer).
    instance.middleware("http")(limit_upload_size_middleware)
    instance.middleware("http")(security_and_rate_limit_middleware)
    instance.add_middleware(
        CORSMiddleware,
        allow_origins=list(init_settings.allowed_origins),
        allow_credentials=bool(init_settings.allowed_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Accept"],
    )
    return instance


_create_app = create_app
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        workers=1,
    )
