"""Startup-loaded predictor registry with idempotent loading and clean shutdown."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from api.config import ApiSettings
from facial_emotion.model import FacialEmotionPredictor
from facial_emotion.resnet_predictor import ResNetFacialEmotionPredictor
from speech_emotion.wav2vec2_model import Wav2Vec2Predictor
from text_emotion.distilbert_model import DistilBertTextPredictor

logger = logging.getLogger("emotion_api.registry")


@dataclass
class LoadedModel:
    predictor: Any | None
    identifier: str | None
    device: str | None
    error: str | None = None
    architecture: str | None = None
    checkpoint_sha256: str | None = None
    active_role: str | None = None
    preprocessing: str | None = None
    pooling_method: str | None = None
    load_ms: float | None = None
    calibrated: bool | None = None


class ModelRegistry:
    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings
        self.models: dict[str, LoadedModel] = {}
        self._lock = threading.RLock()
        self._locks: dict[str, threading.RLock] = {
            "text": threading.RLock(),
            "audio": threading.RLock(),
            "video": threading.RLock(),
        }
        self._loaded = False
        self.load_duration_ms: float | None = None

    def missing_model_paths(self) -> dict[str, str]:
        """Return modality keys whose expected checkpoint files or directories do not exist."""
        missing: dict[str, str] = {}
        if self.settings.text_model_dir is None or not self.settings.text_model_dir.is_dir():
            missing["text"] = str(self.settings.text_model_dir)
        if self.settings.audio_model_dir is None or not self.settings.audio_model_dir.is_dir():
            missing["audio"] = str(self.settings.audio_model_dir)
        if self.settings.facial_backend == "legacy_cnn":
            if self.settings.legacy_video_model_path is None or not self.settings.legacy_video_model_path.is_file():
                missing["video"] = str(self.settings.legacy_video_model_path)
        else:
            primary_ok = self.settings.video_model_path is not None and self.settings.video_model_path.is_file()
            fallback_ok = (
                self.settings.video_fallback_model_path is not None
                and self.settings.video_fallback_model_path.is_file()
            )
            if not primary_ok and not fallback_ok:
                missing["video"] = str(self.settings.video_model_path)
        return missing

    @staticmethod
    def _calibration_kwargs(factory: Any, *, use_calibration: bool, calibration_path: Any) -> dict[str, Any]:
        import inspect

        try:
            params = inspect.signature(factory).parameters
        except (TypeError, ValueError):
            return {}
        accepts_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
        kwargs: dict[str, Any] = {}
        if accepts_var_kw or "calibration_path" in params:
            kwargs["calibration_path"] = calibration_path
        if accepts_var_kw or "use_calibration" in params:
            kwargs["use_calibration"] = use_calibration
        return kwargs

    def _build_audio_predictor(self, device: str) -> Any:
        assert self.settings.audio_model_dir is not None
        extra_kwargs = self._calibration_kwargs(
            Wav2Vec2Predictor,
            use_calibration=self.settings.use_speech_calibration,
            calibration_path=self.settings.speech_calibration_path,
        )
        return Wav2Vec2Predictor(self.settings.audio_model_dir, device=device, **extra_kwargs)

    def _build_video_predictor(self, device: str) -> Any:
        if self.settings.facial_backend == "legacy_cnn":
            assert self.settings.legacy_video_model_path is not None
            return FacialEmotionPredictor(self.settings.legacy_video_model_path, device=device)
        assert self.settings.video_model_path is not None
        assert self.settings.video_fallback_model_path is not None
        extra_kwargs = self._calibration_kwargs(
            ResNetFacialEmotionPredictor,
            use_calibration=self.settings.use_facial_calibration,
            calibration_path=self.settings.facial_calibration_path,
        )
        return ResNetFacialEmotionPredictor(
            model_path=self.settings.video_model_path,
            expected_sha256=self.settings.video_model_sha256,
            fallback_model_path=self.settings.video_fallback_model_path,
            fallback_expected_sha256=self.settings.video_fallback_sha256,
            device=device,
            strict_fp32=self.settings.facial_strict_fp32,
            default_pooling=self.settings.facial_pooling,
            **extra_kwargs,
        )

    def load(self, force: bool = False) -> None:
        with self._lock:
            if self._loaded and not force:
                return
            started_all = time.perf_counter()
            device = self.settings.selected_device()
            assert self.settings.text_model_dir is not None
            assert self.settings.audio_model_dir is not None
            loaders = {
                "text": lambda: DistilBertTextPredictor(self.settings.text_model_dir, device=device),
                "audio": lambda: self._build_audio_predictor(device),
                "video": lambda: self._build_video_predictor(device),
            }
            for modality, loader in loaders.items():
                started_mod = time.perf_counter()
                try:
                    predictor = loader()
                    elapsed_ms = (time.perf_counter() - started_mod) * 1000.0
                    identifier = getattr(predictor, "model_identifier", predictor.__class__.__name__)
                    self.models[modality] = LoadedModel(
                        predictor=predictor,
                        identifier=str(identifier),
                        device=device,
                        architecture=getattr(predictor, "architecture", None),
                        checkpoint_sha256=getattr(predictor, "checkpoint_sha256", None),
                        active_role=getattr(predictor, "active_role", None),
                        preprocessing="imagenet_rgb_96x96" if hasattr(predictor, "checkpoint_sha256") else None,
                        pooling_method=getattr(predictor, "default_pooling", None),
                        load_ms=round(elapsed_ms, 2),
                        calibrated=bool(getattr(predictor, "calibration", None) is not None),
                    )
                    logger.info("Loaded %s model on device=%s in %.1f ms", modality, device, elapsed_ms)
                except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
                    elapsed_ms = (time.perf_counter() - started_mod) * 1000.0
                    logger.error("Failed loading %s model (%s): %s", modality, type(exc).__name__, exc)
                    self.models[modality] = LoadedModel(None, None, device, str(exc), load_ms=round(elapsed_ms, 2))
            self.load_duration_ms = round((time.perf_counter() - started_all) * 1000.0, 2)
            self._loaded = True

    def unload(self) -> None:
        """Release model references cleanly during application shutdown."""
        with self._lock:
            self.models.clear()
            self._loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, OSError, RuntimeError):
            pass

    def all_loaded(self) -> bool:
        return bool(self.models) and all(item.predictor is not None for item in self.models.values())

    def status(self) -> dict[str, Any]:
        return {
            modality: {
                "loaded": item.predictor is not None,
                "model_identifier": item.identifier,
                "device": item.device,
                "error": item.error,
                "architecture": item.architecture,
                "checkpoint_sha256": item.checkpoint_sha256,
                "active_role": item.active_role,
                "preprocessing": item.preprocessing,
                "pooling_method": item.pooling_method,
                "calibrated": item.calibrated,
            }
            for modality, item in self.models.items()
        }

    def require(self, modality: str) -> Any:
        item = self.models.get(modality)
        if item is None or item.predictor is None:
            raise RuntimeError(f"{modality} model is unavailable.")
        return item.predictor

    def predict(self, modality: str, *args: Any, **kwargs: Any) -> Any:
        lock = self._locks.get(modality, self._lock)
        with lock:
            predictor = self.require(modality)
            if modality == "text":
                return predictor.predict(*args, **kwargs)
            if modality == "audio":
                return predictor.predict_file(*args, **kwargs)
            return predictor.predict_video(*args, **kwargs)
