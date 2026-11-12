"""Phase 18D tests for ExtendedMultimodalFusionEngine and /predict/multimodal routes."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.app as api_app
from api.fusion import CLASS_NAMES, Candidate4FusionEngine
from api.multimodal_extension import (
    AUDIO_ONLY_IDENTIFIER,
    AUDIO_VIDEO_FUSION_IDENTIFIER,
    EXTENDED_FUSION_IDENTIFIER,
    VIDEO_ONLY_IDENTIFIER,
    ExtendedMultimodalFusionEngine,
)


def _uniform_plus(peak_idx: int, peak_val: float = 0.70) -> dict[str, float]:
    rest = (1.0 - peak_val) / (len(CLASS_NAMES) - 1)
    return {cls: (peak_val if i == peak_idx else rest) for i, cls in enumerate(CLASS_NAMES)}


def _write_test_wav(path: Path, seconds: float = 1.0, freq: float = 220.0) -> Path:
    sr = 16000
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    samples = (0.25 * np.sin(2 * np.pi * freq * t) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return path


def _write_test_mp4(path: Path, frames: int = 12, fps: float = 12.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for i in range(frames):
        frame = np.full((120, 160, 3), (i * 15) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class _FakeRegistry:
    def __init__(self, video_mode: str = "valid") -> None:
        self.video_mode = video_mode
        self.text_probs = _uniform_plus(3, 0.70)  # Joy
        self.audio_probs = _uniform_plus(0, 0.65)  # Anger
        self.video_probs = _uniform_plus(6, 0.62)  # Surprise

    def predict(self, modality: str, payload: Any) -> dict[str, Any]:
        if modality == "text":
            return {
                "predicted_label": "joy",
                "probabilities": dict(self.text_probs),
                "confidence": 0.70,
                "model": "distilbert-fake",
                "device": "cpu",
                "char_length": len(str(payload)),
                "word_count": len(str(payload).split()),
            }
        if modality == "audio":
            return {
                "predicted_label": "anger",
                "probabilities": dict(self.audio_probs),
                "confidence": 0.65,
                "model": "wav2vec2-fake",
                "device": "cpu",
                "duration_seconds": 1.0,
                "rms": 0.15,
                "sample_rate": 16000,
                "is_silent": False,
            }
        if modality == "video":
            if self.video_mode == "no_face":
                raise ValueError("No detectable faces found in the uploaded video.")
            cov = 0.25 if self.video_mode == "low_coverage" else 0.85
            probs = (
                {c: 1.0 / len(CLASS_NAMES) for c in CLASS_NAMES}
                if self.video_mode == "high_entropy"
                else dict(self.video_probs)
            )
            top = max(probs, key=probs.__getitem__)
            ent = float(-sum(p * np.log(max(p, 1e-12)) for p in probs.values()))
            return {
                "predicted_label": top,
                "probabilities": probs,
                "confidence": float(probs[top]),
                "model": "facial-resnet18-ce-v1",
                "device": "cpu",
                "num_faces": 1 if cov < 0.5 else 10,
                "num_sampled_frames": 12,
                "face_detection_coverage": cov,
                "prediction_entropy": ent,
            }
        raise ValueError(f"Unknown modality: {modality}")


def test_extended_engine_preserves_candidate4_verbatim():
    c4 = Candidate4FusionEngine()
    ext = ExtendedMultimodalFusionEngine(c4)

    text_p = _uniform_plus(4, 0.55)
    audio_p = _uniform_plus(0, 0.72)

    for a_p in (None, audio_p):
        r_c4 = c4.fuse(text_p, a_p)
        r_ext = ext.fuse(text_p, a_p, None)
        assert r_ext.fusion_method == r_c4.fusion_method
        assert r_ext.predicted_emotion == r_c4.predicted_emotion
        assert r_ext.rule_applied == r_c4.rule_applied
        assert r_ext.modality_weights == r_c4.modality_weights
        for cls in CLASS_NAMES:
            assert r_ext.probabilities[cls] == pytest.approx(r_c4.probabilities[cls], abs=1e-12)


def test_extended_engine_all_modality_combinations_and_quality_gates():
    ext = ExtendedMultimodalFusionEngine()
    text_p = _uniform_plus(4, 0.58)
    audio_p = _uniform_plus(6, 0.72)
    video_p = _uniform_plus(0, 0.68)

    # Text + Audio + Video (quality passes)
    res_tav = ext.fuse(
        text_probabilities=text_p,
        audio_probabilities=audio_p,
        video_probabilities=video_p,
        video_quality={"face_detection_coverage": 0.75, "prediction_entropy": 1.15},
    )
    assert res_tav.fusion_method == EXTENDED_FUSION_IDENTIFIER
    assert res_tav.modality_weights["video"] == pytest.approx(0.08, abs=1e-9)
    assert sum(res_tav.modality_weights.values()) == pytest.approx(1.0, abs=1e-9)
    assert sum(res_tav.probabilities.values()) == pytest.approx(1.0, abs=1e-9)
    assert list(res_tav.probabilities.keys()) == list(CLASS_NAMES)

    # Low coverage gates video to advisory-only (w_v = 0.0, Candidate 4 probs preserved)
    res_c4 = Candidate4FusionEngine().fuse(text_p, audio_p)
    res_low = ext.fuse(
        text_probabilities=text_p,
        audio_probabilities=audio_p,
        video_probabilities=video_p,
        video_quality={"face_detection_coverage": 0.25, "prediction_entropy": 1.10},
    )
    assert res_low.fusion_method == EXTENDED_FUSION_IDENTIFIER
    assert res_low.modalities_available["video"] is True
    assert res_low.modality_weights["video"] == 0.0
    assert "video_low_face_coverage" in (res_low.exclusion_reasons["video"] or "")
    for cls in CLASS_NAMES:
        assert res_low.probabilities[cls] == pytest.approx(res_c4.probabilities[cls], abs=1e-12)

    # Video-only, Audio-only, Audio+Video without text
    v_only = ext.fuse(video_probabilities=video_p)
    assert v_only.fusion_method == VIDEO_ONLY_IDENTIFIER
    assert v_only.modality_weights == {"text": 0.0, "audio": 0.0, "video": 1.0}
    assert v_only.exclusion_reasons["text"] == "No text input provided"

    a_only = ext.fuse(audio_probabilities=audio_p)
    assert a_only.fusion_method == AUDIO_ONLY_IDENTIFIER
    assert a_only.modality_weights == {"text": 0.0, "audio": 1.0, "video": 0.0}

    av = ext.fuse(
        audio_probabilities=audio_p,
        video_probabilities=video_p,
        video_quality={"face_detection_coverage": 0.80, "prediction_entropy": 1.10},
    )
    assert av.fusion_method == AUDIO_VIDEO_FUSION_IDENTIFIER
    assert av.modality_weights == pytest.approx({"text": 0.0, "audio": 0.92, "video": 0.08}, abs=1e-9)

    with pytest.raises(ValueError):
        ext.fuse()


def _temp_media_files() -> set[Path]:
    tmp_root = Path(tempfile.gettempdir())
    found: set[Path] = set()
    for pattern in ("*.wav", "*.mp4", "*.webm", "emotion*"):
        found.update(p for p in tmp_root.glob(pattern) if p.is_file())
    return found


def test_multimodal_api_all_combinations_fallbacks_and_cleanup(tmp_path):
    old_registry = getattr(api_app.app.state, "registry", None)
    old_engine = getattr(api_app.app.state, "extended_fusion_engine", None)
    wav_path = _write_test_wav(tmp_path / "sample.wav")
    mp4_path = _write_test_mp4(tmp_path / "sample.mp4")

    temp_before = _temp_media_files()

    try:
        client = TestClient(api_app.app)
        api_app.app.state.registry = _FakeRegistry(video_mode="valid")
        api_app.app.state.extended_fusion_engine = ExtendedMultimodalFusionEngine()

        # 1. Text-only JSON -> Candidate 4
        r_txt = client.post("/predict/multimodal", json={"text": "That was wonderful!"})
        assert r_txt.status_code == 200
        d_txt = r_txt.json()
        assert d_txt["fusion_method"] == "text_anchored_class_conditional_bounded_fusion_v1"
        assert d_txt["modality_weights"]["video"] == 0.0
        assert d_txt["exclusion_reasons"]["video"] == "No video file provided"

        # Whitespace JSON -> 422
        r_blank = client.post("/predict/multimodal", json={"text": "   "})
        assert r_blank.status_code == 422

        # 2. Text + Audio multipart -> exact Candidate 4 parity
        with wav_path.open("rb") as wf:
            r_ta = client.post(
                "/predict/multimodal",
                data={"text": "What on earth are you doing?!"},
                files={"file": ("sample.wav", wf, "audio/wav")},
            )
        assert r_ta.status_code == 200
        d_ta = r_ta.json()
        assert d_ta["fusion_method"] == "text_anchored_class_conditional_bounded_fusion_v1"
        assert d_ta["modalities_available"]["audio"] is True
        assert d_ta["modalities_available"]["video"] is False

        # 3. Text + Video multipart -> extended fusion w_v = 0.08
        with mp4_path.open("rb") as vf:
            r_tv = client.post(
                "/predict/multimodal",
                data={"text": "Oh my goodness!"},
                files={"video": ("sample.mp4", vf, "video/mp4")},
            )
        assert r_tv.status_code == 200
        d_tv = r_tv.json()
        assert d_tv["fusion_method"] == EXTENDED_FUSION_IDENTIFIER
        assert d_tv["modalities_available"] == {"text": True, "audio": False, "video": True}
        assert d_tv["modality_weights"]["video"] == pytest.approx(0.08, abs=1e-6)
        assert d_tv["individual_predictions"]["video"]["modality"] == "video"
        assert d_tv["quality"]["video_num_faces"] == 10

        # 4. Text + Audio + Video -> all 3 active
        with wav_path.open("rb") as wf, mp4_path.open("rb") as vf:
            r_tav = client.post(
                "/predict/multimodal",
                data={"text": "Oh wow!"},
                files={
                    "audio": ("sample.wav", wf, "audio/wav"),
                    "video": ("sample.mp4", vf, "video/mp4"),
                },
            )
        assert r_tav.status_code == 200
        d_tav = r_tav.json()
        assert d_tav["fusion_method"] == EXTENDED_FUSION_IDENTIFIER
        assert d_tav["modalities_available"] == {"text": True, "audio": True, "video": True}
        assert sum(d_tav["modality_weights"].values()) == pytest.approx(1.0, abs=1e-6)

        # 5. Video-only, Audio-only, Audio+Video without text
        with mp4_path.open("rb") as vf:
            r_vo = client.post("/predict/multimodal", files={"video": ("sample.mp4", vf, "video/mp4")})
        assert r_vo.status_code == 200
        assert r_vo.json()["fusion_method"] == VIDEO_ONLY_IDENTIFIER
        assert r_vo.json()["modality_weights"]["video"] == pytest.approx(1.0, abs=1e-6)

        with wav_path.open("rb") as wf:
            r_ao = client.post("/predict/multimodal", files={"file": ("sample.wav", wf, "audio/wav")})
        assert r_ao.status_code == 200
        assert r_ao.json()["fusion_method"] == AUDIO_ONLY_IDENTIFIER

        with wav_path.open("rb") as wf, mp4_path.open("rb") as vf:
            r_av = client.post(
                "/predict/multimodal",
                files={"file": ("sample.wav", wf, "audio/wav"), "video": ("sample.mp4", vf, "video/mp4")},
            )
        assert r_av.status_code == 200
        d_av = r_av.json()
        assert d_av["fusion_method"] == AUDIO_VIDEO_FUSION_IDENTIFIER
        assert d_av["modality_weights"]["audio"] == pytest.approx(0.92, abs=1e-6)
        assert d_av["modality_weights"]["video"] == pytest.approx(0.08, abs=1e-6)

        # 6. Quality-gate advisory downgrade on low coverage
        api_app.app.state.registry = _FakeRegistry(video_mode="low_coverage")
        with mp4_path.open("rb") as vf:
            r_low = client.post(
                "/predict/multimodal",
                data={"text": "I feel fine."},
                files={"video": ("sample.mp4", vf, "video/mp4")},
            )
        assert r_low.status_code == 200
        d_low = r_low.json()
        assert d_low["modalities_available"]["video"] is True
        assert d_low["modality_weights"]["video"] == 0.0
        assert "video_low_face_coverage" in (d_low["exclusion_reasons"]["video"] or "")

        # 7. Corrupt video fallback when paired with text, 422 when video only
        r_corrupt_tv = client.post(
            "/predict/multimodal",
            data={"text": "Still works safely!"},
            files={"video": ("corrupt.mp4", b"not a valid video", "video/mp4")},
        )
        assert r_corrupt_tv.status_code == 200
        assert r_corrupt_tv.json()["modalities_available"] == {"text": True, "audio": False, "video": False}
        assert r_corrupt_tv.json()["exclusion_reasons"]["video"]

        r_corrupt_vo = client.post(
            "/predict/multimodal",
            files={"video": ("corrupt.mp4", b"not a valid video", "video/mp4")},
        )
        assert r_corrupt_vo.status_code == 422

        # 8. No-face video fallback with text vs 422 alone
        api_app.app.state.registry = _FakeRegistry(video_mode="no_face")
        with mp4_path.open("rb") as vf:
            r_nf_tv = client.post(
                "/predict/multimodal",
                data={"text": "Text remains intact."},
                files={"video": ("sample.mp4", vf, "video/mp4")},
            )
        assert r_nf_tv.status_code == 200
        assert "No detectable faces" in (r_nf_tv.json()["exclusion_reasons"]["video"] or "")

        with mp4_path.open("rb") as vf:
            r_nf_vo = client.post("/predict/multimodal", files={"video": ("sample.mp4", vf, "video/mp4")})
        assert r_nf_vo.status_code == 422

        # 9. Audio-only MP4 rejection / fallback via ffmpeg when available
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            audio_only_mp4 = tmp_path / "audio_only.mp4"
            subprocess.run(
                [ffmpeg, "-y", "-i", str(wav_path), "-vn", "-c:a", "aac", str(audio_only_mp4)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with audio_only_mp4.open("rb") as af:
                r_ao_mp4 = client.post(
                    "/predict/multimodal",
                    data={"text": "Audio-only MP4 fallback"},
                    files={"video": ("audio_only.mp4", af, "video/mp4")},
                )
            assert r_ao_mp4.status_code == 200
            assert r_ao_mp4.json()["modalities_available"]["video"] is False
            assert r_ao_mp4.json()["exclusion_reasons"]["video"]
    finally:
        api_app.app.state.registry = old_registry
        api_app.app.state.extended_fusion_engine = old_engine

    assert _temp_media_files() - temp_before == set()


def test_phase18d_real_models_multimodal_video_and_cleanup(tmp_path):
    blank_mp4 = _write_test_mp4(tmp_path / "blank.mp4")
    temp_before = _temp_media_files()

    with TestClient(api_app.app) as client:
        st = client.get("/models/status")
        assert st.status_code == 200
        models = st.json()["models"]
        audio_status = models["audio"]
        video_status = models["video"]
        assert audio_status["loaded"] is True
        assert audio_status["model_identifier"] == "speech-wav2vec2-v2"
        assert audio_status["calibrated"] is True
        assert video_status["loaded"] is True
        assert video_status["model_identifier"] == "facial-resnet18-v2-la_ce_tau040"
        assert video_status["calibrated"] is True

        # Real Text + blank video safely excludes video while preserving DistilBERT text output
        with blank_mp4.open("rb") as vf:
            res = client.post(
                "/predict/multimodal",
                data={"text": "I am delighted by this reliable integration!"},
                files={"video": ("blank.mp4", vf, "video/mp4")},
            )
        assert res.status_code == 200
        body = res.json()
        assert body["modalities_available"] == {"text": True, "audio": False, "video": False}
        assert "No detectable faces" in (body["exclusion_reasons"]["video"] or "")

        # Real webcam/sample video if present locally
        sample_clip = Path("C:/Users/Bhavanth D/OneDrive/Pictures/Camera Roll/WIN_20261111_03_07_13_Pro.mp4")
        if sample_clip.exists():
            with sample_clip.open("rb") as vf:
                res_real = client.post(
                    "/predict/multimodal",
                    data={"text": "Wait, seriously?!"},
                    files={"video": (sample_clip.name, vf, "video/mp4")},
                )
            assert res_real.status_code == 200
            d_real = res_real.json()
            assert d_real["fusion_method"] == EXTENDED_FUSION_IDENTIFIER
            assert d_real["modalities_available"]["video"] is True
            assert sum(d_real["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
            assert list(d_real["probabilities"].keys()) == list(CLASS_NAMES)

    assert _temp_media_files() - temp_before == set()


def test_v1_plus_p1_env_rollback_configuration_and_registry(monkeypatch):
    from api.config import ApiSettings, PRIMARY_RESNET_SHA256
    from api.model_registry import ModelRegistry

    default_cfg = ApiSettings.from_env()
    monkeypatch.setenv(
        "EMOTION_API_AUDIO_MODEL_DIR",
        str(default_cfg.model_root / "speech_emotion" / "wav2vec2" / "best"),
    )
    monkeypatch.setenv(
        "EMOTION_API_VIDEO_MODEL_PATH",
        str(default_cfg.model_root / "facial_emotion" / "resnet18" / "exp5_ce_weighted_best.pt"),
    )
    monkeypatch.delenv("EMOTION_API_VIDEO_SHA256", raising=False)
    monkeypatch.setenv("EMOTION_API_DEVICE", "cpu")

    rollback_cfg = ApiSettings.from_env()
    assert rollback_cfg.video_model_sha256 == PRIMARY_RESNET_SHA256

    rollback_reg = ModelRegistry(rollback_cfg)
    rollback_reg.load()
    try:
        st = rollback_reg.status()
        assert st["audio"]["loaded"] is True
        assert st["audio"]["model_identifier"] == "facebook/wav2vec2-base"
        assert st["audio"]["calibrated"] is True
        assert st["video"]["loaded"] is True
        assert st["video"]["model_identifier"] == "facial-resnet18-ce-v1"
        assert st["video"]["checkpoint_sha256"] == PRIMARY_RESNET_SHA256
        assert st["video"]["calibrated"] is False
    finally:
        rollback_reg.unload()

    monkeypatch.setenv("EMOTION_API_USE_SPEECH_CALIBRATION", "0")
    uncal_cfg = ApiSettings.from_env()
    uncal_reg = ModelRegistry(uncal_cfg)
    uncal_reg.load()
    try:
        assert uncal_reg.status()["audio"]["calibrated"] is False
    finally:
        uncal_reg.unload()
