"""Unit and integration tests for Phase 18B ResNet18 + MTCNN facial model integration."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from api.config import ApiSettings
from api.model_registry import ModelRegistry
from facial_emotion.config import EMOTION_CLASSES
from facial_emotion.mtcnn_detector import MtcnnFaceDetector, sample_video_frames
from facial_emotion.remediation import IMAGENET_MEAN, IMAGENET_STD
from facial_emotion.resnet_predictor import (
    DEFAULT_FALLBACK_CHECKPOINT,
    DEFAULT_PRIMARY_CHECKPOINT,
    FALLBACK_CHECKPOINT_SHA256,
    FALLBACK_MODEL_IDENTIFIER,
    PRIMARY_CHECKPOINT_SHA256,
    PRIMARY_MODEL_IDENTIFIER,
    SUPPORTED_POOLING_METHODS,
    ResNetFacialEmotionPredictor,
    aggregate_crop_outputs,
    compute_sha256,
    preprocess_resnet_face_crops,
)
from project_config import MELD_PREPARED_ROOT


def _first_dev_video_with_face() -> Path:
    manifest = MELD_PREPARED_ROOT / "reports" / "dev" / "video_manifest.csv"
    detector = MtcnnFaceDetector(device="cpu")
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("video_status") != "ok":
                continue
            key = "resolved_video_path" if "resolved_video_path" in row else "output_path"
            candidate = Path(row[key])
            if not candidate.is_file():
                continue
            sampled = sample_video_frames(candidate, n_sample_frames=4)
            for _fi, frame in sampled["sampled_frames"]:
                dets, _fp = detector.detect_frame(frame)
                if dets:
                    return candidate
    raise AssertionError("No MELD dev video with a detectable face found.")


def test_primary_and_fallback_checkpoints_match_sha256():
    assert DEFAULT_PRIMARY_CHECKPOINT.is_file()
    assert DEFAULT_FALLBACK_CHECKPOINT.is_file()
    assert compute_sha256(DEFAULT_PRIMARY_CHECKPOINT) == PRIMARY_CHECKPOINT_SHA256
    assert compute_sha256(DEFAULT_FALLBACK_CHECKPOINT) == FALLBACK_CHECKPOINT_SHA256


def test_startup_fails_on_sha256_mismatch_without_fallback():
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        ResNetFacialEmotionPredictor(
            model_path=DEFAULT_PRIMARY_CHECKPOINT,
            expected_sha256="0" * 64,
            fallback_model_path=None,
            device="cpu",
        )


def test_startup_falls_back_to_verified_focal_checkpoint_when_primary_checksum_fails():
    predictor = ResNetFacialEmotionPredictor(
        model_path=DEFAULT_PRIMARY_CHECKPOINT,
        expected_sha256="0" * 64,
        fallback_model_path=DEFAULT_FALLBACK_CHECKPOINT,
        fallback_expected_sha256=FALLBACK_CHECKPOINT_SHA256,
        device="cpu",
    )
    assert predictor.active_role == "fallback"
    assert predictor.fallback_used is True
    assert predictor.model_identifier == FALLBACK_MODEL_IDENTIFIER
    assert predictor.checkpoint_sha256 == FALLBACK_CHECKPOINT_SHA256


def test_imagenet_preprocessing_swaps_bgr_to_rgb_and_normalizes():
    bgr = np.zeros((1, 48, 48, 3), dtype=np.uint8)
    bgr[0, :, :, 2] = 255  # Pure red in BGR
    tensor = preprocess_resnet_face_crops(bgr, image_size=96)
    assert tensor.shape == (1, 3, 96, 96)
    expected_r = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    expected_g = (0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    expected_b = (0.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    assert np.isclose(float(tensor[0, 0, 0, 0]), expected_r, atol=1e-5)
    assert np.isclose(float(tensor[0, 1, 0, 0]), expected_g, atol=1e-5)
    assert np.isclose(float(tensor[0, 2, 0, 0]), expected_b, atol=1e-5)


def test_predict_faces_outputs_canonical_normalized_probabilities():
    predictor = ResNetFacialEmotionPredictor(device="cpu")
    rng = np.random.default_rng(42)
    crops = [rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8) for _ in range(3)]
    res = predictor.predict_faces(crops)

    assert res["model_identifier"] == PRIMARY_MODEL_IDENTIFIER
    assert res["architecture"] == "resnet18"
    assert res["preprocessing"] == "imagenet_rgb_96x96"
    assert tuple(res["probabilities"].keys()) == tuple(EMOTION_CLASSES)
    vals = list(res["probabilities"].values())
    assert all(np.isfinite(v) and v >= 0.0 for v in vals)
    assert sum(vals) == pytest.approx(1.0, abs=1e-6)
    assert res["predicted_label"] == max(res["probabilities"], key=res["probabilities"].get)
    assert res["confidence"] == pytest.approx(max(vals), abs=1e-6)


def test_predict_image_and_empty_detection_handling():
    predictor = ResNetFacialEmotionPredictor(device="cpu")
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="No detectable faces"):
        predictor.predict_image(blank)

    dev_video = _first_dev_video_with_face()
    sampled = sample_video_frames(dev_video, n_sample_frames=4)
    face_frame = next(
        frame
        for _fi, frame in sampled["sampled_frames"]
        if predictor.detector.detect_frame(frame)[0]
    )
    img_res = predictor.predict_image(face_frame)
    assert tuple(img_res["probabilities"].keys()) == tuple(EMOTION_CLASSES)
    assert sum(img_res["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CPU/CUDA parity verification")
def test_cpu_vs_cuda_strict_fp32_parity():
    rng = np.random.default_rng(123)
    crops = [rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8) for _ in range(8)]

    cpu_pred = ResNetFacialEmotionPredictor(device="cpu", strict_fp32=True)
    cuda_pred = ResNetFacialEmotionPredictor(device="cuda", strict_fp32=True)

    cpu_res = cpu_pred.predict_faces(crops)
    cuda_res = cuda_pred.predict_faces(crops)

    max_diff = max(abs(cpu_res["probabilities"][c] - cuda_res["probabilities"][c]) for c in EMOTION_CLASSES)
    assert max_diff < 1e-4, f"Expected CPU/CUDA diff < 1e-4, got {max_diff}"
    assert cpu_res["predicted_label"] == cuda_res["predicted_label"]


def test_registry_supports_resnet18_default_and_legacy_rollback():
    from api.config import PRIMARY_RESNET_V2_SHA256

    resnet_reg = ModelRegistry(ApiSettings(device="cpu", facial_backend="resnet18"))
    resnet_reg.load()
    resnet_status = resnet_reg.status()["video"]
    assert resnet_status["loaded"] is True
    assert resnet_status["model_identifier"] == "facial-resnet18-v2-la_ce_tau040"
    assert resnet_status["architecture"] == "resnet18"
    assert resnet_status["checkpoint_sha256"] == PRIMARY_RESNET_V2_SHA256
    assert resnet_status["calibrated"] is True

    v1_reg = ModelRegistry(
        ApiSettings(
            device="cpu",
            facial_backend="resnet18",
            video_model_path=PRIMARY_CHECKPOINT_PATH if "PRIMARY_CHECKPOINT_PATH" in globals() else resnet_reg.settings.model_root / "facial_emotion" / "resnet18" / "exp5_ce_weighted_best.pt",
        )
    )
    v1_reg.load()
    v1_status = v1_reg.status()["video"]
    assert v1_status["loaded"] is True
    assert v1_status["model_identifier"] == PRIMARY_MODEL_IDENTIFIER
    assert v1_status["checkpoint_sha256"] == PRIMARY_CHECKPOINT_SHA256

    legacy_reg = ModelRegistry(ApiSettings(device="cpu", facial_backend="legacy_cnn"))
    legacy_reg.load()
    legacy_status = legacy_reg.status()["video"]
    assert legacy_status["loaded"] is True
    assert legacy_status["model_identifier"] == "FacialEmotionPredictor"
    assert legacy_status["architecture"] is None
    assert legacy_reg.require("video").__class__.__name__ == "FacialEmotionPredictor"


def test_raw_mean_logit_pooling_is_supported_and_normalized():
    assert "raw_mean_logit" in SUPPORTED_POOLING_METHODS
    logits = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    probs = aggregate_crop_outputs(logits, method="raw_mean_logit")
    assert probs.shape == (7,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)
    # Mean logit has class 0 (1.0) == class 4 (1.0)
    assert np.isclose(probs[0], probs[4], atol=1e-6)


def test_p1_calibration_remains_opt_in_for_v1_and_applies_explicit_json(tmp_path):
    import json

    predictor_v1 = ResNetFacialEmotionPredictor(device="cpu")
    assert predictor_v1.calibration is None

    rng = np.random.default_rng(7)
    crops = [rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8) for _ in range(2)]
    res_raw = predictor_v1.predict_faces(crops)
    assert res_raw["calibrated"] is False

    cal_file = tmp_path / "test_cal.json"
    cal_file.write_text(
        json.dumps(
            {
                "classes": list(EMOTION_CLASSES),
                "temperature": 0.5,
                "bias": [1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "method": "temp_scaling_test",
            }
        ),
        encoding="utf-8",
    )
    predictor_cal = ResNetFacialEmotionPredictor(device="cpu", calibration_path=cal_file)
    res_cal = predictor_cal.predict_faces(crops)
    assert res_cal["calibrated"] is True
    assert res_cal["calibration_method"] == "temp_scaling_test"
    assert np.isclose(sum(res_cal["probabilities"].values()), 1.0, atol=1e-6)
    assert res_cal["probabilities"]["anger"] > res_raw["probabilities"]["anger"]


def test_p2_facial_v2_training_helpers_preserve_shape_and_safety_gate(tmp_path):
    from facial_emotion.train_resnet18_v2 import (
        LogitAdjustedCrossEntropyLoss,
        _assert_no_test_path,
        augment_face_crop_bgr,
    )

    sample = np.full((96, 96, 3), 120, dtype=np.uint8)
    aug = augment_face_crop_bgr(sample)
    assert aug.shape == (96, 96, 3)
    assert aug.dtype == np.uint8

    priors = np.array([0.11, 0.03, 0.03, 0.17, 0.47, 0.07, 0.12], dtype=np.float64)
    crit = LogitAdjustedCrossEntropyLoss(priors, tau=0.45, label_smoothing=0.04)
    logits = torch.randn(4, 7, requires_grad=True)
    targets = torch.tensor([0, 1, 4, 6], dtype=torch.long)
    loss = crit(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None

    with pytest.raises(PermissionError):
        _assert_no_test_path(tmp_path / "test_sent_emo.csv")

