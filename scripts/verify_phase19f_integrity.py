"""Phase 19F integrity check: protected SHA-256 hashes, canonical labels, zero test-split access."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.fusion import CLASS_NAMES as FUSION_CLASS_NAMES
from facial_emotion.config import EMOTION_CLASSES as FACIAL_EMOTION_CLASSES
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES

CANONICAL_EXPECTED = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")

EXPECTED_HASHES: dict[str, str] = {
    "facial_emotion/model.py": "08df340cc1fa81fd5420511a248f44149961f8b19d0dbed9023621ab0545061b",
    "facial_emotion/detector.py": "24c183be7c61f336940d2e2fb1fd55cd030a5779044e88dd3f964840752f075d",
    "facial_emotion/remediation.py": "3f2fa952afc890af0bc19a4b5800daa56e5f2e237f6cc5665d3aa64f5a947f07",
    "api/fusion.py": "ffa76bea0a7a305c67e4ebd79790730cdc5e7be488f96d76df7768486efb2f2d",
    "models/facial_emotion/facial_cnn_best.pt": "85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760",
    "models/text_emotion/distilbert/best/model.safetensors": "6fbc700eee6cd6936019be7be3c4dd2526a792fd21b79d5d1aa2f22195021474",
    "models/speech_emotion/wav2vec2/best/model.safetensors": "15256b6625bc0aa7602e53335b49e3c3d01bc0a8def0c71cf5620115aa783d6e",
    "models/facial_emotion/resnet18/exp5_ce_weighted_best.pt": "8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64",
    "models/facial_emotion/resnet18/exp5_focal_g3_wt_best.pt": "4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0",
    "models/facial_experiments/exp5_focal_loss/exp5_ce_weighted_best.pt": "8825d865c70351a38c1da1448d9b6c087a0fe090ad586851edf0342fc4761b64",
    "models/facial_experiments/exp5_focal_loss/exp5_focal_g3_wt_best.pt": "4830ac8409e73da149b4eaf169679f2c2b95322316a411751f0567ac478bc9b0",
}

PHASE19_CODE_FILES: list[str] = [
    "scripts/validate_phase19a_hardware.py",
    "scripts/e2e_phase19a_live_capture.cjs",
    "scripts/run_phase19b_uat.py",
    "scripts/e2e_phase19b_uat.cjs",
    "api/tests/test_phase19c_security.py",
    "scripts/benchmark_phase19d_performance.py",
]

TEST_SPLIT_PATTERNS = (
    "test_splits_complete",
    "test_sent_emo.csv",
    "/audio/test/",
    "\\audio\\test\\",
    "reports/test/",
    "reports\\test\\",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    hash_results: dict[str, dict[str, Any]] = {}
    all_hashes_matched = True
    for rel_path, expected in EXPECTED_HASHES.items():
        target = REPO_ROOT / rel_path
        actual = sha256_file(target) if target.exists() else None
        matched = actual == expected
        all_hashes_matched = all_hashes_matched and matched
        hash_results[rel_path] = {
            "exists": target.exists(),
            "expected": expected,
            "actual": actual,
            "matched": matched,
        }

    manifest_path = REPO_ROOT / "models" / "facial_emotion" / "resnet18" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label_checks = {
        "meld_class_names": list(MELD_CLASS_NAMES),
        "fusion_class_names": list(FUSION_CLASS_NAMES),
        "facial_emotion_classes": list(FACIAL_EMOTION_CLASSES),
        "manifest_canonical_classes": list(manifest.get("canonical_classes", [])),
    }
    labels_valid = all(tuple(val) == CANONICAL_EXPECTED for val in label_checks.values())

    test_split_violations: list[dict[str, str]] = []
    for rel_path in PHASE19_CODE_FILES:
        file_path = REPO_ROOT / rel_path
        text = file_path.read_text(encoding="utf-8")
        for pat in TEST_SPLIT_PATTERNS:
            if pat in text:
                test_split_violations.append({"file": rel_path, "pattern": pat})

    report = {
        "phase": "19F",
        "deployed_to_production": False,
        "all_protected_hashes_matched": all_hashes_matched,
        "protected_hashes": hash_results,
        "canonical_labels_matched": labels_valid,
        "label_checks": label_checks,
        "zero_test_split_audit": {
            "checked_files": len(PHASE19_CODE_FILES),
            "violations": test_split_violations,
            "passed": len(test_split_violations) == 0,
        },
        "passed": bool(all_hashes_matched and labels_valid and len(test_split_violations) == 0),
    }

    out_path = REPO_ROOT / "models" / "facial_emotion" / "resnet18" / "phase19f_integrity_validation.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "hashes": len(hash_results), "labels_valid": labels_valid, "test_split_violations": len(test_split_violations)}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

