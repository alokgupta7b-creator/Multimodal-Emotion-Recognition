"""Phase 16 Preflight Verification Script."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

from project_config import MELD_ANNOTATIONS, MELD_PREPARED_ROOT, REPOSITORY_ROOT
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES


def hash_file(path: Path) -> tuple[int, str]:
    with open(path, "rb") as f:
        data = f.read()
    return len(data), hashlib.sha256(data).hexdigest()


def main():
    print("=" * 70)
    print("PHASE 16 PREFLIGHT INTEGRITY & MANIFEST AUDIT")
    print("=" * 70)

    # 1. Component Hashes
    targets = {
        "DistilBERT Checkpoint": REPOSITORY_ROOT / "models" / "text_emotion" / "distilbert" / "best" / "model.safetensors",
        "Wav2Vec2 Checkpoint": REPOSITORY_ROOT / "models" / "speech_emotion" / "wav2vec2" / "best" / "model.safetensors",
        "Facial CNN Checkpoint": REPOSITORY_ROOT / "models" / "facial_emotion" / "facial_cnn_best.pt",
        "Candidate 4 Fusion Code": REPOSITORY_ROOT / "api" / "fusion.py",
        "Label Mapping": REPOSITORY_ROOT / "speech_emotion" / "model" / "label_mapping.py",
    }

    manifest_hashes = {}
    for name, p in targets.items():
        if not p.is_file():
            print(f"ERROR: {name} not found at {p}")
            return
        size, sha = hash_file(p)
        manifest_hashes[name] = {"path": str(p), "size_bytes": size, "sha256": sha}
        print(f"{name}:\n  Size: {size:,} bytes\n  SHA-256: {sha}")

    # Expected frozen hashes from Phase 14 & 15
    expected_distilbert = "6fbc700eee6cd6936019be7be3c4dd2526a792fd21b79d5d1aa2f22195021474"
    expected_wav2vec2 = "15256b6625bc0aa7602e53335b49e3c3d01bc0a8def0c71cf5620115aa783d6e"
    expected_facial = "85267a9b694396d28740d15a8c24061f90490e205b772f816cdc758c10d15760"

    assert manifest_hashes["DistilBERT Checkpoint"]["sha256"] == expected_distilbert, "DistilBERT hash mismatch!"
    assert manifest_hashes["Wav2Vec2 Checkpoint"]["sha256"] == expected_wav2vec2, "Wav2Vec2 hash mismatch!"
    assert manifest_hashes["Facial CNN Checkpoint"]["sha256"] == expected_facial, "Facial CNN hash mismatch!"
    print("\n✔ All model checkpoint SHA-256 hashes match frozen standards perfectly.")

    # 2. Annotation & Manifest Alignment
    test_csv = MELD_ANNOTATIONS / "test_sent_emo.csv"
    with test_csv.open(encoding="utf-8", newline="") as f:
        test_annotations = list(csv.DictReader(f))
    print(f"\nTest Annotations Count: {len(test_annotations)}")
    assert len(test_annotations) == 2610, f"Expected 2,610 test annotations, found {len(test_annotations)}"

    emotion_counts = Counter(row["Emotion"].strip().lower() for row in test_annotations)
    print("Test Split Label Distribution:")
    for emo in MELD_CLASS_NAMES:
        print(f"  {emo:<10}: {emotion_counts[emo]:>5} ({emotion_counts[emo]/len(test_annotations)*100:5.2f}%)")

    # Check Audio Manifest
    audio_manifest_path = MELD_PREPARED_ROOT / "reports" / "test" / "audio_manifest.csv"
    with audio_manifest_path.open(encoding="utf-8", newline="") as f:
        audio_manifest = list(csv.DictReader(f))
    print(f"\nTest Audio Manifest Count: {len(audio_manifest)}")
    assert len(audio_manifest) == 2610, f"Expected 2,610 audio manifest entries, found {len(audio_manifest)}"

    audio_status_counts = Counter(row.get("audio_status") for row in audio_manifest)
    print("Test Split Audio Status Distribution in Manifest:")
    for status, count in audio_status_counts.items():
        print(f"  {status}: {count}")

    # Verify 1-to-1 matching across Dialogue_ID and Utterance_ID
    missing_audio_files = 0
    existing_audio_files = 0
    for idx, (anno, audio_row) in enumerate(zip(test_annotations, audio_manifest)):
        dia = int(anno["Dialogue_ID"])
        utt = int(anno["Utterance_ID"])
        audio_dia = int(audio_row["dialogue_id"])
        audio_utt = int(audio_row["utterance_id"])
        assert (dia, utt) == (audio_dia, audio_utt), f"Alignment mismatch at row {idx}: ({dia}, {utt}) vs ({audio_dia}, {audio_utt})"

        audio_path_str = audio_row.get("output_path", "")
        if audio_path_str and Path(audio_path_str).is_file():
            existing_audio_files += 1
        else:
            missing_audio_files += 1

    print(f"\nAudio File Presence Verification on Disk:")
    print(f"  Existing audio files: {existing_audio_files}")
    print(f"  Missing audio files:  {missing_audio_files}")
    print(f"  Total test samples:   {len(test_annotations)}")
    print("\n✔ Preflight verification completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

