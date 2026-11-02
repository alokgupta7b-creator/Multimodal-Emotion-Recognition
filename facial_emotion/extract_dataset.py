
"""Batch facial feature extraction for MELD."""

from pathlib import Path

import numpy as np
import pandas as pd

from facial_emotion.config import FacialConfig
from facial_emotion.feature_extraction import FacialFeatureExtractor


BATCH_SIZE = 100
SPLITS = ("train", "dev", "test")


def is_true(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def process_split(config, split):
    manifest_path = config.manifest_path(split)
    output_dir = config.output_dir / split
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)

    required = {"sample_id", "video_path", "emotion", "has_video"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {missing}")

    total = len(df)
    print(f"\nProcessing {split}: {total} manifest rows")

    with FacialFeatureExtractor(config) as extractor:
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch_number = start // BATCH_SIZE

            batch_file = output_dir / f"batch_{batch_number:04d}.npz"

            if batch_file.exists():
                print(f"Skipping completed batch {batch_number}")
                continue

            features_list = []
            sample_ids = []
            labels = []
            statuses = []

            for _, row in df.iloc[start:end].iterrows():
                sample_id = str(row["sample_id"])
                video_value = row["video_path"]

                if not is_true(row["has_video"]):
                    statuses.append((sample_id, "unavailable_in_manifest"))
                    continue

                if pd.isna(video_value) or not str(video_value).strip():
                    statuses.append((sample_id, "missing_video_path"))
                    continue

                video_path = config.video_root / str(video_value)

                if not video_path.is_file():
                    statuses.append((sample_id, "missing_file"))
                    continue

                try:
                    features = extractor.extract_from_video(video_path)

                    if features is None:
                        statuses.append((sample_id, "no_face_detected"))
                        continue

                    features_list.append(features)
                    sample_ids.append(sample_id)
                    labels.append(str(row["emotion"]))
                    statuses.append((sample_id, "success"))

                except Exception as exc:
                    print(f"Error: {sample_id}: {exc}")
                    statuses.append((sample_id, "extraction_error"))

            feature_array = (
                np.stack(features_list).astype(np.float32)
                if features_list
                else np.empty((0, 1912), dtype=np.float32)
            )

            status_array = np.array(statuses, dtype=str)

            np.savez_compressed(
                batch_file,
                features=feature_array,
                sample_ids=np.array(sample_ids, dtype=str),
                labels=np.array(labels, dtype=str),
                statuses=status_array,
            )

            print(
                f"Batch {batch_number}: rows {start + 1}-{end}, "
                f"successful features: {len(sample_ids)}"
            )

    print(f"Finished split: {split}")


def main():
    config = FacialConfig()
    config.validate()

    config.output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        process_split(config, split)

    print("\nAll MELD splits processed.")


if __name__ == "__main__":
    main()
