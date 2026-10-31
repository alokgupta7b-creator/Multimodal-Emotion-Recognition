
"""Retry failed MELD facial samples using 20 sampled frames."""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from facial_emotion.config import FacialConfig
from facial_emotion.feature_extraction import FacialFeatureExtractor


BATCH_SIZE = 100
SPLITS = ("train", "dev", "test")
FEATURE_COUNT = 1912

BACKUP_DIR = Path(r"D:\MELD_Facial_Output\backup_before_retry")


def process_split(config, split):
    manifest = pd.read_csv(config.manifest_path(split))
    output_dir = config.output_dir / split

    attempted = 0
    recovered = 0
    still_failed = 0

    print(f"\nProcessing {split}...")

    # Retry with 20 sampled frames.
    config.frame_sample_count = 20

    with FacialFeatureExtractor(config) as extractor:
        for start in range(0, len(manifest), BATCH_SIZE):
            batch_number = start // BATCH_SIZE
            batch_file = output_dir / f"batch_{batch_number:04d}.npz"

            if not batch_file.exists():
                continue

            end = min(start + BATCH_SIZE, len(manifest))
            batch_manifest = manifest.iloc[start:end]

            with np.load(batch_file, allow_pickle=False) as data:
                features = data["features"].copy()
                sample_ids = data["sample_ids"].copy()
                labels = data["labels"].copy()
                statuses = data["statuses"].copy()

            if statuses.ndim != 2 or statuses.shape[1] != 2:
                raise ValueError(
                    f"Unexpected status format in {batch_file}"
                )

            batch_changed = False

            for status_index in range(len(statuses)):
                sample_id = str(statuses[status_index, 0])
                status = str(statuses[status_index, 1])

                if status != "no_face_detected":
                    continue

                matching = batch_manifest[
                    batch_manifest["sample_id"].astype(str) == sample_id
                ]

                if len(matching) != 1:
                    raise ValueError(
                        f"Could not uniquely find {sample_id} in manifest"
                    )

                row = matching.iloc[0]
                video_path = config.video_root / str(row["video_path"])

                attempted += 1

                try:
                    new_features = extractor.extract_from_video(video_path)

                    if new_features is None:
                        still_failed += 1
                        continue

                    if new_features.shape != (FEATURE_COUNT,):
                        raise ValueError(
                            f"Unexpected feature shape: {new_features.shape}"
                        )

                    if sample_id in sample_ids:
                        raise ValueError(
                            f"Duplicate feature ID already exists: {sample_id}"
                        )

                    features = np.vstack(
                        [features, new_features.reshape(1, -1)]
                    ).astype(np.float32)

                    sample_ids = np.append(sample_ids, sample_id)
                    labels = np.append(labels, str(row["emotion"]))

                    statuses[status_index, 1] = "success"

                    recovered += 1
                    batch_changed = True

                    print(f"Recovered: {sample_id}")

                except Exception as exc:
                    print(f"Error on {sample_id}: {exc}")
                    still_failed += 1

            if batch_changed:
                backup_path = BACKUP_DIR / split / batch_file.name
                backup_path.parent.mkdir(parents=True, exist_ok=True)

                if not backup_path.exists():
                    shutil.copy2(batch_file, backup_path)

                # Write to a temporary file before replacing the original.
                temp_file = batch_file.with_name(
                    batch_file.stem + ".tmp"
                )

                with open(temp_file, "wb") as file:
                    np.savez_compressed(
                        file,
                        features=features,
                        sample_ids=sample_ids,
                        labels=labels,
                        statuses=statuses,
                    )

                temp_file.replace(batch_file)

                print(f"Updated batch: {batch_file.name}")

    print(f"\n{split} summary:")
    print("Retried:", attempted)
    print("Recovered:", recovered)
    print("Still unavailable or failed:", still_failed)

    return attempted, recovered, still_failed


def main():
    config = FacialConfig()
    config.validate()

    totals = [0, 0, 0]

    for split in SPLITS:
        result = process_split(config, split)
        totals = [a + b for a, b in zip(totals, result)]

    print("\n========== FINAL RECOVERY SUMMARY ==========")
    print("Total retried:", totals[0])
    print("Total recovered:", totals[1])
    print("Still unavailable or failed:", totals[2])
    print("Backups:", BACKUP_DIR)


if __name__ == "__main__":
    main()
