
from pathlib import Path
import numpy as np

OUTPUT = Path(r"D:\MELD_Facial_Output")
SPLITS = ("train", "dev", "test")
FEATURE_COUNT = 1912

for split in SPLITS:
    split_dir = OUTPUT / split
    all_ids = []
    total_features = 0

    for batch_file in sorted(split_dir.glob("batch_[0-9][0-9][0-9][0-9].npz")):
        with np.load(batch_file, allow_pickle=False) as data:
            features = data["features"]
            sample_ids = data["sample_ids"]
            labels = data["labels"]
            statuses = data["statuses"]

            assert features.ndim == 2
            assert features.shape[1] == FEATURE_COUNT
            assert len(features) == len(sample_ids) == len(labels)
            assert np.isfinite(features).all()
            assert len(set(sample_ids)) == len(sample_ids)
            assert statuses.ndim == 2 and statuses.shape[1] == 2

            success_ids = set(statuses[statuses[:, 1] == "success", 0])

            assert success_ids == set(sample_ids), (
                f"Success status mismatch in {batch_file.name}"
            )

            all_ids.extend(sample_ids.tolist())
            total_features += len(features)

    assert len(all_ids) == len(set(all_ids)), (
        f"Duplicate sample IDs found in {split}"
    )

    print(
        f"{split}: VALID | "
        f"Features: {total_features} | "
        f"Unique IDs: {len(set(all_ids))}"
    )

print("\nAll feature batches validated successfully.")
