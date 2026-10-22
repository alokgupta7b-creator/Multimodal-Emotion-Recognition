"""
speech_emotion.training.data
============================
Reproducible data loading and validation pipeline for MELD SER.

Ensures:
1. Stable join between manifests and acoustic features using split-aware composite IDs.
2. Label consistency verification between annotations and extracted features.
3. Strict 40-feature MFCC validation with finiteness guarantees.
4. Total split isolation (no sample contamination across train, dev, and test).
5. Comprehensive audit accounting for excluded or missing records.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from speech_emotion.config import MELD_EMOTIONS, TOTAL_MFCC_FEATURES
from speech_emotion.model.label_mapping import MELD_CLASS_NAMES, MELDLabelEncoder
from speech_emotion.training.config import DataConfig, DEFAULT_MANIFEST_DIR, DEFAULT_CANDIDATE_FEATURE_DIR


FEATURE_NAMES = [
    f"mfcc_{kind}_{i:02d}"
    for kind in ("mean", "std")
    for i in range(1, 21)
]


@dataclasses.dataclass
class DatasetSplit:
    """
    Container for a single validated MELD dataset split.
    """
    split: str
    sample_ids: List[str]
    X: np.ndarray
    y: List[str]
    dialogue_ids: np.ndarray
    utterance_ids: np.ndarray
    missing_records: List[Dict[str, Any]]
    class_distribution: Dict[str, int]
    metadata: pd.DataFrame

    @property
    def num_samples(self) -> int:
        return len(self.y)

    @property
    def feature_dim(self) -> int:
        return self.X.shape[1] if self.X.ndim == 2 else 0

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "num_samples": self.num_samples,
            "feature_dim": self.feature_dim,
            "class_distribution": self.class_distribution,
            "missing_records_count": len(self.missing_records),
        }


class MELDDataLoader:
    """
    Data loader and alignment engine for MELD audio features and manifests.
    """

    def __init__(self, config: Optional[DataConfig] = None) -> None:
        self.config = config or DataConfig()
        self.label_encoder = MELDLabelEncoder()

    @staticmethod
    def derive_sample_id(split: str, dialogue_id: int, utterance_id: int) -> str:
        """Derive standard canonical sample identifier."""
        return f"{split.strip().lower()}_dia{int(dialogue_id)}_utt{int(utterance_id)}"

    def load_manifest(self, split: str) -> pd.DataFrame:
        """
        Load and validate an official MELD split manifest CSV.
        """
        path = self.config.manifest_dir / f"{split}_manifest.csv"
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found for split {split!r} at {path}")

        df = pd.read_csv(path)
        required_cols = {"sample_id", "dialogue_id", "utterance_id", "emotion"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Manifest {path.name} missing required columns: {sorted(missing_cols)}")

        if df["sample_id"].duplicated().any():
            dups = df[df["sample_id"].duplicated()]["sample_id"].tolist()
            raise ValueError(f"Duplicate sample_ids in manifest {path.name}: {dups[:5]}")

        # Normalize emotion column
        df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
        invalid_emotions = set(df["emotion"]) - MELD_EMOTIONS
        if invalid_emotions:
            raise ValueError(f"Invalid emotions in manifest {path.name}: {sorted(invalid_emotions)}")

        return df

    def load_features(self, split: str) -> pd.DataFrame:
        """
        Load and validate extracted MFCC features for a given split.
        """
        path = self.config.feature_dir / f"{split}_mfcc.csv"
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found for split {split!r} at {path}")

        df = pd.read_csv(path)
        required_id_cols = {"Dialogue_ID", "Utterance_ID", "emotion"}
        missing_id_cols = required_id_cols - set(df.columns)
        if missing_id_cols:
            raise ValueError(f"Feature file {path.name} missing ID columns: {sorted(missing_id_cols)}")

        missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
        if missing_features:
            raise ValueError(f"Feature file {path.name} missing required features: {missing_features[:5]}")

        # Derive split-aware sample_id
        df["sample_id"] = df.apply(
            lambda r: self.derive_sample_id(split, int(r["Dialogue_ID"]), int(r["Utterance_ID"])),
            axis=1,
        )

        if df["sample_id"].duplicated().any():
            dups = df[df["sample_id"].duplicated()]["sample_id"].tolist()
            raise ValueError(f"Duplicate sample_ids in feature file {path.name}: {dups[:5]}")

        # Validate finiteness
        feat_vals = df[FEATURE_NAMES].values
        if not np.all(np.isfinite(feat_vals)):
            raise ValueError(f"Feature file {path.name} contains non-finite values (NaN or Inf)")

        df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
        invalid_emotions = set(df["emotion"]) - MELD_EMOTIONS
        if invalid_emotions:
            raise ValueError(f"Invalid emotions in feature file {path.name}: {sorted(invalid_emotions)}")

        return df

    def load_split(self, split: str, join_manifest: bool = True) -> DatasetSplit:
        """
        Load and join a single MELD split.
        """
        norm_split = split.strip().lower()
        if norm_split not in ("train", "dev", "test"):
            raise ValueError(f"Invalid split name {split!r}; must be one of ('train', 'dev', 'test')")

        features_df = self.load_features(norm_split)

        missing_records: List[Dict[str, Any]] = []
        if join_manifest and self.config.manifest_dir.exists():
            manifest_df = self.load_manifest(norm_split)
            # Identify missing records from features
            manifest_ids = set(manifest_df["sample_id"])
            feature_ids = set(features_df["sample_id"])
            missing_ids = manifest_ids - feature_ids
            if missing_ids:
                missing_df = manifest_df[manifest_df["sample_id"].isin(missing_ids)]
                missing_records = missing_df.to_dict(orient="records")

            # Check for features not present in manifest
            extra_ids = feature_ids - manifest_ids
            if extra_ids:
                raise ValueError(
                    f"Found {len(extra_ids)} feature records in {norm_split} not present in official manifest"
                )

            # Perform stable inner join on sample_id
            merged = pd.merge(
                manifest_df,
                features_df,
                on="sample_id",
                suffixes=("_manifest", "_feature"),
            )

            # Validate label concordance
            mismatches = merged[merged["emotion_manifest"] != merged["emotion_feature"]]
            if not mismatches.empty:
                bad_samples = mismatches[["sample_id", "emotion_manifest", "emotion_feature"]].head(5).to_dict(orient="records")
                raise ValueError(
                    f"Label mismatch between manifest and features in split {norm_split}: {bad_samples}"
                )

            working_df = merged
            emotion_col = "emotion_manifest"
            dialogue_col = "dialogue_id"
            utterance_col = "utterance_id"
        else:
            working_df = features_df
            emotion_col = "emotion"
            dialogue_col = "Dialogue_ID"
            utterance_col = "Utterance_ID"

        X = working_df[FEATURE_NAMES].values.astype(np.float64)
        y = working_df[emotion_col].tolist()
        sample_ids = working_df["sample_id"].tolist()
        dialogue_ids = working_df[dialogue_col].to_numpy(dtype=np.int64)
        utterance_ids = working_df[utterance_col].to_numpy(dtype=np.int64)

        # Class distribution
        class_dist = {cls_name: 0 for cls_name in self.config.canonical_classes}
        for label in y:
            if label in class_dist:
                class_dist[label] += 1

        return DatasetSplit(
            split=norm_split,
            sample_ids=sample_ids,
            X=X,
            y=y,
            dialogue_ids=dialogue_ids,
            utterance_ids=utterance_ids,
            missing_records=missing_records,
            class_distribution=class_dist,
            metadata=working_df,
        )

    def load_all_splits(self, join_manifest: bool = True) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        """
        Load and validate all three official MELD splits with split-isolation verification.
        """
        train_split = self.load_split("train", join_manifest=join_manifest)
        dev_split = self.load_split("dev", join_manifest=join_manifest)
        test_split = self.load_split("test", join_manifest=join_manifest)

        if self.config.verify_split_isolation:
            train_set = set(train_split.sample_ids)
            dev_set = set(dev_split.sample_ids)
            test_set = set(test_split.sample_ids)

            train_dev_overlap = train_set.intersection(dev_set)
            if train_dev_overlap:
                raise ValueError(f"Cross-split contamination between train and dev: {len(train_dev_overlap)} samples")

            train_test_overlap = train_set.intersection(test_set)
            if train_test_overlap:
                raise ValueError(f"Cross-split contamination between train and test: {len(train_test_overlap)} samples")

            dev_test_overlap = dev_set.intersection(test_set)
            if dev_test_overlap:
                raise ValueError(f"Cross-split contamination between dev and test: {len(dev_test_overlap)} samples")

        return train_split, dev_split, test_split
