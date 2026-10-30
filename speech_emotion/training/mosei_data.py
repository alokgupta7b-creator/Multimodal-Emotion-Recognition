"""
speech_emotion.training.mosei_data
==================================
Data loading, interval feature pooling, and label harmonization
for CMU-MOSEI acoustic emotion and sentiment recognition.
"""
from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from speech_emotion.model.label_mapping import MELD_CLASS_NAMES
from speech_emotion.training.data import DatasetSplit

logger = logging.getLogger("speech_emotion.training.mosei_data")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Resolve the CMU-MOSEI dataset directory from an environment variable.
# If not configured, use the repository-relative dataset directory.
DEFAULT_MOSEI_DIR = Path(
    os.environ.get(
        "MOSEI_DIR",
        str(REPO_ROOT / "datasets" / "CMU-MOSEI"),
    )
)

COVAREP_FEATURE_DIM = 74

# Canonical 1-to-1 ontological mapping from MOSEI dominant emotion to canonical 7-class emotion
MOSEI_TO_CANONICAL_EMOTION: Dict[str, str] = {
    "happy": "joy",
    "sad": "sadness",
    "anger": "anger",
    "surprise": "surprise",
    "disgust": "disgust",
    "fear": "fear",
    "none": "neutral",
}

CANONICAL_TO_MOSEI_EMOTION: Dict[str, str] = {
    v: k for k, v in MOSEI_TO_CANONICAL_EMOTION.items()
}

MOSEI_SENTIMENT_CLASSES: Tuple[str, ...] = ("negative", "neutral", "positive")

COVAREP_FEATURE_NAMES = [f"covarep_{i:02d}" for i in range(1, COVAREP_FEATURE_DIM + 1)]


def harmonize_mosei_emotion(label: str) -> str:
    """
    Map CMU-MOSEI dominant emotion to canonical 7-class emotion space.
    """
    norm = str(label).strip().lower()
    if norm not in MOSEI_TO_CANONICAL_EMOTION:
        raise ValueError(
            f"Cannot harmonize CMU-MOSEI emotion {label!r}. "
            f"Must be one of {sorted(MOSEI_TO_CANONICAL_EMOTION.keys())}"
        )
    return MOSEI_TO_CANONICAL_EMOTION[norm]


@dataclasses.dataclass
class MOSEIDataConfig:
    """
    Configuration for CMU-MOSEI manifest and COVAREP feature extraction.
    """
    mosei_dir: Path = DEFAULT_MOSEI_DIR
    csd_path: Optional[Path] = None
    cache_dir: Optional[Path] = None
    expected_features: int = COVAREP_FEATURE_DIM
    target_type: str = "harmonized_emotion"  # 'harmonized_emotion' or 'sentiment_discrete'
    verify_split_isolation: bool = True

    def __post_init__(self) -> None:
        self.mosei_dir = Path(self.mosei_dir)
        if self.csd_path is None:
            self.csd_path = self.mosei_dir / "features" / "CMU_MOSEI_COVAREP.csd"
        else:
            self.csd_path = Path(self.csd_path)

        if self.cache_dir is None:
            self.cache_dir = self.mosei_dir / "features" / "covarep_pooled"
        else:
            self.cache_dir = Path(self.cache_dir)

        if self.target_type not in ("harmonized_emotion", "sentiment_discrete"):
            raise ValueError(
                f"Unsupported target_type: {self.target_type!r}. Must be 'harmonized_emotion' or 'sentiment_discrete'"
            )


class MOSEICovarepExtractor:
    """
    Extracts and temporal-pools utterance-level acoustic features from CMU_MOSEI_COVAREP.csd.
    """

    def __init__(self, csd_path: Path, cache_dir: Path) -> None:
        self.csd_path = Path(csd_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_and_pool_split(
        self,
        manifest_df: pd.DataFrame,
        split: str,
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """
        Extract utterance-level pooled features for all clips in manifest_df.
        Uses cached CSV if available.
        """
        cache_file = self.cache_dir / f"{split}_covarep.csv"
        if cache_file.exists() and not force_recompute:
            logger.info("Loading cached pooled COVAREP features from %s", cache_file)
            return pd.read_csv(cache_file)

        if not self.csd_path.exists():
            raise FileNotFoundError(f"COVAREP CSD archive not found at {self.csd_path}")

        try:
            import h5py
        except ImportError as err:
            raise ImportError("h5py is required to extract features from COVAREP CSD") from err

        rows: List[Dict[str, Any]] = []
        logger.info("Extracting %d clips from %s...", len(manifest_df), self.csd_path.name)

        with h5py.File(self.csd_path, "r") as f:
            data_grp = f["COVAREP"]["data"]

            for _, row in manifest_df.iterrows():
                sample_id = str(row["sample_id"])
                vid = str(row["video_id"])
                start = float(row["start_time"])
                end = float(row["end_time"])

                if vid not in data_grp:
                    continue

                vid_entry = data_grp[vid]
                intervals = vid_entry["intervals"][:]

                # Overlapping frames
                mask = (intervals[:, 0] < end) & (intervals[:, 1] > start)
                matching = np.where(mask)[0]

                if len(matching) == 0:
                    continue

                first_idx = matching[0]
                last_idx = matching[-1] + 1
                raw_feats = vid_entry["features"][first_idx:last_idx]

                # Sanitize non-finite values (handle Inf/NaN in raw COVAREP)
                sanitized = np.nan_to_num(raw_feats, nan=0.0, posinf=0.0, neginf=0.0)

                # Utterance-level temporal mean pooling
                mean_feat = np.mean(sanitized, axis=0)

                record: Dict[str, Any] = {"sample_id": sample_id}
                for dim_idx, col_name in enumerate(COVAREP_FEATURE_NAMES):
                    record[col_name] = float(mean_feat[dim_idx])
                rows.append(record)

        df_out = pd.DataFrame(rows)
        df_out.to_csv(cache_file, index=False)
        logger.info("Saved %d pooled COVAREP feature rows to %s", len(df_out), cache_file)
        return df_out


class MOSEIDataLoader:
    """
    DataLoader and validator for CMU-MOSEI acoustic features and manifests.
    """

    def __init__(self, config: Optional[MOSEIDataConfig] = None) -> None:
        self.config = config or MOSEIDataConfig()
        self.extractor = MOSEICovarepExtractor(
            csd_path=self.config.csd_path,
            cache_dir=self.config.cache_dir,
        )

    def load_manifest(self, split: str) -> pd.DataFrame:
        """Load official CMU-MOSEI manifest CSV."""
        path = self.config.mosei_dir / "manifests" / f"{split}_manifest.csv"
        if not path.exists():
            raise FileNotFoundError(f"CMU-MOSEI manifest not found at {path}")

        df = pd.read_csv(path)
        required = {"sample_id", "video_id", "clip_id", "dominant_emotion", "sentiment_discrete"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CMU-MOSEI manifest missing columns: {sorted(missing)}")
        return df

    def load_split(self, split: str, force_recompute_features: bool = False) -> DatasetSplit:
        """Load and harmonize a single CMU-MOSEI split."""
        norm_split = split.strip().lower()
        if norm_split not in ("train", "dev", "test"):
            raise ValueError(f"Invalid split name {split!r}")

        manifest_df = self.load_manifest(norm_split)
        features_df = self.extractor.extract_and_pool_split(
            manifest_df=manifest_df,
            split=norm_split,
            force_recompute=force_recompute_features,
        )

        merged = pd.merge(manifest_df, features_df, on="sample_id")

        if self.config.target_type == "harmonized_emotion":
            # Harmonize dominant_emotion to canonical 7-class emotion
            y = [harmonize_mosei_emotion(e) for e in merged["dominant_emotion"]]
            classes = list(MELD_CLASS_NAMES)
        elif self.config.target_type == "sentiment_discrete":
            y = [str(s).strip().lower() for s in merged["sentiment_discrete"]]
            classes = list(MOSEI_SENTIMENT_CLASSES)
        else:
            raise ValueError(f"Unknown target_type: {self.config.target_type}")

        X = merged[COVAREP_FEATURE_NAMES].values.astype(np.float64)
        sample_ids = merged["sample_id"].tolist()
        video_ids = np.array(merged["video_id"].astype(str).tolist())
        clip_ids = merged["clip_id"].to_numpy(dtype=np.int64)

        # Finiteness validation
        if not np.all(np.isfinite(X)):
            raise ValueError(f"Non-finite values detected in pooled CMU-MOSEI {norm_split} features")

        class_dist = {cls_name: 0 for cls_name in classes}
        for label in y:
            if label in class_dist:
                class_dist[label] += 1

        missing_records: List[Dict[str, Any]] = []
        if len(merged) < len(manifest_df):
            missing_ids = set(manifest_df["sample_id"]) - set(merged["sample_id"])
            missing_df = manifest_df[manifest_df["sample_id"].isin(missing_ids)]
            missing_records = missing_df.to_dict(orient="records")

        return DatasetSplit(
            split=norm_split,
            sample_ids=sample_ids,
            X=X,
            y=y,
            dialogue_ids=video_ids,
            utterance_ids=clip_ids,
            missing_records=missing_records,
            class_distribution=class_dist,
            metadata=merged,
        )

    def load_all_splits(self, force_recompute_features: bool = False) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        """Load all three official CMU-MOSEI splits with split-isolation verification."""
        train_split = self.load_split("train", force_recompute_features=force_recompute_features)
        dev_split = self.load_split("dev", force_recompute_features=force_recompute_features)
        test_split = self.load_split("test", force_recompute_features=force_recompute_features)

        if self.config.verify_split_isolation:
            train_videos = set(train_split.dialogue_ids)
            dev_videos = set(dev_split.dialogue_ids)
            test_videos = set(test_split.dialogue_ids)

            train_dev_leakage = train_videos.intersection(dev_videos)
            if train_dev_leakage:
                raise ValueError(f"Video leakage between train and dev: {len(train_dev_leakage)} videos")

            train_test_leakage = train_videos.intersection(test_videos)
            if train_test_leakage:
                raise ValueError(f"Video leakage between train and test: {len(train_test_leakage)} videos")

            dev_test_leakage = dev_videos.intersection(test_videos)
            if dev_test_leakage:
                raise ValueError(f"Video leakage between dev and test: {len(dev_test_leakage)} videos")

        return train_split, dev_split, test_split
