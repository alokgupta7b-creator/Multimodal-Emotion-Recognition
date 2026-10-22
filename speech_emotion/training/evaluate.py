"""
speech_emotion.training.evaluate
================================
Independent evaluation and reporting pipeline for Speech Emotion Recognition.

Protocol:
1. Evaluates frozen models on specified splits (development or test).
2. Generates comprehensive audit reports in JSON and Markdown formats.
3. Calculates per-class and overall summary metrics using canonical class ordering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from speech_emotion.model.classifier import SERClassifier
from speech_emotion.training.config import DEFAULT_MODELS_DIR
from speech_emotion.training.data import DatasetSplit, MELDDataLoader
from speech_emotion.training.metrics import EvaluationResult
from speech_emotion.training.train import evaluate_split


def evaluate_model_on_split(
    model: Union[SERClassifier, str, Path],
    split_data: DatasetSplit,
    model_name: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    report_prefix: Optional[str] = None,
) -> EvaluationResult:
    """
    Evaluate a fitted SERClassifier or saved artifact on a dataset split.

    Parameters
    ----------
    model:
        Either a fitted SERClassifier instance or a path to a saved .joblib artifact.
    split_data:
        DatasetSplit container holding features, labels, and metadata.
    model_name:
        Optional display name for the model.
    output_dir:
        Directory where evaluation reports will be saved.
    report_prefix:
        Prefix for report files (defaults to split_data.split).

    Returns
    -------
    EvaluationResult
    """
    saved_path: Optional[Path] = None
    if isinstance(model, (str, Path)):
        saved_path = Path(model)
        classifier = SERClassifier.load(saved_path)
        name = model_name or saved_path.stem
    else:
        classifier = model
        name = model_name or "SER_Classifier"

    res = evaluate_split(
        model=classifier,
        split_data=split_data,
        model_name=name,
        saved_model_path=saved_path,
    )

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        prefix = report_prefix or f"{split_data.split}_{name}"

        json_file = out_path / f"{prefix}_metrics.json"
        md_file = out_path / f"{prefix}_report.md"

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(res.to_dict(), f, indent=2)

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# MELD SER Evaluation Report — Split: {split_data.split.upper()}\n\n")
            f.write(res.to_markdown())

    return res
