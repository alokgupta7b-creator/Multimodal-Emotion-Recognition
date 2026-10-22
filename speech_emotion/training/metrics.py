"""
speech_emotion.training.metrics
===============================
Standardized evaluation metrics for Speech Emotion Recognition (SER).

Implements:
1. Canonical 7-class metrics computation using scikit-learn.
2. Deterministic label order handling (even if some classes are unpredicted).
3. Accuracy, balanced accuracy, macro F1, weighted F1, macro precision, macro recall.
4. Per-class precision, recall, F1, and support.
5. Confusion matrix calculation and ASCII/Markdown formatting.
6. Latency, duration, and model footprint tracking.
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from speech_emotion.model.label_mapping import MELD_CLASS_NAMES


@dataclasses.dataclass
class EvaluationResult:
    """
    Standardized container for SER evaluation metrics.
    """
    model_name: str
    split_name: str
    num_samples: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    per_class_metrics: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]
    classes: List[str]
    training_duration_sec: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    model_size_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a JSON-serializable dictionary."""
        return {
            "model_name": self.model_name,
            "split_name": self.split_name,
            "num_samples": self.num_samples,
            "accuracy": round(float(self.accuracy), 4),
            "balanced_accuracy": round(float(self.balanced_accuracy), 4),
            "macro_f1": round(float(self.macro_f1), 4),
            "weighted_f1": round(float(self.weighted_f1), 4),
            "macro_precision": round(float(self.macro_precision), 4),
            "macro_recall": round(float(self.macro_recall), 4),
            "classes": self.classes,
            "per_class_metrics": {
                cls_name: {k: round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)
                           for k, v in m.items()}
                for cls_name, m in self.per_class_metrics.items()
            },
            "confusion_matrix": self.confusion_matrix,
            "training_duration_sec": round(float(self.training_duration_sec), 3) if self.training_duration_sec else None,
            "inference_latency_ms_per_sample": round(float(self.inference_latency_ms), 3) if self.inference_latency_ms else None,
            "model_size_bytes": self.model_size_bytes,
        }

    def to_markdown(self) -> str:
        """Render evaluation result as a formatted GitHub-style Markdown report."""
        lines = [
            f"### SER Evaluation Report: {self.model_name} on `{self.split_name}`",
            "",
            "#### Summary Metrics",
            "| Metric | Value |",
            "| :--- | :---: |",
            f"| **Samples** | {self.num_samples:,} |",
            f"| **Accuracy** | {self.accuracy * 100:.2f}% |",
            f"| **Balanced Accuracy** | {self.balanced_accuracy * 100:.2f}% |",
            f"| **Macro F1** | {self.macro_f1 * 100:.2f}% |",
            f"| **Weighted F1** | {self.weighted_f1 * 100:.2f}% |",
            f"| **Macro Precision** | {self.macro_precision * 100:.2f}% |",
            f"| **Macro Recall** | {self.macro_recall * 100:.2f}% |",
        ]

        if self.training_duration_sec is not None:
            lines.append(f"| **Training Duration** | {self.training_duration_sec:.2f} s |")
        if self.inference_latency_ms is not None:
            lines.append(f"| **Inference Latency** | {self.inference_latency_ms:.3f} ms / sample |")
        if self.model_size_bytes is not None:
            lines.append(f"| **Model Size** | {self.model_size_bytes / 1024:.1f} KB |")

        lines.extend([
            "",
            "#### Per-Class Performance",
            "| Emotion Class | Precision | Recall | F1-Score | Support |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])

        for cls_name in self.classes:
            m = self.per_class_metrics.get(cls_name, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
            prec = m.get("precision", 0.0) * 100
            rec = m.get("recall", 0.0) * 100
            f1 = m.get("f1-score", 0.0) * 100
            supp = int(m.get("support", 0))
            lines.append(f"| **{cls_name}** | {prec:.2f}% | {rec:.2f}% | {f1:.2f}% | {supp:,} |")

        lines.extend([
            "",
            "#### Confusion Matrix (Rows: True, Columns: Predicted)",
            "| True \\ Pred | " + " | ".join(self.classes) + " |",
            "| :--- | " + " | ".join([":---:"] * len(self.classes)) + " |",
        ])

        for i, cls_name in enumerate(self.classes):
            row_vals = self.confusion_matrix[i]
            lines.append(f"| **{cls_name}** | " + " | ".join(f"{v:,}" for v in row_vals) + " |")

        return "\n".join(lines)


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    model_name: str = "SER_Model",
    split_name: str = "eval",
    classes: Optional[Sequence[str]] = None,
    training_duration_sec: Optional[float] = None,
    inference_latency_ms: Optional[float] = None,
    model_size_bytes: Optional[int] = None,
) -> EvaluationResult:
    """
    Compute comprehensive classification metrics adhering to canonical label order.
    """
    labels = list(classes) if classes is not None else list(MELD_CLASS_NAMES)
    # Clean inputs
    clean_true = [str(y).strip().lower() for y in y_true]
    clean_pred = [str(y).strip().lower() for y in y_pred]

    acc = float(accuracy_score(clean_true, clean_pred))
    bal_acc = float(balanced_accuracy_score(clean_true, clean_pred))
    macro_f1 = float(f1_score(clean_true, clean_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(clean_true, clean_pred, labels=labels, average="weighted", zero_division=0))
    macro_prec = float(precision_score(clean_true, clean_pred, labels=labels, average="macro", zero_division=0))
    macro_rec = float(recall_score(clean_true, clean_pred, labels=labels, average="macro", zero_division=0))

    # Per-class metrics
    raw_report = classification_report(
        clean_true,
        clean_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for cls_name in labels:
        if cls_name in raw_report:
            per_class[cls_name] = {
                "precision": float(raw_report[cls_name]["precision"]),
                "recall": float(raw_report[cls_name]["recall"]),
                "f1-score": float(raw_report[cls_name]["f1-score"]),
                "support": int(raw_report[cls_name]["support"]),
            }
        else:
            per_class[cls_name] = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}

    # Confusion matrix with fixed canonical ordering
    cm = confusion_matrix(clean_true, clean_pred, labels=labels)
    cm_list = cm.tolist()

    return EvaluationResult(
        model_name=model_name,
        split_name=split_name,
        num_samples=len(clean_true),
        accuracy=acc,
        balanced_accuracy=bal_acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        per_class_metrics=per_class,
        confusion_matrix=cm_list,
        classes=labels,
        training_duration_sec=training_duration_sec,
        inference_latency_ms=inference_latency_ms,
        model_size_bytes=model_size_bytes,
    )
