"""Evaluation metrics implemented with numpy only (no sklearn dependency).

Convention: a prediction of ``-1`` means "no prediction" (e.g. every modality
was missing). It counts as wrong for accuracy and is excluded from the
confusion matrix columns, so the per-class recall sees it as a miss.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class. ``-1`` predictions are skipped."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        if 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def per_class_prf(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, np.ndarray]:
    """Per-class precision, recall, F1 and support."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    support = np.zeros(n_classes, dtype=int)
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        support[c] = np.sum(y_true == c)
        precision[c] = tp / (tp + fp) if (tp + fp) else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) else 0.0
        denom = precision[c] + recall[c]
        f1[c] = 2 * precision[c] * recall[c] / denom if denom else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Unweighted mean F1 over classes that have at least one true sample."""
    prf = per_class_prf(y_true, y_pred, n_classes)
    present = prf["support"] > 0
    return float(prf["f1"][present].mean()) if present.any() else float("nan")
