"""
speech_emotion.training.train
=============================
Training, hyperparameter tuning, and benchmark selection pipeline
for Speech Emotion Recognition.

Protocol:
1. All models and preprocessing (StandardScaler) are fitted STRICTLY on train split.
2. Model selection and hyperparameter exploration use the dev split ONLY.
3. The test split is NEVER touched during model selection or tuning.
4. Comprehensive logging of training duration, convergence, and evaluation metrics.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np

from speech_emotion.model.classifier import SERClassifier, SERConfig
from speech_emotion.training.config import TrainingConfig, DEFAULT_MODELS_DIR
from speech_emotion.training.data import DatasetSplit
from speech_emotion.training.metrics import EvaluationResult, compute_metrics

logger = logging.getLogger("speech_emotion.training")


def train_single_model(
    config: SERConfig,
    train_data: DatasetSplit,
) -> Tuple[SERClassifier, float]:
    """
    Train a single SERClassifier on the training split and measure training duration.
    """
    model = SERClassifier(config=config)
    start_time = time.perf_counter()
    model.fit(train_data.X, train_data.y)
    duration = time.perf_counter() - start_time
    return model, duration


def evaluate_split(
    model: SERClassifier,
    split_data: DatasetSplit,
    model_name: str,
    training_duration_sec: Optional[float] = None,
    saved_model_path: Optional[Path] = None,
) -> EvaluationResult:
    """
    Evaluate a fitted SERClassifier on a dataset split and measure latency.
    """
    # Measure inference latency over the split
    start_time = time.perf_counter()
    y_pred = model.predict(split_data.X)
    total_infer_time = time.perf_counter() - start_time
    latency_ms_per_sample = (total_infer_time / len(split_data.X)) * 1000.0 if len(split_data.X) > 0 else 0.0

    model_size = os.path.getsize(saved_model_path) if saved_model_path and saved_model_path.exists() else None

    return compute_metrics(
        y_true=split_data.y,
        y_pred=y_pred,
        model_name=model_name,
        split_name=split_data.split,
        classes=model.classes_,
        training_duration_sec=training_duration_sec,
        inference_latency_ms=latency_ms_per_sample,
        model_size_bytes=model_size,
    )


class SERModelTrainer:
    """
    Orchestrates baseline training, candidate model comparison, and dev-set model selection.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        random_state: int = 42,
    ) -> None:
        self.output_dir = Path(output_dir or DEFAULT_MODELS_DIR)
        self.random_state = random_state
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def define_candidate_configs(self) -> Dict[str, SERConfig]:
        """
        Define candidate models for hyperparameter and architecture benchmarking.
        """
        return {
            "logreg_balanced_C1.0": SERConfig(
                classifier_type="logistic_regression",
                class_weight="balanced",
                C=1.0,
                max_iter=2000,
                random_state=self.random_state,
            ),
            "logreg_unweighted_C1.0": SERConfig(
                classifier_type="logistic_regression",
                class_weight=None,
                C=1.0,
                max_iter=2000,
                random_state=self.random_state,
            ),
            "logreg_balanced_C0.1": SERConfig(
                classifier_type="logistic_regression",
                class_weight="balanced",
                C=0.1,
                max_iter=2000,
                random_state=self.random_state,
            ),
            "svm_rbf_balanced_C1.0": SERConfig(
                classifier_type="svm",
                svm_kernel="rbf",
                class_weight="balanced",
                C=1.0,
                random_state=self.random_state,
            ),
            "svm_linear_balanced_C0.1": SERConfig(
                classifier_type="svm",
                svm_kernel="linear",
                class_weight="balanced",
                C=0.1,
                max_iter=2000,
                random_state=self.random_state,
            ),
            "mlp_relu_balanced": SERConfig(
                classifier_type="mlp",
                mlp_hidden_layers=(64, 32),
                mlp_alpha=0.001,
                max_iter=500,
                mlp_early_stopping=True,
                random_state=self.random_state,
            ),
        }

    def run_benchmark(
        self,
        train_data: DatasetSplit,
        dev_data: DatasetSplit,
        candidate_configs: Optional[Dict[str, SERConfig]] = None,
        selection_metric: str = "macro_f1",
    ) -> Tuple[str, SERClassifier, Dict[str, EvaluationResult]]:
        """
        Fit all candidate models on train_data and evaluate on dev_data.
        Selects the best model configuration based on dev split performance.
        """
        configs = candidate_configs or self.define_candidate_configs()
        results: Dict[str, EvaluationResult] = {}
        fitted_models: Dict[str, SERClassifier] = {}

        print(f"\n{'='*70}")
        print(f"Starting SER Model Benchmark on MELD ({len(configs)} configurations)")
        print(f"Train split: {train_data.num_samples:,} samples | Dev split: {dev_data.num_samples:,} samples")
        print(f"Selection metric: {selection_metric}")
        print(f"{'='*70}\n")

        for name, config in configs.items():
            print(f"--> Training {name} ({config.classifier_type})...", end=" ", flush=True)
            model, duration = train_single_model(config, train_data)
            fitted_models[name] = model

            # Save candidate checkpoint
            temp_path = self.output_dir / f"checkpoint_{name}.joblib"
            model.save(temp_path, overwrite=True)

            # Evaluate on dev
            eval_res = evaluate_split(
                model=model,
                split_data=dev_data,
                model_name=name,
                training_duration_sec=duration,
                saved_model_path=temp_path,
            )
            results[name] = eval_res

            print(
                f"Done ({duration:.1f}s) | "
                f"Dev Acc: {eval_res.accuracy*100:.2f}% | "
                f"Dev Macro F1: {eval_res.macro_f1*100:.2f}% | "
                f"Dev Weighted F1: {eval_res.weighted_f1*100:.2f}%"
            )

        # Select best model based on selection_metric
        best_name = max(
            results.keys(),
            key=lambda k: getattr(results[k], selection_metric),
        )
        best_model = fitted_models[best_name]

        print(f"\n{'='*70}")
        print(f"Benchmark Complete. Top Model on Dev Split: {best_name}")
        best_res = results[best_name]
        print(
            f"Dev Accuracy: {best_res.accuracy*100:.2f}% | "
            f"Dev Balanced Acc: {best_res.balanced_accuracy*100:.2f}% | "
            f"Dev Macro F1: {best_res.macro_f1*100:.2f}%"
        )
        print(f"{'='*70}\n")

        # Save baseline model explicitly
        baseline_name = "logreg_balanced_C1.0"
        if baseline_name in fitted_models:
            baseline_path = self.output_dir / "ser_baseline_logreg.joblib"
            fitted_models[baseline_name].save(baseline_path, overwrite=True)

        # Save selected best model explicitly
        best_path = self.output_dir / "ser_best_model.joblib"
        best_model.save(best_path, overwrite=True)

        # Generate benchmark report
        self._save_benchmark_report(results, best_name, selection_metric)

        return best_name, best_model, results

    def _save_benchmark_report(
        self,
        results: Dict[str, EvaluationResult],
        best_name: str,
        selection_metric: str,
    ) -> None:
        """Save JSON and Markdown benchmark comparison reports."""
        json_path = self.output_dir / "dev_benchmark_report.json"
        md_path = self.output_dir / "dev_benchmark_report.md"

        data = {
            "selection_metric": selection_metric,
            "best_model": best_name,
            "models": {k: res.to_dict() for k, res in results.items()},
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Markdown report
        lines = [
            "# MELD Speech Emotion Recognition (SER) — Development Benchmark",
            "",
            f"**Selected Top Model**: `{best_name}` (selected by `{selection_metric}`)",
            "",
            "## Model Comparison on Development Split",
            "",
            "| Model Configuration | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Training Time | Latency / Sample | Model Size |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        for name, res in results.items():
            is_best = " **(Selected)**" if name == best_name else ""
            train_t = f"{res.training_duration_sec:.1f} s" if res.training_duration_sec else "N/A"
            lat = f"{res.inference_latency_ms:.3f} ms" if res.inference_latency_ms else "N/A"
            size = f"{res.model_size_bytes/1024:.1f} KB" if res.model_size_bytes else "N/A"
            lines.append(
                f"| **`{name}`**{is_best} | {res.accuracy*100:.2f}% | {res.balanced_accuracy*100:.2f}% | "
                f"{res.macro_f1*100:.2f}% | {res.weighted_f1*100:.2f}% | {train_t} | {lat} | {size} |"
            )

        lines.extend([
            "",
            "## Top Model Detailed Breakdown",
            "",
            results[best_name].to_markdown(),
        ])

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
