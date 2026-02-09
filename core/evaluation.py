"""
Evaluation Module for Face Recognition System.

Provides metrics computation and visualization for similarity-based
facial recognition evaluation. Consolidates the evaluation module
originally authored by Evelyn (DS-2).

Usage:
    from core.evaluation import FaceRecognitionEvaluator, EvaluationResult

    evaluator = FaceRecognitionEvaluator(threshold=0.65)
    result = evaluator.evaluate(similarities, labels, plot=True)
    print(f"Accuracy: {result.accuracy:.3f}, EER: {result.eer:.3f}")
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""

    similarities: np.ndarray
    predictions: np.ndarray
    labels: np.ndarray
    threshold: float

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    far: float
    tar: float
    eer: float
    eer_threshold: float
    auc_score: float
    confusion_matrix: np.ndarray


class FaceRecognitionEvaluator:
    """
    Evaluation engine for facial recognition models using similarity scores.

    Computes standard biometric metrics (FAR, TAR, EER, AUC) and generates
    four diagnostic plots: similarity distribution, confusion matrix,
    ROC curve, and metrics summary bar chart.
    """

    def __init__(self, threshold: float = 0.65):
        """
        Args:
            threshold: Similarity threshold for match decision.
        """
        self.threshold = threshold

    def evaluate(
        self,
        similarities: List[float],
        labels: List[int],
        plot: bool = True,
        save_dir: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate matching performance.

        Args:
            similarities: Similarity scores from ScoreFusion (0-1).
            labels: Ground truth labels (1 = genuine, 0 = impostor).
            plot: Whether to display charts.
            save_dir: If provided, save plots as PNG to this directory.

        Returns:
            EvaluationResult with all computed metrics.
        """
        similarities = np.asarray(similarities, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)

        # Predictions based on threshold
        predictions = (similarities >= self.threshold).astype(int)

        # Core metrics
        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions, zero_division=0)
        rec = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        # FAR / TAR
        false_accepts = np.sum((predictions == 1) & (labels == 0))
        false_rejects = np.sum((predictions == 0) & (labels == 1))
        total_impostors = np.sum(labels == 0)
        total_genuine = np.sum(labels == 1)

        far = false_accepts / max(total_impostors, 1)
        tar = 1.0 - (false_rejects / max(total_genuine, 1))

        # EER and AUC
        eer_val, eer_thresh = self._compute_eer(labels, similarities)
        fpr, tpr, _ = roc_curve(labels, similarities)
        auc_val = float(auc(fpr, tpr))

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)

        # Visualization
        if plot or save_dir:
            self._plot_all(
                similarities, labels, cm,
                acc, prec, rec, f1, far, tar,
                show=plot, save_dir=save_dir,
            )

        return EvaluationResult(
            similarities=similarities,
            predictions=predictions,
            labels=labels,
            threshold=self.threshold,
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            far=far,
            tar=tar,
            eer=eer_val,
            eer_threshold=eer_thresh,
            auc_score=auc_val,
            confusion_matrix=cm,
        )

    # ------------------------------------------------------------------
    # EER computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_eer(labels, similarities):
        """Find the Equal Error Rate (threshold where FAR = FRR)."""
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        fnr = 1 - tpr
        # Find where FPR and FNR cross
        eer_index = np.argmin(np.abs(fpr - fnr))
        eer = float((fpr[eer_index] + fnr[eer_index]) / 2)
        eer_threshold = float(thresholds[eer_index]) if len(thresholds) > eer_index else 0.5
        return eer, eer_threshold

    # ------------------------------------------------------------------
    # Visualization (based on Evelyn's original plots)
    # ------------------------------------------------------------------

    def _plot_all(
        self, similarities, labels, cm,
        accuracy, precision, recall, f1, far, tar,
        show=True, save_dir=None,
    ):
        """Generate all four diagnostic plots."""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        self._plot_similarity_distribution(
            similarities, labels, show=show,
            save_path=os.path.join(save_dir, "score_distributions.png") if save_dir else None,
        )
        self._plot_confusion_matrix(
            cm, show=show,
            save_path=os.path.join(save_dir, "confusion_matrix.png") if save_dir else None,
        )
        self._plot_roc_curve(
            labels, similarities, show=show,
            save_path=os.path.join(save_dir, "roc_curve.png") if save_dir else None,
        )
        self._plot_metrics_summary(
            accuracy, precision, recall, f1, far, tar, show=show,
            save_path=os.path.join(save_dir, "metrics_summary.png") if save_dir else None,
        )

    def _plot_similarity_distribution(self, similarities, labels, show=True, save_path=None):
        plt.figure(figsize=(8, 5))
        plt.hist(similarities[labels == 1], bins=25, alpha=0.7, label="Genuine", color="green")
        plt.hist(similarities[labels == 0], bins=25, alpha=0.7, label="Impostor", color="red")
        plt.axvline(self.threshold, color="black", linestyle="--", label=f"Threshold ({self.threshold:.2f})")
        plt.xlabel("Similarity Score")
        plt.ylabel("Count")
        plt.title("Similarity Score Distribution")
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close()

    def _plot_confusion_matrix(self, cm, show=True, save_path=None):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Impostor", "Genuine"],
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close()

    def _plot_roc_curve(self, labels, similarities, show=True, save_path=None):
        fpr, tpr, _ = roc_curve(labels, similarities)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
        plt.xlabel("False Accept Rate (FAR)")
        plt.ylabel("True Accept Rate (TAR)")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close()

    def _plot_metrics_summary(self, accuracy, precision, recall, f1, far, tar, show=True, save_path=None):
        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "FAR": far,
            "TAR": tar,
        }
        plt.figure(figsize=(8, 5))
        bars = plt.bar(metrics.keys(), metrics.values(), color=[
            "#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"
        ])
        plt.ylim(0, 1.05)
        plt.ylabel("Score")
        plt.title("Evaluation Metrics Summary")
        plt.xticks(rotation=30)
        # Add value labels on bars
        for bar, val in zip(bars, metrics.values()):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close()
