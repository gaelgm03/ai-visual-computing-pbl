class FaceRecognitionEvaluator:
    """
    Evaluation engine for facial recognition models using similarity scores.
    """

    def __init__(self, threshold: float = 0.6):
        """
        Args:
            threshold: Similarity threshold for match decision
        """
        self.threshold = threshold

    def evaluate(
        self,
        similarities: List[float],
        labels: List[int],
        plot: bool = True
    ) -> EvaluationResult:
        """
        Args:
            similarities: Similarity scores from ScoreFusion (0–1)
            labels: Ground truth labels (1 = genuine, 0 = impostor)
            plot: Whether to generate charts

        Returns:
            EvaluationResult
        """

        similarities = np.asarray(similarities, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)

        # -----------------------------
        # Predictions
        # -----------------------------
        predictions = (similarities >= self.threshold).astype(int)

        # -----------------------------
        # Core Metrics
        # -----------------------------
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        # -----------------------------
        # FAR / TAR
        # -----------------------------
        false_accepts = np.sum((predictions == 1) & (labels == 0))
        false_rejects = np.sum((predictions == 0) & (labels == 1))

        total_impostors = np.sum(labels == 0)
        total_genuine = np.sum(labels == 1)

        far = false_accepts / max(total_impostors, 1)
        tar = 1.0 - (false_rejects / max(total_genuine, 1))

        # -----------------------------
        # Confusion Matrix
        # -----------------------------
        cm = confusion_matrix(labels, predictions)

        if plot:
            self._plot_similarity_distribution(similarities, labels)
            self._plot_confusion_matrix(cm)
            self._plot_roc_curve(labels, similarities)
            self._plot_metrics_summary(
                accuracy, precision, recall, f1, far, tar
            )

        return EvaluationResult(
            similarities=similarities,
            predictions=predictions,
            labels=labels,
            threshold=self.threshold,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            far=far,
            tar=tar,
            confusion_matrix=cm,
        )