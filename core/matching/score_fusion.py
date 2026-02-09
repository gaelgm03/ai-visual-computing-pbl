"""
Score Fusion: Combine geometric and descriptor scores into a final decision.

DS-1 implementation of the ScoreFusion interface defined in interfaces.py.

Includes:
  - WeightedFusion: Original 2-input fusion (geometric + descriptor)
  - MultiModalFusion: N-input fusion supporting ArcFace embedding + geometric + descriptor

Reference: CS2DS-share.md §5.3
"""

import warnings
from typing import Dict

import numpy as np

from core.matching.interfaces import ScoreFusion, MatchResult


class WeightedFusion(ScoreFusion):
    """
    Weighted linear combination of geometric and descriptor scores.

    final_score = α * geometric_score + β * descriptor_score
    is_match = final_score >= threshold

    Where α + β = 1.0 (normalized if not).
    """

    def __init__(self, config: dict):
        self.geo_weight = config.get("geometric_weight", 0.4)
        self.desc_weight = config.get("descriptor_weight", 0.6)
        self.threshold = config.get("accept_threshold", 0.65)

        # Validate weights sum to ~1.0
        total = self.geo_weight + self.desc_weight
        if abs(total - 1.0) > 0.01:
            warnings.warn(
                f"Fusion weights sum to {total:.3f}, not 1.0. Normalizing."
            )
            self.geo_weight /= total
            self.desc_weight /= total

    def fuse(
        self, geometric_result: MatchResult, descriptor_result: MatchResult
    ) -> MatchResult:
        """
        Combine geometric and descriptor match results.

        Args:
            geometric_result:  MatchResult from GeometricMatcher
            descriptor_result: MatchResult from DescriptorMatcher

        Returns:
            MatchResult with final fused score and accept/reject decision.
        """
        geo_score = geometric_result.score
        desc_score = descriptor_result.score

        final_score = self.geo_weight * geo_score + self.desc_weight * desc_score
        final_score = float(np.clip(final_score, 0.0, 1.0))

        is_match = final_score >= self.threshold

        return MatchResult(
            score=final_score,
            details={
                "method": "weighted_fusion",
                "geometric_score": geo_score,
                "descriptor_score": desc_score,
                "geometric_weight": self.geo_weight,
                "descriptor_weight": self.desc_weight,
                "threshold": self.threshold,
                "geometric_details": geometric_result.details,
                "descriptor_details": descriptor_result.details,
            },
            is_match=is_match,
        )


class MultiModalFusion:
    """
    N-input weighted fusion supporting ArcFace embedding + geometric + descriptor.

    final_score = Σ (weight_i * score_i) for all active channels
    is_match = final_score >= threshold

    Default weights prioritize ArcFace embeddings (the primary identity signal)
    over geometric shape matching (supplementary) and MASt3R descriptors (disabled).

    Args:
        config: Dictionary with keys:
            - embedding_weight: Weight for ArcFace embedding score (default 0.7)
            - geometric_weight: Weight for geometric score (default 0.3)
            - descriptor_weight: Weight for MASt3R descriptor score (default 0.0)
            - accept_threshold: Decision threshold (default 0.55)
    """

    def __init__(self, config: dict):
        self.weights = {
            "embedding": config.get("embedding_weight", 0.7),
            "geometric": config.get("geometric_weight", 0.3),
            "descriptor": config.get("descriptor_weight", 0.0),
        }
        self.threshold = config.get("accept_threshold", 0.55)

        # Normalize weights of active channels (weight > 0)
        active_total = sum(w for w in self.weights.values() if w > 0)
        if active_total > 0 and abs(active_total - 1.0) > 0.01:
            warnings.warn(
                f"Active fusion weights sum to {active_total:.3f}, not 1.0. Normalizing."
            )
            for key in self.weights:
                if self.weights[key] > 0:
                    self.weights[key] /= active_total

    def fuse(self, results: Dict[str, MatchResult]) -> MatchResult:
        """
        Combine multiple match results with weighted fusion.

        Args:
            results: Dictionary mapping channel name to MatchResult.
                     Keys should be from: "embedding", "geometric", "descriptor".
                     Missing channels are skipped (weight redistributed).

        Returns:
            MatchResult with final fused score and accept/reject decision.
        """
        weighted_sum = 0.0
        total_weight = 0.0
        component_scores = {}
        component_details = {}

        for channel, weight in self.weights.items():
            if weight <= 0 or channel not in results:
                continue

            result = results[channel]
            weighted_sum += weight * result.score
            total_weight += weight
            component_scores[f"{channel}_score"] = result.score
            component_details[f"{channel}_details"] = result.details

        # Normalize if not all channels present
        if total_weight > 0:
            final_score = weighted_sum / total_weight
        else:
            final_score = 0.0

        final_score = float(np.clip(final_score, 0.0, 1.0))
        is_match = final_score >= self.threshold

        details = {
            "method": "multi_modal_fusion",
            "threshold": self.threshold,
            "weights": self.weights.copy(),
            "active_weight_total": total_weight,
        }
        details.update(component_scores)
        details.update(component_details)

        return MatchResult(
            score=final_score,
            details=details,
            is_match=is_match,
        )
