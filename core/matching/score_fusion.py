"""
Score Fusion: Combine geometric and descriptor scores into a final decision.

DS-1 implementation of the ScoreFusion interface defined in interfaces.py.

Reference: CS2DS-share.md §5.3
"""

import warnings
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
