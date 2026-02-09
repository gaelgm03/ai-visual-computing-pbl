"""
Matching Interfaces Module

This module defines the abstract interfaces for face matching algorithms.
CS-1 defines these interfaces, and the DS team implements concrete versions.

The matching pipeline has three components:
1. GeometricMatcher - Compares 3D point cloud shapes
2. DescriptorMatcher - Compares learned feature vectors
3. ScoreFusion - Combines both scores into a final decision

Stub implementations are provided for the CS team to use before
the DS team delivers the real implementations.

Usage:
    from core.matching.interfaces import (
        MatchResult,
        GeometricMatcher,
        StubGeometricMatcher,
    )

    # Use stubs for early development
    matcher = StubGeometricMatcher()
    result = matcher.compare(probe_cloud, template_cloud)
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class MatchResult:
    """
    Result of a matching operation.

    Attributes:
        score: Similarity score between 0.0 and 1.0.
               0.0 = completely different (no match)
               1.0 = perfect match (identical)
        details: Dictionary containing algorithm-specific details.
                 Useful for debugging, visualization, and analysis.
                 Examples: {"chamfer_distance": 0.012, "icp_fitness": 0.95}
        is_match: Boolean decision based on threshold comparison.
                  True = the faces are the same person.
                  False = the faces are different people.
    """

    score: float
    details: Dict[str, Any]
    is_match: bool


class GeometricMatcher(ABC):
    """
    Abstract base class for geometric (3D shape) matching.

    Compares two 3D face point clouds to determine shape similarity.
    This measures how geometrically similar two faces are in 3D space.

    DS-1 implements this in: core/matching/geometric_matcher.py

    Typical approach:
        1. Pre-align point clouds (center, optional PCA alignment)
        2. Run ICP (Iterative Closest Point) for fine alignment
        3. Compute distance metric (Chamfer distance)
        4. Convert distance to similarity score [0, 1]
    """

    @abstractmethod
    def compare(
        self, probe_cloud: np.ndarray, template_cloud: np.ndarray
    ) -> MatchResult:
        """
        Compare two 3D point clouds geometrically.

        Args:
            probe_cloud: 3D points from authentication capture.
                         Shape: (N, 3) where N is number of points.
                         Each row is (x, y, z) coordinates.
            template_cloud: 3D points from enrolled template.
                            Shape: (M, 3) where M is number of points.

        Returns:
            MatchResult with geometric similarity score.

        Note:
            - N and M may be different (different point counts)
            - Both clouds should be in the same scale (meters)
            - Pre-alignment (centering) improves results
        """
        pass


class DescriptorMatcher(ABC):
    """
    Abstract base class for descriptor (feature vector) matching.

    Compares learned feature descriptors to determine identity similarity.
    MASt3R produces a descriptor vector for each 3D point, encoding
    local appearance and geometric information.

    DS-1 implements this in: core/matching/descriptor_matcher.py

    Typical approach:
        1. For each descriptor in probe, find nearest neighbor in template
        2. Apply reciprocal matching (mutual nearest neighbor check)
        3. Count match ratio and compute average similarity
        4. Convert to final score [0, 1]
    """

    @abstractmethod
    def compare(
        self,
        probe_desc: np.ndarray,
        template_desc: np.ndarray,
        probe_cloud: np.ndarray,
        template_cloud: np.ndarray,
    ) -> MatchResult:
        """
        Compare descriptor sets from two face captures.

        Args:
            probe_desc: Descriptors from authentication capture.
                        Shape: (N, D) where D is descriptor dimension.
            template_desc: Descriptors from enrolled template.
                           Shape: (M, D).
            probe_cloud: 3D positions of probe points (N, 3).
                         Can be used for spatial filtering.
            template_cloud: 3D positions of template points (M, 3).

        Returns:
            MatchResult with descriptor similarity score.

        Note:
            - MASt3R descriptors are typically unit-normalized
            - Cosine similarity = dot product for unit vectors
            - Spatial positions can help filter outlier matches
        """
        pass


class EmbeddingMatcher(ABC):
    """
    Abstract base class for global face embedding matching.

    Compares pre-extracted identity embeddings (e.g. ArcFace 512-dim)
    to determine if two faces belong to the same person. This is the
    primary identity signal, replacing MASt3R descriptors which lack
    identity discrimination.

    DS-1 implements this in: core/matching/embedding_matcher.py

    Typical approach:
        1. Load pre-extracted embeddings from templates
        2. Compute cosine similarity between probe and template embeddings
        3. Map similarity to [0, 1] score
    """

    @abstractmethod
    def compare(
        self,
        probe_embedding: np.ndarray,
        template_embedding: np.ndarray,
    ) -> MatchResult:
        """
        Compare two face identity embeddings.

        Args:
            probe_embedding: Identity embedding from authentication capture.
                            Shape: (D,) where D is embedding dimension (typically 512).
                            Must be L2-normalized.
            template_embedding: Identity embedding from enrolled template.
                               Shape: (D,).

        Returns:
            MatchResult with identity similarity score.

        Note:
            - Embeddings should be L2-normalized before comparison
            - Cosine similarity of unit vectors = dot product
            - Score remapped from [-1, 1] to [0, 1]
        """
        pass


class ScoreFusion(ABC):
    """
    Abstract base class for combining match scores.

    Combines geometric and descriptor scores into a final decision.
    The fusion strategy can be simple (weighted average) or complex
    (learned fusion, per-path minimums, etc.).

    DS-1 implements this in: core/matching/score_fusion.py

    Typical approach:
        final_score = alpha * geometric_score + beta * descriptor_score
        is_match = final_score >= threshold
    """

    @abstractmethod
    def fuse(
        self, geometric_result: MatchResult, descriptor_result: MatchResult
    ) -> MatchResult:
        """
        Combine geometric and descriptor match results.

        Args:
            geometric_result: MatchResult from GeometricMatcher.
            descriptor_result: MatchResult from DescriptorMatcher.

        Returns:
            MatchResult with final fused score and decision.

        Note:
            - Both input scores should be in [0, 1]
            - The details dict should include component scores
            - Threshold for is_match is typically 0.65 (tuned by DS)
        """
        pass


# ============================================================
# Stub Implementations (Use before DS team delivers real code)
# ============================================================


class StubGeometricMatcher(GeometricMatcher):
    """
    Placeholder geometric matcher that always returns a fixed score.

    Use this for testing the pipeline before DS team implements
    the real ICP + Chamfer distance algorithm.
    """

    def __init__(self, default_score: float = 0.5):
        """
        Initialize stub matcher.

        Args:
            default_score: Score to return for all comparisons.
        """
        self.default_score = default_score

    def compare(
        self, probe_cloud: np.ndarray, template_cloud: np.ndarray
    ) -> MatchResult:
        """Return a fixed score for any input."""
        return MatchResult(
            score=self.default_score,
            details={
                "method": "stub",
                "probe_points": len(probe_cloud),
                "template_points": len(template_cloud),
            },
            is_match=True,  # Stub always matches
        )


class StubDescriptorMatcher(DescriptorMatcher):
    """
    Placeholder descriptor matcher that always returns a fixed score.

    Use this for testing the pipeline before DS team implements
    the real nearest neighbor matching algorithm.
    """

    def __init__(self, default_score: float = 0.5):
        """
        Initialize stub matcher.

        Args:
            default_score: Score to return for all comparisons.
        """
        self.default_score = default_score

    def compare(
        self,
        probe_desc: np.ndarray,
        template_desc: np.ndarray,
        probe_cloud: np.ndarray,
        template_cloud: np.ndarray,
    ) -> MatchResult:
        """Return a fixed score for any input."""
        return MatchResult(
            score=self.default_score,
            details={
                "method": "stub",
                "probe_descriptors": len(probe_desc),
                "template_descriptors": len(template_desc),
                "descriptor_dim": probe_desc.shape[1] if len(probe_desc) > 0 else 0,
            },
            is_match=True,  # Stub always matches
        )


class StubEmbeddingMatcher(EmbeddingMatcher):
    """
    Placeholder embedding matcher that always returns a fixed score.

    Use this for testing the pipeline before ArcFace embeddings are available.
    """

    def __init__(self, default_score: float = 0.5):
        self.default_score = default_score

    def compare(
        self,
        probe_embedding: np.ndarray,
        template_embedding: np.ndarray,
    ) -> MatchResult:
        """Return a fixed score for any input."""
        return MatchResult(
            score=self.default_score,
            details={
                "method": "stub",
                "probe_dim": probe_embedding.shape[0] if probe_embedding is not None else 0,
                "template_dim": template_embedding.shape[0] if template_embedding is not None else 0,
            },
            is_match=True,
        )


class StubScoreFusion(ScoreFusion):
    """
    Placeholder score fusion that uses simple weighted average.

    This is a minimal implementation that DS team will replace
    with optimized weights and potentially more complex fusion logic.
    """

    def __init__(
        self,
        geometric_weight: float = 0.4,
        descriptor_weight: float = 0.6,
        threshold: float = 0.65,
    ):
        """
        Initialize stub fusion.

        Args:
            geometric_weight: Weight for geometric score (alpha).
            descriptor_weight: Weight for descriptor score (beta).
            threshold: Accept/reject threshold for final decision.
        """
        self.geo_weight = geometric_weight
        self.desc_weight = descriptor_weight
        self.threshold = threshold

    def fuse(
        self, geometric_result: MatchResult, descriptor_result: MatchResult
    ) -> MatchResult:
        """Combine scores using weighted average."""
        # Calculate weighted average
        final_score = (
            self.geo_weight * geometric_result.score
            + self.desc_weight * descriptor_result.score
        )

        # Make decision based on threshold
        is_match = final_score >= self.threshold

        return MatchResult(
            score=final_score,
            details={
                "method": "stub_weighted_average",
                "geometric_score": geometric_result.score,
                "descriptor_score": descriptor_result.score,
                "geometric_weight": self.geo_weight,
                "descriptor_weight": self.desc_weight,
                "threshold": self.threshold,
            },
            is_match=is_match,
        )


if __name__ == "__main__":
    # Quick test of stub implementations
    print("Testing stub matchers...")

    # Create dummy data
    probe_cloud = np.random.randn(1000, 3)
    template_cloud = np.random.randn(1200, 3)
    probe_desc = np.random.randn(1000, 128)
    template_desc = np.random.randn(1200, 128)

    # Test geometric matcher
    geo_matcher = StubGeometricMatcher(default_score=0.75)
    geo_result = geo_matcher.compare(probe_cloud, template_cloud)
    print(f"Geometric result: score={geo_result.score}, is_match={geo_result.is_match}")

    # Test descriptor matcher
    desc_matcher = StubDescriptorMatcher(default_score=0.85)
    desc_result = desc_matcher.compare(
        probe_desc, template_desc, probe_cloud, template_cloud
    )
    print(f"Descriptor result: score={desc_result.score}, is_match={desc_result.is_match}")

    # Test fusion
    fusion = StubScoreFusion(geometric_weight=0.4, descriptor_weight=0.6, threshold=0.65)
    final_result = fusion.fuse(geo_result, desc_result)
    print(f"Final result: score={final_result.score:.3f}, is_match={final_result.is_match}")
    print(f"Details: {final_result.details}")
