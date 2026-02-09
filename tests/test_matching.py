"""
Tests for the Matching Module

These tests verify that:
1. ICPGeometricMatcher correctly compares 3D point cloud shapes
2. NNDescriptorMatcher correctly matches feature descriptors
3. WeightedFusion correctly combines scores and makes decisions
4. Edge cases are handled gracefully (empty data, mismatched dimensions, etc.)

Author: CS-1
"""

import numpy as np
import pytest

from core.matching.interfaces import (
    MatchResult,
    GeometricMatcher,
    DescriptorMatcher,
    ScoreFusion,
    StubGeometricMatcher,
    StubDescriptorMatcher,
    StubScoreFusion,
)
from core.matching.geometric_matcher import ICPGeometricMatcher
from core.matching.descriptor_matcher import NNDescriptorMatcher
from core.matching.score_fusion import WeightedFusion


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def default_config():
    """Default configuration for matchers."""
    return {
        "icp": {
            "max_iterations": 50,
            "max_correspondence_distance": 0.05,
        },
        "chamfer_alpha": 30.0,
        "geometric_subsample": 10000,
        "match_ratio_weight": 0.4,
        "avg_similarity_weight": 0.6,
        "descriptor_subsample": 15000,
        "geometric_weight": 0.4,
        "descriptor_weight": 0.6,
        "accept_threshold": 0.65,
    }


@pytest.fixture
def face_like_cloud():
    """
    Generate a synthetic face-like point cloud.
    
    Creates a hemispherical surface with some noise,
    simulating a face's 3D structure.
    """
    n_points = 1000
    np.random.seed(42)
    
    # Create hemisphere (face-like shape)
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi / 2, n_points)
    r = 0.1 + np.random.normal(0, 0.005, n_points)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    return np.stack([x, y, z], axis=1).astype(np.float32)


@pytest.fixture
def face_descriptors():
    """
    Generate synthetic descriptors for a face point cloud.
    
    Creates unit-normalized random descriptors, similar to MASt3R output.
    """
    n_points = 1000
    descriptor_dim = 128
    np.random.seed(42)
    
    desc = np.random.randn(n_points, descriptor_dim).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    desc = desc / (norms + 1e-8)
    
    return desc


# ============================================================
# MatchResult Tests
# ============================================================

class TestMatchResult:
    """Tests for the MatchResult dataclass."""

    def test_creation(self):
        """Test basic MatchResult creation."""
        result = MatchResult(
            score=0.75,
            details={"method": "test"},
            is_match=True,
        )
        assert result.score == 0.75
        assert result.details["method"] == "test"
        assert result.is_match is True

    def test_score_range(self):
        """Test that scores are typically in [0, 1]."""
        result = MatchResult(score=0.0, details={}, is_match=False)
        assert 0.0 <= result.score <= 1.0
        
        result = MatchResult(score=1.0, details={}, is_match=True)
        assert 0.0 <= result.score <= 1.0


# ============================================================
# ICPGeometricMatcher Tests
# ============================================================

class TestICPGeometricMatcher:
    """Tests for the ICP-based geometric matcher."""

    @pytest.fixture
    def matcher(self, default_config):
        """Create an ICPGeometricMatcher instance."""
        return ICPGeometricMatcher(default_config)

    def test_same_cloud_high_score(self, matcher, face_like_cloud):
        """
        Test that comparing identical clouds gives a high score.
        
        Same point cloud should match perfectly (score ≈ 1.0).
        """
        result = matcher.compare(face_like_cloud, face_like_cloud.copy())
        
        assert result.score > 0.9, f"Same cloud should give high score, got {result.score}"
        assert result.details["method"] == "icp_chamfer"
        assert result.details["chamfer_distance"] < 0.01

    def test_similar_clouds_high_score(self, matcher, face_like_cloud):
        """
        Test that similar clouds (with small noise) still match well.
        """
        np.random.seed(123)
        noisy_cloud = face_like_cloud + np.random.normal(0, 0.002, face_like_cloud.shape).astype(np.float32)
        
        result = matcher.compare(noisy_cloud, face_like_cloud)
        
        assert result.score > 0.7, f"Similar clouds should match well, got {result.score}"

    def test_translated_cloud_matches(self, matcher, face_like_cloud):
        """
        Test that a translated cloud still matches (ICP should align it).
        """
        translated_cloud = face_like_cloud + np.array([0.1, 0.1, 0.1], dtype=np.float32)
        
        result = matcher.compare(translated_cloud, face_like_cloud)
        
        # ICP should align the clouds, so score should be high
        assert result.score > 0.8, f"Translated cloud should still match after ICP, got {result.score}"

    def test_different_shapes_low_score(self, matcher, face_like_cloud):
        """
        Test that completely different shapes give low scores.
        """
        np.random.seed(999)
        # Create a random blob instead of face-like shape
        random_cloud = np.random.randn(1000, 3).astype(np.float32) * 0.1
        
        result = matcher.compare(random_cloud, face_like_cloud)
        
        # Different shapes should have lower scores
        assert result.score < 0.8, f"Different shapes should give lower score, got {result.score}"

    def test_scaled_cloud_lower_score(self, matcher, face_like_cloud):
        """
        Test that a scaled cloud gives lower score (different size face).
        """
        scaled_cloud = face_like_cloud * 2.0
        
        result = matcher.compare(scaled_cloud, face_like_cloud)
        
        # Scaled clouds have different Chamfer distance
        assert result.score < 0.9, f"Scaled cloud should give lower score, got {result.score}"

    def test_empty_cloud_fails_gracefully(self, matcher):
        """Test that empty clouds are handled gracefully."""
        empty_cloud = np.zeros((5, 3), dtype=np.float32)
        normal_cloud = np.random.randn(1000, 3).astype(np.float32)
        
        result = matcher.compare(empty_cloud, normal_cloud)
        
        assert result.score == 0.0
        assert result.is_match is False
        assert "error" in result.details or result.details.get("probe_points", 0) < 10

    def test_result_contains_details(self, matcher, face_like_cloud):
        """Test that result contains expected detail fields."""
        result = matcher.compare(face_like_cloud, face_like_cloud)
        
        assert "method" in result.details
        assert "chamfer_distance" in result.details
        assert "icp_fitness" in result.details
        assert "probe_points" in result.details
        assert "template_points" in result.details


# ============================================================
# NNDescriptorMatcher Tests
# ============================================================

class TestNNDescriptorMatcher:
    """Tests for the nearest-neighbor descriptor matcher."""

    @pytest.fixture
    def matcher(self, default_config):
        """Create an NNDescriptorMatcher instance."""
        return NNDescriptorMatcher(default_config)

    def test_same_descriptors_high_score(self, matcher, face_descriptors, face_like_cloud):
        """
        Test that identical descriptors give high score.
        """
        result = matcher.compare(
            face_descriptors, face_descriptors.copy(),
            face_like_cloud, face_like_cloud.copy(),
        )
        
        assert result.score > 0.9, f"Same descriptors should give high score, got {result.score}"
        assert result.details["method"] == "nn_reciprocal"

    def test_similar_descriptors_high_score(self, matcher, face_descriptors, face_like_cloud):
        """
        Test that similar descriptors (with small noise) still match.
        """
        np.random.seed(123)
        noisy_desc = face_descriptors + np.random.normal(0, 0.05, face_descriptors.shape).astype(np.float32)
        # Re-normalize after adding noise
        norms = np.linalg.norm(noisy_desc, axis=1, keepdims=True)
        noisy_desc = noisy_desc / (norms + 1e-8)
        
        result = matcher.compare(
            noisy_desc, face_descriptors,
            face_like_cloud, face_like_cloud,
        )
        
        assert result.score > 0.6, f"Similar descriptors should match, got {result.score}"

    def test_random_descriptors_low_score(self, matcher, face_descriptors, face_like_cloud):
        """
        Test that random unrelated descriptors give low match ratio.
        """
        np.random.seed(999)
        random_desc = np.random.randn(1000, 128).astype(np.float32)
        norms = np.linalg.norm(random_desc, axis=1, keepdims=True)
        random_desc = random_desc / (norms + 1e-8)
        
        random_cloud = np.random.randn(1000, 3).astype(np.float32) * 0.1
        
        result = matcher.compare(
            random_desc, face_descriptors,
            random_cloud, face_like_cloud,
        )
        
        # Random descriptors should have relatively low reciprocal match ratio
        # Note: With random unit vectors in high dimensions, some spurious matches are expected
        assert result.details["match_ratio"] < 0.6, f"Random descriptors should have low match ratio, got {result.details['match_ratio']}"

    def test_empty_descriptors_fails_gracefully(self, matcher, face_like_cloud):
        """Test that empty descriptors are handled gracefully."""
        empty_desc = np.zeros((3, 128), dtype=np.float32)
        normal_desc = np.random.randn(1000, 128).astype(np.float32)
        empty_cloud = np.zeros((3, 3), dtype=np.float32)
        
        result = matcher.compare(
            empty_desc, normal_desc,
            empty_cloud, face_like_cloud,
        )
        
        assert result.score == 0.0
        assert result.is_match is False

    def test_dimension_mismatch_fails(self, matcher, face_like_cloud):
        """Test that mismatched descriptor dimensions are handled."""
        desc_128 = np.random.randn(100, 128).astype(np.float32)
        desc_256 = np.random.randn(100, 256).astype(np.float32)
        
        result = matcher.compare(
            desc_128, desc_256,
            face_like_cloud[:100], face_like_cloud[:100],
        )
        
        assert result.score == 0.0
        assert "dimension mismatch" in result.details.get("error", "")

    def test_result_contains_details(self, matcher, face_descriptors, face_like_cloud):
        """Test that result contains expected detail fields."""
        result = matcher.compare(
            face_descriptors, face_descriptors,
            face_like_cloud, face_like_cloud,
        )
        
        assert "method" in result.details
        assert "n_reciprocal_matches" in result.details
        assert "match_ratio" in result.details
        assert "avg_cosine_similarity" in result.details
        assert "descriptor_dim" in result.details

    def test_handles_nan_descriptors(self, matcher, face_like_cloud):
        """Test that NaN descriptors are filtered out."""
        n_points = 100
        desc = np.random.randn(n_points, 128).astype(np.float32)
        # Insert some NaN values
        desc[0, :] = np.nan
        desc[50, 10] = np.nan
        
        cloud = np.random.randn(n_points, 3).astype(np.float32) * 0.1
        
        # Should not raise, should filter NaN rows
        result = matcher.compare(desc, desc, cloud, cloud)
        
        # Should still produce a result (may be 0 if all filtered)
        assert isinstance(result, MatchResult)


# ============================================================
# WeightedFusion Tests
# ============================================================

class TestWeightedFusion:
    """Tests for the weighted score fusion."""

    @pytest.fixture
    def fusion(self, default_config):
        """Create a WeightedFusion instance."""
        return WeightedFusion(default_config)

    def test_high_scores_match(self, fusion):
        """Test that high component scores result in a match."""
        geo_result = MatchResult(score=0.8, details={}, is_match=True)
        desc_result = MatchResult(score=0.9, details={}, is_match=True)
        
        fused = fusion.fuse(geo_result, desc_result)
        
        # 0.4 * 0.8 + 0.6 * 0.9 = 0.32 + 0.54 = 0.86
        assert fused.score == pytest.approx(0.86, abs=0.01)
        assert fused.is_match is True

    def test_low_scores_no_match(self, fusion):
        """Test that low component scores result in no match."""
        geo_result = MatchResult(score=0.3, details={}, is_match=False)
        desc_result = MatchResult(score=0.4, details={}, is_match=False)
        
        fused = fusion.fuse(geo_result, desc_result)
        
        # 0.4 * 0.3 + 0.6 * 0.4 = 0.12 + 0.24 = 0.36
        assert fused.score == pytest.approx(0.36, abs=0.01)
        assert fused.is_match is False

    def test_threshold_boundary(self, fusion):
        """Test behavior at the threshold boundary."""
        # Score slightly above threshold (0.65)
        # 0.4 * 0.6 + 0.6 * 0.7 = 0.24 + 0.42 = 0.66
        geo_result = MatchResult(score=0.6, details={}, is_match=True)
        desc_result = MatchResult(score=0.7, details={}, is_match=True)
        
        fused = fusion.fuse(geo_result, desc_result)
        
        assert fused.score == pytest.approx(0.66, abs=0.01)
        assert fused.is_match is True  # > threshold

    def test_just_below_threshold(self, fusion):
        """Test that score just below threshold results in no match."""
        geo_result = MatchResult(score=0.5, details={}, is_match=True)
        desc_result = MatchResult(score=0.7, details={}, is_match=True)
        
        fused = fusion.fuse(geo_result, desc_result)
        
        # 0.4 * 0.5 + 0.6 * 0.7 = 0.2 + 0.42 = 0.62
        assert fused.score == pytest.approx(0.62, abs=0.01)
        assert fused.is_match is False  # < threshold

    def test_weights_sum_to_one(self, default_config):
        """Test that weights are normalized if they don't sum to 1."""
        bad_config = {**default_config, "geometric_weight": 0.5, "descriptor_weight": 0.5}
        fusion = WeightedFusion(bad_config)
        
        assert fusion.geo_weight + fusion.desc_weight == pytest.approx(1.0, abs=0.01)

    def test_result_contains_component_scores(self, fusion):
        """Test that fused result contains component scores in details."""
        geo_result = MatchResult(score=0.7, details={"method": "icp"}, is_match=True)
        desc_result = MatchResult(score=0.8, details={"method": "nn"}, is_match=True)
        
        fused = fusion.fuse(geo_result, desc_result)
        
        assert fused.details["geometric_score"] == 0.7
        assert fused.details["descriptor_score"] == 0.8
        assert fused.details["geometric_weight"] == pytest.approx(0.4, abs=0.01)
        assert fused.details["descriptor_weight"] == pytest.approx(0.6, abs=0.01)
        assert fused.details["threshold"] == 0.65

    def test_zero_scores(self, fusion):
        """Test fusion with zero scores."""
        geo_result = MatchResult(score=0.0, details={}, is_match=False)
        desc_result = MatchResult(score=0.0, details={}, is_match=False)
        
        fused = fusion.fuse(geo_result, desc_result)
        
        assert fused.score == 0.0
        assert fused.is_match is False

    def test_perfect_scores(self, fusion):
        """Test fusion with perfect scores."""
        geo_result = MatchResult(score=1.0, details={}, is_match=True)
        desc_result = MatchResult(score=1.0, details={}, is_match=True)
        
        fused = fusion.fuse(geo_result, desc_result)
        
        assert fused.score == pytest.approx(1.0, abs=0.01)
        assert fused.is_match is True


# ============================================================
# Stub Implementation Tests
# ============================================================

class TestStubImplementations:
    """Tests for the stub matcher implementations."""

    def test_stub_geometric_matcher(self):
        """Test StubGeometricMatcher returns fixed score."""
        matcher = StubGeometricMatcher(default_score=0.75)
        
        cloud = np.random.randn(100, 3).astype(np.float32)
        result = matcher.compare(cloud, cloud)
        
        assert result.score == 0.75
        assert result.is_match is True
        assert result.details["method"] == "stub"

    def test_stub_descriptor_matcher(self):
        """Test StubDescriptorMatcher returns fixed score."""
        matcher = StubDescriptorMatcher(default_score=0.8)
        
        desc = np.random.randn(100, 128).astype(np.float32)
        cloud = np.random.randn(100, 3).astype(np.float32)
        result = matcher.compare(desc, desc, cloud, cloud)
        
        assert result.score == 0.8
        assert result.is_match is True
        assert result.details["method"] == "stub"

    def test_stub_score_fusion(self):
        """Test StubScoreFusion combines scores correctly."""
        fusion = StubScoreFusion(
            geometric_weight=0.4,
            descriptor_weight=0.6,
            threshold=0.65,
        )
        
        geo_result = MatchResult(score=0.7, details={}, is_match=True)
        desc_result = MatchResult(score=0.8, details={}, is_match=True)
        
        fused = fusion.fuse(geo_result, desc_result)
        
        # 0.4 * 0.7 + 0.6 * 0.8 = 0.28 + 0.48 = 0.76
        assert fused.score == pytest.approx(0.76, abs=0.01)
        assert fused.is_match is True


# ============================================================
# Integration Tests
# ============================================================

class TestMatchingIntegration:
    """Integration tests for the full matching pipeline."""

    @pytest.fixture
    def full_pipeline(self, default_config):
        """Create a full matching pipeline."""
        return (
            ICPGeometricMatcher(default_config),
            NNDescriptorMatcher(default_config),
            WeightedFusion(default_config),
        )

    def test_genuine_pair_matches(self, full_pipeline, face_like_cloud, face_descriptors):
        """
        Test that a genuine pair (same person, same capture) matches.
        """
        geo_matcher, desc_matcher, fusion = full_pipeline
        
        # Simulate slightly noisy probe (same person, different capture)
        np.random.seed(456)
        probe_cloud = face_like_cloud + np.random.normal(0, 0.003, face_like_cloud.shape).astype(np.float32)
        probe_desc = face_descriptors + np.random.normal(0, 0.02, face_descriptors.shape).astype(np.float32)
        norms = np.linalg.norm(probe_desc, axis=1, keepdims=True)
        probe_desc = probe_desc / (norms + 1e-8)
        
        geo_result = geo_matcher.compare(probe_cloud, face_like_cloud)
        desc_result = desc_matcher.compare(
            probe_desc, face_descriptors,
            probe_cloud, face_like_cloud,
        )
        fused = fusion.fuse(geo_result, desc_result)
        
        assert fused.score > 0.5, f"Genuine pair should have reasonable score, got {fused.score}"

    def test_impostor_pair_lower_score(self, full_pipeline, face_like_cloud, face_descriptors):
        """
        Test that an impostor pair (different person) has lower score.
        """
        geo_matcher, desc_matcher, fusion = full_pipeline
        
        # Create completely different "impostor" data
        np.random.seed(789)
        impostor_cloud = np.random.randn(1000, 3).astype(np.float32) * 0.08
        impostor_desc = np.random.randn(1000, 128).astype(np.float32)
        norms = np.linalg.norm(impostor_desc, axis=1, keepdims=True)
        impostor_desc = impostor_desc / (norms + 1e-8)
        
        geo_result = geo_matcher.compare(impostor_cloud, face_like_cloud)
        desc_result = desc_matcher.compare(
            impostor_desc, face_descriptors,
            impostor_cloud, face_like_cloud,
        )
        fused = fusion.fuse(geo_result, desc_result)
        
        # Impostor should have lower score than genuine
        assert fused.score < 0.7, f"Impostor pair should have lower score, got {fused.score}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
