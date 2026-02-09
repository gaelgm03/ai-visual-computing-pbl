"""
Tests for the Anti-Spoofing Module

These tests verify that:
1. Flat (planar) point clouds are detected as spoofs
2. Curved (3D) point clouds pass anti-spoof checks

Author: CS-1
"""

import numpy as np
import pytest
from core.anti_spoof import AntiSpoof, AntiSpoofResult, get_anti_spoof


class TestAntiSpoofResult:
    """Tests for the AntiSpoofResult dataclass."""

    def test_dataclass_creation(self):
        """Test that AntiSpoofResult can be created with required fields."""
        result = AntiSpoofResult(
            passed=True,
            depth_variance=0.005,
            eigenvalue_ratio=0.1,
            confidence_mean=0.85,
        )
        assert result.passed is True
        assert result.depth_variance == 0.005
        assert result.eigenvalue_ratio == 0.1
        assert result.confidence_mean == 0.85
        assert result.details == {}

    def test_dataclass_with_details(self):
        """Test AntiSpoofResult with details dict."""
        result = AntiSpoofResult(
            passed=False,
            depth_variance=0.001,
            eigenvalue_ratio=0.02,
            confidence_mean=0.6,
            details={"reason": "too flat"},
        )
        assert result.passed is False
        assert result.details["reason"] == "too flat"


class TestAntiSpoof:
    """Tests for the AntiSpoof class."""

    @pytest.fixture
    def default_config(self):
        """Default configuration for anti-spoof checks."""
        return {
            "min_depth_variance": 0.003,
            "min_eigenvalue_ratio": 0.05,
            "min_confidence_mean": 0.7,
            "enabled": True,
        }

    @pytest.fixture
    def checker(self, default_config):
        """Create an AntiSpoof checker with default config."""
        return AntiSpoof(default_config)

    def test_flat_plane_fails(self, checker):
        """
        Test that a completely flat (planar) point cloud fails anti-spoof.
        
        This simulates a photo or screen attack where all points lie on
        a single plane (z = constant).
        """
        n_points = 1000
        
        # Create flat point cloud (z = 0 for all points)
        flat_cloud = np.random.uniform(-0.1, 0.1, (n_points, 3)).astype(np.float32)
        flat_cloud[:, 2] = 0.0  # All z values are zero (perfectly flat)
        
        confidence = np.ones(n_points, dtype=np.float32) * 0.9
        
        result = checker.check(flat_cloud, confidence)
        
        # Should fail due to zero depth variance
        assert result.passed is False, "Flat plane should fail anti-spoof"
        assert result.depth_variance < 0.001, "Flat plane should have near-zero depth variance"
        assert result.eigenvalue_ratio < 0.01, "Flat plane should have low eigenvalue ratio"

    def test_nearly_flat_fails(self, checker):
        """
        Test that a nearly flat point cloud (tiny z variation) fails.
        
        This simulates a slightly noisy photo scan.
        """
        n_points = 1000
        
        # Create nearly flat cloud (very small z variation)
        nearly_flat = np.random.uniform(-0.1, 0.1, (n_points, 3)).astype(np.float32)
        nearly_flat[:, 2] = np.random.uniform(-0.001, 0.001, n_points)  # Tiny z noise
        
        confidence = np.ones(n_points, dtype=np.float32) * 0.9
        
        result = checker.check(nearly_flat, confidence)
        
        # Should fail - variance below threshold
        assert result.passed is False, "Nearly flat cloud should fail"
        assert result.depth_variance < 0.003, "Nearly flat should have low depth variance"

    def test_curved_surface_passes(self, checker):
        """
        Test that a curved (3D) point cloud passes anti-spoof.
        
        This simulates a real face with proper 3D structure.
        Real MASt3R face clouds have depth variance around 0.005-0.02.
        """
        n_points = 1000
        
        # Create face-like point cloud with significant depth variation
        # Simulating a face that spans ~0.15m in x/y and ~0.1m in z (depth)
        x = np.random.uniform(-0.08, 0.08, n_points)
        y = np.random.uniform(-0.1, 0.1, n_points)
        # Z ranges from 0 to 0.12 (nose tip to ears) - this gives var > 0.003
        z = np.random.uniform(0.0, 0.12, n_points)
        
        curved_cloud = np.stack([x, y, z], axis=1).astype(np.float32)
        confidence = np.ones(n_points, dtype=np.float32) * 0.9
        
        result = checker.check(curved_cloud, confidence)
        
        # Should pass - has proper 3D structure
        assert result.passed is True, f"Curved surface should pass. Details: {result.details}, depth_var={result.depth_variance:.6f}"
        assert result.depth_variance > 0.003, f"Curved surface should have significant depth variance, got {result.depth_variance}"
        assert result.eigenvalue_ratio > 0.05, "Curved surface should have reasonable eigenvalue ratio"

    def test_low_confidence_fails(self, checker):
        """Test that low mean confidence causes failure."""
        n_points = 1000
        
        # Create valid 3D cloud
        curved_cloud = np.random.uniform(-0.1, 0.1, (n_points, 3)).astype(np.float32)
        curved_cloud[:, 2] = np.random.uniform(-0.1, 0.1, n_points)
        
        # But with low confidence
        low_confidence = np.ones(n_points, dtype=np.float32) * 0.4
        
        result = checker.check(curved_cloud, low_confidence)
        
        assert result.confidence_mean < 0.7
        assert result.details["confidence_ok"] is False

    def test_empty_cloud_fails(self, checker):
        """Test that empty or tiny point clouds fail gracefully."""
        empty_cloud = np.zeros((5, 3), dtype=np.float32)
        confidence = np.ones(5, dtype=np.float32)
        
        result = checker.check(empty_cloud, confidence)
        
        assert result.passed is False
        assert "error" in result.details or result.depth_variance == 0.0

    def test_disabled_mode_passes(self, default_config):
        """Test that disabled anti-spoof always passes."""
        disabled_config = {**default_config, "enabled": False}
        checker = AntiSpoof(disabled_config)
        
        # Create flat cloud that would normally fail
        flat_cloud = np.random.uniform(-0.1, 0.1, (1000, 3)).astype(np.float32)
        flat_cloud[:, 2] = 0.0
        confidence = np.ones(1000, dtype=np.float32) * 0.9
        
        result = checker.check(flat_cloud, confidence)
        
        # Should pass when disabled, but metrics still computed
        assert result.passed is True, "Disabled checker should always pass"
        assert result.depth_variance < 0.001, "Metrics should still be computed"


class TestGetAntiSpoof:
    """Tests for the factory function."""

    def test_factory_with_config(self):
        """Test factory function with explicit config."""
        config = {
            "min_depth_variance": 0.01,
            "min_eigenvalue_ratio": 0.1,
            "min_confidence_mean": 0.8,
        }
        checker = get_anti_spoof(config)
        
        assert checker.min_depth_variance == 0.01
        assert checker.min_eigenvalue_ratio == 0.1
        assert checker.min_confidence_mean == 0.8

    def test_factory_without_config(self):
        """Test factory function without config (uses defaults or config.yaml)."""
        checker = get_anti_spoof()
        
        # Should have some reasonable defaults
        assert hasattr(checker, 'min_depth_variance')
        assert hasattr(checker, 'min_eigenvalue_ratio')
        assert hasattr(checker, 'min_confidence_mean')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
