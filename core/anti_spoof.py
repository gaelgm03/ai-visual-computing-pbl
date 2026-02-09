"""
Anti-Spoofing Module

This module provides anti-spoofing checks for the MASt3R face authentication system.
It detects presentation attacks (photos, screens, masks) by analyzing the 3D
structure of the reconstructed face point cloud.

The key insight: real faces have 3D depth variation and curvature, while
photos/screens are flat (planar). We measure this via:
1. Depth variance: Real faces have significant z-coordinate variance
2. Eigenvalue ratio: PCA of point cloud - flat surfaces have one tiny eigenvalue
3. Confidence mean: MASt3R confidence scores are typically lower for fake inputs

Usage:
    from core.anti_spoof import AntiSpoof, AntiSpoofResult

    anti_spoof = AntiSpoof(config)
    result = anti_spoof.check(point_cloud, confidence)
    if not result.passed:
        print("Spoofing detected!")

Author: CS-1
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class AntiSpoofResult:
    """
    Result of an anti-spoofing check.

    Attributes:
        passed: True if the input appears to be a real 3D face.
        depth_variance: Variance of z-coordinates (higher = more 3D).
        eigenvalue_ratio: Ratio of smallest to largest PCA eigenvalue.
                          Low ratio (< 0.05) indicates planarity (fake).
        confidence_mean: Mean MASt3R confidence score.
        details: Dictionary with additional diagnostic information.
    """
    passed: bool
    depth_variance: float
    eigenvalue_ratio: float
    confidence_mean: float
    details: Dict[str, Any] = field(default_factory=dict)


class AntiSpoof:
    """
    Anti-spoofing checker for 3D face point clouds.

    Detects flat/planar inputs that indicate photos or screens
    rather than real 3D faces.

    Thresholds are loaded from config and can be tuned by DS-2.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the anti-spoof checker.

        Args:
            config: Configuration dictionary containing:
                - min_depth_variance: Minimum variance of z-coords (default: 0.003)
                - min_eigenvalue_ratio: Minimum eigenvalue ratio (default: 0.05)
                - min_confidence_mean: Minimum mean confidence (default: 0.7)
                - enabled: Whether to enforce checks (default: True)

        Example:
            config = {
                "min_depth_variance": 0.003,
                "min_eigenvalue_ratio": 0.05,
                "min_confidence_mean": 0.7,
            }
            checker = AntiSpoof(config)
        """
        self.min_depth_variance = config.get("depth_std_threshold", config.get("min_depth_variance", 0.003))
        self.min_eigenvalue_ratio = config.get("planarity_ratio_threshold", config.get("min_eigenvalue_ratio", 0.05))
        self.min_confidence_mean = config.get("confidence_mean_threshold", config.get("min_confidence_mean", 0.7))
        self.enabled = config.get("enabled", True)

    def check(
        self,
        point_cloud: np.ndarray,
        confidence: np.ndarray,
    ) -> AntiSpoofResult:
        """
        Check if a point cloud represents a real 3D face.

        Args:
            point_cloud: 3D point coordinates, shape (N, 3).
            confidence: Per-point confidence values, shape (N,).

        Returns:
            AntiSpoofResult with pass/fail decision and metrics.

        Example:
            result = checker.check(probe_cloud, probe_confidence)
            if not result.passed:
                return "Spoofing detected"
        """
        # Handle edge cases
        if point_cloud is None or len(point_cloud) < 10:
            return AntiSpoofResult(
                passed=False,
                depth_variance=0.0,
                eigenvalue_ratio=0.0,
                confidence_mean=0.0,
                details={"error": "Insufficient points", "n_points": len(point_cloud) if point_cloud is not None else 0},
            )

        # 1. Compute depth standard deviation (z-axis)
        # Note: config uses depth_std_threshold, so we compute std not variance
        z_values = point_cloud[:, 2]
        depth_variance = float(np.std(z_values))  # Actually std dev, named for schema compat

        # 2. Compute PCA eigenvalues for planarity check
        eigenvalue_ratio = self._compute_eigenvalue_ratio(point_cloud)

        # 3. Compute mean confidence
        if confidence is not None and len(confidence) > 0:
            confidence_mean = float(np.mean(confidence))
        else:
            confidence_mean = 1.0  # Assume valid if no confidence provided

        # 4. Apply threshold checks
        depth_ok = depth_variance >= self.min_depth_variance
        planarity_ok = eigenvalue_ratio >= self.min_eigenvalue_ratio
        confidence_ok = confidence_mean >= self.min_confidence_mean

        # If disabled, always pass but still compute metrics
        if not self.enabled:
            passed = True
        else:
            passed = depth_ok and planarity_ok and confidence_ok

        return AntiSpoofResult(
            passed=passed,
            depth_variance=depth_variance,
            eigenvalue_ratio=eigenvalue_ratio,
            confidence_mean=confidence_mean,
            details={
                "depth_ok": depth_ok,
                "planarity_ok": planarity_ok,
                "confidence_ok": confidence_ok,
                "thresholds": {
                    "min_depth_variance": self.min_depth_variance,
                    "min_eigenvalue_ratio": self.min_eigenvalue_ratio,
                    "min_confidence_mean": self.min_confidence_mean,
                },
                "n_points": len(point_cloud),
            },
        )

    def _compute_eigenvalue_ratio(self, point_cloud: np.ndarray) -> float:
        """
        Compute the ratio of smallest to largest PCA eigenvalue.

        A low ratio indicates the point cloud is planar (flat),
        which suggests a photo or screen attack.

        Args:
            point_cloud: (N, 3) array of 3D points.

        Returns:
            Ratio of smallest to largest eigenvalue (0 to 1).
            Real faces typically have ratio > 0.05.
        """
        # Center the point cloud
        centered = point_cloud - point_cloud.mean(axis=0)

        # Compute covariance matrix
        cov = np.cov(centered.T)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(cov)

        # Sort eigenvalues (smallest to largest)
        eigenvalues = np.sort(eigenvalues)

        # Avoid division by zero
        largest = eigenvalues[-1]
        smallest = eigenvalues[0]

        if largest < 1e-10:
            return 0.0

        return float(smallest / largest)


def get_anti_spoof(config: Dict[str, Any] = None) -> AntiSpoof:
    """
    Factory function to get an AntiSpoof instance with config.

    Args:
        config: Optional config dict. If None, loads from config.yaml.

    Returns:
        Configured AntiSpoof instance.
    """
    if config is None:
        try:
            from core.config import get_config
            full_config = get_config()
            config = full_config.get("anti_spoof", {})
        except Exception:
            config = {}

    return AntiSpoof(config)


if __name__ == "__main__":
    # Quick test
    print("Testing AntiSpoof module...")

    config = {
        "min_depth_variance": 0.003,
        "min_eigenvalue_ratio": 0.05,
        "min_confidence_mean": 0.7,
        "enabled": True,
    }
    checker = AntiSpoof(config)

    # Test with curved (real face-like) point cloud
    n_points = 1000
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi / 2, n_points)
    r = 0.1 + np.random.normal(0, 0.01, n_points)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    curved_cloud = np.stack([x, y, z], axis=1).astype(np.float32)
    curved_conf = np.ones(n_points, dtype=np.float32) * 0.9

    result = checker.check(curved_cloud, curved_conf)
    print(f"Curved cloud: passed={result.passed}, depth_var={result.depth_variance:.6f}, "
          f"eigen_ratio={result.eigenvalue_ratio:.4f}")

    # Test with flat (photo-like) point cloud
    flat_cloud = np.random.uniform(-0.1, 0.1, (1000, 3)).astype(np.float32)
    flat_cloud[:, 2] = 0.0  # Completely flat in z
    flat_conf = np.ones(1000, dtype=np.float32) * 0.9

    result = checker.check(flat_cloud, flat_conf)
    print(f"Flat cloud: passed={result.passed}, depth_var={result.depth_variance:.6f}, "
          f"eigen_ratio={result.eigenvalue_ratio:.4f}")

    print("AntiSpoof module test complete!")
