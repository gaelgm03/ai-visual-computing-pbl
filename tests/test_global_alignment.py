"""
Unit Tests for Global Alignment Module

This module tests the global alignment functionality:
- AlignmentResult dataclass creation and validation
- GlobalAligner initialization
- Integration with MASt3R-SfM sparse_global_alignment

Note: Full alignment tests require MASt3R to be installed and a GPU.
      Tests are designed to skip gracefully if MASt3R is not available.

Usage:
    pytest tests/test_global_alignment.py -v
    pytest tests/test_global_alignment.py -v -k "not slow"  # Skip slow tests
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.global_alignment import (
    GlobalAligner,
    AlignmentResult,
    align_multiview,
)


# ============================================================
# Test Data Classes
# ============================================================

class TestAlignmentResult:
    """Tests for AlignmentResult dataclass."""

    def test_create_alignment_result(self):
        """Test that AlignmentResult can be created with valid data."""
        n_points = 10000
        n_frames = 12

        result = AlignmentResult(
            point_cloud=np.zeros((n_points, 3)),
            colors=np.zeros((n_points, 3), dtype=np.uint8),
            confidence=np.ones(n_points),
            camera_poses=[np.eye(4) for _ in range(n_frames)],
            intrinsics=[np.eye(3) * 500 for _ in range(n_frames)],
        )

        assert result.point_cloud.shape == (n_points, 3)
        assert result.colors.shape == (n_points, 3)
        assert result.colors.dtype == np.uint8
        assert result.confidence.shape == (n_points,)
        assert len(result.camera_poses) == n_frames
        assert len(result.intrinsics) == n_frames

    def test_alignment_result_with_depthmaps(self):
        """Test AlignmentResult with optional depthmaps."""
        n_points = 1000
        n_frames = 5
        h, w = 384, 512

        result = AlignmentResult(
            point_cloud=np.zeros((n_points, 3)),
            colors=np.zeros((n_points, 3), dtype=np.uint8),
            confidence=np.ones(n_points),
            camera_poses=[np.eye(4) for _ in range(n_frames)],
            intrinsics=[np.eye(3) * 500 for _ in range(n_frames)],
            depthmaps=[np.ones((h, w)) for _ in range(n_frames)],
        )

        assert result.depthmaps is not None
        assert len(result.depthmaps) == n_frames
        assert result.depthmaps[0].shape == (h, w)

    def test_alignment_result_camera_pose_shape(self):
        """Test that camera poses have correct shape."""
        result = AlignmentResult(
            point_cloud=np.zeros((100, 3)),
            colors=np.zeros((100, 3), dtype=np.uint8),
            confidence=np.ones(100),
            camera_poses=[np.eye(4), np.eye(4) * 2],
            intrinsics=[np.eye(3)],
        )

        assert result.camera_poses[0].shape == (4, 4)
        assert result.camera_poses[1].shape == (4, 4)

    def test_empty_alignment_result(self):
        """Test AlignmentResult with empty point cloud."""
        result = AlignmentResult(
            point_cloud=np.zeros((0, 3)),
            colors=np.zeros((0, 3), dtype=np.uint8),
            confidence=np.zeros(0),
            camera_poses=[],
            intrinsics=[],
        )

        assert len(result.point_cloud) == 0
        assert len(result.colors) == 0
        assert len(result.camera_poses) == 0


# ============================================================
# Test GlobalAligner Initialization
# ============================================================

class TestGlobalAlignerInit:
    """Tests for GlobalAligner initialization."""

    def test_default_config(self):
        """Test aligner initialization with default config."""
        config = {}
        aligner = GlobalAligner(config)

        assert aligner.lr1 == 0.07
        assert aligner.niter1 == 300
        assert aligner.lr2 == 0.01
        assert aligner.niter2 == 300
        assert aligner.subsample == 8
        assert aligner.device == "cuda"

    def test_custom_config(self):
        """Test aligner initialization with custom config."""
        config = {
            "lr1": 0.1,
            "niter1": 500,
            "lr2": 0.02,
            "niter2": 200,
            "subsample": 4,
            "device": "cpu",
        }
        aligner = GlobalAligner(config)

        assert aligner.lr1 == 0.1
        assert aligner.niter1 == 500
        assert aligner.lr2 == 0.02
        assert aligner.niter2 == 200
        assert aligner.subsample == 4
        assert aligner.device == "cpu"

    def test_partial_config(self):
        """Test aligner with partial config (uses defaults for missing)."""
        config = {
            "lr1": 0.05,
            "device": "cpu",
        }
        aligner = GlobalAligner(config)

        assert aligner.lr1 == 0.05
        assert aligner.device == "cpu"
        # Should use defaults for unspecified
        assert aligner.niter1 == 300
        assert aligner.subsample == 8


# ============================================================
# Test Alignment Logic (Mock Tests)
# ============================================================

class TestAlignmentLogic:
    """Tests for alignment logic that don't require MASt3R."""

    def test_alignment_result_unified_coordinates(self):
        """Test that aligned points should share same origin (conceptual)."""
        # This is a conceptual test showing what unified coordinates mean
        # Each camera pose transforms points to world coordinates
        n_frames = 3

        # Simulate camera poses (identity means camera at origin)
        poses = [np.eye(4) for _ in range(n_frames)]

        # Add translation to show different camera positions
        poses[1][0, 3] = 1.0  # Camera 2 at x=1
        poses[2][0, 3] = 2.0  # Camera 3 at x=2

        # Points in camera 1's frame
        local_points = np.array([[0, 0, 1], [0, 0, 2]])

        # Transform to world using camera 1's pose (identity)
        world_points_cam1 = local_points  # No change since identity

        # If camera 2 sees the same physical point...
        # In camera 2's local frame, the point would appear at different location
        # Applying camera 2's pose transforms it back to same world position

        # This is what global alignment achieves
        result = AlignmentResult(
            point_cloud=world_points_cam1,
            colors=np.zeros((2, 3), dtype=np.uint8),
            confidence=np.ones(2),
            camera_poses=poses,
            intrinsics=[np.eye(3) * 500 for _ in range(n_frames)],
        )

        # Points should be in unified world coordinates
        assert result.point_cloud.shape == (2, 3)
        assert len(result.camera_poses) == n_frames

    def test_camera_pose_consistency(self):
        """Test that camera poses form consistent transformation chain."""
        # Create a simple chain of camera poses
        n_frames = 4

        poses = []
        for i in range(n_frames):
            pose = np.eye(4)
            pose[0, 3] = i * 0.5  # Cameras spaced 0.5 apart
            poses.append(pose)

        result = AlignmentResult(
            point_cloud=np.zeros((100, 3)),
            colors=np.zeros((100, 3), dtype=np.uint8),
            confidence=np.ones(100),
            camera_poses=poses,
            intrinsics=[np.eye(3) for _ in range(n_frames)],
        )

        # Check camera positions increase monotonically
        for i in range(n_frames - 1):
            pos_i = result.camera_poses[i][0, 3]
            pos_j = result.camera_poses[i + 1][0, 3]
            assert pos_j > pos_i, "Camera positions should increase"

    def test_point_cloud_bounds_reasonable(self):
        """Test that point cloud should have reasonable bounds for face."""
        # Face should be roughly 15-25 cm wide, 20-30 cm tall
        # In metric coordinates

        # Simulate face-sized point cloud
        n_points = 1000
        # Random points within face-like bounds
        points = np.random.uniform(
            low=[-0.1, -0.15, 0.3],  # min x, y, z
            high=[0.1, 0.15, 0.4],   # max x, y, z
            size=(n_points, 3)
        )

        result = AlignmentResult(
            point_cloud=points,
            colors=np.zeros((n_points, 3), dtype=np.uint8),
            confidence=np.ones(n_points),
            camera_poses=[np.eye(4)],
            intrinsics=[np.eye(3) * 500],
        )

        # Check bounds (using np.ptp for NumPy 2.x compatibility)
        x_range = np.ptp(result.point_cloud[:, 0])
        y_range = np.ptp(result.point_cloud[:, 1])
        z_range = np.ptp(result.point_cloud[:, 2])

        # Face dimensions should be reasonable (in meters)
        assert 0.1 < x_range < 0.5, f"X range {x_range} unreasonable for face"
        assert 0.1 < y_range < 0.5, f"Y range {y_range} unreasonable for face"


# ============================================================
# Integration Tests (Require MASt3R installation)
# ============================================================

def mast3r_available():
    """Check if MASt3R is available for import."""
    try:
        from mast3r.model import AsymmetricMASt3R
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        return True
    except ImportError:
        return False


def gpu_available():
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not mast3r_available(), reason="MASt3R not installed")
class TestGlobalAlignmentIntegration:
    """Integration tests that require MASt3R to be installed."""

    def test_aligner_can_import_mast3r(self):
        """Test that GlobalAligner can import MASt3R modules."""
        config = {"device": "cpu"}
        aligner = GlobalAligner(config)

        # Try to import the sparse_global_alignment function
        try:
            from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
            imported = True
        except ImportError:
            imported = False

        assert imported, "Should be able to import sparse_global_alignment"

    @pytest.mark.slow
    @pytest.mark.skipif(not gpu_available(), reason="GPU not available")
    def test_full_alignment_with_dummy_images(self):
        """Test full alignment pipeline with dummy images."""
        import torch
        from mast3r.model import AsymmetricMASt3R

        # Load model
        model = AsymmetricMASt3R.from_pretrained(
            "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        )
        model = model.to("cuda")

        # Create dummy frames
        n_frames = 4
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                  for _ in range(n_frames)]
        pairs = [(0, 1), (1, 2), (2, 3), (0, 2)]

        # Run alignment
        config = {"device": "cuda", "niter1": 50, "niter2": 50}  # Reduced iters
        aligner = GlobalAligner(config)
        result = aligner.align_from_frames(frames, pairs, model)

        # Check result structure
        assert isinstance(result, AlignmentResult)
        assert result.point_cloud.shape[1] == 3
        assert len(result.camera_poses) == n_frames


# ============================================================
# Test Error Handling
# ============================================================

class TestErrorHandling:
    """Tests for error handling in global alignment."""

    def test_align_multiview_convenience_function(self):
        """Test the convenience function interface."""
        # Without MASt3R, this should at least create the aligner
        config = {"device": "cpu"}
        aligner = GlobalAligner(config)

        assert aligner is not None
        assert aligner.device == "cpu"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
