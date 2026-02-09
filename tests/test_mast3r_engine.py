"""
Unit Tests for MASt3R Engine Module

This module tests the MASt3R engine wrapper functionality:
- Data class creation and validation
- Pair index generation
- Image preprocessing
- Engine initialization

Note: Full inference tests require MASt3R to be installed and a GPU.
      Tests are designed to skip gracefully if MASt3R is not available.

Usage:
    pytest tests/test_mast3r_engine.py -v
    pytest tests/test_mast3r_engine.py -v -k "not slow"  # Skip slow tests
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.mast3r_engine import (
    MASt3REngine,
    PairwiseResult,
    MultiViewResult,
    get_engine,
)


# ============================================================
# Test Data Classes
# ============================================================

class TestPairwiseResult:
    """Tests for PairwiseResult dataclass."""

    def test_create_pairwise_result(self):
        """Test that PairwiseResult can be created with valid data."""
        h, w, d = 384, 512, 24

        result = PairwiseResult(
            pointmap1=np.zeros((h, w, 3)),
            pointmap2=np.zeros((h, w, 3)),
            confidence1=np.ones((h, w)),
            confidence2=np.ones((h, w)),
            descriptors1=np.zeros((h, w, d)),
            descriptors2=np.zeros((h, w, d)),
            image1_shape=(h, w),
            image2_shape=(h, w),
        )

        assert result.pointmap1.shape == (h, w, 3)
        assert result.confidence1.shape == (h, w)
        assert result.descriptors1.shape == (h, w, d)
        assert result.image1_shape == (h, w)

    def test_pairwise_result_with_real_data(self):
        """Test PairwiseResult with realistic data values."""
        h, w = 100, 150

        # Simulate realistic data
        pointmap = np.random.randn(h, w, 3).astype(np.float32)
        confidence = np.random.uniform(0, 1, (h, w)).astype(np.float32)
        descriptors = np.random.randn(h, w, 24).astype(np.float32)

        result = PairwiseResult(
            pointmap1=pointmap,
            pointmap2=pointmap,
            confidence1=confidence,
            confidence2=confidence,
            descriptors1=descriptors,
            descriptors2=descriptors,
        )

        # Check data types are preserved
        assert result.pointmap1.dtype == np.float32
        assert result.confidence1.dtype == np.float32
        assert result.descriptors1.dtype == np.float32


class TestMultiViewResult:
    """Tests for MultiViewResult dataclass."""

    def test_create_multiview_result(self):
        """Test that MultiViewResult can be created with valid data."""
        n_points = 10000
        d = 24

        result = MultiViewResult(
            point_cloud=np.zeros((n_points, 3)),
            colors=np.zeros((n_points, 3), dtype=np.uint8),
            descriptors=np.zeros((n_points, d)),
            confidence=np.ones(n_points),
            n_frames=12,
            n_pairs=22,
        )

        assert result.point_cloud.shape == (n_points, 3)
        assert result.colors.shape == (n_points, 3)
        assert result.colors.dtype == np.uint8
        assert result.descriptors.shape == (n_points, d)
        assert result.confidence.shape == (n_points,)
        assert result.n_frames == 12
        assert result.n_pairs == 22

    def test_multiview_result_with_metadata(self):
        """Test MultiViewResult with reconstruction metadata."""
        result = MultiViewResult(
            point_cloud=np.zeros((100, 3)),
            colors=np.zeros((100, 3), dtype=np.uint8),
            descriptors=np.zeros((100, 24)),
            confidence=np.ones(100),
            per_frame_poses=[np.eye(4) for _ in range(5)],
            n_frames=5,
            n_pairs=8,
            reconstruction_metadata={
                "confidence_threshold": 0.5,
                "pairs_processed": [(0, 1), (1, 2)],
            }
        )

        assert len(result.per_frame_poses) == 5
        assert result.reconstruction_metadata["confidence_threshold"] == 0.5

    def test_empty_multiview_result(self):
        """Test MultiViewResult with empty point cloud."""
        result = MultiViewResult(
            point_cloud=np.zeros((0, 3)),
            colors=np.zeros((0, 3), dtype=np.uint8),
            descriptors=np.zeros((0, 24)),
            confidence=np.zeros(0),
        )

        assert len(result.point_cloud) == 0
        assert len(result.colors) == 0


# ============================================================
# Test Pair Generation
# ============================================================

class TestPairGeneration:
    """Tests for generate_pair_indices static method."""

    def test_generate_pairs_minimum(self):
        """Test pair generation with minimum frames (2)."""
        pairs = MASt3REngine.generate_pair_indices(2)

        # Should have exactly 1 sequential pair
        assert len(pairs) == 1
        assert (0, 1) in pairs

    def test_generate_pairs_three_frames(self):
        """Test pair generation with 3 frames."""
        pairs = MASt3REngine.generate_pair_indices(3)

        # Sequential: (0,1), (1,2)
        # Skip-one: (0,2)
        assert (0, 1) in pairs
        assert (1, 2) in pairs
        assert (0, 2) in pairs
        assert len(pairs) == 3

    def test_generate_pairs_twelve_frames(self):
        """Test pair generation with typical enrollment count (12)."""
        pairs = MASt3REngine.generate_pair_indices(12)

        # Check all sequential pairs exist
        for i in range(11):
            assert (i, i + 1) in pairs, f"Missing sequential pair ({i}, {i+1})"

        # Check skip-one pairs exist
        for i in range(10):
            assert (i, i + 2) in pairs, f"Missing skip-one pair ({i}, {i+2})"

        # Check loop closure
        assert (0, 11) in pairs, "Missing loop closure pair"

        # For n_frames <= 20, implementation uses all-pairs for better reconstruction
        # C(12, 2) = 12 * 11 / 2 = 66 pairs
        assert len(pairs) == 66, f"Expected all-pairs (66), got {len(pairs)}"

    def test_generate_pairs_single_frame(self):
        """Test that single frame returns empty list."""
        pairs = MASt3REngine.generate_pair_indices(1)
        assert len(pairs) == 0

    def test_generate_pairs_zero_frames(self):
        """Test that zero frames returns empty list."""
        pairs = MASt3REngine.generate_pair_indices(0)
        assert len(pairs) == 0

    def test_pair_indices_valid(self):
        """Test that all generated pair indices are valid."""
        n_frames = 15
        pairs = MASt3REngine.generate_pair_indices(n_frames)

        for i, j in pairs:
            assert 0 <= i < n_frames, f"Invalid index i={i}"
            assert 0 <= j < n_frames, f"Invalid index j={j}"
            assert i < j, f"Pair should have i < j: ({i}, {j})"


# ============================================================
# Test Engine Initialization
# ============================================================

class TestEngineInitialization:
    """Tests for MASt3REngine initialization."""

    def test_default_config(self):
        """Test engine initialization with default config."""
        config = {
            "device": "cpu",  # Use CPU for tests
        }
        engine = MASt3REngine(config)

        assert engine.image_size == 512
        assert engine.force_fp16 == True
        assert engine.device.type == "cpu"
        assert engine.model is None

    def test_custom_config(self):
        """Test engine initialization with custom config."""
        config = {
            "checkpoint": "custom/model",
            "image_size": 256,
            "device": "cpu",
            "force_fp16": False,
        }
        engine = MASt3REngine(config)

        assert engine.checkpoint == "custom/model"
        assert engine.image_size == 256
        assert engine.force_fp16 == False

    def test_cuda_fallback_to_cpu(self):
        """Test that CUDA falls back to CPU if not available."""
        import torch

        config = {
            "device": "cuda",
        }
        engine = MASt3REngine(config)

        # If CUDA is not available, should fall back to CPU
        if not torch.cuda.is_available():
            assert engine.device.type == "cpu"


# ============================================================
# Test Image Preprocessing
# ============================================================

class TestImagePreprocessing:
    """Tests for image preprocessing functionality."""

    def test_preprocess_bgr_to_rgb(self):
        """Test that BGR images are converted to RGB."""
        config = {"device": "cpu", "image_size": 512}
        engine = MASt3REngine(config)

        # Create a test image with known colors
        # BGR: Blue channel = 255, others = 0
        bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = 255  # Blue in BGR

        rgb_image = engine._preprocess_image(bgr_image)

        # In RGB, the blue should now be in channel 2
        assert rgb_image[:, :, 2].mean() == 255
        assert rgb_image[:, :, 0].mean() == 0

    def test_preprocess_resize_large_image(self):
        """Test that large images are resized."""
        config = {"device": "cpu", "image_size": 512}
        engine = MASt3REngine(config)

        # Create a large image
        large_image = np.zeros((1000, 800, 3), dtype=np.uint8)
        processed = engine._preprocess_image(large_image)

        # Should be resized to fit within 512
        assert max(processed.shape[:2]) <= 512

    def test_preprocess_small_image_unchanged(self):
        """Test that small images are not upscaled."""
        config = {"device": "cpu", "image_size": 512}
        engine = MASt3REngine(config)

        # Create a small image
        small_image = np.zeros((200, 300, 3), dtype=np.uint8)
        processed = engine._preprocess_image(small_image)

        # Should remain the same size
        assert processed.shape[:2] == (200, 300)

    def test_preprocess_maintains_aspect_ratio(self):
        """Test that aspect ratio is maintained during resize."""
        config = {"device": "cpu", "image_size": 512}
        engine = MASt3REngine(config)

        # Create a wide image (2:1 aspect ratio)
        wide_image = np.zeros((500, 1000, 3), dtype=np.uint8)
        processed = engine._preprocess_image(wide_image)

        # Aspect ratio should be preserved
        original_ratio = 1000 / 500  # 2.0
        new_ratio = processed.shape[1] / processed.shape[0]
        assert abs(new_ratio - original_ratio) < 0.01


# ============================================================
# Test Deduplication
# ============================================================

class TestPointDeduplication:
    """Tests for point cloud deduplication."""

    def test_deduplicate_removes_duplicates(self):
        """Test that duplicate points are removed."""
        config = {"device": "cpu"}
        engine = MASt3REngine(config)

        # Create points with some duplicates
        points = np.array([
            [0, 0, 0],
            [0, 0, 0.001],  # Very close to first point
            [1, 1, 1],
            [1, 1, 1.001],  # Very close to third point
            [2, 2, 2],
        ])
        colors = np.array([[255, 0, 0]] * 5, dtype=np.uint8)
        descriptors = np.random.randn(5, 24)
        confidence = np.array([0.9, 0.8, 0.95, 0.85, 0.7])

        result_pts, result_colors, result_desc, result_conf = engine._deduplicate_points(
            points, colors, descriptors, confidence,
            distance_threshold=0.01
        )

        # Should have fewer points after deduplication
        assert len(result_pts) < len(points)

    def test_deduplicate_keeps_high_confidence(self):
        """Test that highest confidence point is kept for each voxel."""
        config = {"device": "cpu"}
        engine = MASt3REngine(config)

        # Create two close points with different confidence
        points = np.array([
            [0, 0, 0],
            [0, 0, 0.001],
        ])
        colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
        descriptors = np.random.randn(2, 24)
        confidence = np.array([0.5, 0.9])  # Second point has higher confidence

        result_pts, result_colors, result_desc, result_conf = engine._deduplicate_points(
            points, colors, descriptors, confidence,
            distance_threshold=0.01
        )

        # Should keep only one point
        assert len(result_pts) == 1
        # Should keep the one with higher confidence
        assert result_conf[0] == 0.9

    def test_deduplicate_max_points(self):
        """Test that max_points limit is respected."""
        config = {"device": "cpu"}
        engine = MASt3REngine(config)

        # Create many points
        n_points = 1000
        points = np.random.randn(n_points, 3)
        colors = np.random.randint(0, 255, (n_points, 3), dtype=np.uint8)
        descriptors = np.random.randn(n_points, 24)
        confidence = np.random.uniform(0, 1, n_points)

        result_pts, _, _, _ = engine._deduplicate_points(
            points, colors, descriptors, confidence,
            distance_threshold=0.001,  # Small threshold to keep more points
            max_points=100
        )

        # Should not exceed max_points
        assert len(result_pts) <= 100


# ============================================================
# Test Singleton Pattern
# ============================================================

class TestSingleton:
    """Tests for the get_engine singleton function."""

    def test_get_engine_returns_same_instance(self):
        """Test that get_engine returns the same instance."""
        # Reset singleton for test
        import core.mast3r_engine as engine_module
        engine_module._engine_instance = None

        config = {"device": "cpu"}
        engine1 = get_engine(config)
        engine2 = get_engine()  # Should return same instance

        assert engine1 is engine2

    def test_get_engine_ignores_config_after_first_call(self):
        """Test that subsequent config is ignored."""
        import core.mast3r_engine as engine_module
        engine_module._engine_instance = None

        config1 = {"device": "cpu", "image_size": 512}
        config2 = {"device": "cpu", "image_size": 256}

        engine1 = get_engine(config1)
        engine2 = get_engine(config2)

        # Should still have original image_size
        assert engine2.image_size == 512


# ============================================================
# Integration Tests (Require MASt3R installation)
# ============================================================

def mast3r_available():
    """Check if MASt3R is available for import."""
    try:
        from mast3r.model import AsymmetricMASt3R
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not mast3r_available(), reason="MASt3R not installed")
class TestMASt3RIntegration:
    """Integration tests that require MASt3R to be installed."""

    @pytest.mark.slow
    def test_model_loading(self):
        """Test that model can be loaded."""
        config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "force_fp16": True,
        }
        engine = MASt3REngine(config)
        engine.load_model()

        assert engine._model_loaded
        assert engine.model is not None

    @pytest.mark.slow
    def test_inference_pair(self):
        """Test pairwise inference with dummy images."""
        import torch

        config = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "force_fp16": True,
        }
        engine = MASt3REngine(config)
        engine.load_model()

        # Create dummy images
        img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = engine.infer_pair(img1, img2)

        # Check output shapes
        assert result.pointmap1.ndim == 3
        assert result.pointmap1.shape[-1] == 3
        assert result.confidence1.ndim == 2
        assert result.descriptors1.ndim == 3


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
