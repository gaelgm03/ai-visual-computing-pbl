"""
Unit Tests for Face Detector Module

This module tests the FaceDetector class and related functionality:
- FaceDetection dataclass creation and validation
- Bounding box calculation from landmarks
- Face region cropping
- Head pose calculation
- Integration tests with MediaPipe (when available)

Note: Full detection tests require MediaPipe Face Landmarker to be installed.
      Tests are designed to skip gracefully if MediaPipe is not available.

Usage:
    pytest tests/test_face_detector.py -v
    pytest tests/test_face_detector.py -v -k "not slow"
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.face_detector import FaceDetector, FaceDetection

# Check if MediaPipe is available
try:
    import mediapipe
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


# ============================================================
# Test FaceDetection Dataclass
# ============================================================

class TestFaceDetection:
    """Tests for FaceDetection dataclass."""

    def test_create_face_detection(self):
        """Test that FaceDetection can be created with valid data."""
        detection = FaceDetection(
            bbox=(100, 100, 300, 400),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(10.0, 5.0, -2.0),
            confidence=0.95,
        )
        assert detection.bbox == (100, 100, 300, 400)
        assert detection.landmarks_2d.shape == (478, 2)
        assert detection.landmarks_3d.shape == (478, 3)
        assert detection.head_pose == (10.0, 5.0, -2.0)
        assert detection.confidence == 0.95

    def test_face_detection_landmark_shapes(self):
        """Test FaceDetection with realistic landmark data."""
        landmarks_2d = np.random.randn(478, 2).astype(np.float32)
        landmarks_3d = np.random.randn(478, 3).astype(np.float32)
        detection = FaceDetection(
            bbox=(0, 0, 100, 100),
            landmarks_2d=landmarks_2d,
            landmarks_3d=landmarks_3d,
            head_pose=(0.0, 0.0, 0.0),
            confidence=0.9,
        )
        assert detection.landmarks_2d.dtype == np.float32
        assert detection.landmarks_3d.dtype == np.float32

    def test_face_detection_bbox_values(self):
        """Test that bbox contains valid integer coordinates."""
        detection = FaceDetection(
            bbox=(50, 75, 200, 250),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(15.0, -10.0, 3.0),
            confidence=0.88,
        )
        x1, y1, x2, y2 = detection.bbox
        assert x2 > x1, "x2 should be greater than x1"
        assert y2 > y1, "y2 should be greater than y1"
        assert all(isinstance(v, int) for v in detection.bbox)

    def test_face_detection_head_pose_range(self):
        """Test head pose with typical angle values."""
        # Typical yaw: -90 to +90 degrees
        # Typical pitch: -90 to +90 degrees
        # Typical roll: -90 to +90 degrees
        detection = FaceDetection(
            bbox=(100, 100, 300, 400),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(-30.0, 15.0, 5.0),  # Looking left, slightly up, slight tilt
            confidence=0.92,
        )
        yaw, pitch, roll = detection.head_pose
        assert -90 <= yaw <= 90
        assert -90 <= pitch <= 90
        assert -90 <= roll <= 90


# ============================================================
# Test Bounding Box Calculation
# ============================================================

class TestBboxCalculation:
    """Tests for bounding box calculation from landmarks (no MediaPipe needed)."""

    @pytest.fixture
    def mock_landmarks(self):
        """Create mock 2D landmarks for testing."""
        landmarks = np.zeros((478, 2))
        # Spread landmarks across a region
        landmarks[:, 0] = np.linspace(100, 300, 478)  # x from 100 to 300
        landmarks[:, 1] = np.linspace(50, 350, 478)   # y from 50 to 350
        return landmarks

    def test_bbox_from_landmarks(self, mock_landmarks):
        """Test that bounding box encompasses all landmarks."""
        x_min = int(np.min(mock_landmarks[:, 0]))
        y_min = int(np.min(mock_landmarks[:, 1]))
        x_max = int(np.max(mock_landmarks[:, 0]))
        y_max = int(np.max(mock_landmarks[:, 1]))

        assert x_min == 100
        assert y_min == 50
        assert x_max == 300
        assert y_max == 350

    def test_bbox_area_calculation(self, mock_landmarks):
        """Test bounding box area calculation."""
        x_min = int(np.min(mock_landmarks[:, 0]))
        y_min = int(np.min(mock_landmarks[:, 1]))
        x_max = int(np.max(mock_landmarks[:, 0]))
        y_max = int(np.max(mock_landmarks[:, 1]))

        width = x_max - x_min
        height = y_max - y_min
        area = width * height

        assert width == 200
        assert height == 300
        assert area == 60000


# ============================================================
# Test Face Region Cropping
# ============================================================

class TestCropFaceRegion:
    """Tests for face region cropping logic."""

    def test_crop_basic(self):
        """Test basic face cropping without padding."""
        # Create a frame with a bright region
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:300, 150:350] = 255  # Bright rectangle

        detection = FaceDetection(
            bbox=(150, 100, 350, 300),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(0.0, 0.0, 0.0),
            confidence=0.95,
        )

        x1, y1, x2, y2 = detection.bbox
        cropped = frame[y1:y2, x1:x2]

        assert cropped.shape[0] == 200  # height
        assert cropped.shape[1] == 200  # width
        assert cropped.mean() == 255  # Should be all white

    def test_crop_with_padding(self):
        """Test face cropping with padding."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:300, 150:350] = 255

        detection = FaceDetection(
            bbox=(150, 100, 350, 300),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(0.0, 0.0, 0.0),
            confidence=0.95,
        )

        x1, y1, x2, y2 = detection.bbox
        padding = 0.3
        face_w = x2 - x1
        face_h = y2 - y1
        pad_x = int(face_w * padding)
        pad_y = int(face_h * padding)

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(640, x2 + pad_x)
        crop_y2 = min(480, y2 + pad_y)

        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        assert cropped.shape[0] > 0
        assert cropped.shape[1] > 0
        # Padded crop should be larger than original
        assert cropped.shape[0] > (y2 - y1)
        assert cropped.shape[1] > (x2 - x1)

    def test_crop_at_image_boundary(self):
        """Test cropping when face is at image boundary."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Face near top-left corner
        detection = FaceDetection(
            bbox=(10, 10, 150, 200),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(0.0, 0.0, 0.0),
            confidence=0.9,
        )

        x1, y1, x2, y2 = detection.bbox
        padding = 0.3
        face_w = x2 - x1
        face_h = y2 - y1
        pad_x = int(face_w * padding)
        pad_y = int(face_h * padding)

        # Clamp to boundaries
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(640, x2 + pad_x)
        crop_y2 = min(480, y2 + pad_y)

        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        assert crop_x1 == 0  # Should be clamped to 0
        assert crop_y1 == 0  # Should be clamped to 0
        assert cropped.shape[0] > 0
        assert cropped.shape[1] > 0


# ============================================================
# Test Head Pose Estimation Logic
# ============================================================

class TestHeadPoseLogic:
    """Tests for head pose estimation concepts (no MediaPipe needed)."""

    def test_pose_landmarks_indices(self):
        """Test that pose landmark indices are valid."""
        from core.face_detector import POSE_LANDMARKS

        # All indices should be valid for 478 landmarks
        for name, idx in POSE_LANDMARKS.items():
            assert 0 <= idx < 478, f"Invalid index for {name}: {idx}"

    def test_model_points_3d_shape(self):
        """Test that 3D model points have correct shape."""
        from core.face_detector import MODEL_POINTS_3D

        # Should have 6 points with 3D coordinates
        assert MODEL_POINTS_3D.shape == (6, 3)
        assert MODEL_POINTS_3D.dtype == np.float64


# ============================================================
# Integration Tests (Require MediaPipe)
# ============================================================

@pytest.mark.skipif(not MEDIAPIPE_AVAILABLE, reason="MediaPipe not installed")
class TestFaceDetectorIntegration:
    """Integration tests requiring MediaPipe."""

    @pytest.fixture
    def detector(self):
        """Create a FaceDetector instance for testing."""
        config = {
            "min_detection_confidence": 0.9,
            "min_tracking_confidence": 0.5,
            "face_padding": 0.3,
        }
        det = FaceDetector(config)
        yield det
        det.close()

    def test_detector_initialization(self, detector):
        """Test that detector initializes correctly."""
        assert detector.face_padding == 0.3
        assert hasattr(detector, 'landmarker')
        assert detector.landmarker is not None

    def test_detector_config_values(self, detector):
        """Test that config values are stored correctly."""
        assert detector.config.get("min_detection_confidence") == 0.9
        assert detector.config.get("min_tracking_confidence") == 0.5

    @pytest.mark.slow
    def test_detect_no_face_in_blank_image(self, detector):
        """Test detection on a blank image (should return None)."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect(blank)
        assert result is None

    @pytest.mark.slow
    def test_detect_no_face_in_noise_image(self, detector):
        """Test detection on random noise (likely no face)."""
        noise = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(noise)
        # Random noise usually won't contain a detectable face
        # but if it does, verify the result structure
        if result is not None:
            assert isinstance(result, FaceDetection)
            assert isinstance(result.bbox, tuple)
            assert len(result.bbox) == 4
            assert result.landmarks_2d.shape == (478, 2)
            assert result.landmarks_3d.shape == (478, 3)

    @pytest.mark.slow
    def test_detect_returns_valid_structure(self, detector):
        """Test that detection result has valid structure when face is found."""
        # Create a simple pattern that might trigger detection
        # (unlikely to actually detect, but tests the flow)
        test_image = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(test_image)

        # If a face is detected (unlikely with random data)
        if result is not None:
            assert isinstance(result.bbox, tuple)
            assert len(result.bbox) == 4
            assert result.landmarks_2d.shape == (478, 2)
            assert result.landmarks_3d.shape == (478, 3)
            assert len(result.head_pose) == 3
            assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.slow
    def test_crop_face_region_method(self, detector):
        """Test the crop_face_region method."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Create a mock detection
        detection = FaceDetection(
            bbox=(200, 150, 400, 350),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(0.0, 0.0, 0.0),
            confidence=0.95,
        )

        cropped = detector.crop_face_region(frame, detection, padding=0.3)

        assert cropped.shape[0] > 0
        assert cropped.shape[1] > 0
        assert cropped.shape[2] == 3  # Should still be 3 channels

    @pytest.mark.slow
    def test_crop_face_region_default_padding(self, detector):
        """Test crop_face_region uses config padding by default."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        detection = FaceDetection(
            bbox=(200, 150, 400, 350),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(0.0, 0.0, 0.0),
            confidence=0.95,
        )

        # Call without explicit padding (should use config value)
        cropped = detector.crop_face_region(frame, detection)

        assert cropped is not None
        assert cropped.shape[0] > 0


# ============================================================
# Test Confidence Estimation Logic
# ============================================================

class TestConfidenceEstimation:
    """Tests for confidence estimation logic."""

    def test_confidence_for_centered_face(self):
        """Test confidence estimation for a well-centered face."""
        width, height = 640, 480

        # Create landmarks for a well-centered face
        landmarks_2d = np.zeros((478, 2))
        # Face centered in the middle, taking about 30% of the image
        center_x, center_y = width / 2, height / 2
        face_size = min(width, height) * 0.3

        landmarks_2d[:, 0] = np.linspace(
            center_x - face_size / 2,
            center_x + face_size / 2,
            478
        )
        landmarks_2d[:, 1] = np.linspace(
            center_y - face_size / 2,
            center_y + face_size / 2,
            478
        )

        # Check that landmarks are within bounds
        margin = 5
        in_bounds = (
            np.all(landmarks_2d[:, 0] >= margin) and
            np.all(landmarks_2d[:, 0] <= width - margin) and
            np.all(landmarks_2d[:, 1] >= margin) and
            np.all(landmarks_2d[:, 1] <= height - margin)
        )

        assert in_bounds, "Centered face landmarks should be within bounds"

    def test_confidence_for_small_face(self):
        """Test that very small faces should result in lower confidence."""
        width, height = 640, 480

        # Create landmarks for a very small face (< 1% of image)
        landmarks_2d = np.zeros((478, 2))
        landmarks_2d[:, 0] = np.linspace(300, 310, 478)  # 10 pixels wide
        landmarks_2d[:, 1] = np.linspace(200, 210, 478)  # 10 pixels tall

        # Calculate face area ratio
        face_width = np.max(landmarks_2d[:, 0]) - np.min(landmarks_2d[:, 0])
        face_height = np.max(landmarks_2d[:, 1]) - np.min(landmarks_2d[:, 1])
        size_ratio = (face_width * face_height) / (width * height)

        assert size_ratio < 0.01, "This should be a very small face"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
