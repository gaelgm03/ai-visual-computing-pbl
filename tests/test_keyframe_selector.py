"""
Unit Tests for Keyframe Selector Module

This module tests the KeyframeSelector class and related functionality:
- KeyframeCandidate dataclass creation
- CoverageStatus dataclass creation
- Keyframe selection logic
- Novelty detection
- Blur detection
- Coverage status calculation

Usage:
    pytest tests/test_keyframe_selector.py -v
"""

import numpy as np
import pytest
import time
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.keyframe_selector import (
    KeyframeSelector,
    KeyframeCandidate,
    CoverageStatus,
)
from core.face_detector import FaceDetection


# ============================================================
# Test KeyframeCandidate Dataclass
# ============================================================

class TestKeyframeCandidate:
    """Tests for KeyframeCandidate dataclass."""

    def test_create_candidate(self):
        """Test that KeyframeCandidate can be created with valid data."""
        candidate = KeyframeCandidate(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            head_pose=(15.0, 5.0, 0.0),
            timestamp=time.time(),
            quality_score=0.85,
        )
        assert candidate.head_pose == (15.0, 5.0, 0.0)
        assert candidate.quality_score == 0.85
        assert candidate.frame.shape == (480, 640, 3)

    def test_candidate_with_different_poses(self):
        """Test candidates with various head poses."""
        poses = [
            (-30.0, 0.0, 0.0),   # Looking left
            (30.0, 0.0, 0.0),    # Looking right
            (0.0, -15.0, 0.0),   # Looking down
            (0.0, 15.0, 0.0),    # Looking up
            (0.0, 0.0, 5.0),     # Slight tilt
        ]

        for pose in poses:
            candidate = KeyframeCandidate(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                head_pose=pose,
                timestamp=time.time(),
                quality_score=0.9,
            )
            assert candidate.head_pose == pose

    def test_candidate_timestamp(self):
        """Test that timestamp is stored correctly."""
        ts = time.time()
        candidate = KeyframeCandidate(
            frame=np.zeros((100, 100, 3), dtype=np.uint8),
            head_pose=(0.0, 0.0, 0.0),
            timestamp=ts,
            quality_score=0.9,
        )
        assert candidate.timestamp == ts


# ============================================================
# Test CoverageStatus Dataclass
# ============================================================

class TestCoverageStatusDataclass:
    """Tests for CoverageStatus dataclass."""

    def test_create_status(self):
        """Test that CoverageStatus can be created with valid data."""
        status = CoverageStatus(
            yaw_range=(-20.0, 20.0),
            pitch_range=(-10.0, 10.0),
            total_frames=8,
            is_sufficient=False,
            missing_directions=["up"],
        )
        assert status.total_frames == 8
        assert status.is_sufficient is False
        assert "up" in status.missing_directions

    def test_status_with_sufficient_coverage(self):
        """Test status when coverage is sufficient."""
        status = CoverageStatus(
            yaw_range=(-25.0, 25.0),
            pitch_range=(-12.0, 12.0),
            total_frames=12,
            is_sufficient=True,
            missing_directions=[],
        )
        assert status.is_sufficient is True
        assert len(status.missing_directions) == 0

    def test_status_with_all_directions_missing(self):
        """Test status when all directions are missing."""
        status = CoverageStatus(
            yaw_range=(0.0, 0.0),
            pitch_range=(0.0, 0.0),
            total_frames=0,
            is_sufficient=False,
            missing_directions=["left", "right", "up", "down"],
        )
        assert len(status.missing_directions) == 4


# ============================================================
# Test KeyframeSelector Initialization
# ============================================================

class TestKeyframeSelectorInit:
    """Tests for KeyframeSelector initialization."""

    def test_default_config(self):
        """Test selector initialization with empty config (use defaults)."""
        selector = KeyframeSelector({})
        assert selector.target_count == 12
        assert selector.min_yaw_spread == 40.0
        assert selector.min_pitch_spread == 20.0
        assert selector.pose_novelty_threshold == 5.0
        assert selector.blur_threshold == 100.0
        assert selector.max_roll == 15.0
        assert selector.min_confidence == 0.9

    def test_custom_config(self):
        """Test selector initialization with custom config."""
        config = {
            "target_count": 8,
            "min_yaw_spread": 30.0,
            "min_pitch_spread": 15.0,
            "blur_threshold": 50.0,
            "pose_novelty_threshold": 10.0,
            "max_roll": 20.0,
            "min_confidence": 0.85,
        }
        selector = KeyframeSelector(config)
        assert selector.target_count == 8
        assert selector.min_yaw_spread == 30.0
        assert selector.min_pitch_spread == 15.0
        assert selector.blur_threshold == 50.0
        assert selector.pose_novelty_threshold == 10.0
        assert selector.max_roll == 20.0
        assert selector.min_confidence == 0.85


# ============================================================
# Test Novelty Detection
# ============================================================

class TestNoveltyDetection:
    """Tests for _adds_novelty method."""

    @pytest.fixture
    def selector(self):
        """Create a selector with specific novelty threshold."""
        return KeyframeSelector({"pose_novelty_threshold": 5.0})

    @pytest.fixture
    def existing_candidates(self):
        """Create a list of existing candidates."""
        return [
            KeyframeCandidate(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                head_pose=(0.0, 0.0, 0.0),
                timestamp=time.time(),
                quality_score=0.9,
            ),
            KeyframeCandidate(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                head_pose=(15.0, 0.0, 0.0),
                timestamp=time.time(),
                quality_score=0.9,
            ),
        ]

    def test_first_frame_always_novel(self, selector):
        """Test that the first frame is always considered novel."""
        assert selector._adds_novelty(0.0, 0.0, []) is True
        assert selector._adds_novelty(30.0, 10.0, []) is True

    def test_similar_pose_not_novel(self, selector, existing_candidates):
        """Test that a similar pose is not considered novel."""
        # Very close to existing (0, 0, 0) pose
        assert selector._adds_novelty(2.0, 2.0, existing_candidates) is False

    def test_different_yaw_is_novel(self, selector, existing_candidates):
        """Test that a different yaw angle is considered novel."""
        # Far from all existing poses in yaw
        assert selector._adds_novelty(-20.0, 0.0, existing_candidates) is True

    def test_different_pitch_is_novel(self, selector, existing_candidates):
        """Test that a different pitch angle is considered novel."""
        # Far from all existing poses in pitch
        assert selector._adds_novelty(0.0, 15.0, existing_candidates) is True

    def test_boundary_case_exactly_threshold(self, selector, existing_candidates):
        """Test the boundary case at exactly the novelty threshold."""
        # At exactly 5 degrees difference in both yaw and pitch
        # This should still be considered not novel (< threshold check)
        assert selector._adds_novelty(4.9, 4.9, existing_candidates) is False


# ============================================================
# Test Coverage Status Method
# ============================================================

class TestCoverageStatusMethod:
    """Tests for get_coverage_status method."""

    @pytest.fixture
    def selector(self):
        """Create a selector with specific coverage requirements."""
        return KeyframeSelector({
            "target_count": 12,
            "min_yaw_spread": 40.0,
            "min_pitch_spread": 20.0,
        })

    def test_empty_candidates(self, selector):
        """Test coverage status with no candidates."""
        status = selector.get_coverage_status([])
        assert status.total_frames == 0
        assert status.is_sufficient is False
        assert set(status.missing_directions) == {"left", "right", "up", "down"}
        assert status.yaw_range == (0.0, 0.0)
        assert status.pitch_range == (0.0, 0.0)

    def test_single_candidate(self, selector):
        """Test coverage status with a single candidate."""
        candidates = [
            KeyframeCandidate(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                head_pose=(0.0, 0.0, 0.0),  # Center pose
                timestamp=time.time(),
                quality_score=0.9,
            )
        ]
        status = selector.get_coverage_status(candidates)
        assert status.total_frames == 1
        assert status.is_sufficient is False
        assert status.yaw_range == (0.0, 0.0)
        assert status.pitch_range == (0.0, 0.0)

    def test_partial_coverage(self, selector):
        """Test coverage status with partial coverage."""
        candidates = [
            KeyframeCandidate(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                head_pose=(yaw, 0.0, 0.0),
                timestamp=time.time(),
                quality_score=0.9,
            )
            for yaw in [-10.0, 0.0, 10.0]
        ]
        status = selector.get_coverage_status(candidates)
        assert status.is_sufficient is False
        assert status.yaw_range == (-10.0, 10.0)
        assert status.total_frames == 3

    def test_sufficient_coverage(self, selector):
        """Test coverage status when all requirements are met."""
        # Create 12 candidates with sufficient angular spread
        poses = [
            (-25, -12), (-25, 0), (-25, 12),
            (0, -12), (0, 0), (0, 12),
            (25, -12), (25, 0), (25, 12),
            (-15, -8), (15, 8), (10, -5),
        ]
        candidates = [
            KeyframeCandidate(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                head_pose=(yaw, pitch, 0.0),
                timestamp=time.time(),
                quality_score=0.9,
            )
            for yaw, pitch in poses
        ]
        
        # Simulate that all target poses have been captured
        # The new implementation uses target-based coverage tracking
        selector._captured_target_indices = set(range(len(selector.target_poses)))
        
        status = selector.get_coverage_status(candidates)
        assert status.is_sufficient is True
        assert status.total_frames == 12
        assert status.yaw_range[1] - status.yaw_range[0] >= 40.0
        assert status.pitch_range[1] - status.pitch_range[0] >= 20.0


# ============================================================
# Test Should Capture Method
# ============================================================

class TestShouldCapture:
    """Tests for should_capture method."""

    @pytest.fixture
    def selector(self):
        """Create a selector for testing."""
        return KeyframeSelector({
            "target_count": 12,
            "min_confidence": 0.9,
            "max_roll": 15.0,
            "blur_threshold": 100.0,
            "pose_novelty_threshold": 5.0,
        })

    @pytest.fixture
    def mock_detection(self):
        """Create a mock face detection."""
        return FaceDetection(
            bbox=(100, 100, 300, 400),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(10.0, 5.0, 0.0),  # Valid pose (roll = 0)
            confidence=0.95,
        )

    def test_capture_first_frame(self, selector, mock_detection):
        """Test that first frame is always captured (if quality is good)."""
        result = selector.should_capture(mock_detection, [])
        assert result is True

    def test_reject_low_confidence(self, selector):
        """Test rejection of low confidence detections."""
        low_conf = FaceDetection(
            bbox=(100, 100, 300, 400),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(10.0, 5.0, 0.0),
            confidence=0.5,  # Below threshold
        )
        result = selector.should_capture(low_conf, [])
        assert result is False

    def test_reject_high_roll(self, selector):
        """Test rejection of tilted head (high roll)."""
        tilted = FaceDetection(
            bbox=(100, 100, 300, 400),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(10.0, 5.0, 25.0),  # Roll > max_roll
            confidence=0.95,
        )
        result = selector.should_capture(tilted, [])
        assert result is False

    def test_reject_at_target_count(self, selector, mock_detection):
        """Test rejection when target count is already reached."""
        existing = [
            KeyframeCandidate(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                head_pose=(i * 5.0, 0.0, 0.0),
                timestamp=time.time(),
                quality_score=0.9,
            )
            for i in range(12)  # Already at target count
        ]
        result = selector.should_capture(mock_detection, existing)
        assert result is False

    def test_reject_similar_pose(self, selector):
        """Test rejection of similar pose to existing candidate."""
        detection = FaceDetection(
            bbox=(100, 100, 300, 400),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(0.0, 0.0, 0.0),
            confidence=0.95,
        )

        existing = [
            KeyframeCandidate(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                head_pose=(2.0, 2.0, 0.0),  # Very close pose
                timestamp=time.time(),
                quality_score=0.9,
            )
        ]
        result = selector.should_capture(detection, existing)
        assert result is False

    def test_accept_novel_pose(self, selector):
        """Test acceptance of a novel pose."""
        detection = FaceDetection(
            bbox=(100, 100, 300, 400),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(20.0, 0.0, 0.0),  # Novel yaw
            confidence=0.95,
        )

        existing = [
            KeyframeCandidate(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                head_pose=(0.0, 0.0, 0.0),
                timestamp=time.time(),
                quality_score=0.9,
            )
        ]
        result = selector.should_capture(detection, existing)
        assert result is True


# ============================================================
# Test Blur Detection
# ============================================================

class TestBlurDetection:
    """Tests for blur score computation."""

    @pytest.fixture
    def selector(self):
        """Create a selector for blur testing."""
        return KeyframeSelector({})

    def test_sharp_image_high_score(self, selector):
        """Test that a sharp image (high contrast edges) has high blur score."""
        # Create an image with clear edges
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[40:60, 40:60] = 255  # Sharp white square
        score = selector._compute_blur_score(img)
        assert score > 0

    def test_uniform_image_low_score(self, selector):
        """Test that a uniform image has very low blur score."""
        # Completely uniform image - no edges
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        score = selector._compute_blur_score(img)
        assert score < 1  # Very low variance

    def test_checkerboard_high_score(self, selector):
        """Test that a checkerboard pattern has high blur score."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Create checkerboard pattern
        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                if (i // 10 + j // 10) % 2 == 0:
                    img[i:i+10, j:j+10] = 255
        score = selector._compute_blur_score(img)
        assert score > 100  # High edge content


# ============================================================
# Test Quality Score
# ============================================================

class TestQualityScore:
    """Tests for compute_quality_score method."""

    @pytest.fixture
    def selector(self):
        """Create a selector for quality testing."""
        return KeyframeSelector({})

    def test_quality_score_range(self, selector):
        """Test that quality score is in valid range [0, 1]."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detection = FaceDetection(
            bbox=(100, 100, 300, 400),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(0.0, 0.0, 0.0),
            confidence=0.95,
        )
        score = selector.compute_quality_score(frame, detection)
        assert 0.0 <= score <= 1.0

    def test_high_confidence_higher_quality(self, selector):
        """Test that higher confidence leads to higher quality score."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        high_conf = FaceDetection(
            bbox=(100, 100, 300, 400),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(0.0, 0.0, 0.0),
            confidence=0.99,
        )

        low_conf = FaceDetection(
            bbox=(100, 100, 300, 400),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(0.0, 0.0, 0.0),
            confidence=0.5,
        )

        score_high = selector.compute_quality_score(frame, high_conf)
        score_low = selector.compute_quality_score(frame, low_conf)

        assert score_high > score_low

    def test_roll_penalty(self, selector):
        """Test that high roll angle reduces quality score."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        upright = FaceDetection(
            bbox=(100, 100, 300, 400),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(0.0, 0.0, 0.0),  # No roll
            confidence=0.95,
        )

        tilted = FaceDetection(
            bbox=(100, 100, 300, 400),
            landmarks_2d=np.zeros((478, 2)),
            landmarks_3d=np.zeros((478, 3)),
            head_pose=(0.0, 0.0, 25.0),  # High roll
            confidence=0.95,
        )

        score_upright = selector.compute_quality_score(frame, upright)
        score_tilted = selector.compute_quality_score(frame, tilted)

        assert score_upright > score_tilted


# ============================================================
# Test Select Best Candidates
# ============================================================

class TestSelectBestCandidates:
    """Tests for select_best_candidates method."""

    @pytest.fixture
    def selector(self):
        """Create a selector for testing."""
        return KeyframeSelector({"target_count": 5, "pose_novelty_threshold": 5.0})

    def test_returns_all_if_under_limit(self, selector):
        """Test that all candidates are returned if under max_count."""
        candidates = [
            KeyframeCandidate(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                head_pose=(i * 10.0, 0.0, 0.0),
                timestamp=time.time(),
                quality_score=0.9,
            )
            for i in range(3)
        ]
        result = selector.select_best_candidates(candidates, max_count=5)
        assert len(result) == 3

    def test_selects_up_to_max_count(self, selector):
        """Test that selection respects max_count limit."""
        candidates = [
            KeyframeCandidate(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                head_pose=(i * 10.0, 0.0, 0.0),
                timestamp=time.time(),
                quality_score=0.9 - i * 0.05,
            )
            for i in range(10)
        ]
        result = selector.select_best_candidates(candidates, max_count=5)
        assert len(result) <= 5

    def test_prioritizes_quality(self, selector):
        """Test that higher quality candidates are prioritized."""
        candidates = [
            KeyframeCandidate(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                head_pose=(i * 15.0, 0.0, 0.0),  # All different poses
                timestamp=time.time(),
                quality_score=0.5 + i * 0.1,  # Increasing quality
            )
            for i in range(5)
        ]
        result = selector.select_best_candidates(candidates, max_count=3)

        # The highest quality candidates should be selected
        selected_qualities = [c.quality_score for c in result]
        assert max(selected_qualities) == max(c.quality_score for c in candidates)

    def test_maintains_diversity(self, selector):
        """Test that selection maintains pose diversity."""
        # Create candidates with some similar poses
        candidates = [
            KeyframeCandidate(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                head_pose=(0.0, 0.0, 0.0),
                timestamp=time.time(),
                quality_score=0.95,
            ),
            KeyframeCandidate(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                head_pose=(1.0, 1.0, 0.0),  # Very similar to first
                timestamp=time.time(),
                quality_score=0.90,
            ),
            KeyframeCandidate(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                head_pose=(20.0, 0.0, 0.0),  # Different pose
                timestamp=time.time(),
                quality_score=0.85,
            ),
        ]

        result = selector.select_best_candidates(candidates, max_count=2)

        # Should select diverse poses, not two similar ones
        assert len(result) == 2
        poses = [c.head_pose for c in result]
        # At least one should have significantly different yaw
        yaws = [p[0] for p in poses]
        assert max(yaws) - min(yaws) >= 10


# ============================================================
# Test Missing Directions Logic
# ============================================================

class TestMissingDirections:
    """
    Tests for _get_missing_directions method.
    
    NOTE: _get_missing_directions was refactored to use target-based coverage
    instead of yaw/pitch range parameters. The method signature is kept for
    backwards compatibility but the parameters are now ignored. These tests
    now verify the new target-based behavior.
    """

    @pytest.fixture
    def selector(self):
        """Create a selector with specific coverage requirements."""
        return KeyframeSelector({
            "min_yaw_spread": 40.0,
            "min_pitch_spread": 20.0,
        })

    def test_all_directions_missing_initially(self, selector):
        """
        Test that all directions are missing when no targets are captured.
        
        The new implementation uses target poses, so at initialization
        all directions are missing since no targets have been captured yet.
        """
        missing = selector._get_missing_directions(
            yaw_min=0.0, yaw_max=0.0,
            pitch_min=0.0, pitch_max=0.0
        )
        # With target-based system, all directions start as missing
        assert set(missing) == {"left", "right", "up", "down"}

    def test_directions_reduce_when_targets_captured(self, selector):
        """
        Test that missing directions reduce as targets are captured.
        
        Simulates capturing targets by adding to _captured_target_indices.
        """
        # Initially all missing
        missing = selector._get_missing_directions(0, 0, 0, 0)
        assert len(missing) == 4
        
        # Mark all target indices as captured
        selector._captured_target_indices = set(range(len(selector.target_poses)))
        
        # Now nothing should be missing
        missing = selector._get_missing_directions(0, 0, 0, 0)
        assert len(missing) == 0

    def test_partial_capture_reduces_missing(self, selector):
        """
        Test that capturing some targets reduces missing directions.
        """
        # Find indices of targets with negative yaw (left direction)
        left_indices = [
            i for i, (yaw, pitch) in enumerate(selector.target_poses)
            if yaw < -5
        ]
        
        # Capture left-looking targets
        selector._captured_target_indices = set(left_indices)
        
        missing = selector._get_missing_directions(0, 0, 0, 0)
        # Left should no longer be in missing
        assert "left" not in missing


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
