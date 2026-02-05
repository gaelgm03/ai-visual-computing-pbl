"""
Keyframe Selection Module

This module selects optimal frames from a webcam stream for face enrollment.
The goal is to capture a diverse set of face images that cover different
viewing angles (yaw, pitch) while maintaining image quality.

During enrollment, the user slowly rotates their head while the system
automatically selects keyframes that:
1. Are not blurry
2. Have good face detection confidence
3. Add angular novelty (different head pose from existing frames)

Usage:
    from core.keyframe_selector import KeyframeSelector, KeyframeCandidate
    from core.face_detector import FaceDetector, FaceDetection

    selector = KeyframeSelector(config)
    candidates = []

    # In your capture loop:
    detection = face_detector.detect(frame)
    if detection and selector.should_capture(detection, candidates):
        candidate = KeyframeCandidate(
            frame=cropped_face,
            head_pose=detection.head_pose,
            timestamp=time.time(),
            quality_score=compute_quality(frame)
        )
        candidates.append(candidate)

    status = selector.get_coverage_status(candidates)
    if status.is_sufficient:
        # Ready for MASt3R processing
        pass
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from core.face_detector import FaceDetection


@dataclass
class KeyframeCandidate:
    """
    A candidate keyframe selected for enrollment.

    Attributes:
        frame: The cropped face image (BGR format, numpy array).
        head_pose: Head rotation angles (yaw, pitch, roll) in degrees.
        timestamp: Unix timestamp when the frame was captured.
        quality_score: Image quality score (higher is better).
                       Based on blur detection and face confidence.
    """

    frame: np.ndarray
    head_pose: Tuple[float, float, float]  # (yaw, pitch, roll) in degrees
    timestamp: float
    quality_score: float


@dataclass
class CoverageStatus:
    """
    Status of the angular coverage achieved by captured keyframes.

    This helps the UI guide the user on which directions to turn.

    Attributes:
        yaw_range: (min_yaw, max_yaw) range of captured yaw angles in degrees.
        pitch_range: (min_pitch, max_pitch) range of captured pitch angles.
        total_frames: Number of keyframes captured so far.
        is_sufficient: True if we have enough diverse keyframes for enrollment.
        missing_directions: List of directions user still needs to turn.
                            Possible values: "left", "right", "up", "down"
    """

    yaw_range: Tuple[float, float]
    pitch_range: Tuple[float, float]
    total_frames: int
    is_sufficient: bool
    missing_directions: List[str]


class KeyframeSelector:
    """
    Selects diverse, high-quality keyframes for face enrollment.

    The selector ensures that captured frames:
    1. Cover a sufficient range of head poses (yaw/pitch angles)
    2. Meet quality requirements (not blurry, good detection confidence)
    3. Add novelty (each frame is sufficiently different from existing ones)

    Attributes:
        target_count: Target number of keyframes to capture (e.g., 12).
        min_yaw_spread: Minimum yaw angle spread required (e.g., 40 degrees).
        min_pitch_spread: Minimum pitch angle spread required (e.g., 20 degrees).
        max_roll: Maximum allowed roll angle (frames with more roll are rejected).
        pose_novelty_threshold: Minimum angular difference from existing frames.
        blur_threshold: Minimum Laplacian variance (higher = sharper image).
        min_confidence: Minimum face detection confidence to accept frame.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the KeyframeSelector.

        Args:
            config: Configuration dictionary containing:
                - target_count: Number of keyframes to capture
                - min_yaw_spread: Required yaw spread in degrees
                - min_pitch_spread: Required pitch spread in degrees
                - pose_novelty_threshold: Min angle delta for new frame
                - blur_threshold: Laplacian variance threshold
                - (optional) max_roll: Maximum roll angle to accept
                - (optional) min_confidence: Minimum detection confidence

        Example:
            config = {
                "target_count": 12,
                "min_yaw_spread": 40.0,
                "min_pitch_spread": 20.0,
                "pose_novelty_threshold": 5.0,
                "blur_threshold": 100.0
            }
            selector = KeyframeSelector(config)
        """
        # Load configuration with defaults
        self.target_count = config.get("target_count", 12)
        self.min_yaw_spread = config.get("min_yaw_spread", 40.0)
        self.min_pitch_spread = config.get("min_pitch_spread", 20.0)
        self.pose_novelty_threshold = config.get("pose_novelty_threshold", 5.0)
        self.blur_threshold = config.get("blur_threshold", 100.0)
        self.max_roll = config.get("max_roll", 15.0)
        self.min_confidence = config.get("min_confidence", 0.9)

    def should_capture(
        self,
        detection: FaceDetection,
        existing: List[KeyframeCandidate],
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Determine if a frame should be captured as a keyframe.

        This method checks multiple criteria to decide if the current frame
        adds value to the enrollment dataset.

        Args:
            detection: FaceDetection result from the face detector.
            existing: List of already captured KeyframeCandidates.
            frame: Optional original frame for blur detection.
                   If not provided, blur check is skipped.

        Returns:
            True if this frame should be captured as a keyframe.

        Criteria checked (in order):
            1. Not already at target count
            2. Detection confidence is high enough
            3. Roll angle is within acceptable range
            4. Image is not blurry (if frame provided)
            5. Head pose adds novelty to existing set
        """
        # Check 1: Don't capture more than target count
        if len(existing) >= self.target_count:
            return False

        # Check 2: Detection confidence must be high enough
        if detection.confidence < self.min_confidence:
            return False

        yaw, pitch, roll = detection.head_pose

        # Check 3: Roll should be near zero (face not tilted)
        if abs(roll) > self.max_roll:
            return False

        # Check 4: Image should not be blurry
        if frame is not None:
            blur_score = self._compute_blur_score(frame)
            if blur_score < self.blur_threshold:
                return False

        # Check 5: Head pose should add novelty
        if not self._adds_novelty(yaw, pitch, existing):
            return False

        return True

    def _compute_blur_score(self, frame: np.ndarray) -> float:
        """
        Compute image sharpness using Laplacian variance.

        The Laplacian operator highlights edges in an image.
        A sharp image has strong edges (high variance), while a
        blurry image has weak edges (low variance).

        Args:
            frame: Image in BGR format.

        Returns:
            Blur score (higher = sharper). Typical threshold is around 100.
        """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Laplacian operator to detect edges
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Variance of Laplacian indicates sharpness
        # Higher variance = more edges = sharper image
        variance = laplacian.var()

        return float(variance)

    def _adds_novelty(
        self, yaw: float, pitch: float, existing: List[KeyframeCandidate]
    ) -> bool:
        """
        Check if the given pose adds novelty to the existing keyframe set.

        A pose is considered novel if it's sufficiently different from
        all existing poses. This ensures we capture diverse viewpoints.

        Args:
            yaw: Current yaw angle in degrees.
            pitch: Current pitch angle in degrees.
            existing: List of existing keyframes.

        Returns:
            True if this pose adds novelty (different enough from all existing).
        """
        # First frame is always novel
        if len(existing) == 0:
            return True

        # Check distance to all existing poses
        for candidate in existing:
            existing_yaw, existing_pitch, _ = candidate.head_pose

            # Calculate angular distance (simple Euclidean in angle space)
            yaw_diff = abs(yaw - existing_yaw)
            pitch_diff = abs(pitch - existing_pitch)

            # If either angle is significantly different, consider it novel
            # Use max instead of sum to be more lenient
            if yaw_diff < self.pose_novelty_threshold and pitch_diff < self.pose_novelty_threshold:
                # Found an existing frame too similar to this one
                return False

        return True

    def get_coverage_status(self, candidates: List[KeyframeCandidate]) -> CoverageStatus:
        """
        Get the current coverage status of captured keyframes.

        This helps the UI know:
        - How many frames have been captured
        - What angular range has been covered
        - Whether enrollment can proceed
        - Which directions the user still needs to turn

        Args:
            candidates: List of captured KeyframeCandidates.

        Returns:
            CoverageStatus object with detailed coverage information.

        Example:
            status = selector.get_coverage_status(candidates)
            if not status.is_sufficient:
                for direction in status.missing_directions:
                    print(f"Please turn your head {direction}")
        """
        # Handle empty case
        if len(candidates) == 0:
            return CoverageStatus(
                yaw_range=(0.0, 0.0),
                pitch_range=(0.0, 0.0),
                total_frames=0,
                is_sufficient=False,
                missing_directions=["left", "right", "up", "down"],
            )

        # Extract yaw and pitch values from all candidates
        yaws = [c.head_pose[0] for c in candidates]
        pitches = [c.head_pose[1] for c in candidates]

        # Calculate ranges
        yaw_min, yaw_max = min(yaws), max(yaws)
        pitch_min, pitch_max = min(pitches), max(pitches)

        yaw_spread = yaw_max - yaw_min
        pitch_spread = pitch_max - pitch_min

        # Determine missing directions
        missing = self._get_missing_directions(yaw_min, yaw_max, pitch_min, pitch_max)

        # Check if coverage is sufficient
        # Need: enough frames AND enough angular spread
        has_enough_frames = len(candidates) >= self.target_count
        has_enough_yaw = yaw_spread >= self.min_yaw_spread
        has_enough_pitch = pitch_spread >= self.min_pitch_spread

        # We require minimum frames OR both angular spreads met
        is_sufficient = has_enough_frames and has_enough_yaw and has_enough_pitch

        return CoverageStatus(
            yaw_range=(yaw_min, yaw_max),
            pitch_range=(pitch_min, pitch_max),
            total_frames=len(candidates),
            is_sufficient=is_sufficient,
            missing_directions=missing,
        )

    def _get_missing_directions(
        self, yaw_min: float, yaw_max: float, pitch_min: float, pitch_max: float
    ) -> List[str]:
        """
        Determine which directions the user still needs to turn.

        Based on current coverage and target ranges, this identifies
        directions where more frames are needed.

        Args:
            yaw_min: Minimum captured yaw angle.
            yaw_max: Maximum captured yaw angle.
            pitch_min: Minimum captured pitch angle.
            pitch_max: Maximum captured pitch angle.

        Returns:
            List of direction strings: "left", "right", "up", "down"
        """
        missing = []

        # Calculate how much more yaw spread we need
        current_yaw_spread = yaw_max - yaw_min
        needed_yaw_spread = self.min_yaw_spread - current_yaw_spread

        if needed_yaw_spread > 0:
            # Check which direction has more room to expand
            # Target range is roughly -20 to +20 for yaw (40 degree spread)
            target_yaw_half = self.min_yaw_spread / 2

            # If not enough coverage on the left (negative yaw)
            if yaw_min > -target_yaw_half:
                missing.append("left")

            # If not enough coverage on the right (positive yaw)
            if yaw_max < target_yaw_half:
                missing.append("right")

        # Calculate how much more pitch spread we need
        current_pitch_spread = pitch_max - pitch_min
        needed_pitch_spread = self.min_pitch_spread - current_pitch_spread

        if needed_pitch_spread > 0:
            # Target range is roughly -10 to +10 for pitch (20 degree spread)
            target_pitch_half = self.min_pitch_spread / 2

            # If not enough coverage looking down (negative pitch)
            if pitch_min > -target_pitch_half:
                missing.append("down")

            # If not enough coverage looking up (positive pitch)
            if pitch_max < target_pitch_half:
                missing.append("up")

        return missing

    def compute_quality_score(
        self, frame: np.ndarray, detection: FaceDetection
    ) -> float:
        """
        Compute an overall quality score for a potential keyframe.

        This combines multiple quality metrics into a single score
        that can be used for ranking or filtering frames.

        Args:
            frame: The image frame (BGR format).
            detection: Face detection result for this frame.

        Returns:
            Quality score between 0.0 and 1.0 (higher is better).
        """
        # Component 1: Blur score (normalized)
        blur = self._compute_blur_score(frame)
        # Normalize: 100 is threshold, 500 is very sharp
        blur_normalized = min(1.0, blur / 500.0)

        # Component 2: Detection confidence
        confidence = detection.confidence

        # Component 3: Roll penalty (face should be upright)
        _, _, roll = detection.head_pose
        roll_score = max(0.0, 1.0 - abs(roll) / 30.0)  # 0 at 30 degrees

        # Component 4: Face size (larger faces are better)
        x1, y1, x2, y2 = detection.bbox
        face_area = (x2 - x1) * (y2 - y1)
        frame_area = frame.shape[0] * frame.shape[1]
        size_ratio = face_area / frame_area
        # Normalize: 0.1 (10% of frame) is good
        size_score = min(1.0, size_ratio / 0.1)

        # Weighted combination
        quality = (
            0.3 * blur_normalized
            + 0.3 * confidence
            + 0.2 * roll_score
            + 0.2 * size_score
        )

        return quality

    def select_best_candidates(
        self, candidates: List[KeyframeCandidate], max_count: Optional[int] = None
    ) -> List[KeyframeCandidate]:
        """
        Select the best candidates from a list, prioritizing diversity and quality.

        This can be used to reduce the number of keyframes while maintaining
        good angular coverage.

        Args:
            candidates: List of all captured keyframe candidates.
            max_count: Maximum number to select. If None, uses target_count.

        Returns:
            Filtered list of best candidates.
        """
        if max_count is None:
            max_count = self.target_count

        if len(candidates) <= max_count:
            return candidates

        # Sort by quality score (highest first)
        sorted_candidates = sorted(
            candidates, key=lambda c: c.quality_score, reverse=True
        )

        # Greedy selection: pick best quality while maintaining diversity
        selected = []

        for candidate in sorted_candidates:
            if len(selected) >= max_count:
                break

            # Check if this candidate adds novelty to selected set
            yaw, pitch, _ = candidate.head_pose
            if self._adds_novelty(yaw, pitch, selected):
                selected.append(candidate)

        return selected


if __name__ == "__main__":
    # Quick test of the keyframe selector
    print("Testing KeyframeSelector...")

    config = {
        "target_count": 12,
        "min_yaw_spread": 40.0,
        "min_pitch_spread": 20.0,
        "pose_novelty_threshold": 5.0,
        "blur_threshold": 100.0,
    }

    selector = KeyframeSelector(config)
    print("KeyframeSelector initialized successfully!")

    # Simulate some keyframe candidates
    import time

    # Create dummy candidates with different poses
    test_candidates = []
    poses = [
        (-20, 0, 0),  # Looking left
        (0, 0, 0),  # Center
        (20, 0, 0),  # Looking right
        (0, -10, 0),  # Looking down
        (0, 10, 0),  # Looking up
    ]

    for i, pose in enumerate(poses):
        candidate = KeyframeCandidate(
            frame=np.zeros((480, 640, 3), dtype=np.uint8),  # Dummy frame
            head_pose=pose,
            timestamp=time.time(),
            quality_score=0.9,
        )
        test_candidates.append(candidate)

    # Get coverage status
    status = selector.get_coverage_status(test_candidates)
    print(f"\nCoverage Status:")
    print(f"  Total frames: {status.total_frames}")
    print(f"  Yaw range: {status.yaw_range}")
    print(f"  Pitch range: {status.pitch_range}")
    print(f"  Is sufficient: {status.is_sufficient}")
    print(f"  Missing directions: {status.missing_directions}")
