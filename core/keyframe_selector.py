"""
Keyframe Selection Module

This module selects optimal frames from a webcam stream for face enrollment.
The goal is to capture a diverse set of face images that cover different
viewing angles (yaw, pitch) while maintaining image quality.

The selector uses a target-pose strategy: N target poses are pre-defined
on a uniform grid across the front-facing yaw/pitch range. The user is
guided to each target, and frames are captured when the head pose is
within tolerance of an uncaptured target.

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

import logging
import math
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from core.face_detector import FaceDetection

logger = logging.getLogger(__name__)


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
        targets_captured: Number of target poses that have been captured.
        targets_total: Total number of target poses.
        next_target: (yaw, pitch) of the recommended next target, or None.
        next_target_direction: Human-readable direction like "RIGHT and UP".
    """

    yaw_range: Tuple[float, float]
    pitch_range: Tuple[float, float]
    total_frames: int
    is_sufficient: bool
    missing_directions: List[str]
    targets_captured: int = 0
    targets_total: int = 0
    next_target: Optional[Tuple[float, float]] = None
    next_target_direction: Optional[str] = None


class KeyframeSelector:
    """
    Selects diverse, high-quality keyframes for face enrollment.

    Uses a target-pose strategy: N target poses are pre-defined on a uniform
    grid across the front-facing yaw/pitch range. Frames are captured when
    the head pose is within tolerance of an uncaptured target. This ensures
    uniform angular coverage regardless of the user's head movement pattern.
    """

    # Practical yaw/pitch ranges for face detection (beyond these, detection fails)
    YAW_RANGE = (-25.0, 25.0)
    PITCH_RANGE = (-15.0, 15.0)
    TARGET_TOLERANCE = 7.0  # degrees from target center to accept

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
        """
        self.target_count = config.get("target_count", 12)
        self.min_yaw_spread = config.get("min_yaw_spread", 40.0)
        self.min_pitch_spread = config.get("min_pitch_spread", 20.0)
        self.pose_novelty_threshold = config.get("pose_novelty_threshold", 5.0)
        self.blur_threshold = config.get("blur_threshold", 100.0)
        self.max_roll = config.get("max_roll", 15.0)
        self.min_confidence = config.get("min_confidence", 0.9)

        # Generate target poses and tracking state
        self.target_poses = self._generate_target_poses(self.target_count)
        self._captured_target_indices: set = set()

        logger.info(f"KeyframeSelector initialized with {len(self.target_poses)} target poses")
        for i, (y, p) in enumerate(self.target_poses):
            logger.debug(f"  Target {i}: yaw={y:.1f}, pitch={p:.1f}")

    def _generate_target_poses(self, n: int) -> List[Tuple[float, float]]:
        """
        Generate N target poses uniformly distributed across the yaw/pitch range.

        Uses a grid layout with the best factorization of N into rows × cols.
        """
        yaw_min, yaw_max = self.YAW_RANGE
        pitch_min, pitch_max = self.PITCH_RANGE

        # Find best grid factorization: prefer more yaw columns (horizontal variety)
        n_yaw, n_pitch = self._best_grid_factors(n)

        # Generate uniform grid points
        if n_yaw == 1:
            yaw_values = [0.0]
        else:
            yaw_values = [
                yaw_min + i * (yaw_max - yaw_min) / (n_yaw - 1)
                for i in range(n_yaw)
            ]

        if n_pitch == 1:
            pitch_values = [0.0]
        else:
            pitch_values = [
                pitch_min + i * (pitch_max - pitch_min) / (n_pitch - 1)
                for i in range(n_pitch)
            ]

        targets = []
        for pitch in pitch_values:
            for yaw in yaw_values:
                targets.append((yaw, pitch))

        return targets[:n]  # Trim to exactly N if factorization gives more

    @staticmethod
    def _best_grid_factors(n: int) -> Tuple[int, int]:
        """Find (n_yaw, n_pitch) factorization of n, preferring wider yaw."""
        best = (n, 1)
        best_ratio = float('inf')

        for cols in range(1, n + 1):
            if n % cols == 0:
                rows = n // cols
                # Prefer roughly 2:1 ratio (more yaw than pitch)
                ratio_diff = abs(cols / max(rows, 1) - 2.0)
                if ratio_diff < best_ratio:
                    best_ratio = ratio_diff
                    best = (cols, rows)  # (n_yaw, n_pitch)

        return best

    def should_capture(
        self,
        detection: FaceDetection,
        existing: List[KeyframeCandidate],
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Determine if a frame should be captured as a keyframe.

        Uses target-pose matching: captures when the head pose is within
        tolerance of an uncaptured target pose.

        Args:
            detection: FaceDetection result from the face detector.
            existing: List of already captured KeyframeCandidates.
            frame: Optional original frame for blur detection.

        Returns:
            True if this frame should be captured as a keyframe.
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
            logger.debug(f"Blur score: {blur_score:.1f} (threshold: {self.blur_threshold})")
            if blur_score < self.blur_threshold:
                return False

        # Check 5: Head pose must match an uncaptured target
        target_idx = self._find_matching_target(yaw, pitch)
        if target_idx < 0:
            return False

        # Mark this target as captured
        self._captured_target_indices.add(target_idx)
        return True

    def _find_matching_target(self, yaw: float, pitch: float) -> int:
        """
        Find the nearest uncaptured target within tolerance.

        Returns the target index, or -1 if no uncaptured target is close enough.
        """
        best_idx = -1
        best_dist = float('inf')

        for i, (t_yaw, t_pitch) in enumerate(self.target_poses):
            if i in self._captured_target_indices:
                continue

            dist = math.sqrt((yaw - t_yaw) ** 2 + (pitch - t_pitch) ** 2)
            if dist < self.TARGET_TOLERANCE and dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx

    def get_next_target(self, current_yaw: float = 0.0, current_pitch: float = 0.0) -> Optional[Tuple[float, float]]:
        """
        Get the nearest uncaptured target pose from the current head position.

        Returns (yaw, pitch) of the next target, or None if all captured.
        """
        best_target = None
        best_dist = float('inf')

        for i, (t_yaw, t_pitch) in enumerate(self.target_poses):
            if i in self._captured_target_indices:
                continue

            dist = math.sqrt((current_yaw - t_yaw) ** 2 + (current_pitch - t_pitch) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_target = (t_yaw, t_pitch)

        return best_target

    @staticmethod
    def _pose_to_direction(target_yaw: float, target_pitch: float) -> str:
        """Convert a target pose to a human-readable direction string."""
        parts = []

        if target_yaw < -5:
            parts.append("LEFT")
        elif target_yaw > 5:
            parts.append("RIGHT")

        if target_pitch < -5:
            parts.append("DOWN")
        elif target_pitch > 5:
            parts.append("UP")

        if not parts:
            return "CENTER"

        return " and ".join(parts)

    def reset(self):
        """Reset captured target tracking (e.g., when user presses 'r')."""
        self._captured_target_indices.clear()

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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)

    def _adds_novelty(
        self, yaw: float, pitch: float, existing: List[KeyframeCandidate]
    ) -> bool:
        """
        Check if the given pose adds novelty to the existing keyframe set.
        Kept for use by select_best_candidates().
        """
        if len(existing) == 0:
            return True

        for candidate in existing:
            existing_yaw, existing_pitch, _ = candidate.head_pose
            yaw_diff = abs(yaw - existing_yaw)
            pitch_diff = abs(pitch - existing_pitch)

            if yaw_diff < self.pose_novelty_threshold and pitch_diff < self.pose_novelty_threshold:
                return False

        return True

    def get_coverage_status(self, candidates: List[KeyframeCandidate]) -> CoverageStatus:
        """
        Get the current coverage status of captured keyframes.

        Reports target-based progress: how many of the pre-defined target
        poses have been captured, and which direction to look next.
        """
        n_captured = len(self._captured_target_indices)
        n_total = len(self.target_poses)

        # Handle empty case
        if len(candidates) == 0:
            next_target = self.get_next_target()
            next_dir = self._pose_to_direction(*next_target) if next_target else None
            return CoverageStatus(
                yaw_range=(0.0, 0.0),
                pitch_range=(0.0, 0.0),
                total_frames=0,
                is_sufficient=False,
                missing_directions=["left", "right", "up", "down"],
                targets_captured=n_captured,
                targets_total=n_total,
                next_target=next_target,
                next_target_direction=next_dir,
            )

        # Extract yaw and pitch values
        yaws = [c.head_pose[0] for c in candidates]
        pitches = [c.head_pose[1] for c in candidates]

        yaw_min, yaw_max = min(yaws), max(yaws)
        pitch_min, pitch_max = min(pitches), max(pitches)

        # Missing directions based on uncaptured targets
        missing = self._get_missing_from_targets()

        # Sufficient when all targets captured
        is_sufficient = n_captured >= n_total

        # Next target closest to current position (use last candidate's pose)
        current_yaw, current_pitch = yaws[-1], pitches[-1]
        next_target = self.get_next_target(current_yaw, current_pitch)
        next_dir = self._pose_to_direction(*next_target) if next_target else None

        return CoverageStatus(
            yaw_range=(yaw_min, yaw_max),
            pitch_range=(pitch_min, pitch_max),
            total_frames=len(candidates),
            is_sufficient=is_sufficient,
            missing_directions=missing,
            targets_captured=n_captured,
            targets_total=n_total,
            next_target=next_target,
            next_target_direction=next_dir,
        )

    def _get_missing_from_targets(self) -> List[str]:
        """Determine missing directions based on uncaptured target poses."""
        directions = set()
        for i, (t_yaw, t_pitch) in enumerate(self.target_poses):
            if i in self._captured_target_indices:
                continue
            if t_yaw < -5:
                directions.add("left")
            if t_yaw > 5:
                directions.add("right")
            if t_pitch < -5:
                directions.add("down")
            if t_pitch > 5:
                directions.add("up")
        return sorted(directions)

    def _get_missing_directions(
        self, yaw_min: float, yaw_max: float, pitch_min: float, pitch_max: float
    ) -> List[str]:
        """Kept for backwards compatibility."""
        return self._get_missing_from_targets()

    def compute_quality_score(
        self, frame: np.ndarray, detection: FaceDetection
    ) -> float:
        """
        Compute an overall quality score for a potential keyframe.
        """
        blur = self._compute_blur_score(frame)
        blur_normalized = min(1.0, blur / 500.0)

        confidence = detection.confidence

        _, _, roll = detection.head_pose
        roll_score = max(0.0, 1.0 - abs(roll) / 30.0)

        x1, y1, x2, y2 = detection.bbox
        face_area = (x2 - x1) * (y2 - y1)
        frame_area = frame.shape[0] * frame.shape[1]
        size_ratio = face_area / frame_area
        size_score = min(1.0, size_ratio / 0.1)

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
        """
        if max_count is None:
            max_count = self.target_count

        if len(candidates) <= max_count:
            return candidates

        sorted_candidates = sorted(
            candidates, key=lambda c: c.quality_score, reverse=True
        )

        selected = []
        for candidate in sorted_candidates:
            if len(selected) >= max_count:
                break
            yaw, pitch, _ = candidate.head_pose
            if self._adds_novelty(yaw, pitch, selected):
                selected.append(candidate)

        return selected


if __name__ == "__main__":
    print("Testing KeyframeSelector...")

    config = {
        "target_count": 12,
        "min_yaw_spread": 40.0,
        "min_pitch_spread": 20.0,
        "pose_novelty_threshold": 5.0,
        "blur_threshold": 100.0,
    }

    selector = KeyframeSelector(config)
    print(f"Generated {len(selector.target_poses)} target poses:")
    for i, (y, p) in enumerate(selector.target_poses):
        print(f"  Target {i}: yaw={y:.1f}, pitch={p:.1f}")

    print(f"\nGrid: {selector._best_grid_factors(12)} (n_yaw x n_pitch)")
    print(f"Tolerance: {selector.TARGET_TOLERANCE} degrees")
