"""
Enrollment guide UI component for MASt3R Face Authentication System.
CS-2 Primary Ownership.

Provides visual guidance for head rotation during enrollment,
including directional arrows and progress indicators.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class HeadPose:
    """Represents head orientation in degrees."""
    yaw: float = 0.0    # Left(-) / Right(+)
    pitch: float = 0.0  # Down(-) / Up(+)
    roll: float = 0.0   # Tilt


@dataclass
class CoverageStatus:
    """Status of head pose coverage during enrollment."""
    yaw_range: Tuple[float, float] = (-30.0, 30.0)  # Target range
    pitch_range: Tuple[float, float] = (-15.0, 15.0)  # Target range
    captured_yaw: Tuple[float, float] = (0.0, 0.0)  # Actual captured
    captured_pitch: Tuple[float, float] = (0.0, 0.0)  # Actual captured
    total_captured: int = 0
    target_count: int = 12
    is_sufficient: bool = False
    missing_directions: List[str] = field(default_factory=list)


@dataclass
class EnrollmentConfig:
    """Configuration for enrollment guidance."""
    target_frames: int = 12
    min_yaw_spread: float = 40.0  # degrees
    min_pitch_spread: float = 20.0  # degrees
    max_roll: float = 15.0  # reject frames with excessive roll


class EnrollmentGuide:
    """
    Guides the user through the enrollment process with visual feedback.
    
    Responsibilities:
    - Display directional arrows prompting head movement
    - Show progress meter for captured frames
    - Indicate pose coverage (which angles have been captured)
    - Provide real-time feedback on face detection status
    """
    
    # Direction names for missing coverage feedback
    DIRECTIONS = {
        "left": (-30, -15),   # yaw range
        "right": (15, 30),
        "up": (8, 15),        # pitch range
        "down": (-15, -8),
        "center": (-10, 10),  # both yaw and pitch
    }
    
    def __init__(self, config: Optional[EnrollmentConfig] = None):
        self.config = config or EnrollmentConfig()
        self._captured_poses: List[HeadPose] = []
        self._coverage = CoverageStatus(target_count=self.config.target_frames)
    
    def reset(self) -> None:
        """Reset enrollment state for a new session."""
        self._captured_poses = []
        self._coverage = CoverageStatus(target_count=self.config.target_frames)
    
    def update_coverage(self, poses: List[HeadPose]) -> CoverageStatus:
        """
        Update coverage status based on captured poses.
        
        Args:
            poses: List of head poses captured so far.
        
        Returns:
            Updated CoverageStatus.
        """
        self._captured_poses = poses
        
        if not poses:
            self._coverage = CoverageStatus(target_count=self.config.target_frames)
            return self._coverage
        
        # Calculate captured ranges
        yaws = [p.yaw for p in poses]
        pitches = [p.pitch for p in poses]
        
        captured_yaw = (min(yaws), max(yaws))
        captured_pitch = (min(pitches), max(pitches))
        
        yaw_spread = captured_yaw[1] - captured_yaw[0]
        pitch_spread = captured_pitch[1] - captured_pitch[0]
        
        # Determine missing directions
        missing = []
        if captured_yaw[0] > -15:
            missing.append("left")
        if captured_yaw[1] < 15:
            missing.append("right")
        if captured_pitch[1] < 8:
            missing.append("up")
        if captured_pitch[0] > -8:
            missing.append("down")
        
        # Check if coverage is sufficient
        is_sufficient = (
            len(poses) >= self.config.target_frames and
            yaw_spread >= self.config.min_yaw_spread and
            pitch_spread >= self.config.min_pitch_spread
        )
        
        self._coverage = CoverageStatus(
            captured_yaw=captured_yaw,
            captured_pitch=captured_pitch,
            total_captured=len(poses),
            target_count=self.config.target_frames,
            is_sufficient=is_sufficient,
            missing_directions=missing,
        )
        
        return self._coverage
    
    def get_next_direction(self) -> Optional[str]:
        """
        Get the next direction the user should turn.
        
        Returns:
            Direction string ('left', 'right', 'up', 'down') or None if sufficient.
        """
        if self._coverage.is_sufficient:
            return None
        
        if self._coverage.missing_directions:
            return self._coverage.missing_directions[0]
        
        return "center"
    
    def get_progress_percent(self) -> float:
        """Get enrollment progress as a percentage (0-100)."""
        return min(100.0, (self._coverage.total_captured / self._coverage.target_count) * 100)
    
    def get_direction_arrow_angle(self, direction: str) -> float:
        """
        Get the rotation angle for the direction arrow.
        
        Args:
            direction: One of 'left', 'right', 'up', 'down', 'center'
        
        Returns:
            Rotation angle in degrees (0 = right, 90 = down, etc.)
        """
        angles = {
            "right": 0,
            "down": 90,
            "left": 180,
            "up": 270,
            "center": -1,  # Special case: no arrow
        }
        return angles.get(direction, -1)
    
    def format_status_message(self) -> str:
        """Get a human-readable status message for the UI."""
        if self._coverage.is_sufficient:
            return "✅ Coverage complete! Processing enrollment..."
        
        captured = self._coverage.total_captured
        target = self._coverage.target_count
        
        direction = self.get_next_direction()
        if direction == "center":
            instruction = "Look at the camera"
        elif direction:
            instruction = f"Turn your head {direction}"
        else:
            instruction = "Hold steady"
        
        return f"📷 {captured}/{target} frames | {instruction}"
    
    def get_coverage_visualization_data(self) -> Dict:
        """
        Get data for visualizing pose coverage (for Gradio/Plotly).
        
        Returns:
            Dict with coverage data suitable for visualization.
        """
        # Captured poses as scatter points
        poses_data = [
            {"yaw": p.yaw, "pitch": p.pitch}
            for p in self._captured_poses
        ]
        
        return {
            "poses": poses_data,
            "target_yaw_range": self._coverage.yaw_range,
            "target_pitch_range": self._coverage.pitch_range,
            "captured_yaw_range": self._coverage.captured_yaw,
            "captured_pitch_range": self._coverage.captured_pitch,
            "progress_percent": self.get_progress_percent(),
            "is_sufficient": self._coverage.is_sufficient,
        }
    
    @property
    def coverage(self) -> CoverageStatus:
        """Get current coverage status."""
        return self._coverage
