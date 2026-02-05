"""
Core Module for MASt3R Face Authentication System

This package contains the core functionality for face detection,
keyframe selection, 3D reconstruction, and template management.

Main components:
    - config: Configuration loading and management
    - face_detector: Face detection using MediaPipe
    - keyframe_selector: Optimal frame selection for enrollment
    - mast3r_engine: MASt3R model wrapper (CS-1 Phase 2)
    - template_manager: Face template storage (CS-1 Phase 2)
    - matching: Matching algorithms (DS team implements)

Usage:
    from core.config import get_config
    from core.face_detector import FaceDetector
    from core.keyframe_selector import KeyframeSelector
"""

from core.config import (
    get_config,
    get_section,
    get_face_detection_config,
    get_keyframe_config,
    get_mast3r_config,
    get_matching_config,
    get_storage_config,
)

from core.face_detector import FaceDetector, FaceDetection

from core.keyframe_selector import (
    KeyframeSelector,
    KeyframeCandidate,
    CoverageStatus,
)

__all__ = [
    # Configuration
    "get_config",
    "get_section",
    "get_face_detection_config",
    "get_keyframe_config",
    "get_mast3r_config",
    "get_matching_config",
    "get_storage_config",
    # Face Detection
    "FaceDetector",
    "FaceDetection",
    # Keyframe Selection
    "KeyframeSelector",
    "KeyframeCandidate",
    "CoverageStatus",
]
