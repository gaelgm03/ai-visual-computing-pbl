"""
Pydantic Schemas for API Request/Response Models

This module defines the data models used for API communication between
the frontend (CS-2) and backend (CS-1).

These schemas provide:
- Type validation
- Automatic documentation in OpenAPI/Swagger
- Clear interface contracts

Author: CS-1
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============================================================
# Enrollment Schemas
# ============================================================

class HeadPose(BaseModel):
    """Head pose angles in degrees."""
    yaw: float = Field(..., description="Left(-) / Right(+) rotation in degrees")
    pitch: float = Field(..., description="Down(-) / Up(+) rotation in degrees")
    roll: float = Field(..., description="Left tilt(-) / Right tilt(+) in degrees")


class Coverage(BaseModel):
    """Angular coverage status during enrollment."""
    yaw_range: List[float] = Field(..., description="[min, max] yaw angles captured")
    pitch_range: List[float] = Field(..., description="[min, max] pitch angles captured")
    is_sufficient: bool = Field(..., description="True if coverage is sufficient for enrollment")
    missing_directions: List[str] = Field(
        default_factory=list,
        description="Directions user still needs to turn: 'left', 'right', 'up', 'down'"
    )


class FrameMessage(BaseModel):
    """Message sent by client for each frame during enrollment."""
    type: str = Field(default="frame", description="Message type, should be 'frame'")
    data: str = Field(..., description="Base64-encoded JPEG image data")


class FrameStatusResponse(BaseModel):
    """Response sent to client for each processed frame."""
    type: str = Field(default="frame_status", description="Message type")
    face_detected: bool = Field(..., description="Whether a face was detected")
    head_pose: Optional[HeadPose] = Field(None, description="Head pose if face detected")
    captured: bool = Field(False, description="Whether this frame was selected as keyframe")
    total_captured: int = Field(0, description="Total keyframes captured so far")
    target_count: int = Field(12, description="Target number of keyframes")
    coverage: Optional[Coverage] = Field(None, description="Current coverage status")


class EnrollmentCompleteResponse(BaseModel):
    """Response sent when enrollment is complete."""
    type: str = Field(default="enrollment_complete", description="Message type")
    user_id: str = Field(..., description="Generated unique user ID")
    user_name: str = Field(..., description="User's display name")
    point_cloud_preview: Optional[str] = Field(
        None,
        description="Base64-encoded point cloud preview (JSON format with points and colors)"
    )
    n_points: int = Field(..., description="Number of 3D points in template")
    n_frames_used: int = Field(..., description="Number of keyframes used")
    reconstruction_time_sec: float = Field(..., description="Time taken for 3D reconstruction")


class EnrollmentErrorResponse(BaseModel):
    """Error response during enrollment."""
    type: str = Field(default="error", description="Message type")
    error: str = Field(..., description="Error message")
    code: str = Field(default="ENROLLMENT_ERROR", description="Error code")


class BatchEnrollRequest(BaseModel):
    """Request for batch (non-streaming) enrollment."""
    user_name: str = Field(..., description="Display name for the user")
    frames: List[str] = Field(
        ...,
        min_length=2,
        max_length=12,
        description="List of base64-encoded JPEG images (2-12 frames)"
    )


# ============================================================
# Authentication Schemas
# ============================================================

class AuthRequest(BaseModel):
    """Request for authentication."""
    user_id: Optional[str] = Field(
        None,
        description="User ID for 1:1 verification. Null for 1:N identification."
    )
    frames: List[str] = Field(
        ...,
        min_length=2,
        max_length=4,
        description="List of base64-encoded JPEG images (2-4 frames)"
    )


class AntiSpoofResult(BaseModel):
    """Anti-spoofing check results."""
    passed: bool = Field(..., description="Whether anti-spoofing check passed")
    depth_variance: float = Field(..., description="Depth variance of reconstructed face")
    planarity_ratio: float = Field(..., description="Planarity ratio (lower = more 3D)")


class VisualizationData(BaseModel):
    """Data for result visualization."""
    probe_cloud: Optional[str] = Field(None, description="Base64-encoded probe point cloud")
    matched_points: Optional[List[List[float]]] = Field(
        None,
        description="List of matched point pairs for visualization"
    )


class AuthResponse(BaseModel):
    """Response from authentication attempt."""
    is_match: bool = Field(..., description="Whether authentication succeeded")
    matched_user_id: Optional[str] = Field(None, description="ID of matched user (if any)")
    matched_user_name: Optional[str] = Field(None, description="Name of matched user (if any)")
    final_score: float = Field(..., description="Final fused match score (0-1)")
    geometric_score: float = Field(..., description="Geometric matching score (0-1)")
    descriptor_score: float = Field(..., description="Descriptor matching score (0-1)")
    embedding_score: float = Field(0.0, description="ArcFace embedding match score (0-1)")
    anti_spoof: AntiSpoofResult = Field(..., description="Anti-spoofing results")
    processing_time_sec: float = Field(..., description="Total processing time in seconds")
    visualization_data: Optional[VisualizationData] = Field(
        None,
        description="Data for visualizing match results"
    )


# ============================================================
# User Management Schemas
# ============================================================

class UserInfo(BaseModel):
    """User information summary."""
    user_id: str = Field(..., description="Unique user identifier")
    user_name: str = Field(..., description="User's display name")
    enrolled_at: str = Field(..., description="ISO timestamp of enrollment")
    n_points: Optional[int] = Field(None, description="Number of 3D points in template")
    n_frames_used: Optional[int] = Field(None, description="Number of keyframes used")


class UserListResponse(BaseModel):
    """Response containing list of enrolled users."""
    users: List[UserInfo] = Field(default_factory=list)
    total: int = Field(0, description="Total number of enrolled users")


class UserDetailResponse(UserInfo):
    """Detailed user information."""
    template_path: Optional[str] = Field(None, description="Path to template file")
    enrollment_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional enrollment metadata"
    )


class DeleteUserResponse(BaseModel):
    """Response from user deletion."""
    success: bool = Field(..., description="Whether deletion was successful")
    user_id: str = Field(..., description="ID of deleted user")
    message: str = Field(..., description="Status message")


# ============================================================
# Health Check Schemas
# ============================================================

class HealthResponse(BaseModel):
    """System health check response."""
    model_config = {"protected_namespaces": ()}  # Allow 'model_' prefix in field names
    
    status: str = Field(..., description="Overall status: 'healthy' or 'unhealthy'")
    model_loaded: bool = Field(..., description="Whether MASt3R model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_memory_total_gb: Optional[float] = Field(None, description="Total GPU memory in GB")
    gpu_memory_used_gb: Optional[float] = Field(None, description="Used GPU memory in GB")
    enrolled_users: int = Field(..., description="Number of enrolled users")
