"""
API client for MASt3R Face Authentication System.
CS-2 Primary Ownership.

Handles WebSocket and REST API communication with the backend.
Includes mock mode for development without backend.
"""

import asyncio
import json
import base64
import numpy as np
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import time

try:
    import websockets
    from websockets.sync.client import connect as ws_connect
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class ConnectionMode(Enum):
    """API connection mode."""
    MOCK = "mock"          # Simulated responses (no backend needed)
    LIVE = "live"          # Real backend connection


@dataclass
class EnrollmentFrame:
    """A frame sent during enrollment with backend response."""
    frame_id: int
    timestamp: float
    # Response from backend (or mock)
    face_detected: bool = False
    face_bbox: Optional[tuple] = None  # (x, y, w, h)
    head_pose: Optional[Dict[str, float]] = None  # {yaw, pitch, roll}
    quality_score: float = 0.0
    is_keyframe: bool = False
    partial_points: Optional[np.ndarray] = None  # Incremental point cloud


@dataclass
class EnrollmentSession:
    """State for an enrollment session."""
    user_name: str
    session_id: str = ""
    is_active: bool = False
    frames_sent: int = 0
    keyframes_captured: int = 0
    coverage_status: Dict[str, Any] = field(default_factory=dict)
    point_cloud: Optional[np.ndarray] = None
    point_colors: Optional[np.ndarray] = None
    error: Optional[str] = None


@dataclass 
class AuthResponse:
    """Response from authentication endpoint."""
    success: bool = False
    is_match: bool = False
    matched_user_id: Optional[str] = None
    matched_user_name: Optional[str] = None
    final_score: float = 0.0
    geometric_score: float = 0.0
    descriptor_score: float = 0.0
    anti_spoof_passed: bool = True
    processing_time_ms: float = 0.0
    error: Optional[str] = None


class MockBackend:
    """
    Simulates backend responses for development without the real API.
    Mimics the expected behavior of CS-1's endpoints.
    """
    
    def __init__(self):
        self._frame_count = 0
        self._keyframes: List[EnrollmentFrame] = []
        self._accumulated_points: List[np.ndarray] = []
        self._accumulated_colors: List[np.ndarray] = []
    
    def reset(self):
        """Reset mock state for new session."""
        self._frame_count = 0
        self._keyframes = []
        self._accumulated_points = []
        self._accumulated_colors = []
    
    def process_enrollment_frame(self, frame_b64: str) -> EnrollmentFrame:
        """Simulate processing an enrollment frame."""
        self._frame_count += 1
        
        # Simulate face detection (90% success rate)
        face_detected = np.random.random() > 0.1
        
        # Simulate head pose with some temporal consistency
        base_yaw = 30 * np.sin(self._frame_count * 0.1)
        base_pitch = 15 * np.cos(self._frame_count * 0.15)
        
        head_pose = {
            "yaw": base_yaw + np.random.normal(0, 3),
            "pitch": base_pitch + np.random.normal(0, 2),
            "roll": np.random.normal(0, 2),
        } if face_detected else None
        
        # Simulate keyframe selection (capture every ~8 frames when face detected)
        is_keyframe = (
            face_detected and 
            len(self._keyframes) < 12 and
            self._frame_count % 8 == 0
        )
        
        # Generate partial point cloud for keyframes
        partial_points = None
        if is_keyframe:
            # Generate random face-like point cloud segment
            n_points = np.random.randint(200, 400)
            # Create points roughly in face shape
            theta = np.random.uniform(0, 2*np.pi, n_points)
            phi = np.random.uniform(0, np.pi/2, n_points)
            r = 0.08 + np.random.normal(0, 0.01, n_points)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta) 
            z = r * np.cos(phi) - 0.1
            
            partial_points = np.stack([x, y, z], axis=1).astype(np.float32)
            colors = np.random.randint(150, 255, (n_points, 3), dtype=np.uint8)
            
            self._accumulated_points.append(partial_points)
            self._accumulated_colors.append(colors)
        
        result = EnrollmentFrame(
            frame_id=self._frame_count,
            timestamp=time.time(),
            face_detected=face_detected,
            face_bbox=(160, 120, 320, 320) if face_detected else None,
            head_pose=head_pose,
            quality_score=np.random.uniform(0.7, 1.0) if face_detected else 0.0,
            is_keyframe=is_keyframe,
            partial_points=partial_points,
        )
        
        if is_keyframe:
            self._keyframes.append(result)
        
        return result
    
    def get_accumulated_cloud(self) -> tuple:
        """Get all accumulated points and colors."""
        if not self._accumulated_points:
            return None, None
        points = np.vstack(self._accumulated_points)
        colors = np.vstack(self._accumulated_colors)
        return points, colors
    
    def finalize_enrollment(self, user_name: str) -> Dict[str, Any]:
        """Simulate finalizing enrollment."""
        points, colors = self.get_accumulated_cloud()
        
        if points is None or len(self._keyframes) < 3:
            return {
                "success": False,
                "error": "Not enough keyframes captured",
            }
        
        return {
            "success": True,
            "user_id": f"usr_{hash(user_name) % 10000:04d}",
            "user_name": user_name,
            "keyframes_count": len(self._keyframes),
            "point_count": len(points),
            "template_path": f"storage/templates/{user_name}.npz",
        }
    
    def authenticate(self, frames_b64: List[str], target_user: Optional[str]) -> AuthResponse:
        """Simulate authentication."""
        # Simulate processing time
        time.sleep(0.3)
        
        # Random match result (70% match rate for demo)
        is_match = np.random.random() > 0.3
        
        return AuthResponse(
            success=True,
            is_match=is_match,
            matched_user_id="usr_0001" if is_match else None,
            matched_user_name=target_user if is_match else None,
            final_score=np.random.uniform(0.55, 0.95) if is_match else np.random.uniform(0.3, 0.6),
            geometric_score=np.random.uniform(0.5, 0.9),
            descriptor_score=np.random.uniform(0.5, 0.95),
            anti_spoof_passed=True,
            processing_time_ms=np.random.uniform(300, 800),
        )


class APIClient:
    """
    Client for communicating with the MASt3R backend API.
    
    Supports both live (real backend) and mock (simulated) modes.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        ws_url: str = "ws://localhost:8000",
        mode: ConnectionMode = ConnectionMode.MOCK,
        timeout_sec: float = 30.0,
    ):
        self.base_url = base_url
        self.ws_url = ws_url
        self.mode = mode
        self.timeout_sec = timeout_sec
        
        self._mock = MockBackend()
        self._ws_connection = None
        self._enrollment_session: Optional[EnrollmentSession] = None
        
        # For async frame processing
        self._frame_queue: queue.Queue = queue.Queue(maxsize=30)
        self._result_queue: queue.Queue = queue.Queue(maxsize=30)
        self._ws_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def set_mode(self, mode: ConnectionMode) -> None:
        """Switch between mock and live mode."""
        self.mode = mode
        if mode == ConnectionMode.MOCK:
            print("[APIClient] Switched to MOCK mode")
        else:
            print("[APIClient] Switched to LIVE mode")
    
    def check_backend_available(self) -> bool:
        """Check if the backend server is reachable."""
        if not HTTPX_AVAILABLE:
            return False
        
        try:
            import httpx
            response = httpx.get(f"{self.base_url}/health", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False
    
    # ==================== Enrollment ====================
    
    def start_enrollment(self, user_name: str) -> EnrollmentSession:
        """Start a new enrollment session."""
        self._mock.reset()
        
        session = EnrollmentSession(
            user_name=user_name,
            session_id=f"enroll_{int(time.time())}",
            is_active=True,
        )
        self._enrollment_session = session
        
        if self.mode == ConnectionMode.LIVE:
            # TODO: Connect to WebSocket ws://localhost:8000/ws/enroll/{user_name}
            # For now, fall back to mock
            pass
        
        return session
    
    def process_enrollment_frame(self, frame: np.ndarray) -> EnrollmentFrame:
        """
        Process a single enrollment frame.
        
        Args:
            frame: BGR numpy array from webcam
            
        Returns:
            EnrollmentFrame with detection results
        """
        if self._enrollment_session is None or not self._enrollment_session.is_active:
            return EnrollmentFrame(frame_id=-1, timestamp=time.time())
        
        # Convert frame to base64
        from frontend.components.webcam_capture import WebcamCapture
        frame_b64 = WebcamCapture.frame_to_base64(frame)
        
        if self.mode == ConnectionMode.MOCK:
            result = self._mock.process_enrollment_frame(frame_b64)
        else:
            # TODO: Send via WebSocket and await response
            result = self._mock.process_enrollment_frame(frame_b64)
        
        # Update session state
        self._enrollment_session.frames_sent += 1
        if result.is_keyframe:
            self._enrollment_session.keyframes_captured += 1
        
        return result
    
    def get_enrollment_cloud(self) -> tuple:
        """Get current accumulated point cloud during enrollment."""
        if self.mode == ConnectionMode.MOCK:
            return self._mock.get_accumulated_cloud()
        # TODO: Get from backend
        return None, None
    
    def complete_enrollment(self) -> Dict[str, Any]:
        """Finalize the enrollment session."""
        if self._enrollment_session is None:
            return {"success": False, "error": "No active session"}
        
        user_name = self._enrollment_session.user_name
        self._enrollment_session.is_active = False
        
        if self.mode == ConnectionMode.MOCK:
            result = self._mock.finalize_enrollment(user_name)
        else:
            # TODO: Call finalize endpoint
            result = self._mock.finalize_enrollment(user_name)
        
        return result
    
    def cancel_enrollment(self) -> None:
        """Cancel the current enrollment session."""
        if self._enrollment_session:
            self._enrollment_session.is_active = False
            self._enrollment_session = None
        self._mock.reset()
    
    # ==================== Authentication ====================
    
    def authenticate(
        self,
        frames: List[np.ndarray],
        target_user: Optional[str] = None,
    ) -> AuthResponse:
        """
        Run authentication on captured frames.
        
        Args:
            frames: List of BGR numpy arrays
            target_user: Optional specific user to authenticate against
            
        Returns:
            AuthResponse with match results
        """
        from frontend.components.webcam_capture import WebcamCapture
        
        frames_b64 = [WebcamCapture.frame_to_base64(f) for f in frames]
        
        if self.mode == ConnectionMode.MOCK:
            return self._mock.authenticate(frames_b64, target_user)
        else:
            # TODO: POST to /authenticate
            return self._mock.authenticate(frames_b64, target_user)
    
    # ==================== User Management ====================
    
    def list_users(self) -> List[Dict[str, Any]]:
        """Get list of enrolled users."""
        if self.mode == ConnectionMode.MOCK:
            # Return placeholder users
            return [
                {"user_id": "usr_0001", "name": "Alice", "enrolled_at": "2026-02-01 10:00"},
                {"user_id": "usr_0002", "name": "Bob", "enrolled_at": "2026-02-02 14:30"},
            ]
        else:
            # TODO: GET /users
            return []
    
    def delete_user(self, user_id: str) -> bool:
        """Delete an enrolled user."""
        if self.mode == ConnectionMode.MOCK:
            return True
        else:
            # TODO: DELETE /users/{user_id}
            return False
    
    def get_user_template(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user's point cloud template for visualization."""
        if self.mode == ConnectionMode.MOCK:
            # Generate random face-like point cloud
            n_points = 2000
            theta = np.random.uniform(0, 2*np.pi, n_points)
            phi = np.random.uniform(0, np.pi/2, n_points)
            r = 0.1 + np.random.normal(0, 0.01, n_points)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi) - 0.12
            
            return {
                "points": np.stack([x, y, z], axis=1).astype(np.float32),
                "colors": np.random.randint(150, 255, (n_points, 3), dtype=np.uint8),
            }
        else:
            # TODO: GET /users/{user_id}/template
            return None


# Global client instance
_api_client: Optional[APIClient] = None


def get_api_client() -> APIClient:
    """Get or create the global API client instance."""
    global _api_client
    if _api_client is None:
        _api_client = APIClient(mode=ConnectionMode.MOCK)
    return _api_client
