"""
Authentication panel component for MASt3R Face Authentication System.
CS-2 Primary Ownership.

Handles the authentication trigger UI and result display.
"""

from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class AuthResult:
    """Result of an authentication attempt."""
    is_match: bool = False
    matched_user_id: Optional[str] = None
    matched_user_name: Optional[str] = None
    final_score: float = 0.0
    geometric_score: float = 0.0
    descriptor_score: float = 0.0
    anti_spoof_passed: bool = True
    processing_time_sec: float = 0.0
    error_message: Optional[str] = None


@dataclass
class AuthConfig:
    """Configuration for authentication panel."""
    capture_frames: int = 3  # Number of frames to capture for auth
    frame_delay_ms: int = 200  # Delay between captures
    api_endpoint: str = "http://localhost:8000/authenticate"


class AuthPanel:
    """
    Manages the authentication UI flow.
    
    Responsibilities:
    - Capture frames for authentication
    - Send authentication request to backend
    - Display authentication results (score gauge, match/no-match)
    - Show anti-spoofing status
    """
    
    def __init__(self, config: Optional[AuthConfig] = None):
        self.config = config or AuthConfig()
        self._last_result: Optional[AuthResult] = None
        self._is_authenticating: bool = False
        self._enrolled_users: List[Dict] = []
    
    def set_enrolled_users(self, users: List[Dict]) -> None:
        """Update the list of enrolled users for selection."""
        self._enrolled_users = users
    
    def get_enrolled_users(self) -> List[Dict]:
        """Get list of enrolled users."""
        return self._enrolled_users
    
    def format_result_message(self, result: AuthResult) -> str:
        """Format authentication result for display."""
        if result.error_message:
            return f"❌ Error: {result.error_message}"
        
        if not result.anti_spoof_passed:
            return "🚫 Spoofing detected! Please use a real face."
        
        if result.is_match:
            return (
                f"✅ Authenticated as **{result.matched_user_name}**\n"
                f"Score: {result.final_score:.2f} "
                f"(Geo: {result.geometric_score:.2f}, Desc: {result.descriptor_score:.2f})"
            )
        else:
            return (
                f"❌ Authentication failed\n"
                f"Score: {result.final_score:.2f} (below threshold)"
            )
    
    def get_score_color(self, score: float) -> str:
        """Get color for score visualization."""
        if score >= 0.8:
            return "#22c55e"  # Green
        elif score >= 0.65:
            return "#eab308"  # Yellow
        else:
            return "#ef4444"  # Red
    
    def format_processing_time(self, seconds: float) -> str:
        """Format processing time for display."""
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        return f"{seconds:.1f}s"
    
    @property
    def last_result(self) -> Optional[AuthResult]:
        """Get the last authentication result."""
        return self._last_result
    
    @property
    def is_authenticating(self) -> bool:
        """Check if authentication is in progress."""
        return self._is_authenticating


# Placeholder function for creating the auth panel UI in Gradio
def create_auth_panel_ui():
    """
    Creates the Gradio components for the auth panel.
    To be called from app_gradio.py
    
    Returns:
        Dict of Gradio components.
    """
    # This will be implemented when we connect to the actual API
    pass
