"""
Webcam capture component for MASt3R Face Authentication System.
CS-2 Primary Ownership.

Handles webcam access and frame dispatch to the backend API.
"""

import cv2
import numpy as np
import base64
from typing import Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class CaptureConfig:
    """Configuration for webcam capture."""
    width: int = 640
    height: int = 480
    fps: int = 30
    device_id: int = 0


class WebcamCapture:
    """
    Manages webcam access and frame capture for the face authentication UI.
    
    This component handles:
    - Opening/closing the webcam device
    - Capturing frames at specified resolution
    - Converting frames to base64 for WebSocket transmission
    - Providing frames to Gradio's streaming interface
    """
    
    def __init__(self, config: Optional[CaptureConfig] = None):
        self.config = config or CaptureConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running: bool = False
    
    def open(self) -> bool:
        """
        Open the webcam device.
        
        Returns:
            True if webcam opened successfully, False otherwise.
        """
        if self._cap is not None:
            self.close()
        
        self._cap = cv2.VideoCapture(self.config.device_id)
        
        if not self._cap.isOpened():
            print(f"[WebcamCapture] Failed to open camera {self.config.device_id}")
            return False
        
        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        self._is_running = True
        print(f"[WebcamCapture] Opened camera {self.config.device_id} at {self.config.width}x{self.config.height}")
        return True
    
    def close(self) -> None:
        """Release the webcam device."""
        self._is_running = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            print("[WebcamCapture] Camera closed")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the webcam.
        
        Returns:
            Tuple of (success, frame) where frame is BGR numpy array or None.
        """
        if self._cap is None or not self._is_running:
            return False, None
        
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        
        return True, frame
    
    def read_frame_rgb(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame and convert to RGB.
        
        Returns:
            Tuple of (success, frame) where frame is RGB numpy array or None.
        """
        success, frame = self.read_frame()
        if success and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return success, frame
    
    @staticmethod
    def frame_to_base64(frame: np.ndarray, format: str = "jpeg") -> str:
        """
        Convert a frame to base64-encoded string for WebSocket transmission.
        
        Args:
            frame: BGR or RGB numpy array
            format: Image format ('jpeg' or 'png')
        
        Returns:
            Base64-encoded string of the image.
        """
        if format == "jpeg":
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            success, buffer = cv2.imencode('.jpg', frame, encode_param)
        else:
            success, buffer = cv2.imencode('.png', frame)
        
        if not success:
            raise ValueError("Failed to encode frame")
        
        return base64.b64encode(buffer).decode('utf-8')
    
    @staticmethod
    def base64_to_frame(b64_string: str) -> np.ndarray:
        """
        Convert a base64-encoded string back to a frame.
        
        Args:
            b64_string: Base64-encoded image string
        
        Returns:
            BGR numpy array.
        """
        img_bytes = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    
    @property
    def is_open(self) -> bool:
        """Check if webcam is currently open."""
        return self._cap is not None and self._cap.isOpened()
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def get_available_cameras(max_check: int = 5) -> list:
    """
    Probe for available camera devices.
    
    Args:
        max_check: Maximum device IDs to check.
    
    Returns:
        List of available camera device IDs.
    """
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available
