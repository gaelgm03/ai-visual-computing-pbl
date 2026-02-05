"""
Face Detection Module

This module provides face detection and landmark extraction using MediaPipe Face Mesh.
It detects faces in images, extracts 478 facial landmarks (both 2D and 3D),
and calculates head pose (yaw, pitch, roll).

The FaceDetector class is the main interface for face detection operations.

Note: MediaPipe 0.10.x uses the new Tasks API (mp.tasks.vision.FaceLandmarker)
instead of the legacy Solutions API (mp.solutions.face_mesh).

Usage:
    from core.face_detector import FaceDetector

    detector = FaceDetector(config)
    detection = detector.detect(frame)
    if detection:
        cropped_face = detector.crop_face_region(frame, detection)
"""

import os
import urllib.request
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# MediaPipe Tasks API imports
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision


@dataclass
class FaceDetection:
    """
    Data class to hold face detection results.

    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2) in pixels.
              (x1, y1) is the top-left corner, (x2, y2) is the bottom-right.
        landmarks_2d: 478 facial landmarks in 2D pixel coordinates.
                      Shape: (478, 2) where each row is (x, y).
        landmarks_3d: 478 facial landmarks in 3D normalized coordinates.
                      Shape: (478, 3) where each row is (x, y, z).
                      x, y are normalized to [0, 1] relative to image size.
                      z represents depth (distance from camera plane).
        head_pose: Head rotation angles in degrees: (yaw, pitch, roll).
                   - yaw: Left/right rotation (-90 to +90, negative=left)
                   - pitch: Up/down rotation (-90 to +90, negative=down)
                   - roll: Tilt rotation (-90 to +90, negative=tilt left)
        confidence: Detection confidence score (0.0 to 1.0).
    """

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    landmarks_2d: np.ndarray  # (478, 2) pixel coordinates
    landmarks_3d: np.ndarray  # (478, 3) normalized 3D coordinates
    head_pose: Tuple[float, float, float]  # (yaw, pitch, roll) in degrees
    confidence: float


# Key landmark indices for head pose estimation
# These landmarks are stable and well-distributed across the face
# Note: MediaPipe Face Landmarker uses 478 landmarks (468 base + 10 iris)
POSE_LANDMARKS = {
    "nose_tip": 1,  # Tip of the nose
    "chin": 152,  # Bottom of the chin
    "left_eye_outer": 263,  # Left eye outer corner
    "right_eye_outer": 33,  # Right eye outer corner
    "left_mouth": 287,  # Left corner of mouth
    "right_mouth": 57,  # Right corner of mouth
}

# 3D model points for head pose estimation (standard face model)
# These are approximate 3D coordinates of facial landmarks in a canonical face
# Units are in millimeters, centered at the nose tip
MODEL_POINTS_3D = np.array(
    [
        [0.0, 0.0, 0.0],  # Nose tip
        [0.0, -63.6, -12.5],  # Chin
        [-43.3, 32.7, -26.0],  # Left eye outer corner
        [43.3, 32.7, -26.0],  # Right eye outer corner
        [-28.9, -28.9, -24.1],  # Left mouth corner
        [28.9, -28.9, -24.1],  # Right mouth corner
    ],
    dtype=np.float64,
)

# URL for the face landmarker model
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_FILENAME = "face_landmarker.task"


def get_model_path() -> str:
    """
    Get the path to the MediaPipe face landmarker model file.
    Downloads the model if it doesn't exist locally.

    Returns:
        Path to the model file.
    """
    # Store model in the project's storage directory
    from core.config import get_project_root

    project_root = get_project_root()
    model_dir = project_root / "storage" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / MODEL_FILENAME

    if not model_path.exists():
        print(f"Downloading MediaPipe face landmarker model...")
        print(f"URL: {MODEL_URL}")
        print(f"Saving to: {model_path}")
        urllib.request.urlretrieve(MODEL_URL, str(model_path))
        print("Download complete!")

    return str(model_path)


class FaceDetector:
    """
    Face detection and landmark extraction using MediaPipe Face Landmarker.

    This class wraps MediaPipe's Face Landmarker (Tasks API) to provide:
    - Face detection in RGB/BGR images
    - 478 facial landmark extraction (2D and 3D)
    - Head pose estimation (yaw, pitch, roll)
    - Face region cropping with padding

    Attributes:
        config: Configuration dictionary with detection parameters.
        landmarker: MediaPipe FaceLandmarker object for detection.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FaceDetector.

        Args:
            config: Configuration dictionary containing:
                - min_detection_confidence: Minimum confidence for detection (0-1)
                - min_tracking_confidence: Minimum confidence for tracking (0-1)
                - face_padding: Padding ratio around detected face bbox

        Example:
            config = {
                "min_detection_confidence": 0.9,
                "min_tracking_confidence": 0.5,
                "face_padding": 0.3
            }
            detector = FaceDetector(config)
        """
        self.config = config

        # Get configuration values with defaults
        min_detection_conf = config.get("min_detection_confidence", 0.9)
        min_tracking_conf = config.get("min_tracking_confidence", 0.5)
        self.face_padding = config.get("face_padding", 0.3)

        # Get the model path (downloads if necessary)
        model_path = get_model_path()

        # Create FaceLandmarker options
        # Using IMAGE mode for single image processing
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,  # Detect only one face
            min_face_detection_confidence=min_detection_conf,
            min_face_presence_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
            output_face_blendshapes=False,  # We don't need blendshapes
            output_facial_transformation_matrixes=True,  # For pose estimation
        )

        # Create the FaceLandmarker
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray) -> Optional[FaceDetection]:
        """
        Detect face and extract landmarks from an image.

        This method processes an image to find a face, extract its landmarks,
        and calculate the head pose. If no face is found, returns None.

        Args:
            frame: Input image as BGR numpy array with shape (H, W, 3).
                   This is the standard format from cv2.imread() or webcam capture.

        Returns:
            FaceDetection object containing all detection results,
            or None if no face is detected.

        Example:
            frame = cv2.imread("face.jpg")
            detection = detector.detect(frame)
            if detection:
                print(f"Head pose: yaw={detection.head_pose[0]:.1f}")
        """
        # Get image dimensions
        h, w = frame.shape[:2]

        # MediaPipe expects RGB, but OpenCV uses BGR
        # So we need to convert the color space
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image from numpy array
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process the image with MediaPipe Face Landmarker
        results = self.landmarker.detect(mp_image)

        # Check if any face was detected
        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return None

        # Get the first (and only) detected face
        face_landmarks = results.face_landmarks[0]

        # Extract 2D and 3D landmarks
        landmarks_2d, landmarks_3d = self._extract_landmarks(face_landmarks, w, h)

        # Calculate bounding box from landmarks
        bbox = self._calculate_bbox(landmarks_2d, w, h)

        # Calculate head pose (yaw, pitch, roll)
        head_pose = self._calculate_head_pose(landmarks_2d, w, h)

        # Estimate detection confidence based on landmark consistency
        confidence = self._estimate_confidence(landmarks_2d, w, h)

        return FaceDetection(
            bbox=bbox,
            landmarks_2d=landmarks_2d,
            landmarks_3d=landmarks_3d,
            head_pose=head_pose,
            confidence=confidence,
        )

    def _extract_landmarks(
        self, face_landmarks, width: int, height: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 2D and 3D landmarks from MediaPipe results.

        Args:
            face_landmarks: List of MediaPipe NormalizedLandmark objects.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Tuple of (landmarks_2d, landmarks_3d) numpy arrays.
        """
        num_landmarks = len(face_landmarks)
        landmarks_2d = np.zeros((num_landmarks, 2), dtype=np.float32)
        landmarks_3d = np.zeros((num_landmarks, 3), dtype=np.float32)

        for i, landmark in enumerate(face_landmarks):
            # 2D coordinates: convert normalized coords to pixel coords
            landmarks_2d[i] = [landmark.x * width, landmark.y * height]

            # 3D coordinates: keep as normalized (x, y) + depth (z)
            # Note: z is the estimated depth, negative values are closer to camera
            landmarks_3d[i] = [landmark.x, landmark.y, landmark.z]

        return landmarks_2d, landmarks_3d

    def _calculate_bbox(
        self, landmarks_2d: np.ndarray, width: int, height: int
    ) -> Tuple[int, int, int, int]:
        """
        Calculate face bounding box from 2D landmarks.

        The bounding box is computed as the min/max of all landmark coordinates,
        ensuring it stays within image boundaries.

        Args:
            landmarks_2d: Array of 2D landmarks with shape (N, 2).
            width: Image width.
            height: Image height.

        Returns:
            Tuple (x1, y1, x2, y2) representing the bounding box corners.
        """
        # Find min and max coordinates across all landmarks
        x_min = int(np.min(landmarks_2d[:, 0]))
        y_min = int(np.min(landmarks_2d[:, 1]))
        x_max = int(np.max(landmarks_2d[:, 0]))
        y_max = int(np.max(landmarks_2d[:, 1]))

        # Clamp to image boundaries
        x1 = max(0, x_min)
        y1 = max(0, y_min)
        x2 = min(width, x_max)
        y2 = min(height, y_max)

        return (x1, y1, x2, y2)

    def _calculate_head_pose(
        self, landmarks_2d: np.ndarray, width: int, height: int
    ) -> Tuple[float, float, float]:
        """
        Calculate head pose (yaw, pitch, roll) from facial landmarks.

        This uses the Perspective-n-Point (PnP) algorithm to estimate
        the 3D rotation of the head based on known 3D face model points
        and their corresponding 2D projections in the image.

        Args:
            landmarks_2d: Array of 2D landmarks with shape (N, 2).
            width: Image width.
            height: Image height.

        Returns:
            Tuple (yaw, pitch, roll) in degrees.
            - yaw: Left(-) / Right(+) rotation
            - pitch: Down(-) / Up(+) rotation
            - roll: Left tilt(-) / Right tilt(+) rotation
        """
        # Extract the 6 key landmarks for pose estimation
        image_points = np.array(
            [
                landmarks_2d[POSE_LANDMARKS["nose_tip"]],
                landmarks_2d[POSE_LANDMARKS["chin"]],
                landmarks_2d[POSE_LANDMARKS["left_eye_outer"]],
                landmarks_2d[POSE_LANDMARKS["right_eye_outer"]],
                landmarks_2d[POSE_LANDMARKS["left_mouth"]],
                landmarks_2d[POSE_LANDMARKS["right_mouth"]],
            ],
            dtype=np.float64,
        )

        # Camera matrix approximation
        # We assume the camera center is at the image center
        # and focal length is proportional to image width
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64,
        )

        # Assume no lens distortion
        dist_coeffs = np.zeros((4, 1))

        # Solve PnP to get rotation and translation vectors
        success, rotation_vec, translation_vec = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return (0.0, 0.0, 0.0)

        # Convert rotation vector to rotation matrix
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)

        # Calculate Euler angles from rotation matrix
        # The decomposition gives angles in radians
        proj_matrix = np.hstack((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        # Extract pitch, yaw, roll
        # euler_angles from decomposeProjectionMatrix returns shape (3, 1)
        # Each element is a single-element array, so we need [0][0] to get scalar
        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        return (yaw, pitch, roll)

    def _estimate_confidence(
        self, landmarks_2d: np.ndarray, width: int, height: int
    ) -> float:
        """
        Estimate detection confidence based on landmark positions.

        This provides a rough confidence measure based on:
        1. Whether key landmarks are within image boundaries
        2. Face size relative to image (very small faces = lower confidence)
        3. Landmark spread (very narrow = potentially profile view = lower conf)

        Args:
            landmarks_2d: Array of 2D landmarks.
            width: Image width.
            height: Image height.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        # Check if landmarks are within bounds (with small margin)
        margin = 5
        in_bounds = np.all(landmarks_2d[:, 0] >= margin) and np.all(
            landmarks_2d[:, 0] <= width - margin
        )
        in_bounds = in_bounds and np.all(landmarks_2d[:, 1] >= margin)
        in_bounds = in_bounds and np.all(landmarks_2d[:, 1] <= height - margin)

        # Calculate face size ratio
        face_width = np.max(landmarks_2d[:, 0]) - np.min(landmarks_2d[:, 0])
        face_height = np.max(landmarks_2d[:, 1]) - np.min(landmarks_2d[:, 1])
        size_ratio = (face_width * face_height) / (width * height)

        # Start with base confidence
        confidence = 0.95 if in_bounds else 0.7

        # Reduce confidence for very small faces
        if size_ratio < 0.01:  # Less than 1% of image
            confidence *= 0.5
        elif size_ratio < 0.05:  # Less than 5% of image
            confidence *= 0.8

        return min(1.0, max(0.0, confidence))

    def crop_face_region(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
        padding: Optional[float] = None,
    ) -> np.ndarray:
        """
        Crop the face region from an image with padding.

        This extracts the face area plus some surrounding context,
        which is useful for MASt3R processing.

        Args:
            frame: Original image (BGR format).
            detection: FaceDetection object with bounding box.
            padding: Padding ratio around the face (0.0 to 1.0).
                     If None, uses the value from config.
                     For example, 0.3 means add 30% padding on each side.

        Returns:
            Cropped face image as numpy array.

        Example:
            detection = detector.detect(frame)
            if detection:
                face_crop = detector.crop_face_region(frame, detection, padding=0.3)
        """
        if padding is None:
            padding = self.face_padding

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = detection.bbox

        # Calculate padding in pixels
        face_w = x2 - x1
        face_h = y2 - y1
        pad_x = int(face_w * padding)
        pad_y = int(face_h * padding)

        # Apply padding while staying within image bounds
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(w, x2 + pad_x)
        crop_y2 = min(h, y2 + pad_y)

        # Extract the cropped region
        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        return cropped

    def close(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, "landmarker"):
            self.landmarker.close()

    def __del__(self):
        """Clean up MediaPipe resources on deletion."""
        self.close()


if __name__ == "__main__":
    # Quick test of the face detector
    print("Testing FaceDetector...")

    # Create detector with default config
    config = {
        "min_detection_confidence": 0.9,
        "min_tracking_confidence": 0.5,
        "face_padding": 0.3,
    }

    detector = FaceDetector(config)
    print("FaceDetector initialized successfully!")

    # If you have a test image, uncomment below:
    # frame = cv2.imread("test_face.jpg")
    # detection = detector.detect(frame)
    # if detection:
    #     print(f"Face detected!")
    #     print(f"  Bounding box: {detection.bbox}")
    #     print(f"  Head pose (yaw, pitch, roll): {detection.head_pose}")
    #     print(f"  Confidence: {detection.confidence:.2f}")
    # else:
    #     print("No face detected")
