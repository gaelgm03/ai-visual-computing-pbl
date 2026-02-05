"""
Face Detection Demo Script

This script demonstrates the face detection and keyframe selection
functionality using a webcam feed. It shows:
- Real-time face detection with landmarks
- Head pose estimation (yaw, pitch, roll)
- Keyframe capture based on pose novelty
- Coverage status for enrollment

This uses the core modules implemented in Phase 1:
- core.face_detector.FaceDetector
- core.keyframe_selector.KeyframeSelector

Usage:
    # Activate virtual environment first
    cd c:\\Users\\sekit\\ai-visual-computing-pbl
    .\\mast3r-face-auth\\Scripts\\Activate.ps1

    # Run the demo
    python scripts/demo_face_detector.py

Controls:
    - Press 'q' to quit
    - Press 'r' to reset keyframe collection
    - Press 's' to save current frame
    - Press 'm' to toggle mesh display
    - Press 'a' to toggle all landmarks display

Author: CS-1
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.face_detector import FaceDetector, FaceDetection
from core.keyframe_selector import KeyframeSelector, KeyframeCandidate, CoverageStatus
from core.config import get_face_detection_config, get_keyframe_config


def draw_landmarks(frame: np.ndarray, detection: FaceDetection, draw_all: bool = False):
    """
    Draw facial landmarks on the frame.

    Args:
        frame: BGR image to draw on (modified in place)
        detection: FaceDetection object with landmarks
        draw_all: If True, draw all landmarks; if False, draw key points only
    """
    # Key landmark indices for visualization
    KEY_POINTS = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_outer": 263,
        "right_eye_outer": 33,
        "left_eye_inner": 362,
        "right_eye_inner": 133,
        "left_mouth": 287,
        "right_mouth": 57,
        "upper_lip": 13,
        "lower_lip": 14,
        "left_eyebrow": 70,
        "right_eyebrow": 300,
    }

    if draw_all:
        # Draw all landmarks as small dots
        for i, (x, y) in enumerate(detection.landmarks_2d):
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    else:
        # Draw only key landmarks with different colors
        colors = {
            "nose_tip": (0, 0, 255),      # Red
            "chin": (255, 0, 0),           # Blue
            "left_eye_outer": (0, 255, 255),   # Yellow
            "right_eye_outer": (0, 255, 255),
            "left_eye_inner": (0, 255, 255),
            "right_eye_inner": (0, 255, 255),
            "left_mouth": (255, 0, 255),   # Magenta
            "right_mouth": (255, 0, 255),
            "upper_lip": (255, 0, 255),
            "lower_lip": (255, 0, 255),
            "left_eyebrow": (255, 255, 0), # Cyan
            "right_eyebrow": (255, 255, 0),
        }

        for name, idx in KEY_POINTS.items():
            if idx < len(detection.landmarks_2d):
                x, y = detection.landmarks_2d[idx]
                color = colors.get(name, (0, 255, 0))
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                cv2.circle(frame, (int(x), int(y)), 6, color, 1)


def draw_face_mesh(frame: np.ndarray, detection: FaceDetection):
    """
    Draw a simple face mesh connecting key landmarks.
    """
    # Define connections for a simple face outline
    FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]

    # Draw face outline
    points = detection.landmarks_2d
    for i in range(len(FACE_OUTLINE) - 1):
        idx1, idx2 = FACE_OUTLINE[i], FACE_OUTLINE[i + 1]
        if idx1 < len(points) and idx2 < len(points):
            pt1 = (int(points[idx1][0]), int(points[idx1][1]))
            pt2 = (int(points[idx2][0]), int(points[idx2][1]))
            cv2.line(frame, pt1, pt2, (0, 200, 0), 1)


def draw_head_pose_axes(frame: np.ndarray, detection: FaceDetection):
    """
    Draw 3D axes to visualize head pose.
    """
    yaw, pitch, roll = detection.head_pose

    # Get nose tip as origin
    nose_idx = 1
    if nose_idx < len(detection.landmarks_2d):
        origin = detection.landmarks_2d[nose_idx]
        origin = (int(origin[0]), int(origin[1]))

        # Axis length
        axis_length = 80

        # Convert angles to radians
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)

        # Simplified axis projection
        # X-axis (red) - pointing right
        x_end = (
            int(origin[0] + axis_length * np.cos(yaw_rad) * np.cos(roll_rad)),
            int(origin[1] + axis_length * np.sin(roll_rad))
        )

        # Y-axis (green) - pointing down
        y_end = (
            int(origin[0] - axis_length * np.sin(roll_rad)),
            int(origin[1] - axis_length * np.cos(pitch_rad) * np.cos(roll_rad))
        )

        # Z-axis (blue) - pointing out of the face
        z_end = (
            int(origin[0] + axis_length * np.sin(yaw_rad) * 0.5),
            int(origin[1] + axis_length * np.sin(pitch_rad) * 0.5)
        )

        # Draw axes
        cv2.arrowedLine(frame, origin, x_end, (0, 0, 255), 2, tipLength=0.2)  # X: Red
        cv2.arrowedLine(frame, origin, y_end, (0, 255, 0), 2, tipLength=0.2)  # Y: Green
        cv2.arrowedLine(frame, origin, z_end, (255, 0, 0), 2, tipLength=0.2)  # Z: Blue


def draw_info_panel(frame: np.ndarray, detection: FaceDetection,
                    status: CoverageStatus, fps: float, keyframe_captured: bool):
    """
    Draw information panel on the frame.
    """
    h, w = frame.shape[:2]

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Head pose info
    yaw, pitch, roll = detection.head_pose
    pose_text = f"Yaw: {yaw:+6.1f}  Pitch: {pitch:+6.1f}  Roll: {roll:+6.1f}"
    cv2.putText(frame, pose_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Keyframe count and coverage
    count_text = f"Keyframes: {status.total_frames}/12"
    cv2.putText(frame, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Progress bar
    progress = status.total_frames / 12
    bar_width = 150
    bar_x = 160
    cv2.rectangle(frame, (bar_x, 40), (bar_x + bar_width, 55), (100, 100, 100), -1)
    cv2.rectangle(frame, (bar_x, 40), (bar_x + int(bar_width * progress), 55), (0, 255, 0), -1)

    # Coverage ranges
    yaw_min, yaw_max = status.yaw_range
    pitch_min, pitch_max = status.pitch_range
    coverage_text = f"Yaw: [{yaw_min:+.0f}, {yaw_max:+.0f}]  Pitch: [{pitch_min:+.0f}, {pitch_max:+.0f}]"
    cv2.putText(frame, coverage_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 100, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Confidence
    conf_text = f"Conf: {detection.confidence:.2f}"
    cv2.putText(frame, conf_text, (w - 100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Keyframe captured indicator
    if keyframe_captured:
        cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 255, 0), 8)
        cv2.putText(frame, "KEYFRAME!", (w//2 - 80, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3, cv2.LINE_AA)

    # Missing directions
    if status.missing_directions:
        dirs_text = "Turn: " + ", ".join(status.missing_directions)
        cv2.putText(frame, dirs_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)
    elif status.is_sufficient:
        cv2.putText(frame, "Coverage complete!", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)


def draw_no_face_message(frame: np.ndarray):
    """Draw message when no face is detected."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 0, 100), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "No Face Detected", (w//2 - 120, h//2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Position your face in the frame", (w//2 - 160, h//2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)


def main():
    """Main demo function."""
    print("=" * 60)
    print("Face Detection Demo - CS-1 Core Modules Test")
    print("=" * 60)
    print()
    print("Controls:")
    print("  q - Quit")
    print("  r - Reset keyframe collection")
    print("  s - Save current frame")
    print("  m - Toggle mesh display")
    print("  a - Toggle all landmarks")
    print()

    # Load configurations
    face_config = get_face_detection_config()
    keyframe_config = get_keyframe_config()

    print(f"Face detection config: {face_config}")
    print(f"Keyframe config: {keyframe_config}")
    print()

    # Initialize modules
    print("Initializing FaceDetector...")
    detector = FaceDetector(face_config)
    print("FaceDetector ready!")

    print("Initializing KeyframeSelector...")
    selector = KeyframeSelector(keyframe_config)
    print("KeyframeSelector ready!")
    print()

    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        print("Make sure your webcam is connected and not used by another application.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Webcam opened successfully!")
    print()
    print("Starting live demo... Press 'q' to quit.")

    # State
    keyframe_candidates = []
    show_mesh = True
    show_all_landmarks = False
    last_keyframe_time = 0
    keyframe_flash_duration = 0.3

    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam")
                break

            # Mirror the frame for more intuitive interaction
            frame = cv2.flip(frame, 1)

            # Update FPS
            fps_frame_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = fps_frame_count / elapsed
                fps_frame_count = 0
                fps_start_time = time.time()

            # Detect face
            detection = detector.detect(frame)

            # Check if keyframe was captured recently (for flash effect)
            keyframe_captured = (time.time() - last_keyframe_time) < keyframe_flash_duration

            if detection is not None:
                # Draw bounding box
                x1, y1, x2, y2 = detection.bbox
                color = (0, 255, 0) if detection.confidence > 0.8 else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw face mesh
                if show_mesh:
                    draw_face_mesh(frame, detection)

                # Draw landmarks
                draw_landmarks(frame, detection, draw_all=show_all_landmarks)

                # Draw head pose axes
                draw_head_pose_axes(frame, detection)

                # Check if should capture keyframe
                should_capture = selector.should_capture(detection, keyframe_candidates, frame)

                if should_capture:
                    # Create keyframe candidate
                    cropped = detector.crop_face_region(frame, detection)
                    candidate = KeyframeCandidate(
                        frame=cropped,
                        head_pose=detection.head_pose,
                        timestamp=time.time(),
                        quality_score=selector.compute_quality_score(frame, detection)
                    )
                    keyframe_candidates.append(candidate)
                    last_keyframe_time = time.time()
                    print(f"Keyframe {len(keyframe_candidates)} captured! "
                          f"Pose: yaw={detection.head_pose[0]:.1f}, "
                          f"pitch={detection.head_pose[1]:.1f}")

                # Get coverage status
                status = selector.get_coverage_status(keyframe_candidates)

                # Draw info panel
                draw_info_panel(frame, detection, status, current_fps, keyframe_captured)
            else:
                # No face detected
                draw_no_face_message(frame)
                status = CoverageStatus(
                    yaw_range=(0, 0),
                    pitch_range=(0, 0),
                    total_frames=len(keyframe_candidates),
                    is_sufficient=False,
                    missing_directions=["center"]
                )

            # Show frame
            cv2.imshow("Face Detection Demo - CS-1", frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('r'):
                keyframe_candidates.clear()
                print("Keyframe collection reset!")
            elif key == ord('s'):
                filename = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved to {filename}")
            elif key == ord('m'):
                show_mesh = not show_mesh
                print(f"Mesh display: {'ON' if show_mesh else 'OFF'}")
            elif key == ord('a'):
                show_all_landmarks = not show_all_landmarks
                print(f"All landmarks: {'ON' if show_all_landmarks else 'OFF'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

        # Summary
        print()
        print("=" * 60)
        print("Session Summary")
        print("=" * 60)
        print(f"Total keyframes captured: {len(keyframe_candidates)}")
        if keyframe_candidates:
            status = selector.get_coverage_status(keyframe_candidates)
            print(f"Yaw range: {status.yaw_range}")
            print(f"Pitch range: {status.pitch_range}")
            print(f"Coverage sufficient: {status.is_sufficient}")


if __name__ == "__main__":
    main()
