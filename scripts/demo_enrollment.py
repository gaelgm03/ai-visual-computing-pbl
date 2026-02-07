"""
Enrollment Demo Script - Full Pipeline Demo

This script demonstrates the complete enrollment pipeline:
1. Real-time face detection with keyframe capture
2. MASt3R 3D reconstruction
3. Point cloud visualization

This combines Phase 1 (face detection) and Phase 2 (3D reconstruction) components
to show the end-to-end enrollment process.

Usage:
    # On WSL2 with GPU
    cd /mnt/c/Users/sekit/ai-visual-computing-pbl
    source mast3r-env/bin/activate
    python scripts/demo_enrollment.py

    # Options
    python scripts/demo_enrollment.py --min-keyframes 8  # Fewer frames for faster demo
    python scripts/demo_enrollment.py --skip-capture     # Use saved keyframes

Controls (during capture):
    - Press 'q' to quit
    - Press 'r' to reset keyframe collection
    - Press SPACE to manually trigger capture
    - Press ENTER when ready to run reconstruction (if enough keyframes)

Author: CS-1
"""

import cv2
import numpy as np
import time
import sys
import argparse
import json
import webbrowser
import tempfile
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.face_detector import FaceDetector, FaceDetection
from core.keyframe_selector import KeyframeSelector, KeyframeCandidate, CoverageStatus
from core.config import get_face_detection_config, get_keyframe_config, get_mast3r_config


def draw_capture_ui(frame: np.ndarray, detection: Optional[FaceDetection],
                    status: CoverageStatus, fps: float, keyframe_flash: bool,
                    min_keyframes: int):
    """Draw the capture UI overlay."""
    h, w = frame.shape[:2]

    # Semi-transparent header
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if detection is not None:
        # Draw bounding box
        x1, y1, x2, y2 = detection.bbox
        color = (0, 255, 0) if detection.confidence > 0.8 else (0, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Head pose info
        yaw, pitch, roll = detection.head_pose
        pose_text = f"Yaw: {yaw:+6.1f}  Pitch: {pitch:+6.1f}  Roll: {roll:+6.1f}"
        cv2.putText(frame, pose_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw head pose axes
        nose_idx = 1
        if nose_idx < len(detection.landmarks_2d):
            origin = detection.landmarks_2d[nose_idx]
            origin = (int(origin[0]), int(origin[1]))
            axis_length = 60

            yaw_rad = np.radians(yaw)
            pitch_rad = np.radians(pitch)
            roll_rad = np.radians(roll)

            x_end = (int(origin[0] + axis_length * np.cos(yaw_rad)),
                     int(origin[1] + axis_length * np.sin(roll_rad)))
            y_end = (int(origin[0] - axis_length * np.sin(roll_rad)),
                     int(origin[1] - axis_length * np.cos(pitch_rad)))

            cv2.arrowedLine(frame, origin, x_end, (0, 0, 255), 2, tipLength=0.2)
            cv2.arrowedLine(frame, origin, y_end, (0, 255, 0), 2, tipLength=0.2)
    else:
        cv2.putText(frame, "No face detected - position your face in frame",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    # Keyframe count and progress bar
    count_text = f"Keyframes: {status.total_frames}/{min_keyframes}"
    cv2.putText(frame, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    progress = min(1.0, status.total_frames / min_keyframes)
    bar_width = 150
    bar_x = 180
    cv2.rectangle(frame, (bar_x, 40), (bar_x + bar_width, 55), (100, 100, 100), -1)
    cv2.rectangle(frame, (bar_x, 40), (bar_x + int(bar_width * progress), 55),
                  (0, 255, 0) if progress >= 1.0 else (0, 200, 255), -1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Keyframe flash effect
    if keyframe_flash:
        cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 255, 0), 8)
        cv2.putText(frame, "CAPTURED!", (w//2 - 70, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

    # Instructions at bottom
    if status.total_frames >= min_keyframes:
        cv2.putText(frame, "Press ENTER to start 3D reconstruction, or continue capturing",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    elif status.missing_directions:
        dirs_text = "Turn: " + ", ".join(status.missing_directions)
        cv2.putText(frame, dirs_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)


def capture_keyframes(detector: FaceDetector, selector: KeyframeSelector,
                      min_keyframes: int = 8) -> List[KeyframeCandidate]:
    """Capture keyframes from webcam with real-time UI."""
    print("\n" + "=" * 60)
    print("PHASE 1: Keyframe Capture")
    print("=" * 60)
    print("Controls:")
    print("  q     - Quit")
    print("  r     - Reset collection")
    print("  SPACE - Force capture current frame")
    print("  ENTER - Start reconstruction (when enough keyframes)")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return []

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Webcam opened. Starting capture...")

    keyframes: List[KeyframeCandidate] = []
    last_keyframe_time = 0
    fps_start = time.time()
    fps_count = 0
    current_fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror

            # Update FPS
            fps_count += 1
            if time.time() - fps_start >= 1.0:
                current_fps = fps_count
                fps_count = 0
                fps_start = time.time()

            # Detect face
            detection = detector.detect(frame)
            keyframe_flash = (time.time() - last_keyframe_time) < 0.3

            # Check for auto-capture
            if detection is not None:
                should_capture = selector.should_capture(detection, keyframes, frame)
                if should_capture:
                    cropped = detector.crop_face_region(frame, detection)
                    candidate = KeyframeCandidate(
                        frame=cropped,
                        head_pose=detection.head_pose,
                        timestamp=time.time(),
                        quality_score=selector.compute_quality_score(frame, detection)
                    )
                    keyframes.append(candidate)
                    last_keyframe_time = time.time()
                    print(f"  Keyframe {len(keyframes)} captured: "
                          f"yaw={detection.head_pose[0]:+.1f}, pitch={detection.head_pose[1]:+.1f}")

            # Get coverage status
            status = selector.get_coverage_status(keyframes)

            # Draw UI
            draw_capture_ui(frame, detection, status, current_fps, keyframe_flash, min_keyframes)

            cv2.imshow("Enrollment Demo - Keyframe Capture", frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Cancelled by user.")
                keyframes.clear()
                break
            elif key == ord('r'):
                keyframes.clear()
                print("Keyframe collection reset.")
            elif key == ord(' ') and detection is not None:
                # Force capture
                cropped = detector.crop_face_region(frame, detection)
                candidate = KeyframeCandidate(
                    frame=cropped,
                    head_pose=detection.head_pose,
                    timestamp=time.time(),
                    quality_score=selector.compute_quality_score(frame, detection)
                )
                keyframes.append(candidate)
                last_keyframe_time = time.time()
                print(f"  Keyframe {len(keyframes)} (manual): "
                      f"yaw={detection.head_pose[0]:+.1f}, pitch={detection.head_pose[1]:+.1f}")
            elif key == 13 and len(keyframes) >= min_keyframes:  # ENTER
                print(f"\nCapture complete! {len(keyframes)} keyframes collected.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return keyframes


def run_reconstruction(keyframes: List[KeyframeCandidate]):
    """Run MASt3R 3D reconstruction on captured keyframes."""
    print("\n" + "=" * 60)
    print("PHASE 2: 3D Reconstruction with MASt3R")
    print("=" * 60)

    if len(keyframes) < 2:
        print("ERROR: Need at least 2 keyframes for reconstruction.")
        return None, None

    # Import MASt3R engine (lazy import for faster startup)
    print("Loading MASt3R engine...")
    from core.mast3r_engine import get_engine

    engine = get_engine()
    if not engine.is_loaded:
        print("Loading MASt3R model (this may take 30-60 seconds)...")
        engine.load_model()

    # Extract frames
    frames = [kf.frame for kf in keyframes]
    print(f"Running reconstruction with {len(frames)} frames...")

    start_time = time.time()
    result = engine.reconstruct_multiview(frames)
    elapsed = time.time() - start_time

    print(f"Reconstruction complete in {elapsed:.1f} seconds!")
    print(f"  Points: {result.point_cloud.shape[0]:,}")
    print(f"  Descriptor dim: {result.descriptors.shape[1] if result.descriptors is not None else 'N/A'}")

    return result.point_cloud, result.colors


def visualize_point_cloud(points: np.ndarray, colors: Optional[np.ndarray],
                          title: str = "3D Face Reconstruction"):
    """Visualize point cloud in browser using Plotly."""
    print("\n" + "=" * 60)
    print("PHASE 3: Visualization")
    print("=" * 60)

    # Subsample for performance
    max_points = 20000
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]
        print(f"Subsampled to {max_points} points for visualization.")

    # Generate color strings
    if colors is not None:
        color_strs = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in colors]
    else:
        # Color by depth
        z_norm = (points[:, 2] - points[:, 2].min()) / (points[:, 2].ptp() + 1e-6)
        color_strs = [f'rgb({int(255*z)},{int(100*(1-z))},{int(255*(1-z))})' for z in z_norm]

    # Create Plotly figure
    figure = {
        "data": [{
            "type": "scatter3d",
            "x": points[:, 0].tolist(),
            "y": points[:, 1].tolist(),
            "z": points[:, 2].tolist(),
            "mode": "markers",
            "marker": {
                "size": 1.5,
                "color": color_strs,
            },
            "hoverinfo": "skip",
        }],
        "layout": {
            "title": {"text": title, "font": {"size": 20}},
            "scene": {
                "aspectmode": "data",
                "xaxis": {"title": "X", "showgrid": True},
                "yaxis": {"title": "Y", "showgrid": True},
                "zaxis": {"title": "Z", "showgrid": True},
                "camera": {
                    "eye": {"x": 0, "y": 0, "z": -1.5},
                    "up": {"x": 0, "y": -1, "z": 0},
                },
            },
            "margin": {"l": 0, "r": 0, "t": 50, "b": 0},
            "width": 900,
            "height": 700,
            "paper_bgcolor": "#1a1a1a",
            "plot_bgcolor": "#1a1a1a",
            "font": {"color": "white"},
        }
    }

    # Create HTML file
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            margin: 0;
            background-color: #1a1a1a;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }}
        #plot {{ width: 100%; max-width: 900px; }}
        .info {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            color: #888;
            font-family: sans-serif;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div id="plot"></div>
    <div class="info">
        Points: {len(points):,} | Drag to rotate, scroll to zoom
    </div>
    <script>
        var figure = {json.dumps(figure)};
        Plotly.newPlot('plot', figure.data, figure.layout, {{responsive: true}});
    </script>
</body>
</html>"""

    # Save and open
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(html_content)
        html_path = f.name

    print(f"Opening visualization in browser...")
    print(f"  File: {html_path}")

    # Try to open in browser
    try:
        webbrowser.open(f'file://{html_path}')
        print("  Browser opened!")
    except Exception as e:
        print(f"  Could not open browser automatically: {e}")
        print(f"  Please open the file manually: {html_path}")

    return html_path


def save_keyframes(keyframes: List[KeyframeCandidate], output_dir: Path):
    """Save keyframes to disk for later use."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, kf in enumerate(keyframes):
        img_path = output_dir / f"keyframe_{i:02d}.jpg"
        cv2.imwrite(str(img_path), kf.frame)

    # Save metadata
    meta = {
        "count": len(keyframes),
        "poses": [list(kf.head_pose) for kf in keyframes],
        "quality_scores": [kf.quality_score for kf in keyframes],
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(keyframes)} keyframes to {output_dir}")


def load_keyframes(input_dir: Path) -> List[KeyframeCandidate]:
    """Load keyframes from disk."""
    meta_path = input_dir / "metadata.json"
    if not meta_path.exists():
        print(f"ERROR: No metadata.json found in {input_dir}")
        return []

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    keyframes = []
    for i in range(meta["count"]):
        img_path = input_dir / f"keyframe_{i:02d}.jpg"
        if not img_path.exists():
            print(f"WARNING: Missing {img_path}")
            continue

        frame = cv2.imread(str(img_path))
        kf = KeyframeCandidate(
            frame=frame,
            head_pose=tuple(meta["poses"][i]),
            timestamp=0.0,
            quality_score=meta["quality_scores"][i],
        )
        keyframes.append(kf)

    print(f"Loaded {len(keyframes)} keyframes from {input_dir}")
    return keyframes


def main():
    parser = argparse.ArgumentParser(
        description="Enrollment Demo - Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--min-keyframes", type=int, default=8,
        help="Minimum keyframes to capture (default: 8)"
    )
    parser.add_argument(
        "--skip-capture", action="store_true",
        help="Skip capture and load from saved keyframes"
    )
    parser.add_argument(
        "--keyframe-dir", type=str, default="storage/demo_keyframes",
        help="Directory for saving/loading keyframes"
    )
    parser.add_argument(
        "--save-keyframes", action="store_true",
        help="Save captured keyframes for later use"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MASt3R Face Authentication - Enrollment Demo")
    print("=" * 60)
    print()

    keyframe_dir = project_root / args.keyframe_dir

    # Phase 1: Capture or load keyframes
    if args.skip_capture:
        keyframes = load_keyframes(keyframe_dir)
        if not keyframes:
            print("No saved keyframes found. Run without --skip-capture first.")
            return 1
    else:
        # Initialize detectors
        print("Initializing face detector...")
        face_config = get_face_detection_config()
        detector = FaceDetector(face_config)

        print("Initializing keyframe selector...")
        keyframe_config = get_keyframe_config()
        # Override target count
        keyframe_config["target_count"] = args.min_keyframes
        selector = KeyframeSelector(keyframe_config)

        try:
            keyframes = capture_keyframes(detector, selector, args.min_keyframes)
        finally:
            detector.close()

        if not keyframes:
            print("No keyframes captured. Exiting.")
            return 1

        if args.save_keyframes:
            save_keyframes(keyframes, keyframe_dir)

    # Phase 2: 3D Reconstruction
    points, colors = run_reconstruction(keyframes)
    if points is None:
        return 1

    # Phase 3: Visualization
    html_path = visualize_point_cloud(points, colors,
                                       f"3D Face Reconstruction ({len(points):,} points)")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print(f"Total keyframes: {len(keyframes)}")
    print(f"Total 3D points: {len(points):,}")
    print(f"Visualization: {html_path}")
    print()
    print("The 3D point cloud should be displayed in your browser.")
    print("You can rotate, pan, and zoom the visualization.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
