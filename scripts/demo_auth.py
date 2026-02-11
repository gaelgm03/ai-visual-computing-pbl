"""
Authentication Demo Script - Full Pipeline Demo

This script demonstrates the complete authentication pipeline:
1. Real-time face capture with pose-guided targeting (yaw ±15°, pitch ±5°)
2. MASt3R 3D reconstruction
3. ArcFace embedding extraction
4. Anti-spoofing check
5. Template matching against enrolled users

The capture phase guides the user to turn their head to 5 target
directions (tolerance ±10°) so MASt3R receives sufficient parallax
for a proper 3D reconstruction.

Usage:
    # Capture frames from webcam and authenticate (requires GPU)
    python scripts/demo_auth.py

    # Capture-only mode (Windows, no GPU needed) — saves frames for WSL
    python scripts/demo_auth.py --capture-only --output-dir storage/auth_capture

    # Run authentication on WSL2 using saved keyframes
    python scripts/demo_auth.py --skip-capture --keyframe-dir storage/auth_capture

    # 1:1 verification against a specific user
    python scripts/demo_auth.py --skip-capture --user-id usr_abc123

    # Customize number of target poses (max 5)
    python scripts/demo_auth.py --num-frames 3

Controls (during capture):
    - Turn your head to the displayed direction to auto-capture
    - Press ENTER when ready to authenticate (if enough frames)
    - Press 'q' to quit

Author: CS-1
"""

import cv2
import numpy as np
import time
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.face_detector import FaceDetector, FaceDetection
from core.keyframe_selector import KeyframeCandidate
from core.config import get_face_detection_config, get_config
from core.ui_overlay import draw_face_guide, draw_pose_grid


def draw_auth_ui(frame: np.ndarray, detection: Optional[FaceDetection],
                 num_captured: int, num_target: int, fps: float,
                 keyframe_flash: bool,
                 guide_direction: str = "",
                 target_poses: Optional[List] = None,
                 captured_mask: Optional[List[bool]] = None):
    """Draw the authentication capture UI overlay with pose guidance."""
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

        # Pose info
        yaw, pitch, roll = detection.head_pose
        cv2.putText(frame, f"Yaw:{yaw:+.0f} Pitch:{pitch:+.0f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No face detected",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1, cv2.LINE_AA)

    # Capture count + progress
    count_text = f"Captured: {num_captured}/{num_target}"
    cv2.putText(frame, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)

    progress = num_captured / max(num_target, 1)
    bar_w = 120
    bar_x = 200
    cv2.rectangle(frame, (bar_x, 38), (bar_x + bar_w, 53), (100, 100, 100), -1)
    cv2.rectangle(frame, (bar_x, 38), (bar_x + int(bar_w * progress), 53),
                  (0, 255, 0) if progress >= 1.0 else (0, 200, 255), -1)

    # Direction guidance
    if guide_direction and num_captured < num_target:
        cv2.putText(frame, f"Look {guide_direction}",
                    (10, 73), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 90, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Keyframe flash
    if keyframe_flash:
        cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 255, 0), 6)

    # Instructions
    if num_captured >= num_target:
        cv2.putText(frame, "Ready! Press ENTER to authenticate",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "ENTER=authenticate  q=quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Pose grid overlay (bottom-right)
    if target_poses is not None and captured_mask is not None:
        cur = (detection.head_pose[0], detection.head_pose[1]) if detection else None
        draw_pose_grid(frame, target_poses, captured_mask, cur,
                       yaw_range=(-20.0, 20.0), pitch_range=(-10.0, 10.0))


# ---------------------------------------------------------------------------
# Auth capture target-pose system
# ---------------------------------------------------------------------------
# 5 target poses: center + 4 diagonal corners (yaw ±15°, pitch ±5°).
# The corner layout maximises angular diversity in both axes, giving
# MASt3R strong parallax for 3D reconstruction and anti-spoofing.
AUTH_TARGETS = [
    (  0.0,  0.0),   # Center
    (-15.0,  5.0),   # Upper-right  (user perspective)
    ( 15.0,  5.0),   # Upper-left   (user perspective)
    (-15.0, -5.0),   # Lower-right  (user perspective)
    ( 15.0, -5.0),   # Lower-left   (user perspective)
]

# Tolerance (degrees) — capture triggers when head pose is within this
# distance of an uncaptured target.  More lenient than enrollment's 7°.
AUTH_TARGET_TOLERANCE = 10.0


def _direction_label(yaw: float, pitch: float) -> str:
    """Human-readable direction from yaw/pitch (mirrored for webcam)."""
    parts = []
    if yaw < -5:
        parts.append("RIGHT")
    elif yaw > 5:
        parts.append("LEFT")
    if pitch < -3:
        parts.append("DOWN")
    elif pitch > 3:
        parts.append("UP")
    return " & ".join(parts) if parts else "CENTER"


def capture_auth_frames(detector: FaceDetector,
                        num_frames: int = 5) -> List[KeyframeCandidate]:
    """
    Capture frames from webcam for authentication with pose targeting.

    Uses pre-defined target poses (yaw ±15°, pitch ±5°).  A frame is
    auto-captured when the user's head pose comes within
    AUTH_TARGET_TOLERANCE (±10°) of an uncaptured target.

    This ensures enough angular diversity for MASt3R to produce a solid
    3D reconstruction and reliably pass anti-spoofing checks.

    Args:
        detector: FaceDetector instance.
        num_frames: Number of target poses to use (max 5, default 5).

    Returns:
        List of KeyframeCandidate objects.
    """
    # Select targets (use first num_frames targets from the list)
    targets = AUTH_TARGETS[:min(num_frames, len(AUTH_TARGETS))]
    captured_mask = [False] * len(targets)
    target_labels = [_direction_label(y, p) for y, p in targets]

    print("\n" + "=" * 60)
    print("PHASE 1: Frame Capture for Authentication")
    print("=" * 60)
    print("Controls:")
    print("  ENTER - Start authentication (need at least 2 frames)")
    print("  q     - Quit")
    print()
    print(f"Turn your head to each direction (tolerance ±{AUTH_TARGET_TOLERANCE:.0f}°):")
    for i, ((y, p), lbl) in enumerate(zip(targets, target_labels)):
        print(f"  [{i+1}] yaw={y:+.0f}° pitch={p:+.0f}°  -> Look {lbl}")
    print()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return []

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Webcam opened.")

    # ------------------------------------------------------------------
    # Alignment phase: show face guide, wait for SPACE to start capture
    # ------------------------------------------------------------------
    print("Align your face to the guide ellipse, then press SPACE to start.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cv2.destroyAllWindows()
                return []
            frame = cv2.flip(frame, 1)
            detection = detector.detect(frame.copy())
            draw_face_guide(frame, face_detected=(detection is not None))
            cv2.imshow("Authentication - Frame Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                print("Starting capture...\n")
                break
            elif key == ord('q'):
                print("Cancelled by user.")
                cap.release()
                cv2.destroyAllWindows()
                return []
    except Exception:
        cap.release()
        cv2.destroyAllWindows()
        return []

    # ------------------------------------------------------------------
    # Capture phase: target-pose based keyframe collection
    # ------------------------------------------------------------------
    keyframes: List[KeyframeCandidate] = []
    last_capture_time = 0
    fps_start = time.time()
    fps_count = 0
    current_fps = 0.0

    def _find_matching_target(yaw, pitch):
        """Return index of closest uncaptured target within tolerance, or -1."""
        best_idx, best_dist = -1, AUTH_TARGET_TOLERANCE
        for idx, ((ty, tp), done) in enumerate(zip(targets, captured_mask)):
            if done:
                continue
            dist = ((yaw - ty) ** 2 + (pitch - tp) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def _next_uncaptured_label():
        for idx, done in enumerate(captured_mask):
            if not done:
                return target_labels[idx]
        return ""

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
            clean_frame = frame.copy()
            detection = detector.detect(clean_frame)
            keyframe_flash = (time.time() - last_capture_time) < 0.3

            # Auto-capture when face matches an uncaptured target
            if (detection is not None
                    and len(keyframes) < len(targets)
                    and time.time() - last_capture_time > 0.5):
                yaw, pitch, _roll = detection.head_pose
                target_idx = _find_matching_target(yaw, pitch)
                if target_idx >= 0:
                    cropped = detector.crop_face_region(clean_frame, detection)
                    candidate = KeyframeCandidate(
                        frame=cropped,
                        head_pose=detection.head_pose,
                        timestamp=time.time(),
                        quality_score=detection.confidence,
                    )
                    keyframes.append(candidate)
                    captured_mask[target_idx] = True
                    last_capture_time = time.time()
                    lbl = target_labels[target_idx]
                    print(f"  Captured [{target_idx+1}] {lbl}: "
                          f"yaw={yaw:+.1f}, pitch={pitch:+.1f}")

            # Draw UI with pose grid
            guide = _next_uncaptured_label()
            draw_auth_ui(frame, detection, len(keyframes), len(targets),
                         current_fps, keyframe_flash,
                         guide_direction=guide,
                         target_poses=targets,
                         captured_mask=captured_mask)
            cv2.imshow("Authentication - Frame Capture", frame)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Cancelled by user.")
                keyframes.clear()
                break
            elif key == 13 and len(keyframes) >= 2:  # ENTER
                print(f"\nCapture complete! {len(keyframes)} frames collected.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return keyframes


def load_keyframes(input_dir: Path) -> List[KeyframeCandidate]:
    """Load keyframes from disk (same format as demo_enrollment.py)."""
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


def run_auth_pipeline(keyframes: List[KeyframeCandidate],
                      user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Run the full authentication pipeline.

    Args:
        keyframes: List of captured KeyframeCandidate objects.
        user_id: If provided, 1:1 verification. Otherwise 1:N identification.

    Returns:
        Dictionary with authentication results.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Authentication Pipeline")
    print("=" * 60)

    if len(keyframes) < 2:
        print("ERROR: Need at least 2 frames for authentication.")
        return {"error": "Not enough frames"}

    frames = [kf.frame for kf in keyframes]
    head_poses = [kf.head_pose for kf in keyframes]
    config = get_config()
    matching_config = config.get("matching", {})

    # --- Step 1: MASt3R 3D Reconstruction ---
    print("\n[1/5] MASt3R 3D Reconstruction...")
    from core.mast3r_engine import get_engine

    engine = get_engine()
    if not engine.is_loaded:
        print("  Loading MASt3R model (this may take 30-60 seconds)...")
        engine.load_model()

    start_time = time.time()
    result = engine.reconstruct_multiview(frames, head_poses=head_poses)
    recon_time = time.time() - start_time

    probe_cloud = result.point_cloud
    probe_descriptors = result.descriptors
    probe_confidence = result.confidence

    print(f"  Reconstruction: {len(probe_cloud):,} points in {recon_time:.1f}s")

    # --- Step 2: ArcFace Embedding ---
    print("\n[2/5] ArcFace Embedding Extraction...")
    probe_embedding = None
    try:
        from core.face_embedder import FaceEmbedder

        embedding_config = config.get("face_embedding", {})
        embedder = FaceEmbedder(embedding_config)
        embedder.load_model()
        # frames are BGR (from OpenCV crop)
        probe_embedding = embedder.extract_multi_frame(frames)
        if probe_embedding is not None:
            print(f"  Embedding: shape={probe_embedding.shape}, "
                  f"norm={np.linalg.norm(probe_embedding):.4f}")
        else:
            print("  WARNING: Embedding extraction returned None")
    except ImportError:
        print("  INFO: insightface not installed — skipping ArcFace")
    except Exception as e:
        print(f"  WARNING: Embedding extraction failed: {e}")

    # --- Step 3: Anti-Spoofing ---
    print("\n[3/5] Anti-Spoofing Check...")
    from core.anti_spoof import get_anti_spoof

    anti_spoof = get_anti_spoof()
    spoof_result = anti_spoof.check(probe_cloud, probe_confidence)

    print(f"  Passed: {spoof_result.passed}")
    print(f"  Depth variance: {spoof_result.depth_variance:.6f}")
    print(f"  Planarity ratio: {spoof_result.eigenvalue_ratio:.6f}")
    print(f"  Confidence mean: {spoof_result.confidence_mean:.4f}")

    if not spoof_result.passed:
        print("\n  ANTI-SPOOFING FAILED — possible presentation attack!")
        return {
            "is_match": False,
            "matched_user_id": None,
            "matched_user_name": None,
            "final_score": 0.0,
            "embedding_score": 0.0,
            "geometric_score": 0.0,
            "descriptor_score": 0.0,
            "accept_threshold": matching_config.get("accept_threshold", 0.65),
            "weights": {
                "embedding": matching_config.get("embedding_weight", 0.40),
                "geometric": matching_config.get("geometric_weight", 0.10),
                "descriptor": matching_config.get("descriptor_weight", 0.50),
            },
            "anti_spoof_passed": False,
            "depth_variance": spoof_result.depth_variance,
            "planarity_ratio": spoof_result.eigenvalue_ratio,
            "n_probe_points": len(probe_cloud),
            "reconstruction_time_sec": recon_time,
        }

    # --- Step 4: Load Templates ---
    print("\n[4/5] Loading Templates...")
    from core.template_manager import get_template_manager

    template_manager = get_template_manager()

    if user_id:
        print(f"  1:1 verification mode — loading template for {user_id}")
        template = template_manager.load_template(user_id)
        if template is None:
            print(f"  ERROR: User {user_id} not found!")
            return {"error": f"User {user_id} not found"}
        templates = [template]
    else:
        print("  1:N identification mode — loading all templates")
        templates = template_manager.load_all_templates()

    print(f"  Loaded {len(templates)} template(s)")

    if not templates:
        print("  ERROR: No enrolled templates found!")
        return {"error": "No enrolled templates"}

    # --- Step 5: Matching ---
    print("\n[5/5] Running Matchers...")
    from core.matching import (
        ICPGeometricMatcher,
        NNDescriptorMatcher,
        ArcFaceEmbeddingMatcher,
        MultiModalFusion,
        WeightedFusion,
        StubGeometricMatcher,
        StubDescriptorMatcher,
        StubScoreFusion,
    )

    # Initialize matchers
    geo_matcher = ICPGeometricMatcher(matching_config) if ICPGeometricMatcher else StubGeometricMatcher()
    desc_matcher = NNDescriptorMatcher(matching_config) if NNDescriptorMatcher else StubDescriptorMatcher()
    emb_matcher = ArcFaceEmbeddingMatcher(matching_config) if ArcFaceEmbeddingMatcher else None

    if MultiModalFusion:
        fusion = MultiModalFusion(matching_config)
    elif WeightedFusion:
        fusion = WeightedFusion(matching_config)
    else:
        fusion = StubScoreFusion()

    legacy_fusion = WeightedFusion(matching_config) if WeightedFusion else StubScoreFusion()

    accept_threshold = matching_config.get("accept_threshold", 0.65)

    # Read configured weights (MultiModalFusion normalizes these internally)
    w_emb = matching_config.get("embedding_weight", 0.40)
    w_geo = matching_config.get("geometric_weight", 0.10)
    w_desc = matching_config.get("descriptor_weight", 0.50)

    best_match = None
    best_score = -1.0
    best_geo_score = 0.0
    best_desc_score = 0.0
    best_emb_score = 0.0

    for template in templates:
        try:
            # Geometric
            geo_result = geo_matcher.compare(probe_cloud, template.point_cloud)
            # Descriptor
            desc_result = desc_matcher.compare(
                probe_descriptors, template.descriptors,
                probe_cloud, template.point_cloud,
            )

            # Embedding + fusion
            use_embedding = (
                emb_matcher is not None
                and probe_embedding is not None
                and template.face_embedding is not None
            )

            if use_embedding:
                emb_result = emb_matcher.compare(probe_embedding, template.face_embedding)
                fused = fusion.fuse({
                    "embedding": emb_result,
                    "geometric": geo_result,
                    "descriptor": desc_result,
                })
                emb_score = emb_result.score
            else:
                fused = legacy_fusion.fuse(geo_result, desc_result)
                emb_score = 0.0

            print(f"  vs {template.user_name} (id={template.user_id}): "
                  f"emb={emb_score:.3f}, geo={geo_result.score:.3f}, "
                  f"desc={desc_result.score:.3f}, fused={fused.score:.3f} "
                  f"{'MATCH' if fused.is_match else 'NO MATCH'}")

            if fused.score > best_score:
                best_score = fused.score
                best_geo_score = geo_result.score
                best_desc_score = desc_result.score
                best_emb_score = emb_score
                best_match = template if fused.is_match else None

        except Exception as e:
            print(f"  ERROR matching vs {template.user_id}: {e}")
            continue

    return {
        "is_match": best_match is not None,
        "matched_user_id": best_match.user_id if best_match else None,
        "matched_user_name": best_match.user_name if best_match else None,
        "final_score": max(0.0, best_score),
        "embedding_score": best_emb_score,
        "geometric_score": best_geo_score,
        "descriptor_score": best_desc_score,
        "accept_threshold": accept_threshold,
        "weights": {"embedding": w_emb, "geometric": w_geo, "descriptor": w_desc},
        "anti_spoof_passed": True,
        "depth_variance": spoof_result.depth_variance,
        "planarity_ratio": spoof_result.eigenvalue_ratio,
        "n_probe_points": len(probe_cloud),
        "reconstruction_time_sec": recon_time,
    }


def print_results(results: Dict[str, Any]):
    """Print authentication results in a formatted table."""
    print("\n" + "=" * 60)
    print("AUTHENTICATION RESULTS")
    print("=" * 60)

    if "error" in results:
        print(f"  ERROR: {results['error']}")
        return

    is_match = results["is_match"]

    # Decision
    if is_match:
        print(f"  Decision:     MATCH")
        print(f"  Matched User: {results['matched_user_name']} ({results['matched_user_id']})")
    else:
        print(f"  Decision:     NO MATCH")

    print()

    # Scores
    threshold = results.get("accept_threshold", 0.65)
    weights = results.get("weights", {"embedding": 0.7, "geometric": 0.3, "descriptor": 0.0})
    final = results["final_score"]
    print(f"  Final Score:      {final:.4f}  {'>' if final >= threshold else '<'} {threshold} (threshold)")
    print(f"  Embedding Score:  {results['embedding_score']:.4f}  (weight: {weights['embedding']})")
    print(f"  Geometric Score:  {results['geometric_score']:.4f}  (weight: {weights['geometric']})")
    print(f"  Descriptor Score: {results['descriptor_score']:.4f}  (weight: {weights['descriptor']})")

    print()

    # Anti-spoofing
    print(f"  Anti-Spoof:       {'PASSED' if results['anti_spoof_passed'] else 'FAILED'}")
    print(f"  Depth Variance:   {results['depth_variance']:.6f}")
    print(f"  Planarity Ratio:  {results['planarity_ratio']:.6f}")

    print()

    # Reconstruction
    print(f"  Probe Points:     {results['n_probe_points']:,}")
    print(f"  Recon Time:       {results['reconstruction_time_sec']:.1f}s")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Authentication Demo - Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--user-id", type=str, default=None,
        help="User ID for 1:1 verification (omit for 1:N identification)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=5,
        help="Number of target poses to capture (default: 5, max: 5)",
    )
    parser.add_argument(
        "--skip-capture", action="store_true",
        help="Skip webcam capture and load from saved keyframes",
    )
    parser.add_argument(
        "--keyframe-dir", type=str, default="storage/demo_keyframes",
        help="Directory for loading keyframes (with --skip-capture)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save captured frames",
    )
    parser.add_argument(
        "--capture-only", action="store_true",
        help="Capture frames and save to --output-dir, then exit "
             "(no MASt3R/matching). Use on Windows where GPU is unavailable.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MASt3R Face Authentication - Authentication Demo")
    print("=" * 60)
    print()

    mode = "1:1 verification" if args.user_id else "1:N identification"
    print(f"Mode: {mode}")
    if args.user_id:
        print(f"Target user: {args.user_id}")
    print()

    # Phase 1: Capture or load keyframes
    if args.skip_capture:
        keyframe_dir = project_root / args.keyframe_dir
        keyframes = load_keyframes(keyframe_dir)
        if not keyframes:
            print("No saved keyframes found. Run without --skip-capture first.")
            return 1
    else:
        print("Initializing face detector...")
        face_config = get_face_detection_config()
        detector = FaceDetector(face_config)

        try:
            keyframes = capture_auth_frames(detector, num_frames=args.num_frames)
        finally:
            detector.close()

        if not keyframes:
            print("No frames captured. Exiting.")
            return 1

        # Save captured frames (required for --capture-only, optional otherwise)
        if args.output_dir or args.capture_only:
            out_dir = Path(args.output_dir or "storage/auth_capture")
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, kf in enumerate(keyframes):
                cv2.imwrite(str(out_dir / f"keyframe_{i:02d}.jpg"), kf.frame)
            meta = {
                "count": len(keyframes),
                "poses": [list(kf.head_pose) for kf in keyframes],
                "quality_scores": [kf.quality_score for kf in keyframes],
            }
            with open(out_dir / "metadata.json", 'w') as f:
                json.dump(meta, f, indent=2)
            print(f"Saved {len(keyframes)} frames to {out_dir}")

            if args.capture_only:
                print("\nCapture-only mode: frames saved. "
                      "Run authentication on WSL2 with:")
                print(f"  python scripts/demo_auth.py --skip-capture "
                      f"--keyframe-dir {out_dir}")
                return 0

    if len(keyframes) < 2:
        print("ERROR: Need at least 2 frames. Exiting.")
        return 1

    # Phase 2: Run authentication pipeline
    results = run_auth_pipeline(keyframes, user_id=args.user_id)

    # Phase 3: Print results
    print_results(results)

    return 0 if results.get("is_match", False) else 1


if __name__ == "__main__":
    sys.exit(main())
