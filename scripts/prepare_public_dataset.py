"""
Public Dataset → .npz Conversion Script

Processes a public face dataset into MASt3R-based .npz enrollment templates
and authentication probes for DS team evaluation.

Pipeline per person:
  1. Load all images from person directory
  2. Run MediaPipe face detection on each → head_pose, confidence, bbox
  3. Filter: no-face, low-confidence, excessive-roll images discarded
  4. Select 12 diverse enrollment frames (iterative replacement algorithm)
  5. Select 4 diverse authentication frames from remaining images
  6. Crop faces with padding → run MASt3R reconstruction → post-process
  7. Save as .npz (point_cloud, descriptors, confidence, colors, metadata)

Output structure (ready for Google Drive upload):
  <output_dir>/
  ├── mast3r_outputs/
  │   ├── <person1>_enrollment.npz
  │   └── ...
  └── auth_probes/
      ├── <person1>_probe.npz
      └── ...

Usage (WSL):
  source ~/mast3r-face-auth/bin/activate
  cd /mnt/c/Users/sekit/ai-visual-computing-pbl

  python scripts/prepare_public_dataset.py \\
    --dataset-dir /mnt/c/Users/sekit/Downloads/<dataset_name> \\
    --output-dir /mnt/c/Users/sekit/Downloads

Author: CS-1
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.face_detector import FaceDetector, FaceDetection
from core.config import get_face_detection_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ============================================================
# Data Structures
# ============================================================

@dataclass
class ImageInfo:
    """Metadata for a single image after face detection."""
    path: Path
    head_pose: Tuple[float, float, float]  # (yaw, pitch, roll)
    confidence: float
    detection: FaceDetection
    quality_score: float = 0.0


@dataclass
class PersonResult:
    """Processing result for a single person."""
    name: str
    enrollment_images: List[ImageInfo]
    auth_images: List[ImageInfo]
    enrollment_npz_path: Optional[str] = None
    auth_npz_path: Optional[str] = None
    enrollment_points: int = 0
    auth_points: int = 0
    elapsed_sec: float = 0.0
    error: Optional[str] = None


# ============================================================
# Face Detection & Filtering
# ============================================================

def scan_person_images(
    person_dir: Path,
    detector: FaceDetector,
    min_confidence: float = 0.9,
    max_roll: float = 15.0,
) -> List[ImageInfo]:
    """
    Load all images for a person and run face detection.

    Returns only images with a valid face detection that passes
    confidence and roll filters.
    """
    image_paths = sorted(
        p for p in person_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        logger.warning(f"  No images found in {person_dir}")
        return []

    logger.info(f"  Scanning {len(image_paths)} images...")
    valid_images: List[ImageInfo] = []

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        detection = detector.detect(frame)
        if detection is None:
            continue

        # Filter by confidence
        if detection.confidence < min_confidence:
            continue

        # Filter by roll
        _, _, roll = detection.head_pose
        if abs(roll) > max_roll:
            continue

        # Compute quality score (blur detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        valid_images.append(ImageInfo(
            path=img_path,
            head_pose=detection.head_pose,
            confidence=detection.confidence,
            detection=detection,
            quality_score=blur_score,
        ))

    logger.info(f"  {len(valid_images)}/{len(image_paths)} images passed detection + filters")
    return valid_images


# ============================================================
# Diversity-Aware Frame Selection
# ============================================================

def angular_distance(pose_a: Tuple[float, float, float],
                     pose_b: Tuple[float, float, float]) -> float:
    """Euclidean distance in (yaw, pitch) space (ignoring roll)."""
    dy = pose_a[0] - pose_b[0]
    dp = pose_a[1] - pose_b[1]
    return math.sqrt(dy * dy + dp * dp)


def select_diverse_frames(
    candidates: List[ImageInfo],
    n_select: int,
    diversity_threshold: float,
    max_iterations: int = 200,
    rng: Optional[random.Random] = None,
) -> List[ImageInfo]:
    """
    Select n_select diverse frames from candidates using iterative replacement.

    Algorithm (user-specified):
    1. Randomly select n_select images
    2. Check all pairs for angular distance
    3. If any 2+ have distance < threshold: keep highest-quality, remove rest
    4. Replace removed with new random picks from unused pool
    5. Repeat until all pairs are diverse or max_iterations reached

    Args:
        candidates: Pool of valid images to select from.
        n_select: Number of frames to select.
        diversity_threshold: Minimum angular distance (degrees) between any pair.
        max_iterations: Maximum replacement iterations.
        rng: Random number generator (for reproducibility).

    Returns:
        List of selected ImageInfo objects.
    """
    if rng is None:
        rng = random.Random()

    if len(candidates) <= n_select:
        logger.warning(
            f"  Only {len(candidates)} valid images, need {n_select}. Using all."
        )
        return list(candidates)

    # Index-based selection for efficiency
    all_indices = list(range(len(candidates)))
    selected_indices = rng.sample(all_indices, n_select)
    used_set = set(selected_indices)

    for iteration in range(max_iterations):
        # Find all conflicting pairs (distance < threshold)
        conflicts: Dict[int, List[int]] = {}  # position → list of conflicting positions
        for i in range(len(selected_indices)):
            for j in range(i + 1, len(selected_indices)):
                dist = angular_distance(
                    candidates[selected_indices[i]].head_pose,
                    candidates[selected_indices[j]].head_pose,
                )
                if dist < diversity_threshold:
                    conflicts.setdefault(i, []).append(j)
                    conflicts.setdefault(j, []).append(i)

        if not conflicts:
            break

        # For each conflict group: keep the one with highest quality, remove rest
        to_remove_positions = set()
        visited = set()
        for pos in sorted(conflicts.keys()):
            if pos in visited or pos in to_remove_positions:
                continue
            # Gather the conflict cluster
            cluster = {pos}
            queue = [pos]
            while queue:
                current = queue.pop(0)
                for neighbor in conflicts.get(current, []):
                    if neighbor not in cluster:
                        cluster.add(neighbor)
                        queue.append(neighbor)
            visited |= cluster

            # Keep the one with highest quality in the cluster
            best_pos = max(cluster, key=lambda p: candidates[selected_indices[p]].quality_score)
            to_remove_positions |= (cluster - {best_pos})

        if not to_remove_positions:
            break

        # Build pool of unused indices
        unused_indices = [idx for idx in all_indices if idx not in used_set]
        rng.shuffle(unused_indices)

        # Replace removed positions
        replaced = 0
        for pos in sorted(to_remove_positions):
            if unused_indices:
                new_idx = unused_indices.pop()
                old_idx = selected_indices[pos]
                used_set.discard(old_idx)
                selected_indices[pos] = new_idx
                used_set.add(new_idx)
                replaced += 1
            # If no more unused candidates, the position keeps its current value
            # (which is suboptimal but the best we can do)

        if replaced == 0:
            logger.warning(f"  No replacement candidates available at iteration {iteration}")
            break

    # Check final diversity
    min_dist = float("inf")
    for i in range(len(selected_indices)):
        for j in range(i + 1, len(selected_indices)):
            dist = angular_distance(
                candidates[selected_indices[i]].head_pose,
                candidates[selected_indices[j]].head_pose,
            )
            min_dist = min(min_dist, dist)

    if min_dist < diversity_threshold:
        logger.warning(
            f"  Could not achieve full diversity (min distance: {min_dist:.1f}°, "
            f"threshold: {diversity_threshold:.1f}°). Using best available."
        )

    return [candidates[idx] for idx in selected_indices]


# ============================================================
# MASt3R Reconstruction
# ============================================================

def run_mast3r_reconstruction(
    images: List[ImageInfo],
    detector: FaceDetector,
    label: str = "enrollment",
    face_embedder=None,
) -> Optional[dict]:
    """
    Run MASt3R reconstruction on selected images.

    Returns dict with point_cloud, descriptors, confidence, colors,
    and optionally face_embedding (if face_embedder is provided).
    Returns None on failure.
    """
    if len(images) < 2:
        logger.error(f"  Need at least 2 frames for {label} reconstruction, got {len(images)}")
        return None

    # Lazy import MASt3R engine (heavy)
    from core.mast3r_engine import get_engine

    engine = get_engine()
    if not engine.is_loaded:
        logger.info("  Loading MASt3R model (30-60 seconds)...")
        engine.load_model()

    # Load and crop faces
    frames = []
    head_poses = []
    for img_info in images:
        frame = cv2.imread(str(img_info.path))
        if frame is None:
            logger.warning(f"  Could not read {img_info.path}, skipping")
            continue
        cropped = detector.crop_face_region(frame, img_info.detection, padding=0.3)
        frames.append(cropped)
        head_poses.append(img_info.head_pose)

    if len(frames) < 2:
        logger.error(f"  Only {len(frames)} valid frames after cropping for {label}")
        return None

    # Extract ArcFace embedding from face crops (before MASt3R processing)
    face_embedding = None
    if face_embedder is not None:
        logger.info(f"  Extracting ArcFace embedding from {len(frames)} face crops...")
        face_embedding = face_embedder.extract_multi_frame(frames)
        if face_embedding is not None:
            logger.info(f"  ArcFace embedding: dim={face_embedding.shape[0]}, norm={np.linalg.norm(face_embedding):.4f}")
        else:
            logger.warning(f"  ArcFace failed to detect any face in {label} frames")

    logger.info(f"  Running MASt3R {label} reconstruction with {len(frames)} frames...")
    start = time.time()

    result = engine.reconstruct_multiview(frames, head_poses=head_poses)

    elapsed = time.time() - start
    n_points = result.point_cloud.shape[0]
    logger.info(f"  {label.capitalize()} reconstruction: {n_points:,} points in {elapsed:.1f}s")

    output = {
        "point_cloud": result.point_cloud,
        "descriptors": result.descriptors,
        "confidence": result.confidence,
        "colors": result.colors,
    }
    if face_embedding is not None:
        output["face_embedding"] = face_embedding

    return output


# ============================================================
# .npz Export
# ============================================================

def save_npz(
    data: dict,
    output_path: Path,
    person_name: str,
    n_frames: int,
    head_poses: List[Tuple[float, float, float]],
    source_images: List[str],
    dataset_name: str,
    label: str = "enrollment",
):
    """Save MASt3R reconstruction result as .npz following architecture format."""
    yaws = [p[0] for p in head_poses]
    pitches = [p[1] for p in head_poses]

    metadata = {
        "user_id": person_name,
        "user_name": person_name,
        "enrolled_at": datetime.now().isoformat(),
        "n_frames": n_frames,
        "yaw_range": [float(min(yaws)), float(max(yaws))],
        "pitch_range": [float(min(pitches)), float(max(pitches))],
        "mast3r_version": "ViTLarge_metric",
        "template_version": "1.0",
        "source_dataset": dataset_name,
        "source_images": source_images,
        "label": label,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = dict(
        point_cloud=data["point_cloud"],
        descriptors=data["descriptors"],
        confidence=data["confidence"],
        colors=data["colors"],
    )
    if "face_embedding" in data and data["face_embedding"] is not None:
        save_kwargs["face_embedding"] = data["face_embedding"]
        metadata["template_version"] = "2.0"
        metadata["embedding_model"] = "buffalo_l"
        metadata["embedding_dim"] = int(data["face_embedding"].shape[0])

    save_kwargs["metadata"] = json.dumps(metadata)
    np.savez_compressed(str(output_path), **save_kwargs)

    emb_info = ""
    if "face_embedding" in save_kwargs:
        emb_info = f", embedding={save_kwargs['face_embedding'].shape}"
    logger.info(f"  Saved {output_path.name} ({data['point_cloud'].shape[0]:,} points{emb_info})")


# ============================================================
# Main Processing
# ============================================================

def process_person(
    person_dir: Path,
    detector: FaceDetector,
    enrollment_dir: Path,
    auth_dir: Path,
    dataset_name: str,
    enrollment_frames: int = 12,
    auth_frames: int = 4,
    diversity_threshold: float = 8.0,
    auth_diversity_threshold: float = 8.0,
    min_confidence: float = 0.9,
    max_roll: float = 15.0,
    rng: Optional[random.Random] = None,
    skip_existing: bool = True,
    face_embedder=None,
) -> PersonResult:
    """Process a single person's images into enrollment + auth .npz files."""
    person_name = person_dir.name
    result = PersonResult(
        name=person_name,
        enrollment_images=[],
        auth_images=[],
    )

    enrollment_path = enrollment_dir / f"{person_name}_enrollment.npz"
    auth_path = auth_dir / f"{person_name}_probe.npz"

    # Resume support: skip if both files already exist
    if skip_existing and enrollment_path.exists() and auth_path.exists():
        logger.info(f"  Skipping {person_name} (already processed)")
        result.enrollment_npz_path = str(enrollment_path)
        result.auth_npz_path = str(auth_path)
        return result

    start_time = time.time()

    # Step 1: Scan all images
    valid_images = scan_person_images(person_dir, detector, min_confidence, max_roll)

    if len(valid_images) < 2:
        result.error = f"Only {len(valid_images)} valid images (need at least 2)"
        logger.error(f"  {result.error}")
        return result

    # Step 2: Select diverse enrollment frames
    logger.info(f"  Selecting {enrollment_frames} enrollment frames (threshold={diversity_threshold}°)...")
    enrollment_images = select_diverse_frames(
        valid_images, enrollment_frames, diversity_threshold, rng=rng,
    )
    result.enrollment_images = enrollment_images

    # Step 3: Select diverse authentication frames from REMAINING images
    used_paths = {img.path for img in enrollment_images}
    remaining = [img for img in valid_images if img.path not in used_paths]

    if len(remaining) < 2:
        logger.warning(
            f"  Only {len(remaining)} images remaining after enrollment selection. "
            f"Cannot create auth probe (need >= 2)."
        )
        result.error = f"Not enough remaining images for auth ({len(remaining)})"
    else:
        logger.info(f"  Selecting {auth_frames} auth frames from {len(remaining)} remaining (threshold={auth_diversity_threshold}°)...")
        auth_images = select_diverse_frames(
            remaining, auth_frames, auth_diversity_threshold, rng=rng,
        )
        result.auth_images = auth_images

    # Step 4+5: MASt3R reconstruction + save enrollment
    if not (skip_existing and enrollment_path.exists()):
        enrollment_data = run_mast3r_reconstruction(
            enrollment_images, detector, label="enrollment",
            face_embedder=face_embedder,
        )
        if enrollment_data:
            save_npz(
                enrollment_data, enrollment_path, person_name,
                n_frames=len(enrollment_images),
                head_poses=[img.head_pose for img in enrollment_images],
                source_images=[img.path.name for img in enrollment_images],
                dataset_name=dataset_name,
                label="enrollment",
            )
            result.enrollment_npz_path = str(enrollment_path)
            result.enrollment_points = enrollment_data["point_cloud"].shape[0]
    else:
        logger.info(f"  Enrollment .npz already exists, skipping reconstruction")
        result.enrollment_npz_path = str(enrollment_path)

    # Step 4+5: MASt3R reconstruction + save auth
    if result.auth_images:
        if not (skip_existing and auth_path.exists()):
            auth_data = run_mast3r_reconstruction(
                result.auth_images, detector, label="auth",
                face_embedder=face_embedder,
            )
            if auth_data:
                save_npz(
                    auth_data, auth_path, person_name,
                    n_frames=len(result.auth_images),
                    head_poses=[img.head_pose for img in result.auth_images],
                    source_images=[img.path.name for img in result.auth_images],
                    dataset_name=dataset_name,
                    label="auth_probe",
                )
                result.auth_npz_path = str(auth_path)
                result.auth_points = auth_data["point_cloud"].shape[0]
        else:
            logger.info(f"  Auth .npz already exists, skipping reconstruction")
            result.auth_npz_path = str(auth_path)

    result.elapsed_sec = time.time() - start_time
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert public face dataset to .npz enrollment templates and auth probes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/prepare_public_dataset.py \\
    --dataset-dir /mnt/c/Users/sekit/Downloads/multi_pie \\
    --output-dir /mnt/c/Users/sekit/Downloads

The dataset directory should have this structure:
  <dataset_name>/
    person1/
      view_001.jpg
      view_002.jpg
      ...
    person2/
      ...
        """,
    )

    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="Path to the dataset root directory containing person subdirectories.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for .npz files. Defaults to same as dataset-dir parent.",
    )
    parser.add_argument(
        "--enrollment-frames", type=int, default=12,
        help="Number of enrollment keyframes to select per person (default: 12).",
    )
    parser.add_argument(
        "--auth-frames", type=int, default=4,
        help="Number of authentication keyframes to select per person (default: 4).",
    )
    parser.add_argument(
        "--diversity-threshold", type=float, default=8.0,
        help="Min angular distance (degrees) between enrollment keyframes (default: 8.0).",
    )
    parser.add_argument(
        "--auth-diversity-threshold", type=float, default=8.0,
        help="Min angular distance (degrees) between auth keyframes (default: 8.0).",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.5,
        help="Min face detection confidence to accept an image (default: 0.5). "
             "Lower for pre-cropped face datasets where landmarks may fall outside image.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true",
        help="Reprocess persons whose .npz files already exist.",
    )
    parser.add_argument(
        "--persons", type=str, nargs="*", default=None,
        help="Process only these person directories (by name). If not set, process all.",
    )

    args = parser.parse_args()

    # Resolve paths
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.is_dir():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    dataset_name = dataset_dir.name

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = dataset_dir.parent

    enrollment_dir = output_dir / "mast3r_outputs"
    auth_dir = output_dir / "auth_probes"
    enrollment_dir.mkdir(parents=True, exist_ok=True)
    auth_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset:    {dataset_dir}")
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Enrollment: {enrollment_dir}")
    logger.info(f"Auth:       {auth_dir}")
    logger.info(f"Enrollment frames: {args.enrollment_frames}")
    logger.info(f"Auth frames:       {args.auth_frames}")
    logger.info(f"Diversity threshold (enrollment): {args.diversity_threshold}°")
    logger.info(f"Diversity threshold (auth):       {args.auth_diversity_threshold}°")
    logger.info(f"Seed: {args.seed}")

    # Set random seed
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # Initialize face detector
    face_config = get_face_detection_config()
    detector = FaceDetector(face_config)

    # Initialize ArcFace embedder (optional, graceful fallback)
    face_embedder = None
    try:
        from core.face_embedder import FaceEmbedder
        face_embedder = FaceEmbedder({"device": "cuda", "backend": "auto"})
        face_embedder.load_model()
        logger.info("ArcFace embedder loaded — .npz files will include face_embedding")
    except Exception as e:
        logger.warning(f"ArcFace not available ({e}), .npz files will NOT include face_embedding")

    # Find person directories
    person_dirs = sorted(
        d for d in dataset_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )

    if args.persons:
        person_filter = set(args.persons)
        person_dirs = [d for d in person_dirs if d.name in person_filter]

    if not person_dirs:
        logger.error("No person directories found!")
        sys.exit(1)

    logger.info(f"Found {len(person_dirs)} person(s) to process")
    logger.info("=" * 60)

    # Process each person
    results: List[PersonResult] = []
    total_start = time.time()

    for i, person_dir in enumerate(person_dirs, 1):
        logger.info(f"\n[{i}/{len(person_dirs)}] Processing: {person_dir.name}")
        logger.info("-" * 40)

        result = process_person(
            person_dir=person_dir,
            detector=detector,
            enrollment_dir=enrollment_dir,
            auth_dir=auth_dir,
            dataset_name=dataset_name,
            enrollment_frames=args.enrollment_frames,
            auth_frames=args.auth_frames,
            diversity_threshold=args.diversity_threshold,
            auth_diversity_threshold=args.auth_diversity_threshold,
            min_confidence=args.min_confidence,
            max_roll=15.0,
            rng=rng,
            skip_existing=not args.no_skip_existing,
            face_embedder=face_embedder,
        )
        results.append(result)

    total_elapsed = time.time() - total_start

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Person':<20} {'Enrollment Pts':>15} {'Auth Pts':>10} {'Time (s)':>10} {'Status':<20}")
    print("-" * 80)

    success_count = 0
    for r in results:
        status = "OK" if r.enrollment_npz_path and r.auth_npz_path else (r.error or "PARTIAL")
        if r.enrollment_npz_path and r.auth_npz_path:
            success_count += 1
        print(f"{r.name:<20} {r.enrollment_points:>15,} {r.auth_points:>10,} {r.elapsed_sec:>10.1f} {status:<20}")

    print("-" * 80)
    print(f"Total: {success_count}/{len(results)} persons processed successfully")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"\nEnrollment .npz files: {enrollment_dir}")
    print(f"Auth probe .npz files: {auth_dir}")


if __name__ == "__main__":
    main()
