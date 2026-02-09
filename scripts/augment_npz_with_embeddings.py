"""
Augment existing .npz templates with ArcFace identity embeddings.

Reads pre-computed .npz files (enrollment and probe), locates the original
source images from the dataset directory, extracts ArcFace embeddings,
and saves augmented .npz files with a new `face_embedding` field.

This allows the DS team to evaluate ArcFace-based matching on Colab
without re-running the full MASt3R pipeline.

Usage (WSL or Colab):
  python scripts/augment_npz_with_embeddings.py \
    --npz-dir /path/to/mast3r_outputs \
    --dataset-dir /path/to/dataset \
    --output-dir /path/to/augmented_outputs

  # Or augment in-place (overwrites originals):
  python scripts/augment_npz_with_embeddings.py \
    --npz-dir /path/to/mast3r_outputs \
    --dataset-dir /path/to/dataset \
    --in-place

Author: DS-1 / CS-1
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def find_source_images(
    person_name: str,
    source_filenames: List[str],
    dataset_dir: Path,
) -> List[Path]:
    """
    Locate original source images for a person in the dataset directory.

    Searches in <dataset_dir>/<person_name>/ for the filenames listed
    in the .npz metadata.source_images field.

    Returns list of found image paths (may be shorter than source_filenames
    if some images are missing).
    """
    person_dir = dataset_dir / person_name
    found = []

    if not person_dir.is_dir():
        # Try case-insensitive search
        for d in dataset_dir.iterdir():
            if d.is_dir() and d.name.lower() == person_name.lower():
                person_dir = d
                break

    if not person_dir.is_dir():
        logger.warning(f"Person directory not found: {person_dir}")
        return found

    for filename in source_filenames:
        img_path = person_dir / filename
        if img_path.exists():
            found.append(img_path)
        else:
            # Try without extension matching (different extensions)
            stem = Path(filename).stem
            for ext in IMAGE_EXTENSIONS:
                alt_path = person_dir / (stem + ext)
                if alt_path.exists():
                    found.append(alt_path)
                    break
            else:
                logger.warning(f"  Source image not found: {filename} in {person_dir}")

    return found


def extract_embeddings_from_images(
    image_paths: List[Path],
    face_embedder,
    face_detector=None,
    face_padding: float = 0.3,
) -> Optional[np.ndarray]:
    """
    Extract ArcFace embeddings from source images and aggregate.

    Args:
        image_paths: List of paths to face images.
        face_embedder: FaceEmbedder instance (already loaded).
        face_detector: Optional FaceDetector for cropping. If None,
                       images are passed directly to ArcFace.
        face_padding: Padding ratio for face cropping.

    Returns:
        Aggregated 512-dim embedding, or None if no faces detected.
    """
    face_crops = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"  Could not read image: {img_path}")
            continue

        if face_detector is not None:
            detection = face_detector.detect(img)
            if detection:
                crop = face_detector.crop_face_region(img, detection, padding=face_padding)
                face_crops.append(crop)
            else:
                # No face detected by MediaPipe, pass full image to ArcFace
                # (insightface has its own detector)
                face_crops.append(img)
        else:
            face_crops.append(img)

    if not face_crops:
        return None

    return face_embedder.extract_multi_frame(face_crops)


def augment_single_npz(
    npz_path: Path,
    dataset_dir: Path,
    output_path: Path,
    face_embedder,
    face_detector=None,
    face_padding: float = 0.3,
) -> bool:
    """
    Augment a single .npz file with ArcFace embedding.

    Returns True if successful, False otherwise.
    """
    # Load existing data
    data = np.load(str(npz_path), allow_pickle=True)
    metadata = json.loads(str(data["metadata"]))

    person_name = metadata.get("user_id", metadata.get("user_name", "unknown"))
    source_filenames = metadata.get("source_images", [])

    if not source_filenames:
        logger.warning(f"  No source_images in metadata for {npz_path.name}")
        return False

    # Check if already augmented
    if "face_embedding" in data:
        logger.info(f"  {npz_path.name} already has face_embedding, skipping")
        return True

    # Find source images
    image_paths = find_source_images(person_name, source_filenames, dataset_dir)
    if not image_paths:
        logger.error(f"  No source images found for {person_name}")
        return False

    logger.info(f"  Found {len(image_paths)}/{len(source_filenames)} source images for {person_name}")

    # Extract embedding
    embedding = extract_embeddings_from_images(
        image_paths, face_embedder, face_detector, face_padding
    )

    if embedding is None:
        logger.error(f"  Failed to extract embedding for {person_name}")
        return False

    # Save augmented .npz
    save_kwargs = {}
    for key in data.files:
        save_kwargs[key] = data[key]
    save_kwargs["face_embedding"] = embedding.astype(np.float32)

    # Update metadata with embedding info
    metadata["template_version"] = "2.0"
    metadata["embedding_model"] = "buffalo_l"
    metadata["embedding_dim"] = int(embedding.shape[0])
    save_kwargs["metadata"] = json.dumps(metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **save_kwargs)

    logger.info(
        f"  Saved augmented {output_path.name} "
        f"(embedding: {embedding.shape}, norm={np.linalg.norm(embedding):.4f})"
    )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Augment .npz templates with ArcFace embeddings"
    )
    parser.add_argument(
        "--npz-dir", type=str, required=True,
        help="Directory containing .npz files to augment",
    )
    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="Root directory of the source dataset (contains person subdirectories)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for augmented .npz files (default: <npz-dir>_augmented)",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite original .npz files instead of creating new ones",
    )
    parser.add_argument(
        "--probe-dir", type=str, default=None,
        help="Optional separate directory for probe .npz files",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for ArcFace inference ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--face-padding", type=float, default=0.3,
        help="Face crop padding ratio (default: 0.3)",
    )
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    dataset_dir = Path(args.dataset_dir)

    if args.in_place:
        output_dir = npz_dir
    elif args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = npz_dir.parent / (npz_dir.name + "_augmented")

    # Initialize face embedder
    from core.face_embedder import FaceEmbedder

    embedder_config = {
        "model": "buffalo_l",
        "device": args.device,
        "backend": "auto",
    }
    face_embedder = FaceEmbedder(embedder_config)
    face_embedder.load_model()

    # Optionally initialize face detector for cropping
    face_detector = None
    try:
        from core.face_detector import FaceDetector
        face_detector = FaceDetector()
        logger.info("Using MediaPipe face detector for cropping")
    except Exception as e:
        logger.info(f"MediaPipe not available ({e}), using insightface's built-in detector")

    # Process enrollment .npz files
    npz_files = sorted(npz_dir.glob("*.npz"))
    logger.info(f"Found {len(npz_files)} .npz files in {npz_dir}")

    success_count = 0
    for npz_path in npz_files:
        logger.info(f"Processing {npz_path.name}...")
        out_path = output_dir / npz_path.name
        ok = augment_single_npz(
            npz_path, dataset_dir, out_path,
            face_embedder, face_detector, args.face_padding,
        )
        if ok:
            success_count += 1

    # Process probe .npz files if separate directory provided
    if args.probe_dir:
        probe_dir = Path(args.probe_dir)
        probe_files = sorted(probe_dir.glob("*.npz"))
        logger.info(f"\nFound {len(probe_files)} probe .npz files in {probe_dir}")

        if args.in_place:
            probe_output = probe_dir
        else:
            probe_output = output_dir.parent / (probe_dir.name + "_augmented" if not args.output_dir else probe_dir.name)

        for npz_path in probe_files:
            logger.info(f"Processing {npz_path.name}...")
            out_path = probe_output / npz_path.name
            ok = augment_single_npz(
                npz_path, dataset_dir, out_path,
                face_embedder, face_detector, args.face_padding,
            )
            if ok:
                success_count += 1

    total = len(npz_files) + (len(probe_files) if args.probe_dir else 0)
    logger.info(f"\nDone: {success_count}/{total} files augmented successfully")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
