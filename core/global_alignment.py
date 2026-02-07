"""
Global Alignment Module

This module wraps MASt3R-SfM's sparse_global_alignment function to unify
multi-view 3D reconstructions into a single coordinate frame.

The key problem this solves:
- Each pairwise MASt3R inference produces 3D points in the reference frame
  of the first image in the pair
- When we have multiple pairs, each result is in a different coordinate system
- Global alignment transforms all results into a unified coordinate frame

This module uses MASt3R-SfM's sparse global alignment which:
1. Computes correspondences between all image pairs
2. Optimizes camera poses to minimize reprojection error
3. Produces globally consistent 3D point clouds

Usage:
    from core.global_alignment import GlobalAligner

    aligner = GlobalAligner(config)
    aligned_points, camera_poses = aligner.align(
        image_paths=["img1.jpg", "img2.jpg", ...],
        pairs=[(0, 1), (1, 2), ...],
        model=mast3r_model,
        cache_path="./cache"
    )

Owner: CS-1
"""

import numpy as np
import torch
import tempfile
import os
import sys
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """
    Result from global alignment.

    Attributes:
        point_cloud: Aligned 3D points in unified coordinate frame.
                     Shape: (N, 3).
        colors: RGB colors for each point.
                Shape: (N, 3), uint8.
        confidence: Per-point confidence values.
                    Shape: (N,).
        camera_poses: List of 4x4 camera-to-world transformation matrices,
                      one per input frame.
        intrinsics: List of 3x3 camera intrinsic matrices.
        depthmaps: List of depth maps per frame (optional).
    """
    point_cloud: np.ndarray
    colors: np.ndarray
    confidence: np.ndarray
    camera_poses: List[np.ndarray]
    intrinsics: List[np.ndarray]
    depthmaps: Optional[List[np.ndarray]] = None


class GlobalAligner:
    """
    Align multiple pairwise 3D reconstructions into unified coordinates.

    Uses MASt3R-SfM's sparse_global_alignment for camera pose optimization
    and point cloud fusion. This resolves scale/rotation ambiguity across
    different pairwise reconstructions.

    The alignment process:
    1. Forward pass: Run MASt3R inference on all image pairs
    2. Extract correspondences: Find matching points between views
    3. Optimize: Jointly optimize camera poses using 3D and 2D losses
    4. Fuse: Combine all views into a unified point cloud
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the global aligner.

        Args:
            config: Configuration dictionary containing:
                - lr1: Learning rate for coarse alignment (default: 0.07)
                - niter1: Iterations for coarse alignment (default: 300)
                - lr2: Learning rate for refinement (default: 0.01)
                - niter2: Iterations for refinement (default: 300)
                - subsample: Subsampling factor for points (default: 8)
                - device: Compute device (default: "cuda")
        """
        self.lr1 = config.get("lr1", 0.07)
        self.niter1 = config.get("niter1", 300)
        self.lr2 = config.get("lr2", 0.01)
        self.niter2 = config.get("niter2", 300)
        self.subsample = config.get("subsample", 8)
        self.device = config.get("device", "cuda")

        # Verify MASt3R paths are set up
        self._setup_paths()

    def _setup_paths(self):
        """Setup Python paths for MASt3R imports."""
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent

        mast3r_path = project_root / "third_party" / "mast3r"
        dust3r_path = mast3r_path / "dust3r"

        for path in [mast3r_path, dust3r_path]:
            path_str = str(path)
            if path.exists() and path_str not in sys.path:
                sys.path.insert(0, path_str)

    def align_from_images(
        self,
        image_paths: List[str],
        pairs: List[Tuple[int, int]],
        model: Any,
        cache_path: Optional[str] = None,
    ) -> AlignmentResult:
        """
        Perform global alignment on a set of images.

        This is the primary method for global alignment. It takes image paths
        and produces a unified 3D reconstruction.

        Args:
            image_paths: List of paths to input images.
            pairs: List of (i, j) index pairs indicating which images to match.
            model: Loaded MASt3R model instance.
            cache_path: Directory for caching intermediate results. If None,
                       uses a temporary directory.

        Returns:
            AlignmentResult containing aligned point cloud and camera poses.

        Raises:
            ImportError: If MASt3R-SfM modules cannot be imported.
            RuntimeError: If alignment fails.
        """
        try:
            from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
            from dust3r.utils.image import load_images
        except ImportError as e:
            raise ImportError(
                "Failed to import MASt3R-SfM modules. "
                "Make sure MASt3R is properly installed."
            ) from e

        # Use temp directory if no cache path provided
        if cache_path is None:
            cache_path = tempfile.mkdtemp(prefix="mast3r_ga_")
        else:
            os.makedirs(cache_path, exist_ok=True)

        logger.info(f"Running global alignment on {len(image_paths)} images, "
                   f"{len(pairs)} pairs")

        # Normalize pairs to ensure consistent ordering (smaller index first)
        # This helps avoid KeyError issues in MASt3R's sparse_scene_optimizer
        normalized_pairs = []
        seen_pairs = set()
        for i, j in pairs:
            # Ensure i < j for consistent ordering
            pair_key = (min(i, j), max(i, j))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                normalized_pairs.append(pair_key)

        logger.info(f"Normalized to {len(normalized_pairs)} unique pairs")

        # Load images for pairs
        pairs_in = []
        for i, j in normalized_pairs:
            images = load_images([image_paths[i], image_paths[j]], size=512)
            pair = [
                {"img": images[0]["img"], "idx": i, "instance": image_paths[i],
                 "true_shape": images[0]["true_shape"]},
                {"img": images[1]["img"], "idx": j, "instance": image_paths[j],
                 "true_shape": images[1]["true_shape"]},
            ]
            pairs_in.append(pair)

        # Run sparse global alignment
        try:
            scene = sparse_global_alignment(
                imgs=image_paths,
                pairs_in=pairs_in,
                cache_path=cache_path,
                model=model,
                subsample=self.subsample,
                device=self.device,
                lr1=self.lr1,
                niter1=self.niter1,
                lr2=self.lr2,
                niter2=self.niter2,
            )
        except Exception as e:
            logger.error(f"Global alignment failed: {e}")
            raise RuntimeError(f"Global alignment failed: {e}") from e

        return self._extract_result(scene, image_paths)

    def align_from_frames(
        self,
        frames: List[np.ndarray],
        pairs: List[Tuple[int, int]],
        model: Any,
        cache_path: Optional[str] = None,
    ) -> AlignmentResult:
        """
        Perform global alignment on numpy array frames.

        This is a convenience method that saves frames to temporary files
        before running alignment. Useful when working with live captures.

        Args:
            frames: List of RGB images as numpy arrays, shape (H, W, 3).
            pairs: List of (i, j) index pairs.
            model: Loaded MASt3R model instance.
            cache_path: Directory for caching intermediate results.

        Returns:
            AlignmentResult containing aligned point cloud and camera poses.
        """
        # Create temporary directory for images
        temp_dir = tempfile.mkdtemp(prefix="mast3r_frames_")
        image_paths = []

        try:
            # Save frames as temporary images
            for i, frame in enumerate(frames):
                path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
                # Convert RGB to BGR for OpenCV if needed, then save
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                Image.fromarray(frame).save(path, quality=95)
                image_paths.append(path)

            # Use default cache path if not provided
            if cache_path is None:
                cache_path = os.path.join(temp_dir, "cache")

            return self.align_from_images(image_paths, pairs, model, cache_path)

        finally:
            # Clean up temp files
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp dir {temp_dir}: {e}")

    def _extract_result(self, scene: Any, image_paths: List[str]) -> AlignmentResult:
        """
        Extract alignment results from SparseGA scene object.

        Uses get_dense_pts3d() to extract dense point clouds for face reconstruction.
        The sparse correspondences (get_sparse_pts3d) are used internally by
        sparse_global_alignment for camera pose optimization.

        Args:
            scene: SparseGA object from sparse_global_alignment.
            image_paths: Original image paths for color extraction.

        Returns:
            AlignmentResult with point cloud, colors, and camera poses.
        """
        from dust3r.utils.device import to_numpy

        # Get camera poses (camera-to-world)
        cam2w = to_numpy(scene.get_im_poses())  # (N, 4, 4)
        # Note: intrinsics is an attribute, not a method in SparseGA
        intrinsics = to_numpy(scene.intrinsics)  # (N, 3, 3)

        # Try to get dense 3D points (preferred for face reconstruction)
        # Fall back to sparse if dense extraction fails
        use_dense = False
        try:
            logger.info("Extracting dense point cloud...")
            dense_pts3d, depthmaps_dense, _ = scene.get_dense_pts3d(
                clean_depth=True,
                subsample=self.subsample
            )
            use_dense = True
            logger.info(f"Dense extraction successful: {len(dense_pts3d)} views")
        except Exception as e:
            logger.warning(f"Dense point extraction failed: {e}")
            logger.warning("Falling back to sparse points")
            dense_pts3d = scene.get_sparse_pts3d()

        # Get images and colors from scene
        imgs = scene.imgs  # List of RGB images in [0, 1] range

        # Concatenate all points from all views
        all_points = []
        all_colors = []
        all_confidence = []

        for i, pts in enumerate(dense_pts3d):
            pts_np = to_numpy(pts)

            # Filter out invalid points (NaN or Inf)
            valid_mask = np.isfinite(pts_np).all(axis=1)
            valid_pts = pts_np[valid_mask]

            if len(valid_pts) == 0:
                continue

            all_points.append(valid_pts)

            # Extract colors by projecting 3D points to image plane
            if i < len(imgs) and i < len(intrinsics):
                img = imgs[i]
                H, W = img.shape[:2]
                K = intrinsics[i]  # 3x3 intrinsic matrix

                # Project valid 3D points to image plane
                # pts3d are in world coordinates, cam2w transforms camera to world
                # So we need world to camera: inv(cam2w)
                cam2w_i = cam2w[i]  # 4x4
                w2cam = np.linalg.inv(cam2w_i)

                # Transform points to camera coordinates
                pts_homo = np.hstack([valid_pts, np.ones((len(valid_pts), 1))])
                pts_cam = (w2cam @ pts_homo.T).T[:, :3]  # (N, 3)

                # Project to image plane
                pts_img = (K @ pts_cam.T).T  # (N, 3)
                pts_img = pts_img[:, :2] / (pts_img[:, 2:3] + 1e-8)  # (N, 2)

                # Get pixel coordinates
                pixel_x = np.clip(pts_img[:, 0].astype(int), 0, W - 1)
                pixel_y = np.clip(pts_img[:, 1].astype(int), 0, H - 1)

                # Sample colors from image
                colors_view = img[pixel_y, pixel_x]

                # Convert from [0, 1] to uint8 if needed
                if colors_view.max() <= 1.0:
                    colors_view = (colors_view * 255).astype(np.uint8)
                all_colors.append(colors_view)
            else:
                # Default gray color
                all_colors.append(np.full((len(valid_pts), 3), 128, dtype=np.uint8))

            # Set confidence to 1.0 for all valid points
            # Note: confs from get_dense_pts3d() is at full resolution, not matching
            # subsampled points, so we use uniform confidence for simplicity
            all_confidence.append(np.ones(len(valid_pts), dtype=np.float32))

        if all_points:
            point_cloud = np.concatenate(all_points, axis=0)
            colors = np.concatenate(all_colors, axis=0)
            confidence = np.concatenate(all_confidence, axis=0)
        else:
            point_cloud = np.zeros((0, 3), dtype=np.float32)
            colors = np.zeros((0, 3), dtype=np.uint8)
            confidence = np.zeros(0, dtype=np.float32)

        # Get depthmaps
        try:
            depthmaps = [to_numpy(d) for d in scene.get_depthmaps()]
        except Exception:
            depthmaps = None

        logger.info(f"Global alignment complete: {len(point_cloud)} points, "
                   f"{len(cam2w)} camera poses (dense={use_dense})")

        return AlignmentResult(
            point_cloud=point_cloud.astype(np.float32),
            colors=colors.astype(np.uint8),
            confidence=confidence.astype(np.float32),
            camera_poses=[cam2w[i] for i in range(len(cam2w))],
            intrinsics=[intrinsics[i] for i in range(len(intrinsics))],
            depthmaps=depthmaps,
        )


def align_multiview(
    frames: List[np.ndarray],
    pairs: List[Tuple[int, int]],
    model: Any,
    config: Optional[Dict[str, Any]] = None,
) -> AlignmentResult:
    """
    Convenience function for multi-view alignment.

    This is a simpler interface that creates a GlobalAligner internally
    and runs alignment on the provided frames.

    Args:
        frames: List of RGB images as numpy arrays.
        pairs: List of (i, j) index pairs.
        model: Loaded MASt3R model instance.
        config: Optional configuration dictionary.

    Returns:
        AlignmentResult with aligned point cloud and camera poses.

    Example:
        from core.global_alignment import align_multiview

        frames = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in keyframes]
        pairs = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3)]
        result = align_multiview(frames, pairs, model)
        print(f"Aligned {len(result.point_cloud)} points")
    """
    if config is None:
        config = {}

    aligner = GlobalAligner(config)
    return aligner.align_from_frames(frames, pairs, model)


if __name__ == "__main__":
    # Quick test of the global alignment module
    print("=" * 60)
    print(" Global Alignment Module Test")
    print("=" * 60)

    # Test data structures
    print("\n1. Testing AlignmentResult dataclass...")
    result = AlignmentResult(
        point_cloud=np.zeros((1000, 3)),
        colors=np.zeros((1000, 3), dtype=np.uint8),
        confidence=np.ones(1000),
        camera_poses=[np.eye(4) for _ in range(5)],
        intrinsics=[np.eye(3) * 500 for _ in range(5)],
    )
    print(f"   Point cloud shape: {result.point_cloud.shape}")
    print(f"   Camera poses: {len(result.camera_poses)}")
    print(f"   AlignmentResult created successfully")

    # Test GlobalAligner initialization
    print("\n2. Testing GlobalAligner initialization...")
    config = {
        "lr1": 0.07,
        "niter1": 300,
        "lr2": 0.01,
        "niter2": 300,
        "subsample": 8,
        "device": "cuda",
    }
    aligner = GlobalAligner(config)
    print(f"   LR1: {aligner.lr1}")
    print(f"   Iterations 1: {aligner.niter1}")
    print(f"   Device: {aligner.device}")
    print(f"   GlobalAligner initialized successfully")

    print("\n" + "=" * 60)
    print(" Basic tests passed!")
    print("=" * 60)
    print("\nNote: Full alignment tests require MASt3R model to be loaded.")
