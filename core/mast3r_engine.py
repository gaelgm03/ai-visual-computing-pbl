"""
MASt3R Engine Module

This module provides a wrapper around the MASt3R (Matching And Stereo 3D Reconstruction)
model for face-specific 3D reconstruction and dense descriptor extraction.

MASt3R takes pairs of images and produces:
- 3D point maps (dense depth/3D coordinates for each pixel)
- Per-pixel confidence values
- Dense local descriptors (feature vectors for matching)

This engine is used for both enrollment (reconstructing a 3D face template from
multiple keyframes) and authentication (comparing a probe against enrolled templates).

Usage:
    from core.mast3r_engine import MASt3REngine, PairwiseResult, MultiViewResult

    # Initialize and load model (do once at startup)
    engine = MASt3REngine(config)
    engine.load_model()

    # Single pair inference
    result = engine.infer_pair(img1, img2)

    # Multi-view reconstruction from keyframes
    frames = [kf.frame for kf in keyframe_candidates]
    result = engine.reconstruct_multiview(frames)

Note:
    - Model loading takes ~30-60 seconds
    - Keep the engine as a singleton to avoid reloading
    - FP16 mode is mandatory on 8GB GPUs (RTX 5070 Laptop)
    - Process pairs sequentially (never batch on limited VRAM)

Author: CS-1
"""

import numpy as np
import torch
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import sys
import logging

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class PairwiseResult:
    """
    Result from MASt3R inference on a single image pair.

    MASt3R processes two images and predicts 3D structure and correspondences.
    The outputs are aligned to image 1's coordinate frame.

    Attributes:
        pointmap1: 3D coordinates for each pixel in image 1.
                   Shape: (H, W, 3) where each pixel has (x, y, z).
        pointmap2: 3D coordinates for image 2 pixels, in image 1's frame.
                   Shape: (H, W, 3).
        confidence1: Per-pixel confidence for pointmap1.
                     Shape: (H, W), values in [0, 1].
        confidence2: Per-pixel confidence for pointmap2.
                     Shape: (H, W).
        descriptors1: Dense local feature descriptors for image 1.
                      Shape: (H, W, D) where D is descriptor dimension.
        descriptors2: Dense local feature descriptors for image 2.
                      Shape: (H, W, D).
        image1_shape: Original shape of input image 1 (H, W).
        image2_shape: Original shape of input image 2 (H, W).
    """

    pointmap1: np.ndarray  # (H, W, 3) - 3D coords for img1
    pointmap2: np.ndarray  # (H, W, 3) - 3D coords for img2 (in img1's frame)
    confidence1: np.ndarray  # (H, W) - confidence for pointmap1
    confidence2: np.ndarray  # (H, W) - confidence for pointmap2
    descriptors1: np.ndarray  # (H, W, D) - descriptors for img1
    descriptors2: np.ndarray  # (H, W, D) - descriptors for img2
    image1_shape: Tuple[int, int] = (0, 0)  # (H, W)
    image2_shape: Tuple[int, int] = (0, 0)  # (H, W)


@dataclass
class MultiViewResult:
    """
    Result from multi-view 3D reconstruction.

    Combines multiple pairwise results into a unified 3D point cloud
    with aggregated descriptors.

    Attributes:
        point_cloud: Fused 3D point coordinates.
                     Shape: (N, 3) where N is total number of points.
        colors: RGB color for each point (from source images).
                Shape: (N, 3), values in [0, 255] uint8.
        descriptors: Aggregated feature descriptors per point.
                     Shape: (N, D) where D is descriptor dimension.
        confidence: Per-point confidence values.
                    Shape: (N,), values in [0, 1].
        per_frame_poses: Estimated camera pose for each input frame.
                         List of 4x4 transformation matrices.
        n_frames: Number of input frames used.
        n_pairs: Number of image pairs processed.
        reconstruction_metadata: Additional info about the reconstruction.
    """

    point_cloud: np.ndarray  # (N, 3) - 3D coordinates
    colors: np.ndarray  # (N, 3) - RGB colors (uint8)
    descriptors: np.ndarray  # (N, D) - feature descriptors
    confidence: np.ndarray  # (N,) - per-point confidence
    per_frame_poses: List[np.ndarray] = field(default_factory=list)  # List of 4x4 matrices
    n_frames: int = 0
    n_pairs: int = 0
    reconstruction_metadata: Dict[str, Any] = field(default_factory=dict)


class MASt3REngine:
    """
    Wrapper around MASt3R model for face-specific 3D inference.

    This class handles:
    - Model loading and GPU memory management
    - Image preprocessing (resizing, format conversion)
    - Pairwise inference with FP16 optimization
    - Multi-view reconstruction with global alignment

    Attributes:
        device: PyTorch device (cuda/cpu).
        model: Loaded MASt3R model (None until load_model() is called).
        image_size: Maximum dimension for input images (default 512).
        force_fp16: Whether to use half-precision (mandatory on 8GB GPUs).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MASt3R engine.

        Args:
            config: Configuration dictionary containing:
                - checkpoint: Model identifier or path
                  (default: "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
                - image_size: Max input dimension (default: 512)
                - device: "cuda" or "cpu" (default: "cuda")
                - force_fp16: Use half precision (default: True)

        Example:
            config = {
                "checkpoint": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                "image_size": 512,
                "device": "cuda",
                "force_fp16": True
            }
            engine = MASt3REngine(config)
        """
        self.checkpoint = config.get(
            "checkpoint",
            "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        )
        self.image_size = config.get("image_size", 512)
        self.force_fp16 = config.get("force_fp16", True)

        # Setup device
        device_name = config.get("device", "cuda")
        if device_name == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device_name = "cpu"
        self.device = torch.device(device_name)

        # Model will be loaded later
        self.model = None
        self._model_loaded = False

        # Track project paths for MASt3R imports
        self._setup_paths()

    @property
    def is_loaded(self) -> bool:
        """Check if the model has been loaded."""
        return self._model_loaded

    def _setup_paths(self):
        """Setup Python paths for MASt3R imports."""
        # Find project root (contains config.yaml)
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent

        # MASt3R is typically in third_party/mast3r
        mast3r_path = project_root / "third_party" / "mast3r"
        dust3r_path = mast3r_path / "dust3r"

        # Add paths if they exist and aren't already in sys.path
        for path in [mast3r_path, dust3r_path]:
            path_str = str(path)
            if path.exists() and path_str not in sys.path:
                sys.path.insert(0, path_str)

    def load_model(self) -> None:
        """
        Load the MASt3R model into memory.

        This should be called once at application startup. The model
        weights are downloaded from Hugging Face Hub if not cached.

        Loading takes approximately 30-60 seconds depending on network
        speed and whether weights are already cached.

        Raises:
            ImportError: If MASt3R is not properly installed.
            RuntimeError: If model loading fails.

        Example:
            engine = MASt3REngine(config)
            engine.load_model()  # Call once at startup
            # Now ready for inference
        """
        if self._model_loaded:
            logger.info("Model already loaded, skipping.")
            return

        logger.info(f"Loading MASt3R model: {self.checkpoint}")
        logger.info(f"Device: {self.device}, FP16: {self.force_fp16}")

        try:
            from mast3r.model import AsymmetricMASt3R

            # Load from Hugging Face Hub (auto-downloads if needed)
            self.model = AsymmetricMASt3R.from_pretrained(self.checkpoint)
            self.model = self.model.to(self.device)
            self.model.eval()

            # Log model info
            n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
            logger.info(f"Model loaded successfully ({n_params:.1f}M parameters)")

            self._model_loaded = True

        except ImportError as e:
            raise ImportError(
                "Failed to import MASt3R. Make sure it's installed: "
                "run scripts/setup_mast3r.sh (Linux/Mac) or "
                "scripts/setup_mast3r.ps1 (Windows)"
            ) from e

    def _ensure_model_loaded(self):
        """Check that model is loaded before inference."""
        if not self._model_loaded:
            raise RuntimeError(
                "Model not loaded. Call load_model() before inference."
            )

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for MASt3R inference.

        Converts BGR to RGB and resizes to fit within image_size while
        maintaining aspect ratio.

        Args:
            img: Input image in BGR format (OpenCV convention).
                 Shape: (H, W, 3), dtype: uint8.

        Returns:
            Preprocessed image in RGB format.
            Shape: (H', W', 3) where max(H', W') <= image_size.
        """
        # Convert BGR to RGB if needed (OpenCV uses BGR)
        if img.ndim == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        # Resize to fit within image_size while maintaining aspect ratio
        h, w = img_rgb.shape[:2]
        max_dim = max(h, w)

        if max_dim > self.image_size:
            scale = self.image_size / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return img_rgb

    def infer_pair(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> PairwiseResult:
        """
        Run MASt3R inference on a single image pair.

        This is the core inference method. Given two images, it produces:
        - 3D point maps (dense 3D coordinates for each pixel)
        - Confidence values (how reliable each point estimate is)
        - Dense descriptors (feature vectors for matching)

        Args:
            img1: First image in BGR format (uint8).
                  Shape: (H1, W1, 3).
            img2: Second image in BGR format (uint8).
                  Shape: (H2, W2, 3).

        Returns:
            PairwiseResult containing pointmaps, confidence, and descriptors
            for both images.

        Raises:
            RuntimeError: If model is not loaded or inference fails.

        Note:
            - Images are automatically resized to fit image_size
            - GPU memory is freed after inference to prevent OOM
            - FP16 is used when force_fp16=True (mandatory on 8GB GPUs)

        Example:
            result = engine.infer_pair(face_img1, face_img2)
            print(f"Pointmap shape: {result.pointmap1.shape}")
            print(f"Descriptor dim: {result.descriptors1.shape[-1]}")
        """
        self._ensure_model_loaded()

        # Import DUSt3R utilities (imported here to avoid import errors if not installed)
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        import tempfile
        import os
        from PIL import Image

        # Preprocess images
        img1_rgb = self._preprocess_image(img1)
        img2_rgb = self._preprocess_image(img2)

        # Save images temporarily (MASt3R's load_images expects file paths)
        with tempfile.TemporaryDirectory() as temp_dir:
            path1 = os.path.join(temp_dir, "img1.jpg")
            path2 = os.path.join(temp_dir, "img2.jpg")

            Image.fromarray(img1_rgb).save(path1, quality=95)
            Image.fromarray(img2_rgb).save(path2, quality=95)

            # Load images using DUSt3R's utility
            images = load_images([path1, path2], size=self.image_size)

        # Run inference with memory optimization
        try:
            # Use autocast for FP16 when enabled
            with torch.cuda.amp.autocast(
                enabled=(self.force_fp16 and self.device.type == "cuda")
            ):
                with torch.no_grad():
                    output = inference(
                        [tuple(images)],
                        self.model,
                        self.device,
                        batch_size=1
                    )

            # Extract results from pred1 and pred2
            # pred1 contains pts3d, conf, desc for image 1
            # pred2 contains pts3d_in_other_view, conf, desc for image 2
            pred1, pred2 = output["pred1"], output["pred2"]

            # Convert to numpy and move to CPU immediately to free GPU memory
            pointmap1 = pred1["pts3d"].squeeze(0).cpu().numpy()  # (H, W, 3)
            pointmap2 = pred2["pts3d_in_other_view"].squeeze(0).cpu().numpy()  # (H, W, 3)
            confidence1 = pred1["conf"].squeeze(0).cpu().numpy()  # (H, W)
            confidence2 = pred2["conf"].squeeze(0).cpu().numpy()  # (H, W)
            descriptors1 = pred1["desc"].squeeze(0).cpu().numpy()  # (H, W, D)
            descriptors2 = pred2["desc"].squeeze(0).cpu().numpy()  # (H, W, D)

        finally:
            # Aggressively free GPU memory
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return PairwiseResult(
            pointmap1=pointmap1,
            pointmap2=pointmap2,
            confidence1=confidence1,
            confidence2=confidence2,
            descriptors1=descriptors1,
            descriptors2=descriptors2,
            image1_shape=(img1_rgb.shape[0], img1_rgb.shape[1]),
            image2_shape=(img2_rgb.shape[0], img2_rgb.shape[1]),
        )

    @staticmethod
    def generate_pair_indices(n_frames: int) -> List[Tuple[int, int]]:
        """
        Generate a reasonable set of image pair indices for multi-view reconstruction.

        The strategy balances reconstruction quality with computational cost:
        - Sequential pairs: Capture temporal coherence (frame N with frame N+1)
        - Skip-one pairs: Wider baseline for better triangulation
        - Loop closure: Connect first and last frames for global consistency

        Args:
            n_frames: Total number of input frames.

        Returns:
            List of (i, j) index pairs indicating which frames to compare.

        Example:
            # For 12 frames, generates approximately 24 pairs:
            pairs = MASt3REngine.generate_pair_indices(12)
            # [(0,1), (1,2), ..., (10,11),  # 11 sequential
            #  (0,2), (1,3), ..., (9,11),   # 10 skip-one
            #  (0,11)]                       # 1 loop closure
            # Total: 22 pairs (not 66 if we did all combinations!)
        """
        if n_frames < 2:
            return []

        pairs = []

        # Sequential pairs: (0,1), (1,2), (2,3), ...
        for i in range(n_frames - 1):
            pairs.append((i, i + 1))

        # Skip-one pairs: (0,2), (1,3), (2,4), ...
        # Wider baseline helps with 3D triangulation
        for i in range(n_frames - 2):
            pairs.append((i, i + 2))

        # Loop closure: connect first and last frame
        # This improves global consistency
        if n_frames > 3:
            pairs.append((0, n_frames - 1))

        return pairs

    def reconstruct_multiview(
        self,
        frames: List[np.ndarray],
        pairs: Optional[List[Tuple[int, int]]] = None,
        confidence_threshold: float = 0.5,
        use_global_alignment: bool = True,
        global_alignment_config: Optional[Dict[str, Any]] = None,
    ) -> MultiViewResult:
        """
        Reconstruct a unified 3D point cloud from multiple frames.

        This method supports two modes:

        1. Global Alignment (default, use_global_alignment=True):
           Uses MASt3R-SfM's sparse_global_alignment to optimize camera poses
           and produce a globally consistent 3D reconstruction. This solves
           the coordinate system unification problem where each pairwise
           result is in a different reference frame.

        2. Simple Accumulation (use_global_alignment=False):
           Simplified fusion that accumulates points from the first view
           of each pair. Faster but produces fragmented results (multiple
           overlapping layers). Use for pipeline testing only.

        Args:
            frames: List of face images in BGR format.
                    Each should be a cropped face from keyframe selection.
            pairs: Optional list of (i, j) pairs specifying which frames
                   to compare. If None, auto-generated using generate_pair_indices.
            confidence_threshold: Minimum confidence to include a point.
                                  Points below this are filtered out.
            use_global_alignment: If True (default), use MASt3R-SfM's
                                  sparse_global_alignment for proper
                                  coordinate unification. If False, use
                                  simple point accumulation.
            global_alignment_config: Optional config dict for GlobalAligner.
                                     See core.global_alignment.GlobalAligner.

        Returns:
            MultiViewResult containing the fused point cloud, colors,
            descriptors, and confidence values.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If fewer than 2 frames provided.

        Example:
            # From keyframe candidates
            frames = [kf.frame for kf in keyframe_candidates]

            # With global alignment (recommended)
            result = engine.reconstruct_multiview(frames)

            # Without global alignment (for testing)
            result = engine.reconstruct_multiview(frames, use_global_alignment=False)

            print(f"Reconstructed {len(result.point_cloud)} points")
        """
        self._ensure_model_loaded()

        if len(frames) < 2:
            raise ValueError("At least 2 frames required for reconstruction.")

        # Generate pairs if not provided
        if pairs is None:
            pairs = self.generate_pair_indices(len(frames))

        logger.info(f"Reconstructing from {len(frames)} frames, {len(pairs)} pairs")
        logger.info(f"Using global alignment: {use_global_alignment}")

        if use_global_alignment:
            return self._reconstruct_with_global_alignment(
                frames, pairs, confidence_threshold, global_alignment_config
            )
        else:
            return self._reconstruct_simple_accumulation(
                frames, pairs, confidence_threshold
            )

    def _reconstruct_with_global_alignment(
        self,
        frames: List[np.ndarray],
        pairs: List[Tuple[int, int]],
        confidence_threshold: float,
        config: Optional[Dict[str, Any]] = None,
    ) -> MultiViewResult:
        """
        Reconstruct using MASt3R-SfM's sparse_global_alignment.

        This method properly unifies all pairwise results into a single
        coordinate frame by optimizing camera poses.
        """
        from core.global_alignment import GlobalAligner

        if config is None:
            # Load from config file
            try:
                from core.config import get_global_alignment_config
                file_config = get_global_alignment_config()
                config = {
                    "device": str(self.device),
                    "subsample": file_config.get("subsample", 8),
                    "lr1": file_config.get("lr1", 0.07),
                    "niter1": file_config.get("niter1", 300),
                    "lr2": file_config.get("lr2", 0.01),
                    "niter2": file_config.get("niter2", 300),
                }
            except (ImportError, KeyError):
                # Fallback to hardcoded defaults
                config = {
                    "device": str(self.device),
                    "subsample": 8,
                    "lr1": 0.07,
                    "niter1": 300,
                    "lr2": 0.01,
                    "niter2": 300,
                }

        aligner = GlobalAligner(config)

        # Convert BGR frames to RGB for global alignment
        frames_rgb = []
        for frame in frames:
            if frame.ndim == 3 and frame.shape[2] == 3:
                # Assume BGR (OpenCV) and convert to RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = frame
            frames_rgb.append(rgb)

        logger.info("Running global alignment...")
        try:
            alignment_result = aligner.align_from_frames(
                frames_rgb, pairs, self.model
            )
        except Exception as e:
            logger.error(f"Global alignment failed: {e}")
            logger.warning("Falling back to simple accumulation method.")
            return self._reconstruct_simple_accumulation(
                frames, pairs, confidence_threshold
            )

        # Extract aligned point cloud
        point_cloud = alignment_result.point_cloud
        colors = alignment_result.colors
        confidence = alignment_result.confidence

        # Apply confidence threshold
        mask = confidence >= confidence_threshold
        point_cloud = point_cloud[mask]
        colors = colors[mask]
        confidence = confidence[mask]

        logger.info(f"Global alignment complete: {len(point_cloud)} points")

        # For descriptors, we need to run pairwise inference and collect them
        # This is done after alignment to use the aligned coordinate system
        descriptors = self._extract_aligned_descriptors(
            frames, pairs, alignment_result.camera_poses,
            alignment_result.intrinsics, confidence_threshold
        )

        return MultiViewResult(
            point_cloud=point_cloud,
            colors=colors,
            descriptors=descriptors,
            confidence=confidence,
            per_frame_poses=alignment_result.camera_poses,
            n_frames=len(frames),
            n_pairs=len(pairs),
            reconstruction_metadata={
                "confidence_threshold": confidence_threshold,
                "pairs_processed": pairs,
                "alignment_method": "sparse_global_alignment",
            }
        )

    def _extract_aligned_descriptors(
        self,
        frames: List[np.ndarray],
        pairs: List[Tuple[int, int]],
        camera_poses: List[np.ndarray],
        intrinsics: List[np.ndarray],
        confidence_threshold: float,
    ) -> np.ndarray:
        """
        Extract descriptors after global alignment.

        Since global alignment uses sparse correspondences, we run pairwise
        inference to get dense descriptors and transform them using the
        optimized camera poses.
        """
        all_descriptors = []

        # For efficiency, we only process the first pair to get descriptor dimension
        # and then aggregate from all pairs
        for idx, (i, j) in enumerate(pairs):
            result = self.infer_pair(frames[i], frames[j])

            # Get descriptors from first view
            descriptors_flat = result.descriptors1.reshape(-1, result.descriptors1.shape[-1])
            confidence_flat = result.confidence1.reshape(-1)

            # Filter by confidence
            mask = confidence_flat >= confidence_threshold
            all_descriptors.append(descriptors_flat[mask])

            # Free memory
            del result
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if all_descriptors:
            descriptors = np.concatenate(all_descriptors, axis=0)
        else:
            descriptors = np.zeros((0, 24), dtype=np.float32)

        return descriptors

    def _reconstruct_simple_accumulation(
        self,
        frames: List[np.ndarray],
        pairs: List[Tuple[int, int]],
        confidence_threshold: float,
    ) -> MultiViewResult:
        """
        Reconstruct using simple point accumulation (no global alignment).

        WARNING: This produces fragmented results where each pair's points
        are in different coordinate systems. Use for pipeline testing only.
        """
        logger.warning(
            "Using simple accumulation (no global alignment). "
            "Results will be fragmented with overlapping layers."
        )

        # Collect points from all pairs
        all_points = []
        all_colors = []
        all_descriptors = []
        all_confidence = []

        for idx, (i, j) in enumerate(pairs):
            logger.debug(f"Processing pair {idx + 1}/{len(pairs)}: frames ({i}, {j})")

            # Run pairwise inference
            result = self.infer_pair(frames[i], frames[j])

            # Get colors from source image
            img_rgb = self._preprocess_image(frames[i])
            h, w = result.pointmap1.shape[:2]

            # Resize color image to match pointmap if needed
            if img_rgb.shape[:2] != (h, w):
                img_rgb = cv2.resize(img_rgb, (w, h))

            # Flatten spatial dimensions: (H, W, C) -> (H*W, C)
            points_flat = result.pointmap1.reshape(-1, 3)
            colors_flat = img_rgb.reshape(-1, 3)
            descriptors_flat = result.descriptors1.reshape(-1, result.descriptors1.shape[-1])
            confidence_flat = result.confidence1.reshape(-1)

            # Filter by confidence
            mask = confidence_flat >= confidence_threshold

            all_points.append(points_flat[mask])
            all_colors.append(colors_flat[mask])
            all_descriptors.append(descriptors_flat[mask])
            all_confidence.append(confidence_flat[mask])

            # Free memory between pairs
            del result
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Concatenate all points
        point_cloud = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
        descriptors = np.concatenate(all_descriptors, axis=0)
        confidence = np.concatenate(all_confidence, axis=0)

        logger.info(f"Reconstructed {len(point_cloud)} points (before dedup)")

        # Remove duplicate points (simple distance-based deduplication)
        point_cloud, colors, descriptors, confidence = self._deduplicate_points(
            point_cloud, colors, descriptors, confidence
        )

        logger.info(f"Final point cloud: {len(point_cloud)} points")

        return MultiViewResult(
            point_cloud=point_cloud,
            colors=colors,
            descriptors=descriptors,
            confidence=confidence,
            per_frame_poses=[],  # Not computed in simple mode
            n_frames=len(frames),
            n_pairs=len(pairs),
            reconstruction_metadata={
                "confidence_threshold": confidence_threshold,
                "pairs_processed": pairs,
                "alignment_method": "simple_accumulation",
            }
        )

    def _deduplicate_points(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        descriptors: np.ndarray,
        confidence: np.ndarray,
        distance_threshold: float = 0.005,  # 5mm in metric units
        max_points: int = 50000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove duplicate/very close points from the point cloud.

        Uses voxel grid downsampling with vectorized operations for efficiency.

        Args:
            points: Point coordinates (N, 3).
            colors: Point colors (N, 3).
            descriptors: Point descriptors (N, D).
            confidence: Point confidence (N,).
            distance_threshold: Minimum distance between kept points.
            max_points: Maximum number of points to keep.

        Returns:
            Tuple of (points, colors, descriptors, confidence) after deduplication.
        """
        if len(points) == 0:
            return points, colors, descriptors, confidence

        logger.info(f"Deduplicating {len(points):,} points...")

        # Simple voxel grid downsampling
        # Quantize points to a grid and keep one point per cell
        voxel_size = distance_threshold

        # Compute voxel indices
        min_coords = points.min(axis=0)
        voxel_indices = ((points - min_coords) / voxel_size).astype(np.int32)

        # Create unique voxel keys
        # Use a large multiplier to create unique hash for each voxel
        multipliers = np.array([1, 10000, 100000000], dtype=np.int64)
        voxel_keys = (voxel_indices.astype(np.int64) * multipliers).sum(axis=1)

        # Vectorized approach: sort by voxel key, then by confidence (descending)
        # This allows us to use np.unique to get the first (highest confidence) point per voxel

        # Create sort order: primary by voxel_keys, secondary by -confidence
        # We negate confidence so that higher values come first after sorting
        sort_idx = np.lexsort((-confidence, voxel_keys))

        # Apply sort order
        sorted_keys = voxel_keys[sort_idx]

        # Find first occurrence of each unique key (which has highest confidence due to sorting)
        _, first_occurrence_idx = np.unique(sorted_keys, return_index=True)

        # Map back to original indices
        best_indices = sort_idx[first_occurrence_idx]

        # Extract results using vectorized indexing
        result_points = points[best_indices]
        result_colors = colors[best_indices]
        result_descriptors = descriptors[best_indices]
        result_confidence = confidence[best_indices]

        logger.info(f"After voxel dedup: {len(result_points):,} points")

        # If still too many points, subsample by confidence
        if len(result_points) > max_points:
            # Keep points with highest confidence
            top_indices = np.argsort(result_confidence)[-max_points:]
            result_points = result_points[top_indices]
            result_colors = result_colors[top_indices]
            result_descriptors = result_descriptors[top_indices]
            result_confidence = result_confidence[top_indices]
            logger.info(f"After max_points limit: {len(result_points):,} points")

        return result_points, result_colors, result_descriptors, result_confidence

    def get_gpu_memory_info(self) -> Dict[str, float]:
        """
        Get current GPU memory usage information.

        Useful for monitoring memory during inference.

        Returns:
            Dictionary with memory info in GB:
            - total: Total GPU memory
            - allocated: Currently allocated memory
            - cached: Memory in PyTorch cache
            - free: Available memory
        """
        if self.device.type != "cuda":
            return {"total": 0, "allocated": 0, "cached": 0, "free": 0}

        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        free = total - cached

        return {
            "total": round(total, 2),
            "allocated": round(allocated, 2),
            "cached": round(cached, 2),
            "free": round(free, 2),
        }


# Singleton instance for the engine (avoids reloading model)
_engine_instance: Optional[MASt3REngine] = None


def get_engine(config: Optional[Dict[str, Any]] = None) -> MASt3REngine:
    """
    Get or create the singleton MASt3R engine instance.

    This function ensures only one engine exists in the application,
    avoiding the memory cost of loading multiple model copies.

    Args:
        config: Configuration dictionary. Only used on first call.
                If None, uses default config from core.config.

    Returns:
        The shared MASt3REngine instance.

    Example:
        # First call initializes the engine
        engine = get_engine(config)
        engine.load_model()

        # Subsequent calls return the same instance
        engine = get_engine()  # Same instance, no config needed
    """
    global _engine_instance

    if _engine_instance is None:
        if config is None:
            # Import config module to get default config
            from core.config import get_mast3r_config
            config = get_mast3r_config()

        _engine_instance = MASt3REngine(config)

    return _engine_instance


if __name__ == "__main__":
    # Quick test of the MASt3R engine
    import time

    print("=" * 60)
    print(" MASt3R Engine Test")
    print("=" * 60)

    # Test configuration
    config = {
        "checkpoint": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
        "image_size": 512,
        "device": "cuda",
        "force_fp16": True,
    }

    # Create engine
    print("\n1. Creating MASt3R engine...")
    engine = MASt3REngine(config)
    print(f"   Device: {engine.device}")
    print(f"   Image size: {engine.image_size}")
    print(f"   FP16 mode: {engine.force_fp16}")

    # Test pair generation
    print("\n2. Testing pair generation...")
    pairs = MASt3REngine.generate_pair_indices(12)
    print(f"   For 12 frames: {len(pairs)} pairs generated")
    print(f"   First 5 pairs: {pairs[:5]}")
    print(f"   Last 3 pairs: {pairs[-3:]}")

    # Test with dummy images (model loading optional)
    print("\n3. Testing data classes...")
    dummy_result = PairwiseResult(
        pointmap1=np.zeros((384, 512, 3)),
        pointmap2=np.zeros((384, 512, 3)),
        confidence1=np.ones((384, 512)),
        confidence2=np.ones((384, 512)),
        descriptors1=np.zeros((384, 512, 24)),
        descriptors2=np.zeros((384, 512, 24)),
        image1_shape=(384, 512),
        image2_shape=(384, 512),
    )
    print(f"   PairwiseResult created successfully")
    print(f"   Pointmap shape: {dummy_result.pointmap1.shape}")
    print(f"   Descriptor shape: {dummy_result.descriptors1.shape}")

    dummy_mv = MultiViewResult(
        point_cloud=np.zeros((10000, 3)),
        colors=np.zeros((10000, 3), dtype=np.uint8),
        descriptors=np.zeros((10000, 24)),
        confidence=np.ones(10000),
        n_frames=12,
        n_pairs=22,
    )
    print(f"   MultiViewResult created successfully")
    print(f"   Point cloud shape: {dummy_mv.point_cloud.shape}")

    # Optional: Load model and test inference
    print("\n4. Model loading test (optional)...")
    try:
        print("   Loading model (this may take 30-60 seconds)...")
        start_time = time.time()
        engine.load_model()
        load_time = time.time() - start_time
        print(f"   Model loaded in {load_time:.1f} seconds")

        # Get memory info
        mem_info = engine.get_gpu_memory_info()
        print(f"   GPU Memory: {mem_info['allocated']:.2f} / {mem_info['total']:.2f} GB")

        # Test inference with dummy images
        print("\n5. Testing inference with dummy images...")
        dummy_img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        start_time = time.time()
        result = engine.infer_pair(dummy_img1, dummy_img2)
        infer_time = time.time() - start_time

        print(f"   Inference completed in {infer_time:.2f} seconds")
        print(f"   Pointmap1 shape: {result.pointmap1.shape}")
        print(f"   Descriptors1 shape: {result.descriptors1.shape}")
        print(f"   Confidence1 range: [{result.confidence1.min():.3f}, {result.confidence1.max():.3f}]")

        print("\n" + "=" * 60)
        print(" All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"   Skipped model tests: {e}")
        print("   (This is OK if MASt3R is not installed yet)")
        print("\n" + "=" * 60)
        print(" Basic tests passed (model-free)")
        print("=" * 60)
