"""
Geometric Matcher: Compare 3D face shapes using ICP alignment + Chamfer distance.

DS-1 implementation of the GeometricMatcher interface defined in interfaces.py.
Uses Open3D for ICP when available, with a scipy SVD-based fallback.

Reference: CS2DS-share.md §5.1, §10
"""

import logging
import numpy as np
from scipy.spatial import cKDTree

from core.matching.interfaces import GeometricMatcher, MatchResult

logger = logging.getLogger(__name__)

# Try importing open3d; fall back to scipy-based ICP if unavailable
try:
    import open3d as o3d
    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False
    logger.info("open3d not available; using scipy SVD-based ICP fallback")


class ICPGeometricMatcher(GeometricMatcher):
    """
    Compare two face point clouds by aligning them with ICP and
    measuring remaining Chamfer distance.

    Pipeline:
      1. Subsample for performance
      2. Center both clouds at origin
      3. PCA pre-alignment of principal axes
      4. ICP refinement
      5. Chamfer distance on aligned clouds
      6. Exponential decay: distance → similarity score in [0, 1]
    """

    def __init__(self, config: dict):
        icp_config = config.get("icp", config)
        self.max_iterations = icp_config.get("max_iterations", 50)
        self.max_correspondence_distance = icp_config.get(
            "max_correspondence_distance", 0.05
        )
        self.chamfer_alpha = config.get("chamfer_alpha", 30.0)
        self.subsample_limit = config.get("geometric_subsample", 10000)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self, probe_cloud: np.ndarray, template_cloud: np.ndarray
    ) -> MatchResult:
        """
        Compare two 3D point clouds geometrically.

        Args:
            probe_cloud:    (N, 3) — the authentication capture
            template_cloud: (M, 3) — the enrolled template

        Returns:
            MatchResult with score in [0, 1] (higher = more similar).
        """
        try:
            return self._compare_impl(probe_cloud, template_cloud)
        except Exception as e:
            logger.warning(f"Geometric matching failed: {e}")
            return MatchResult(
                score=0.0,
                details={"method": "icp_chamfer", "error": str(e)},
                is_match=False,
            )

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _compare_impl(
        self, probe_cloud: np.ndarray, template_cloud: np.ndarray
    ) -> MatchResult:
        # 1. Input validation
        if len(probe_cloud) < 10 or len(template_cloud) < 10:
            return MatchResult(
                score=0.0,
                details={
                    "method": "icp_chamfer",
                    "error": "too few points",
                    "probe_points": len(probe_cloud),
                    "template_points": len(template_cloud),
                },
                is_match=False,
            )

        probe = probe_cloud.astype(np.float64)
        template = template_cloud.astype(np.float64)

        # 2. Subsample for CPU performance
        probe = self._subsample(probe)
        template = self._subsample(template)

        # 3. Center both clouds
        probe_centroid = probe.mean(axis=0)
        template_centroid = template.mean(axis=0)
        probe_c = probe - probe_centroid
        template_c = template - template_centroid

        # 4. PCA pre-alignment
        probe_aligned, template_aligned = self._pca_prealign(probe_c, template_c)

        # 5. ICP refinement
        if _HAS_OPEN3D:
            aligned, fitness, rmse = self._open3d_icp(
                probe_aligned, template_aligned
            )
        else:
            aligned, fitness, rmse = self._scipy_icp(
                probe_aligned, template_aligned
            )

        # 6. Chamfer distance
        chamfer = self._chamfer_distance(aligned, template_aligned)

        # 7. Distance → score
        score = float(np.exp(-self.chamfer_alpha * chamfer))
        score = float(np.clip(score, 0.0, 1.0))

        return MatchResult(
            score=score,
            details={
                "method": "icp_chamfer",
                "chamfer_distance": float(chamfer),
                "icp_fitness": float(fitness),
                "icp_inlier_rmse": float(rmse),
                "probe_points": len(probe_cloud),
                "template_points": len(template_cloud),
                "subsample_used": len(probe),
                "used_open3d": _HAS_OPEN3D,
            },
            is_match=True,  # Fusion layer handles thresholding
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _subsample(self, cloud: np.ndarray) -> np.ndarray:
        if len(cloud) > self.subsample_limit:
            idx = np.random.choice(len(cloud), self.subsample_limit, replace=False)
            return cloud[idx]
        return cloud

    @staticmethod
    def _pca_axes(cloud: np.ndarray) -> np.ndarray:
        """Compute PCA axes (3x3 matrix, rows = principal components)."""
        _, _, Vt = np.linalg.svd(cloud, full_matrices=False)
        # Ensure right-handed coordinate system
        if np.linalg.det(Vt) < 0:
            Vt[-1] *= -1
        return Vt

    def _pca_prealign(
        self, probe: np.ndarray, template: np.ndarray
    ) -> tuple:
        """
        Align probe's principal axes to template's.
        Try both sign orientations of the first axis and pick the better one.
        """
        Vt_p = self._pca_axes(probe)
        Vt_t = self._pca_axes(template)

        # Rotation to align probe PCA to template PCA
        R = Vt_t.T @ Vt_p
        probe_rot = probe @ R.T

        # Try flipping the first principal axis (sign ambiguity)
        Vt_p_flip = Vt_p.copy()
        Vt_p_flip[0] *= -1
        if np.linalg.det(Vt_p_flip) < 0:
            Vt_p_flip[-1] *= -1
        R_flip = Vt_t.T @ Vt_p_flip
        probe_rot_flip = probe @ R_flip.T

        # Pick the orientation that gives smaller initial mean distance
        d1 = cKDTree(template).query(probe_rot)[0].mean()
        d2 = cKDTree(template).query(probe_rot_flip)[0].mean()

        if d2 < d1:
            return probe_rot_flip, template
        return probe_rot, template

    # ------------------------------------------------------------------
    # ICP implementations
    # ------------------------------------------------------------------

    def _open3d_icp(
        self, source: np.ndarray, target: np.ndarray
    ) -> tuple:
        """ICP using Open3D (primary path)."""
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(source)
        tgt = o3d.geometry.PointCloud()
        tgt.points = o3d.utility.Vector3dVector(target)

        result = o3d.pipelines.registration.registration_icp(
            src,
            tgt,
            max_correspondence_distance=self.max_correspondence_distance,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.max_iterations,
            ),
        )

        aligned = np.asarray(
            src.transform(result.transformation).points
        )
        return aligned, result.fitness, result.inlier_rmse

    def _scipy_icp(
        self, source: np.ndarray, target: np.ndarray
    ) -> tuple:
        """SVD-based ICP fallback when Open3D is not available."""
        current = source.copy()
        tree = cKDTree(target)
        threshold = self.max_correspondence_distance
        prev_error = float("inf")

        for _ in range(self.max_iterations):
            dists, indices = tree.query(current)
            mask = dists < threshold
            if mask.sum() < 3:
                break

            src_matched = current[mask]
            tgt_matched = target[indices[mask]]

            # Optimal rotation + translation via SVD
            src_centroid = src_matched.mean(axis=0)
            tgt_centroid = tgt_matched.mean(axis=0)
            H = (src_matched - src_centroid).T @ (tgt_matched - tgt_centroid)
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = Vt.T @ U.T
            t = tgt_centroid - R @ src_centroid

            current = (R @ current.T).T + t

            # Convergence check
            mean_error = dists[mask].mean()
            if abs(prev_error - mean_error) < 1e-8:
                break
            prev_error = mean_error

        # Final statistics
        final_dists, _ = tree.query(current)
        inlier_mask = final_dists < threshold
        fitness = float(inlier_mask.mean())
        if inlier_mask.any():
            rmse = float(np.sqrt((final_dists[inlier_mask] ** 2).mean()))
        else:
            rmse = float("inf")

        return current, fitness, rmse

    @staticmethod
    def _chamfer_distance(cloud_a: np.ndarray, cloud_b: np.ndarray) -> float:
        """Bidirectional Chamfer distance (CS2DS-share.md §10)."""
        d_a2b, _ = cKDTree(cloud_b).query(cloud_a)
        d_b2a, _ = cKDTree(cloud_a).query(cloud_b)
        return float((d_a2b.mean() + d_b2a.mean()) / 2)
