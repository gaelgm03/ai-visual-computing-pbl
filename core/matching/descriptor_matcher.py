"""
Descriptor Matcher: Compare identity feature vectors via reciprocal NN matching.

DS-1 implementation of the DescriptorMatcher interface defined in interfaces.py.
Uses GPU-accelerated matrix multiply when CUDA is available, with scipy cKDTree fallback.

Reference: CS2DS-share.md §5.2, §10
"""

import logging
import numpy as np
from scipy.spatial import cKDTree

from core.matching.interfaces import DescriptorMatcher, MatchResult

logger = logging.getLogger(__name__)

# Check for GPU availability
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_CUDA = False


class NNDescriptorMatcher(DescriptorMatcher):
    """
    Compare descriptor sets using mutual nearest neighbor matching.

    Pipeline:
      1. L2-normalize descriptors (MASt3R outputs are NOT unit-normalized)
      2. Subsample for performance
      3. Forward + backward nearest-neighbor search (GPU or CPU)
      4. Reciprocal matching (mutual NN check)
      5. Score = weighted combination of match_ratio + avg_cosine_similarity
    """

    def __init__(self, config: dict):
        self.match_ratio_weight = config.get("match_ratio_weight", 0.4)
        self.avg_similarity_weight = config.get("avg_similarity_weight", 0.6)
        self.subsample_limit = config.get("descriptor_subsample", 15000)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        probe_desc: np.ndarray,
        template_desc: np.ndarray,
        probe_cloud: np.ndarray,
        template_cloud: np.ndarray,
    ) -> MatchResult:
        """
        Compare descriptor sets from two face captures.

        Args:
            probe_desc:     (N, D) — descriptors from authentication capture
            template_desc:  (M, D) — descriptors from enrolled template
            probe_cloud:    (N, 3) — 3D positions (for spatial context)
            template_cloud: (M, 3) — 3D positions

        Returns:
            MatchResult with score in [0, 1] (higher = more similar).
        """
        try:
            return self._compare_impl(
                probe_desc, template_desc, probe_cloud, template_cloud
            )
        except Exception as e:
            logger.warning(f"Descriptor matching failed: {e}")
            return MatchResult(
                score=0.0,
                details={"method": "nn_reciprocal", "error": str(e)},
                is_match=False,
            )

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _compare_impl(
        self,
        probe_desc: np.ndarray,
        template_desc: np.ndarray,
        probe_cloud: np.ndarray,
        template_cloud: np.ndarray,
    ) -> MatchResult:
        # 1. Input validation
        if len(probe_desc) < 5 or len(template_desc) < 5:
            return MatchResult(
                score=0.0,
                details={
                    "method": "nn_reciprocal",
                    "error": "too few descriptors",
                    "probe_descriptors": len(probe_desc),
                    "template_descriptors": len(template_desc),
                },
                is_match=False,
            )

        if probe_desc.shape[1] != template_desc.shape[1]:
            return MatchResult(
                score=0.0,
                details={
                    "method": "nn_reciprocal",
                    "error": f"dimension mismatch: {probe_desc.shape[1]} vs {template_desc.shape[1]}",
                },
                is_match=False,
            )

        # 2. Filter NaN/Inf descriptors (MASt3R can produce invalid values
        #    from FP16 inference, image edges, or padding regions)
        probe_desc, probe_cloud = self._filter_invalid(probe_desc, probe_cloud)
        template_desc, template_cloud = self._filter_invalid(
            template_desc, template_cloud
        )

        if len(probe_desc) < 5 or len(template_desc) < 5:
            return MatchResult(
                score=0.0,
                details={
                    "method": "nn_reciprocal",
                    "error": "too few valid descriptors after NaN filtering",
                },
                is_match=False,
            )

        # 3. L2-normalize (critical: MASt3R descriptors are NOT unit-normalized)
        probe_norm = self._normalize(probe_desc)
        template_norm = self._normalize(template_desc)

        # 4. Subsample
        probe_norm, _ = self._subsample(probe_norm, probe_cloud)
        template_norm, _ = self._subsample(template_norm, template_cloud)

        # 5. Nearest-neighbor search (GPU or CPU)
        # Primary: GPU matmul on Colab T4 / compatible CUDA GPUs
        # Fallback: CPU cKDTree (for envs where CUDA arch is incompatible)
        use_gpu = _HAS_CUDA and len(probe_norm) * len(template_norm) < 500_000_000
        backend = "cpu_kdtree"
        if use_gpu:
            try:
                idx_p2t, sims_p2t, idx_t2p, sims_t2p = self._gpu_nn(
                    probe_norm, template_norm
                )
                backend = "gpu_matmul"
            except RuntimeError:
                idx_p2t, sims_p2t, idx_t2p, sims_t2p = self._cpu_nn(
                    probe_norm, template_norm
                )
                backend = "cpu_kdtree (gpu_fallback)"
        else:
            idx_p2t, sims_p2t, idx_t2p, sims_t2p = self._cpu_nn(
                probe_norm, template_norm
            )

        # 6. Reciprocal matching (with bounds check as safety net against
        #    any remaining invalid indices from edge cases)
        n_template = len(template_norm)
        n_probe = len(probe_norm)
        reciprocal_sims = []
        for i in range(n_probe):
            j = idx_p2t[i]
            if 0 <= j < n_template and idx_t2p[j] == i:
                reciprocal_sims.append(sims_p2t[i])

        n_reciprocal = len(reciprocal_sims)
        min_size = min(len(probe_norm), len(template_norm))
        match_ratio = n_reciprocal / max(min_size, 1)

        if n_reciprocal > 0:
            avg_similarity = float(np.mean(reciprocal_sims))
        else:
            avg_similarity = 0.0

        # 7. Combined score
        score = float(
            self.match_ratio_weight * match_ratio
            + self.avg_similarity_weight * avg_similarity
        )
        score = float(np.clip(score, 0.0, 1.0))

        return MatchResult(
            score=score,
            details={
                "method": "nn_reciprocal",
                "n_reciprocal_matches": n_reciprocal,
                "match_ratio": float(match_ratio),
                "avg_cosine_similarity": float(avg_similarity),
                "probe_descriptors": len(probe_desc),
                "template_descriptors": len(template_desc),
                "descriptor_dim": int(probe_desc.shape[1]),
                "backend": backend,
            },
            is_match=True,  # Fusion layer handles thresholding
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_invalid(
        desc: np.ndarray, cloud: np.ndarray
    ) -> tuple:
        """
        Remove rows containing NaN/Inf or zero-norm descriptors.

        scipy cKDTree.query returns undefined indices for NaN inputs,
        which causes IndexError in reciprocal matching. This filter
        ensures all descriptor rows are finite and non-degenerate.
        """
        # Finite check (no NaN or Inf)
        finite_mask = np.isfinite(desc).all(axis=1)
        # Zero-norm check (degenerate descriptors)
        norms = np.linalg.norm(desc, axis=1)
        nonzero_mask = norms > 1e-8
        valid = finite_mask & nonzero_mask

        n_removed = len(desc) - valid.sum()
        if n_removed > 0:
            logger.info(
                f"Filtered {n_removed} invalid descriptors "
                f"({len(desc)} -> {valid.sum()})"
            )

        # Truncate cloud to desc length if they differ (architectural mismatch)
        min_len = min(len(desc), len(cloud))
        desc = desc[:min_len]
        cloud = cloud[:min_len]
        valid = valid[:min_len]

        return desc[valid], cloud[valid]

    @staticmethod
    def _normalize(desc: np.ndarray) -> np.ndarray:
        """L2-normalize each descriptor row."""
        norms = np.linalg.norm(desc, axis=1, keepdims=True)
        return desc / (norms + 1e-8)

    def _subsample(
        self, desc: np.ndarray, cloud: np.ndarray
    ) -> tuple:
        """Subsample descriptors and corresponding cloud together."""
        if len(desc) > self.subsample_limit:
            idx = np.random.choice(len(desc), self.subsample_limit, replace=False)
            return desc[idx], cloud[idx]
        return desc, cloud

    # ------------------------------------------------------------------
    # NN search backends
    # ------------------------------------------------------------------

    @staticmethod
    def _gpu_nn(
        probe: np.ndarray, template: np.ndarray
    ) -> tuple:
        """
        GPU-accelerated nearest-neighbor via cosine similarity matrix.
        For unit-normalized vectors: similarity = probe @ template.T
        """
        device = torch.device("cuda")
        p = torch.from_numpy(probe.astype(np.float32)).to(device)
        t = torch.from_numpy(template.astype(np.float32)).to(device)

        # (N, D) @ (D, M) → (N, M) cosine similarity matrix
        sim_matrix = p @ t.T  # all-pairs cosine similarity

        # Forward: for each probe, find most similar template
        sims_p2t, idx_p2t = sim_matrix.max(dim=1)

        # Backward: for each template, find most similar probe
        sims_t2p, idx_t2p = sim_matrix.max(dim=0)

        return (
            idx_p2t.cpu().numpy(),
            sims_p2t.cpu().numpy(),
            idx_t2p.cpu().numpy(),
            sims_t2p.cpu().numpy(),
        )

    @staticmethod
    def _cpu_nn(
        probe: np.ndarray, template: np.ndarray
    ) -> tuple:
        """CPU nearest-neighbor using scipy cKDTree."""
        tree_t = cKDTree(template)
        tree_p = cKDTree(probe)

        dist_p2t, idx_p2t = tree_t.query(probe)
        dist_t2p, idx_t2p = tree_p.query(template)

        # Convert Euclidean distance to cosine similarity for unit vectors:
        # ||a - b||^2 = 2 - 2*cos(a,b)  →  cos(a,b) = 1 - ||a-b||^2 / 2
        sims_p2t = 1.0 - (dist_p2t ** 2) / 2.0
        sims_t2p = 1.0 - (dist_t2p ** 2) / 2.0

        return idx_p2t, sims_p2t, idx_t2p, sims_t2p
