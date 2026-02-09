"""
Embedding Matcher: Compare ArcFace identity embeddings via cosine similarity.

DS-1 implementation of the EmbeddingMatcher interface defined in interfaces.py.

This matcher compares pre-extracted 512-dim ArcFace embeddings using cosine
similarity (dot product of L2-normalized vectors). It is the primary identity
signal, replacing MASt3R descriptors which lack identity discrimination.

Reference: CS2DS-share.md §5.2 (extended)
"""

import logging

import numpy as np

from core.matching.interfaces import EmbeddingMatcher, MatchResult

logger = logging.getLogger(__name__)


class ArcFaceEmbeddingMatcher(EmbeddingMatcher):
    """
    Compare ArcFace face identity embeddings via cosine similarity.

    ArcFace embeddings are L2-normalized 512-dim vectors trained with
    Additive Angular Margin Loss to maximize inter-class separation
    and minimize intra-class variation in face identity space.

    For L2-normalized vectors: cosine_similarity = dot product.
    Raw cosine similarity ranges [-1, 1]; remapped to [0, 1] for scoring.

    Args:
        config: Dictionary with optional keys:
            - embedding_dim: Expected dimension (default 512, for validation)
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        self.embedding_dim = config.get("embedding_dim", 512)

    def compare(
        self,
        probe_embedding: np.ndarray,
        template_embedding: np.ndarray,
    ) -> MatchResult:
        """
        Compare two face identity embeddings.

        Args:
            probe_embedding: (D,) float32, L2-normalized ArcFace embedding.
            template_embedding: (D,) float32, L2-normalized ArcFace embedding.

        Returns:
            MatchResult with cosine similarity score mapped to [0, 1].
        """
        # --- Input validation ---
        if probe_embedding is None or template_embedding is None:
            logger.warning("Received None embedding")
            return MatchResult(
                score=0.0,
                details={"method": "arcface_cosine", "error": "null_embedding"},
                is_match=False,
            )

        probe_embedding = np.asarray(probe_embedding, dtype=np.float32).ravel()
        template_embedding = np.asarray(template_embedding, dtype=np.float32).ravel()

        if probe_embedding.shape[0] != template_embedding.shape[0]:
            logger.error(
                f"Embedding dimension mismatch: probe={probe_embedding.shape[0]}, "
                f"template={template_embedding.shape[0]}"
            )
            return MatchResult(
                score=0.0,
                details={"method": "arcface_cosine", "error": "dim_mismatch"},
                is_match=False,
            )

        # --- Ensure L2-normalization ---
        probe_norm = np.linalg.norm(probe_embedding)
        template_norm = np.linalg.norm(template_embedding)

        if probe_norm < 1e-8 or template_norm < 1e-8:
            logger.warning("Zero-norm embedding detected")
            return MatchResult(
                score=0.0,
                details={"method": "arcface_cosine", "error": "zero_norm"},
                is_match=False,
            )

        probe_normalized = probe_embedding / probe_norm
        template_normalized = template_embedding / template_norm

        # --- Cosine similarity (dot product of unit vectors) ---
        raw_cosine = float(np.dot(probe_normalized, template_normalized))

        # Clamp to [-1, 1] for numerical stability
        raw_cosine = max(-1.0, min(1.0, raw_cosine))

        # Map [-1, 1] → [0, 1]
        score = (raw_cosine + 1.0) / 2.0

        return MatchResult(
            score=score,
            details={
                "method": "arcface_cosine",
                "raw_cosine_similarity": raw_cosine,
                "probe_norm": float(probe_norm),
                "template_norm": float(template_norm),
                "embedding_dim": probe_embedding.shape[0],
            },
            is_match=True,  # Fusion layer decides final match
        )
