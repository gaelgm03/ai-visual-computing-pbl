"""
Authentication API Routes

This module provides the POST /authenticate endpoint for face authentication.
It processes captured frames, runs anti-spoofing checks, and compares against
enrolled templates using the DS-1 matchers.

Author: CS-1
"""

import base64
import time
import logging
import numpy as np
import cv2
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from api.schemas import (
    AuthRequest,
    AuthResponse,
    AntiSpoofResult as AntiSpoofResultSchema,
    VisualizationData,
)
from core.face_detector import FaceDetector
from core.mast3r_engine import get_engine
from core.template_manager import get_template_manager, FaceTemplate
from core.anti_spoof import get_anti_spoof
from core.config import get_face_detection_config, get_config

# Import DS-1 matchers (with fallback to stubs)
from core.matching import (
    ICPGeometricMatcher,
    NNDescriptorMatcher,
    ArcFaceEmbeddingMatcher,
    WeightedFusion,
    MultiModalFusion,
    StubGeometricMatcher,
    StubDescriptorMatcher,
    StubScoreFusion,
)

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["authentication"])


def decode_frame(frame_b64: str) -> Optional[np.ndarray]:
    """
    Decode a base64-encoded JPEG image to numpy array.

    Args:
        frame_b64: Base64-encoded JPEG string.

    Returns:
        BGR numpy array or None if decoding fails.
    """
    try:
        img_bytes = base64.b64decode(frame_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.warning(f"Failed to decode frame: {e}")
        return None


def get_matchers():
    """
    Get matcher instances, using DS-1 implementations or fallback to stubs.

    Returns:
        Tuple of (geo_matcher, desc_matcher, emb_matcher, fusion, legacy_fusion).
        emb_matcher is None if ArcFaceEmbeddingMatcher is unavailable.
    """
    config = get_config()
    matching_config = config.get("matching", {})

    # Use DS-1 implementations if available
    if ICPGeometricMatcher is not None:
        geo_matcher = ICPGeometricMatcher(matching_config)
    else:
        logger.warning("ICPGeometricMatcher not available, using stub")
        geo_matcher = StubGeometricMatcher()

    if NNDescriptorMatcher is not None:
        desc_matcher = NNDescriptorMatcher(matching_config)
    else:
        logger.warning("NNDescriptorMatcher not available, using stub")
        desc_matcher = StubDescriptorMatcher()

    # ArcFace embedding matcher (optional — graceful degradation)
    emb_matcher = None
    if ArcFaceEmbeddingMatcher is not None:
        emb_matcher = ArcFaceEmbeddingMatcher(matching_config)

    # Multi-modal fusion (3-way with ArcFace), with legacy 2-way fallback
    if MultiModalFusion is not None:
        fusion = MultiModalFusion(matching_config)
    elif WeightedFusion is not None:
        fusion = WeightedFusion(matching_config)
    else:
        logger.warning("No fusion implementation available, using stub")
        fusion = StubScoreFusion()

    # Legacy 2-way fusion for templates without embeddings
    if WeightedFusion is not None:
        legacy_fusion = WeightedFusion(matching_config)
    else:
        legacy_fusion = StubScoreFusion()

    return geo_matcher, desc_matcher, emb_matcher, fusion, legacy_fusion


@router.post("/authenticate", response_model=AuthResponse)
async def authenticate(request: AuthRequest):
    """
    Authenticate a user by comparing captured frames against enrolled templates.

    This endpoint:
    1. Decodes base64 JPEG frames
    2. Detects and crops faces
    3. Runs MASt3R 3D reconstruction
    4. Performs anti-spoofing check
    5. Compares against enrolled template(s) using DS-1 matchers
    6. Returns match decision with scores

    Args:
        request: AuthRequest with frames and optional user_id.

    Returns:
        AuthResponse with match result and scores.
    """
    start_time = time.time()

    # 1. Decode frames
    frames = []
    for i, frame_b64 in enumerate(request.frames):
        frame = decode_frame(frame_b64)
        if frame is None:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode frame {i}"
            )
        frames.append(frame)

    logger.info(f"Authentication request: {len(frames)} frames, user_id={request.user_id}")

    # 2. Detect and crop faces
    face_config = get_face_detection_config()
    face_detector = FaceDetector(face_config)

    cropped_faces = []
    cropped_faces_bgr = []  # Keep BGR copies for ArcFace (insightface expects BGR)
    for i, frame in enumerate(frames):
        detection = face_detector.detect(frame)
        if detection is None:
            logger.warning(f"No face detected in frame {i}")
            continue
        cropped = face_detector.crop_face_region(frame, detection)
        cropped_faces_bgr.append(cropped)
        # Convert BGR to RGB for MASt3R
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped_faces.append(cropped_rgb)

    face_detector.close()

    if len(cropped_faces) < 2:
        return AuthResponse(
            is_match=False,
            matched_user_id=None,
            matched_user_name=None,
            final_score=0.0,
            geometric_score=0.0,
            descriptor_score=0.0,
            embedding_score=0.0,
            anti_spoof=AntiSpoofResultSchema(
                passed=False,
                depth_variance=0.0,
                planarity_ratio=0.0,
            ),
            processing_time_sec=time.time() - start_time,
            visualization_data=None,
        )

    # 3. Build probe via MASt3R reconstruction
    engine = get_engine()
    if not engine._model_loaded:
        logger.info("Loading MASt3R model...")
        engine.load_model()

    try:
        probe_result = engine.reconstruct_multiview(cropped_faces)
        probe_cloud = probe_result.point_cloud
        probe_descriptors = probe_result.descriptors
        probe_confidence = probe_result.confidence
    except Exception as e:
        logger.error(f"MASt3R reconstruction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"3D reconstruction failed: {str(e)}"
        )

    logger.info(f"Probe reconstruction: {len(probe_cloud)} points")

    # 3b. Extract probe ArcFace embedding
    probe_embedding = None
    try:
        from core.face_embedder import FaceEmbedder

        embedding_config = get_config().get("face_embedding", {})
        embedder = FaceEmbedder(embedding_config)
        embedder.load_model()
        probe_embedding = embedder.extract_multi_frame(cropped_faces_bgr)
        if probe_embedding is not None:
            logger.info(f"Probe ArcFace embedding: norm={np.linalg.norm(probe_embedding):.4f}")
        else:
            logger.warning("Probe ArcFace embedding extraction returned None")
    except ImportError:
        logger.info("insightface not installed — skipping ArcFace embedding")
    except Exception as e:
        logger.warning(f"Probe ArcFace embedding extraction failed: {e}")

    # 4. Run anti-spoofing check
    anti_spoof = get_anti_spoof()
    spoof_result = anti_spoof.check(probe_cloud, probe_confidence)

    if not spoof_result.passed:
        logger.warning(f"Anti-spoofing check failed: {spoof_result.details}")
        return AuthResponse(
            is_match=False,
            matched_user_id=None,
            matched_user_name=None,
            final_score=0.0,
            geometric_score=0.0,
            descriptor_score=0.0,
            embedding_score=0.0,
            anti_spoof=AntiSpoofResultSchema(
                passed=False,
                depth_variance=spoof_result.depth_variance,
                planarity_ratio=spoof_result.eigenvalue_ratio,
            ),
            processing_time_sec=time.time() - start_time,
            visualization_data=None,
        )

    # 5. Load templates
    template_manager = get_template_manager()

    if request.user_id:
        # 1:1 verification - load specific template
        template = template_manager.load_template(request.user_id)
        if template is None:
            raise HTTPException(
                status_code=404,
                detail=f"User {request.user_id} not found"
            )
        templates = [template]
    else:
        # 1:N identification - load all templates
        templates = template_manager.load_all_templates()
        if not templates:
            return AuthResponse(
                is_match=False,
                matched_user_id=None,
                matched_user_name=None,
                final_score=0.0,
                geometric_score=0.0,
                descriptor_score=0.0,
                embedding_score=0.0,
                anti_spoof=AntiSpoofResultSchema(
                    passed=True,
                    depth_variance=spoof_result.depth_variance,
                    planarity_ratio=spoof_result.eigenvalue_ratio,
                ),
                processing_time_sec=time.time() - start_time,
                visualization_data=None,
            )

    # 6. Run matching against each template
    geo_matcher, desc_matcher, emb_matcher, fusion, legacy_fusion = get_matchers()

    best_match = None
    best_score = -1.0
    best_geo_score = 0.0
    best_desc_score = 0.0
    best_emb_score = 0.0

    for template in templates:
        try:
            # Geometric matching
            geo_result = geo_matcher.compare(probe_cloud, template.point_cloud)

            # Descriptor matching
            desc_result = desc_matcher.compare(
                probe_descriptors,
                template.descriptors,
                probe_cloud,
                template.point_cloud,
            )

            # ArcFace embedding matching + fusion
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
                logger.debug(
                    f"Match vs {template.user_name}: "
                    f"emb={emb_result.score:.3f}, geo={geo_result.score:.3f}, "
                    f"desc={desc_result.score:.3f}, fused={fused.score:.3f}"
                )
            else:
                fused = legacy_fusion.fuse(geo_result, desc_result)
                emb_score = 0.0
                logger.debug(
                    f"Match vs {template.user_name} (no embedding): "
                    f"geo={geo_result.score:.3f}, desc={desc_result.score:.3f}, "
                    f"fused={fused.score:.3f}"
                )

            if fused.score > best_score:
                best_score = fused.score
                best_geo_score = geo_result.score
                best_desc_score = desc_result.score
                best_emb_score = emb_score
                best_match = template if fused.is_match else None

        except Exception as e:
            logger.warning(f"Matching failed for {template.user_id}: {e}")
            continue

    # 7. Build response
    processing_time = time.time() - start_time

    return AuthResponse(
        is_match=best_match is not None,
        matched_user_id=best_match.user_id if best_match else None,
        matched_user_name=best_match.user_name if best_match else None,
        final_score=max(0.0, best_score),
        geometric_score=best_geo_score,
        descriptor_score=best_desc_score,
        embedding_score=best_emb_score,
        anti_spoof=AntiSpoofResultSchema(
            passed=True,
            depth_variance=spoof_result.depth_variance,
            planarity_ratio=spoof_result.eigenvalue_ratio,
        ),
        processing_time_sec=processing_time,
        visualization_data=None,  # TODO: Add visualization data if needed
    )
