"""
Enrollment API Routes

This module provides:
- WebSocket endpoint for real-time enrollment with frame-by-frame feedback
- REST endpoint for batch enrollment (upload all frames at once)

The WebSocket endpoint is the primary way to enroll users, as it provides
real-time feedback on head pose and coverage during the enrollment process.

Author: CS-1
"""

import base64
import time
import logging
import json
import numpy as np
import cv2
from typing import List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import ValidationError

from api.schemas import (
    FrameMessage,
    FrameStatusResponse,
    EnrollmentCompleteResponse,
    EnrollmentErrorResponse,
    HeadPose,
    Coverage,
)
from core.face_detector import FaceDetector, FaceDetection
from core.keyframe_selector import KeyframeSelector, KeyframeCandidate
from core.mast3r_engine import MASt3REngine, get_engine
from core.template_manager import (
    TemplateManager,
    FaceTemplate,
    get_template_manager,
    generate_user_id,
)
from core.config import (
    get_face_detection_config,
    get_keyframe_config,
)

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/ws", tags=["enrollment"])


class EnrollmentSession:
    """
    Manages state for a single enrollment session.

    This class tracks:
    - Captured keyframes and their metadata
    - Face detector and keyframe selector instances
    - Coverage status

    Attributes:
        user_name: Name of the user being enrolled.
        face_detector: FaceDetector instance for this session.
        keyframe_selector: KeyframeSelector instance.
        keyframe_candidates: List of captured keyframes.
        start_time: Session start timestamp.
    """

    def __init__(self, user_name: str):
        """
        Initialize an enrollment session.

        Args:
            user_name: Display name for the user being enrolled.
        """
        self.user_name = user_name
        self.start_time = time.time()

        # Initialize face detector
        face_config = get_face_detection_config()
        self.face_detector = FaceDetector(face_config)

        # Initialize keyframe selector
        keyframe_config = get_keyframe_config()
        self.keyframe_selector = KeyframeSelector(keyframe_config)

        # Track captured keyframes
        self.keyframe_candidates: List[KeyframeCandidate] = []

        logger.info(f"Enrollment session started for user: {user_name}")

    def process_frame(self, frame_data: bytes) -> FrameStatusResponse:
        """
        Process a single frame from the client.

        Args:
            frame_data: Raw JPEG bytes from the client.

        Returns:
            FrameStatusResponse with detection and coverage status.
        """
        # Decode JPEG to numpy array
        np_arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return FrameStatusResponse(
                type="frame_status",
                face_detected=False,
                captured=False,
                total_captured=len(self.keyframe_candidates),
                target_count=self.keyframe_selector.target_count,
            )

        # Detect face
        detection = self.face_detector.detect(frame)

        if detection is None:
            return FrameStatusResponse(
                type="frame_status",
                face_detected=False,
                captured=False,
                total_captured=len(self.keyframe_candidates),
                target_count=self.keyframe_selector.target_count,
            )

        # Check if we should capture this frame
        captured = self.keyframe_selector.should_capture(
            detection, self.keyframe_candidates, frame
        )

        if captured:
            # Crop face region
            cropped = self.face_detector.crop_face_region(frame, detection)

            # Compute quality score
            quality = self.keyframe_selector.compute_quality_score(frame, detection)

            # Create keyframe candidate
            candidate = KeyframeCandidate(
                frame=cropped,
                head_pose=detection.head_pose,
                timestamp=time.time(),
                quality_score=quality,
            )
            self.keyframe_candidates.append(candidate)

            logger.debug(
                f"Captured keyframe {len(self.keyframe_candidates)}: "
                f"pose=({detection.head_pose[0]:.1f}, {detection.head_pose[1]:.1f}), "
                f"quality={quality:.2f}"
            )

        # Get coverage status
        coverage_status = self.keyframe_selector.get_coverage_status(
            self.keyframe_candidates
        )

        # Build response
        return FrameStatusResponse(
            type="frame_status",
            face_detected=True,
            head_pose=HeadPose(
                yaw=detection.head_pose[0],
                pitch=detection.head_pose[1],
                roll=detection.head_pose[2],
            ),
            captured=captured,
            total_captured=len(self.keyframe_candidates),
            target_count=self.keyframe_selector.target_count,
            coverage=Coverage(
                yaw_range=list(coverage_status.yaw_range),
                pitch_range=list(coverage_status.pitch_range),
                is_sufficient=coverage_status.is_sufficient,
                missing_directions=coverage_status.missing_directions,
            ),
        )

    def is_ready_for_enrollment(self) -> bool:
        """Check if we have enough keyframes for enrollment."""
        status = self.keyframe_selector.get_coverage_status(self.keyframe_candidates)
        return status.is_sufficient

    def get_frames(self) -> List[np.ndarray]:
        """Get all captured frames for MASt3R processing."""
        return [kf.frame for kf in self.keyframe_candidates]

    def get_coverage_metadata(self) -> dict:
        """Get coverage metadata for template storage."""
        status = self.keyframe_selector.get_coverage_status(self.keyframe_candidates)
        return {
            "n_frames": len(self.keyframe_candidates),
            "yaw_range": list(status.yaw_range),
            "pitch_range": list(status.pitch_range),
        }

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'face_detector'):
            self.face_detector.close()
        logger.info(f"Enrollment session ended for user: {self.user_name}")


async def run_enrollment(
    session: EnrollmentSession,
    engine: MASt3REngine,
    template_manager: TemplateManager,
) -> EnrollmentCompleteResponse:
    """
    Run the MASt3R reconstruction and save the template.

    This is called when enough keyframes have been captured.

    Args:
        session: The enrollment session with captured keyframes.
        engine: MASt3R engine for 3D reconstruction.
        template_manager: Manager for saving templates.

    Returns:
        EnrollmentCompleteResponse with the enrollment result.
    """
    start_time = time.time()

    # Get frames from session
    frames = session.get_frames()
    logger.info(f"Starting 3D reconstruction with {len(frames)} frames")

    # Run MASt3R multiview reconstruction
    result = engine.reconstruct_multiview(frames)

    reconstruction_time = time.time() - start_time
    logger.info(
        f"Reconstruction complete: {len(result.point_cloud)} points "
        f"in {reconstruction_time:.1f}s"
    )

    # Generate user ID
    user_id = generate_user_id()

    # Create template
    template = FaceTemplate(
        user_id=user_id,
        user_name=session.user_name,
        point_cloud=result.point_cloud,
        descriptors=result.descriptors,
        confidence=result.confidence,
        colors=result.colors,
        enrollment_metadata={
            **session.get_coverage_metadata(),
            "reconstruction_time_sec": reconstruction_time,
            "mast3r_version": "ViTLarge_metric",
        },
    )

    # Save template
    template_manager.save_template(template)
    logger.info(f"Template saved for user {session.user_name} (id={user_id})")

    # Generate point cloud preview for frontend visualization
    # Subsample to max 5000 points for reasonable transfer size
    max_preview_points = 5000
    n_total = len(result.point_cloud)
    if n_total > max_preview_points:
        indices = np.random.choice(n_total, max_preview_points, replace=False)
        preview_points = result.point_cloud[indices].tolist()
        preview_colors = result.colors[indices].tolist()
    else:
        preview_points = result.point_cloud.tolist()
        preview_colors = result.colors.tolist()

    preview_data = {
        "points": preview_points,
        "colors": preview_colors,
    }
    point_cloud_preview = base64.b64encode(
        json.dumps(preview_data).encode()
    ).decode()

    return EnrollmentCompleteResponse(
        type="enrollment_complete",
        user_id=user_id,
        user_name=session.user_name,
        point_cloud_preview=point_cloud_preview,
        n_points=template.n_points,
        n_frames_used=len(frames),
        reconstruction_time_sec=reconstruction_time,
    )


@router.websocket("/enroll/{user_name}")
async def websocket_enroll(websocket: WebSocket, user_name: str):
    """
    WebSocket endpoint for real-time face enrollment.

    The client sends frames as base64-encoded JPEG images. For each frame,
    the server responds with:
    - Whether a face was detected
    - Current head pose
    - Whether the frame was captured as a keyframe
    - Current coverage status (which directions still need to be covered)

    When enrollment is complete (enough keyframes with sufficient coverage),
    the server runs MASt3R reconstruction, saves the template, and sends
    the final result.

    Protocol:
        Client -> Server (per frame):
        {
            "type": "frame",
            "data": "<base64-encoded JPEG>"
        }

        Server -> Client (per frame):
        {
            "type": "frame_status",
            "face_detected": true/false,
            "head_pose": {"yaw": float, "pitch": float, "roll": float},
            "captured": true/false,
            "total_captured": int,
            "target_count": int,
            "coverage": {...}
        }

        Server -> Client (on completion):
        {
            "type": "enrollment_complete",
            "user_id": "usr_xxx",
            "user_name": "...",
            "n_points": int,
            "n_frames_used": int,
            "reconstruction_time_sec": float
        }

    Args:
        websocket: The WebSocket connection.
        user_name: Display name for the user being enrolled.
    """
    await websocket.accept()

    session: Optional[EnrollmentSession] = None

    try:
        # Check if user name already exists
        template_manager = get_template_manager()
        if template_manager.user_exists_by_name(user_name):
            error_response = EnrollmentErrorResponse(
                type="error",
                error=f"User '{user_name}' already exists. Please use a different name.",
                code="USER_EXISTS",
            )
            await websocket.send_json(error_response.model_dump())
            await websocket.close()
            return

        # Create enrollment session
        session = EnrollmentSession(user_name)

        # NOTE: MASt3R model loading is deferred until keyframes are captured
        # This allows face detection/keyframe capture to work without MASt3R installed

        # Process frames until enrollment is complete
        while True:
            # Receive message from client
            try:
                message = await websocket.receive_json()
            except Exception as e:
                logger.warning(f"Failed to receive message: {e}")
                break

            # Validate message type
            if message.get("type") != "frame":
                logger.warning(f"Unknown message type: {message.get('type')}")
                continue

            # Decode base64 image data
            try:
                image_data = base64.b64decode(message.get("data", ""))
            except Exception as e:
                logger.warning(f"Failed to decode image data: {e}")
                error_response = EnrollmentErrorResponse(
                    type="error",
                    error="Invalid image data",
                    code="INVALID_IMAGE",
                )
                await websocket.send_json(error_response.model_dump())
                continue

            # Process frame
            response = session.process_frame(image_data)
            await websocket.send_json(response.model_dump())

            # Check if enrollment is ready
            if session.is_ready_for_enrollment():
                logger.info(
                    f"Enrollment ready for {user_name}: "
                    f"{len(session.keyframe_candidates)} keyframes captured"
                )

                # Load MASt3R model now (deferred until needed)
                engine = get_engine()
                if not engine._model_loaded:
                    logger.info("Loading MASt3R model for reconstruction...")
                    engine.load_model()

                # Run reconstruction and save template
                try:
                    complete_response = await run_enrollment(
                        session, engine, template_manager
                    )
                    await websocket.send_json(complete_response.model_dump())
                    logger.info(f"Enrollment complete for {user_name}")
                    break

                except Exception as e:
                    logger.error(f"Enrollment failed: {e}")
                    error_response = EnrollmentErrorResponse(
                        type="error",
                        error=f"Enrollment failed: {str(e)}",
                        code="ENROLLMENT_FAILED",
                    )
                    await websocket.send_json(error_response.model_dump())
                    break

    except WebSocketDisconnect:
        logger.info(f"Client disconnected during enrollment: {user_name}")

    except Exception as e:
        logger.error(f"Unexpected error during enrollment: {e}")
        try:
            error_response = EnrollmentErrorResponse(
                type="error",
                error=f"Unexpected error: {str(e)}",
                code="UNEXPECTED_ERROR",
            )
            await websocket.send_json(error_response.model_dump())
        except Exception:
            pass

    finally:
        # Cleanup
        if session:
            session.cleanup()
        try:
            await websocket.close()
        except Exception:
            pass
