"""
Shared UI overlay helpers for face capture screens.

Provides:
- draw_face_guide()  — alignment phase ellipse + instructions
- draw_pose_grid()   — mini yaw/pitch dot-map showing target coverage
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Set, Union


# ---------------------------------------------------------------------------
# Face guide ellipse (alignment phase)
# ---------------------------------------------------------------------------

def draw_face_guide(
    frame: np.ndarray,
    face_detected: bool = False,
    message: str = "Place your face in the guide, then press SPACE",
) -> None:
    """Draw a guide ellipse in the center of the frame.

    Args:
        frame: BGR image to draw on (modified in-place).
        face_detected: If True the ellipse is drawn green, otherwise gray.
        message: Instruction text shown below the ellipse.
    """
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)

    # Portrait ellipse sized for a human face — vertical axis = 3/4 of frame height
    semi_v = h * 3 // 8          # vertical semi-axis  (full axis = 3/4 h)
    semi_h = semi_v * 2 // 3     # horizontal semi-axis (face ~2:3 aspect)
    axes = (semi_h, semi_v)

    color = (0, 200, 0) if face_detected else (160, 160, 160)

    # Draw the ellipse (semi-transparent via overlay)
    overlay = frame.copy()
    cv2.ellipse(overlay, center, axes, 0, 0, 360, color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Instruction text
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    tx = (w - text_size[0]) // 2
    ty = center[1] + axes[1] + 30
    cv2.putText(frame, message, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Pose grid overlay (capture phase)
# ---------------------------------------------------------------------------

_GRID_W = 180      # px width of the grid box
_GRID_H = 120      # px height
_GRID_MARGIN = 15  # px from frame edges
_DOT_R = 8         # target dot radius
_CUR_R = 5         # current-pose dot radius


def draw_pose_grid(
    frame: np.ndarray,
    targets: List[Tuple[float, float]],
    captured: Union[Set[int], List[bool]],
    current_pose: Optional[Tuple[float, float]] = None,
    yaw_range: Tuple[float, float] = (-30.0, 30.0),
    pitch_range: Tuple[float, float] = (-20.0, 20.0),
) -> None:
    """Draw a mini yaw/pitch scatter-plot in the bottom-right corner.

    Each target is shown as a circle (green=captured, gray=pending).
    The user's current head pose is a moving cyan dot.

    Args:
        frame: BGR image to draw on (modified in-place).
        targets: List of (yaw, pitch) target poses.
        captured: Either a set of captured target indices **or** a list of
                  booleans (True = captured) of the same length as *targets*.
        current_pose: (yaw, pitch) of the user's current head pose, or None.
        yaw_range: (min_yaw, max_yaw) used for axis scaling.
        pitch_range: (min_pitch, max_pitch) used for axis scaling.
    """
    if not targets:
        return

    fh, fw = frame.shape[:2]

    # Box origin (top-left corner of the grid area)
    box_x = fw - _GRID_W - _GRID_MARGIN
    box_y = fh - _GRID_H - _GRID_MARGIN

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x, box_y),
                  (box_x + _GRID_W, box_y + _GRID_H),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Thin border
    cv2.rectangle(frame, (box_x, box_y),
                  (box_x + _GRID_W, box_y + _GRID_H),
                  (80, 80, 80), 1)

    # Inner padding so dots don't touch the edge
    pad = 20
    inner_x0 = box_x + pad
    inner_y0 = box_y + pad
    inner_w = _GRID_W - 2 * pad
    inner_h = _GRID_H - 2 * pad

    yaw_min, yaw_max = yaw_range
    pitch_min, pitch_max = pitch_range
    yaw_span = max(yaw_max - yaw_min, 1.0)
    pitch_span = max(pitch_max - pitch_min, 1.0)

    def _map(yaw: float, pitch: float) -> Tuple[int, int]:
        """Map (yaw, pitch) to pixel coordinates inside the grid box."""
        # Reversed x-axis: camera is fixed, user moves face.
        # Positive yaw (look left) maps to LEFT side of grid so user
        # intuitively knows to turn their head that way.
        px = inner_x0 + int((yaw_max - yaw) / yaw_span * inner_w)
        # Reversed y-axis: positive pitch (look up) maps to TOP of grid.
        py = inner_y0 + int((pitch - pitch_min) / pitch_span * inner_h)
        return px, py

    # Normalise captured to a lookup function
    if isinstance(captured, set):
        _is_captured = lambda i: i in captured
    else:
        _is_captured = lambda i: bool(captured[i]) if i < len(captured) else False

    # Draw target dots
    for i, (ty, tp) in enumerate(targets):
        cx, cy = _map(ty, tp)
        if _is_captured(i):
            cv2.circle(frame, (cx, cy), _DOT_R, (0, 220, 0), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), _DOT_R, (120, 120, 120), 1, cv2.LINE_AA)

    # Draw current head-pose indicator
    if current_pose is not None:
        cur_yaw, cur_pitch = current_pose
        # Clamp to visible range
        cur_yaw = max(yaw_min, min(yaw_max, cur_yaw))
        cur_pitch = max(pitch_min, min(pitch_max, cur_pitch))
        cx, cy = _map(cur_yaw, cur_pitch)
        cv2.circle(frame, (cx, cy), _CUR_R, (255, 255, 0), -1, cv2.LINE_AA)

    # Axis labels (user perspective: left side of grid = user turns left)
    label_y = box_y + _GRID_H - 5
    cv2.putText(frame, "L", (box_x + 4, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, "R", (box_x + _GRID_W - 14, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
