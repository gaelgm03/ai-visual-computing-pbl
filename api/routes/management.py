"""
User Management API Routes

This module provides REST endpoints for managing enrolled users:
- GET /users: List all enrolled users
- GET /users/{user_id}: Get user details
- GET /users/{user_id}/template: Get user's point cloud template
- DELETE /users/{user_id}: Delete an enrolled user

Author: CS-1
"""

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas import (
    UserInfo,
    UserListResponse,
    UserDetailResponse,
    DeleteUserResponse,
)
from core.template_manager import get_template_manager

# Create router
router = APIRouter(tags=["users"])


@router.get("/users", response_model=UserListResponse)
async def list_users():
    """
    List all enrolled users.

    Returns summary information for each enrolled user including
    their ID, name, enrollment time, and template statistics.
    """
    template_manager = get_template_manager()
    users = template_manager.list_users()

    return UserListResponse(
        users=[
            UserInfo(
                user_id=u["user_id"],
                user_name=u["user_name"],
                enrolled_at=u["enrolled_at"],
                n_points=u["n_points"],
                n_frames_used=u["n_frames_used"],
            )
            for u in users
        ],
        total=len(users),
    )


@router.get("/users/{user_id}", response_model=UserDetailResponse)
async def get_user(user_id: str):
    """
    Get detailed information about a specific user.

    Args:
        user_id: The user's unique identifier.

    Returns:
        Detailed user information including template metadata.

    Raises:
        404: If the user is not found.
    """
    template_manager = get_template_manager()
    user = template_manager.get_user(user_id)

    if user is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    # Load full template to get enrollment metadata
    template = template_manager.load_template(user_id)
    enrollment_metadata = template.enrollment_metadata if template else None

    return UserDetailResponse(
        user_id=user["user_id"],
        user_name=user["user_name"],
        enrolled_at=user["enrolled_at"],
        n_points=user["n_points"],
        n_frames_used=user["n_frames_used"],
        template_path=user.get("template_path"),
        enrollment_metadata=enrollment_metadata,
    )


@router.get("/users/{user_id}/template")
async def get_user_template(user_id: str):
    """
    Get a user's point cloud template for 3D visualization.

    Returns subsampled points and colors for efficient transfer.
    """
    template_manager = get_template_manager()
    template = template_manager.load_template(user_id)

    if template is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    # Subsample for reasonable transfer size
    max_points = 5000
    n_total = len(template.point_cloud)
    if n_total > max_points:
        indices = np.random.choice(n_total, max_points, replace=False)
        points = template.point_cloud[indices].tolist()
        colors = template.colors[indices].tolist() if template.colors is not None else None
    else:
        points = template.point_cloud.tolist()
        colors = template.colors.tolist() if template.colors is not None else None

    return {"points": points, "colors": colors}


@router.delete("/users/{user_id}", response_model=DeleteUserResponse)
async def delete_user(user_id: str):
    """
    Delete an enrolled user and their template.

    This permanently removes the user's enrollment data.

    Args:
        user_id: The user's unique identifier.

    Returns:
        Confirmation of deletion.

    Raises:
        404: If the user is not found.
    """
    template_manager = get_template_manager()

    if not template_manager.user_exists(user_id):
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    success = template_manager.delete_template(user_id)

    return DeleteUserResponse(
        success=success,
        user_id=user_id,
        message=f"User {user_id} deleted successfully" if success else "Deletion failed",
    )
