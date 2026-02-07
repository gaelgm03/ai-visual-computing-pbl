"""
FastAPI Application Entry Point

This module creates and configures the FastAPI application for the
MASt3R Face Authentication API.

The application provides:
- WebSocket endpoint for real-time enrollment
- REST endpoints for authentication (TODO)
- REST endpoints for user management
- Health check endpoint

Usage:
    # From project root:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

    # Or run directly:
    python -m api.app

Author: CS-1
"""

import logging
import torch
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.routes.enrollment import router as enrollment_router
from api.schemas import (
    UserInfo,
    UserListResponse,
    UserDetailResponse,
    DeleteUserResponse,
    HealthResponse,
)
from core.mast3r_engine import get_engine
from core.template_manager import get_template_manager
from core.config import get_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Runs on startup:
    - Load MASt3R model into GPU memory (takes ~30-60s)
    - Initialize template manager

    Runs on shutdown:
    - Cleanup resources
    """
    logger.info("=" * 60)
    logger.info("Starting MASt3R Face Authentication API")
    logger.info("=" * 60)

    # Initialize MASt3R engine (singleton)
    logger.info("Initializing MASt3R engine...")
    engine = get_engine()

    # Optionally pre-load the model (can also be lazy-loaded on first request)
    # Uncomment below to preload at startup:
    # logger.info("Pre-loading MASt3R model (this may take 30-60 seconds)...")
    # engine.load_model()
    # logger.info("Model loaded successfully!")

    # Initialize template manager (singleton)
    logger.info("Initializing template manager...")
    template_manager = get_template_manager()
    stats = template_manager.get_stats()
    logger.info(f"Template manager ready: {stats['total_users']} users enrolled")

    # Log GPU status
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("No GPU available - MASt3R will run on CPU (slow)")

    logger.info("API startup complete!")
    logger.info("=" * 60)

    yield

    # Cleanup on shutdown
    logger.info("Shutting down API...")
    template_manager.close()
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="MASt3R Face Authentication API",
    description="""
API for face authentication using MASt3R 3D reconstruction.

## Features
- **Enrollment**: Register a new user with real-time WebSocket feedback
- **Authentication**: Verify a user's identity (coming soon)
- **User Management**: List, view, and delete enrolled users

## WebSocket Enrollment
Connect to `/ws/enroll/{user_name}` to start an enrollment session.
Send frames as JSON: `{"type": "frame", "data": "<base64 JPEG>"}`
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(enrollment_router)


# ============================================================
# User Management Endpoints
# ============================================================

@app.get("/users", response_model=UserListResponse, tags=["users"])
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


@app.get("/users/{user_id}", response_model=UserDetailResponse, tags=["users"])
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


@app.delete("/users/{user_id}", response_model=DeleteUserResponse, tags=["users"])
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


# ============================================================
# Health Check Endpoint
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """
    Check the health of the API and its dependencies.

    Returns status of:
    - MASt3R model (loaded/not loaded)
    - GPU availability and memory
    - Number of enrolled users
    """
    engine = get_engine()
    template_manager = get_template_manager()
    stats = template_manager.get_stats()

    # Check GPU
    gpu_available = torch.cuda.is_available()
    gpu_memory_total = None
    gpu_memory_used = None

    if gpu_available:
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_memory_used = torch.cuda.memory_allocated() / 1e9

    # Determine overall status
    status = "healthy" if gpu_available else "degraded"

    return HealthResponse(
        status=status,
        model_loaded=engine._model_loaded,
        gpu_available=gpu_available,
        gpu_memory_total_gb=round(gpu_memory_total, 2) if gpu_memory_total is not None else None,
        gpu_memory_used_gb=round(gpu_memory_used, 2) if gpu_memory_used is not None else None,
        enrolled_users=stats["total_users"],
    )


@app.get("/", tags=["system"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MASt3R Face Authentication API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    api_config = config.get("api", {})

    # Parse host/port from base_url or use defaults
    host = "0.0.0.0"
    port = 8000

    base_url = api_config.get("base_url", "http://localhost:8000")
    if ":" in base_url.split("//")[-1]:
        port_str = base_url.split(":")[-1].rstrip("/")
        try:
            port = int(port_str)
        except ValueError:
            pass

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
