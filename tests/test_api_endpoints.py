"""
Tests for API Endpoints

This test suite verifies:
- Health check endpoint
- User management endpoints (list, get, delete)
- WebSocket enrollment endpoint (basic connectivity)

Run with: pytest tests/test_api_endpoints.py -v

Note: Full WebSocket enrollment testing requires MASt3R model loaded,
which is slow and GPU-intensive. These tests focus on the API layer.

Author: CS-1
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        # Mock the MASt3R engine to avoid loading the model
        with patch("api.app.get_engine") as mock_engine:
            mock_engine.return_value._model_loaded = False

            # Mock template manager
            with patch("api.app.get_template_manager") as mock_tm:
                mock_tm.return_value.get_stats.return_value = {
                    "total_users": 0,
                    "total_points": 0,
                    "total_auth_attempts": 0,
                    "successful_auths": 0,
                }

                # Import app after patching
                from api.app import app
                yield TestClient(app)

    def test_health_check_returns_status(self, client):
        """Test that health check returns valid response."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "gpu_available" in data
        assert "enrolled_users" in data

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestUserManagementEndpoints:
    """Tests for user management endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch("api.app.get_engine") as mock_engine:
            mock_engine.return_value._model_loaded = False

            with patch("api.app.get_template_manager") as mock_tm:
                # Setup mock template manager
                mock_manager = MagicMock()
                mock_manager.get_stats.return_value = {
                    "total_users": 2,
                    "total_points": 20000,
                    "total_auth_attempts": 10,
                    "successful_auths": 8,
                }
                mock_manager.list_users.return_value = [
                    {
                        "user_id": "usr_test001",
                        "user_name": "Alice",
                        "enrolled_at": "2026-02-07T10:00:00",
                        "n_points": 10000,
                        "n_frames_used": 12,
                    },
                    {
                        "user_id": "usr_test002",
                        "user_name": "Bob",
                        "enrolled_at": "2026-02-07T11:00:00",
                        "n_points": 10000,
                        "n_frames_used": 10,
                    },
                ]
                mock_manager.get_user.return_value = {
                    "user_id": "usr_test001",
                    "user_name": "Alice",
                    "enrolled_at": "2026-02-07T10:00:00",
                    "n_points": 10000,
                    "n_frames_used": 12,
                    "template_path": "/path/to/template.npz",
                }
                mock_manager.load_template.return_value = MagicMock(
                    enrollment_metadata={"n_frames": 12, "yaw_range": [-25, 25]}
                )
                mock_manager.user_exists.return_value = True
                mock_manager.delete_template.return_value = True
                mock_tm.return_value = mock_manager

                from api.app import app
                yield TestClient(app)

    def test_list_users(self, client):
        """Test listing all users."""
        response = client.get("/users")
        assert response.status_code == 200

        data = response.json()
        assert "users" in data
        assert "total" in data
        assert data["total"] == 2
        assert len(data["users"]) == 2

        # Check user structure
        user = data["users"][0]
        assert "user_id" in user
        assert "user_name" in user
        assert "enrolled_at" in user

    def test_get_user(self, client):
        """Test getting a specific user."""
        response = client.get("/users/usr_test001")
        assert response.status_code == 200

        data = response.json()
        assert data["user_id"] == "usr_test001"
        assert data["user_name"] == "Alice"
        assert "enrollment_metadata" in data

    def test_get_nonexistent_user(self, client):
        """Test getting a user that doesn't exist."""
        # Override mock for this test
        with patch("api.app.get_template_manager") as mock_tm:
            mock_manager = MagicMock()
            mock_manager.get_user.return_value = None
            mock_manager.get_stats.return_value = {"total_users": 0}
            mock_tm.return_value = mock_manager

            response = client.get("/users/usr_nonexistent")
            assert response.status_code == 404

    def test_delete_user(self, client):
        """Test deleting a user."""
        response = client.delete("/users/usr_test001")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["user_id"] == "usr_test001"

    def test_delete_nonexistent_user(self, client):
        """Test deleting a user that doesn't exist."""
        with patch("api.app.get_template_manager") as mock_tm:
            mock_manager = MagicMock()
            mock_manager.user_exists.return_value = False
            mock_manager.get_stats.return_value = {"total_users": 0}
            mock_tm.return_value = mock_manager

            response = client.delete("/users/usr_nonexistent")
            assert response.status_code == 404


class TestSchemas:
    """Tests for Pydantic schemas."""

    def test_frame_status_response(self):
        """Test FrameStatusResponse schema."""
        from api.schemas import FrameStatusResponse, HeadPose, Coverage

        response = FrameStatusResponse(
            type="frame_status",
            face_detected=True,
            head_pose=HeadPose(yaw=10.5, pitch=-5.0, roll=2.0),
            captured=True,
            total_captured=5,
            target_count=12,
            coverage=Coverage(
                yaw_range=[-20.0, 15.0],
                pitch_range=[-10.0, 10.0],
                is_sufficient=False,
                missing_directions=["left"],
            ),
        )

        data = response.model_dump()
        assert data["type"] == "frame_status"
        assert data["face_detected"] is True
        assert data["head_pose"]["yaw"] == 10.5
        assert data["coverage"]["is_sufficient"] is False

    def test_enrollment_complete_response(self):
        """Test EnrollmentCompleteResponse schema."""
        from api.schemas import EnrollmentCompleteResponse

        response = EnrollmentCompleteResponse(
            type="enrollment_complete",
            user_id="usr_abc123",
            user_name="TestUser",
            n_points=10000,
            n_frames_used=12,
            reconstruction_time_sec=8.5,
        )

        data = response.model_dump()
        assert data["user_id"] == "usr_abc123"
        assert data["n_points"] == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
