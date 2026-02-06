"""
Tests for the TemplateManager module.

This test suite verifies:
- FaceTemplate dataclass validation
- Template save/load roundtrip
- Template deletion
- User listing and lookup
- Authentication logging
- Edge cases and error handling

Run with: pytest tests/test_template_manager.py -v
"""

import os
import sys
import tempfile
import shutil
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.template_manager import (
    TemplateManager,
    FaceTemplate,
    generate_user_id,
)


class TestFaceTemplate:
    """Tests for the FaceTemplate dataclass."""

    def test_create_valid_template(self):
        """Test creating a valid FaceTemplate."""
        n_points = 1000
        descriptor_dim = 24

        template = FaceTemplate(
            user_id="usr_test123",
            user_name="Test User",
            point_cloud=np.random.randn(n_points, 3).astype(np.float32),
            descriptors=np.random.randn(n_points, descriptor_dim).astype(np.float32),
            confidence=np.random.rand(n_points).astype(np.float32),
            colors=np.random.randint(0, 255, (n_points, 3), dtype=np.uint8),
            enrollment_metadata={"n_frames": 12},
        )

        assert template.user_id == "usr_test123"
        assert template.user_name == "Test User"
        assert template.n_points == n_points
        assert template.descriptor_dim == descriptor_dim
        assert template.version == "1.0"

    def test_template_auto_dtype_conversion(self):
        """Test that FaceTemplate converts dtypes automatically."""
        n_points = 100

        # Create with wrong dtypes
        template = FaceTemplate(
            user_id="usr_test",
            user_name="Test",
            point_cloud=np.random.randn(n_points, 3).astype(np.float64),  # Wrong dtype
            descriptors=np.random.randn(n_points, 24).astype(np.float64),
            confidence=np.random.rand(n_points).astype(np.float64),
            colors=np.random.randint(0, 255, (n_points, 3), dtype=np.int32),  # Wrong dtype
        )

        # Should be converted to correct dtypes
        assert template.point_cloud.dtype == np.float32
        assert template.descriptors.dtype == np.float32
        assert template.confidence.dtype == np.float32
        assert template.colors.dtype == np.uint8

    def test_template_invalid_shapes(self):
        """Test that FaceTemplate rejects invalid shapes."""
        n_points = 100

        with pytest.raises(AssertionError):
            # Wrong shape for point_cloud
            FaceTemplate(
                user_id="usr_test",
                user_name="Test",
                point_cloud=np.random.randn(n_points, 4).astype(np.float32),  # Should be (N, 3)
                descriptors=np.random.randn(n_points, 24).astype(np.float32),
                confidence=np.random.rand(n_points).astype(np.float32),
                colors=np.random.randint(0, 255, (n_points, 3), dtype=np.uint8),
            )

    def test_template_mismatched_lengths(self):
        """Test that FaceTemplate rejects mismatched array lengths."""
        with pytest.raises(AssertionError):
            FaceTemplate(
                user_id="usr_test",
                user_name="Test",
                point_cloud=np.random.randn(100, 3).astype(np.float32),
                descriptors=np.random.randn(200, 24).astype(np.float32),  # Different length
                confidence=np.random.rand(100).astype(np.float32),
                colors=np.random.randint(0, 255, (100, 3), dtype=np.uint8),
            )


class TestGenerateUserId:
    """Tests for the generate_user_id function."""

    def test_generate_unique_ids(self):
        """Test that generated IDs are unique."""
        ids = [generate_user_id() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique

    def test_id_format(self):
        """Test that generated IDs have correct format."""
        user_id = generate_user_id()
        assert user_id.startswith("usr_")
        assert len(user_id) == 12  # "usr_" + 8 hex chars


class TestTemplateManager:
    """Tests for the TemplateManager class."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp(prefix="template_test_")
        yield {
            "storage_dir": os.path.join(temp_dir, "templates"),
            "db_path": os.path.join(temp_dir, "test.sqlite"),
            "temp_dir": temp_dir,
        }
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def manager(self, temp_storage):
        """Create a TemplateManager instance for testing."""
        m = TemplateManager(
            storage_dir=temp_storage["storage_dir"],
            db_path=temp_storage["db_path"],
        )
        yield m
        m.close()

    @pytest.fixture
    def sample_template(self):
        """Create a sample FaceTemplate for testing."""
        n_points = 500
        return FaceTemplate(
            user_id=generate_user_id(),
            user_name="Test User",
            point_cloud=np.random.randn(n_points, 3).astype(np.float32),
            descriptors=np.random.randn(n_points, 24).astype(np.float32),
            confidence=np.random.rand(n_points).astype(np.float32),
            colors=np.random.randint(0, 255, (n_points, 3), dtype=np.uint8),
            enrollment_metadata={
                "n_frames": 12,
                "yaw_range": [-25.0, 28.0],
                "pitch_range": [-10.0, 15.0],
            },
        )

    def test_init_creates_directories(self, temp_storage):
        """Test that initialization creates storage directories."""
        manager = TemplateManager(
            storage_dir=temp_storage["storage_dir"],
            db_path=temp_storage["db_path"],
        )

        assert os.path.exists(temp_storage["storage_dir"])
        assert os.path.exists(temp_storage["db_path"])
        manager.close()

    def test_save_and_load_template(self, manager, sample_template):
        """Test saving and loading a template."""
        # Save
        path = manager.save_template(sample_template)
        assert os.path.exists(path)

        # Load
        loaded = manager.load_template(sample_template.user_id)
        assert loaded is not None
        assert loaded.user_id == sample_template.user_id
        assert loaded.user_name == sample_template.user_name
        assert loaded.n_points == sample_template.n_points
        assert np.allclose(loaded.point_cloud, sample_template.point_cloud)
        assert np.allclose(loaded.descriptors, sample_template.descriptors)
        assert np.allclose(loaded.confidence, sample_template.confidence)
        assert np.array_equal(loaded.colors, sample_template.colors)

    def test_save_duplicate_raises_error(self, manager, sample_template):
        """Test that saving a duplicate template raises an error."""
        manager.save_template(sample_template)

        with pytest.raises(ValueError, match="already exists"):
            manager.save_template(sample_template)

    def test_load_nonexistent_template(self, manager):
        """Test loading a template that doesn't exist."""
        result = manager.load_template("usr_nonexistent")
        assert result is None

    def test_list_users(self, manager, sample_template):
        """Test listing all users."""
        # Initially empty
        users = manager.list_users()
        assert len(users) == 0

        # Add template
        manager.save_template(sample_template)

        users = manager.list_users()
        assert len(users) == 1
        assert users[0]["user_id"] == sample_template.user_id
        assert users[0]["user_name"] == sample_template.user_name
        assert users[0]["n_points"] == sample_template.n_points

    def test_load_all_templates(self, manager):
        """Test loading all templates."""
        # Create and save multiple templates
        templates = []
        for i in range(3):
            t = FaceTemplate(
                user_id=generate_user_id(),
                user_name=f"User {i}",
                point_cloud=np.random.randn(100, 3).astype(np.float32),
                descriptors=np.random.randn(100, 24).astype(np.float32),
                confidence=np.random.rand(100).astype(np.float32),
                colors=np.random.randint(0, 255, (100, 3), dtype=np.uint8),
            )
            manager.save_template(t)
            templates.append(t)

        # Load all
        loaded = manager.load_all_templates()
        assert len(loaded) == 3

        loaded_ids = {t.user_id for t in loaded}
        expected_ids = {t.user_id for t in templates}
        assert loaded_ids == expected_ids

    def test_delete_template(self, manager, sample_template, temp_storage):
        """Test deleting a template."""
        path = manager.save_template(sample_template)

        # Delete
        success = manager.delete_template(sample_template.user_id)
        assert success is True

        # Verify deletion
        assert not os.path.exists(path)
        assert manager.load_template(sample_template.user_id) is None
        assert len(manager.list_users()) == 0

    def test_delete_nonexistent_template(self, manager):
        """Test deleting a template that doesn't exist."""
        success = manager.delete_template("usr_nonexistent")
        assert success is False

    def test_user_exists(self, manager, sample_template):
        """Test checking if a user exists."""
        assert not manager.user_exists(sample_template.user_id)

        manager.save_template(sample_template)

        assert manager.user_exists(sample_template.user_id)

    def test_user_exists_by_name(self, manager, sample_template):
        """Test checking if a user exists by name."""
        assert not manager.user_exists_by_name(sample_template.user_name)

        manager.save_template(sample_template)

        assert manager.user_exists_by_name(sample_template.user_name)

    def test_get_user(self, manager, sample_template):
        """Test getting user details."""
        assert manager.get_user(sample_template.user_id) is None

        manager.save_template(sample_template)

        user = manager.get_user(sample_template.user_id)
        assert user is not None
        assert user["user_id"] == sample_template.user_id
        assert user["user_name"] == sample_template.user_name
        assert user["n_points"] == sample_template.n_points

    def test_log_authentication(self, manager, sample_template):
        """Test logging authentication attempts."""
        manager.save_template(sample_template)

        log_id = manager.log_authentication(
            user_id=sample_template.user_id,
            final_score=0.85,
            geometric_score=0.80,
            descriptor_score=0.88,
            is_match=True,
            anti_spoof_passed=True,
            processing_time_ms=1500,
        )

        assert log_id > 0

        logs = manager.get_auth_logs(user_id=sample_template.user_id)
        assert len(logs) == 1
        assert logs[0]["final_score"] == 0.85
        assert logs[0]["is_match"] is True

    def test_get_stats(self, manager, sample_template):
        """Test getting database statistics."""
        stats = manager.get_stats()
        assert stats["total_users"] == 0
        assert stats["total_points"] == 0

        manager.save_template(sample_template)
        manager.log_authentication(
            user_id=sample_template.user_id,
            final_score=0.85,
            geometric_score=0.80,
            descriptor_score=0.88,
            is_match=True,
            anti_spoof_passed=True,
            processing_time_ms=1500,
        )

        stats = manager.get_stats()
        assert stats["total_users"] == 1
        assert stats["total_points"] == sample_template.n_points
        assert stats["total_auth_attempts"] == 1
        assert stats["successful_auths"] == 1

    def test_metadata_serialization(self, manager):
        """Test that numpy arrays in metadata are properly serialized."""
        template = FaceTemplate(
            user_id=generate_user_id(),
            user_name="Test",
            point_cloud=np.random.randn(100, 3).astype(np.float32),
            descriptors=np.random.randn(100, 24).astype(np.float32),
            confidence=np.random.rand(100).astype(np.float32),
            colors=np.random.randint(0, 255, (100, 3), dtype=np.uint8),
            enrollment_metadata={
                "yaw_range": np.array([-25.0, 28.0]),  # numpy array
                "pitch_range": np.array([-10.0, 15.0]),
            },
        )

        manager.save_template(template)
        loaded = manager.load_template(template.user_id)

        assert loaded is not None
        assert "yaw_range" in loaded.enrollment_metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
