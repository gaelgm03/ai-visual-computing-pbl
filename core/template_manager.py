"""
Template Manager Module

This module handles persistence and retrieval of face templates for the
MASt3R-based face authentication system.

Templates are stored as:
- .npz files: Contain the 3D point cloud, descriptors, confidence, colors, and metadata
- SQLite database: User metadata for efficient querying and management

The TemplateManager class provides CRUD operations:
- save_template: Save a new enrollment template
- load_template: Load a single user's template
- load_all_templates: Load all enrolled templates (for 1:N identification)
- delete_template: Remove a user's enrollment
- list_users: List all enrolled users

Usage:
    from core.template_manager import TemplateManager, FaceTemplate

    manager = TemplateManager(storage_dir="storage/templates", db_path="storage/db.sqlite")

    # Save a template
    template = FaceTemplate(
        user_id="usr_abc123",
        user_name="Alice",
        point_cloud=point_cloud,
        descriptors=descriptors,
        confidence=confidence,
        colors=colors,
        enrollment_metadata={"n_frames": 12, ...}
    )
    path = manager.save_template(template)

    # Load a template
    loaded = manager.load_template("usr_abc123")

    # List all users
    users = manager.list_users()

Author: CS-1
"""

import os
import json
import sqlite3
import uuid
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class FaceTemplate:
    """
    Data class representing an enrolled face template.

    A template contains all the information needed to authenticate a user:
    - 3D point cloud of the face surface
    - Per-point descriptors for identity matching
    - Per-point confidence values
    - Colors for visualization
    - Metadata about the enrollment process

    Attributes:
        user_id: Unique identifier for the user (e.g., "usr_a1b2c3").
        user_name: Human-readable name (e.g., "Alice").
        point_cloud: 3D coordinates of face surface points.
                     Shape: (N, 3), dtype: float32.
        descriptors: Dense feature descriptors per point.
                     Shape: (N, D), dtype: float32.
        confidence: Per-point confidence values.
                    Shape: (N,), dtype: float32, range [0, 1].
        colors: RGB colors per point for visualization.
                Shape: (N, 3), dtype: uint8, range [0, 255].
        enrollment_metadata: Dictionary with enrollment details:
            - enrolled_at: ISO timestamp of enrollment
            - n_frames: Number of keyframes used
            - yaw_range: [min, max] yaw angles captured
            - pitch_range: [min, max] pitch angles captured
            - mast3r_version: Model version used
        version: Template format version for future compatibility.
    """

    user_id: str
    user_name: str
    point_cloud: np.ndarray  # (N, 3) float32
    descriptors: np.ndarray  # (N, D) float32
    confidence: np.ndarray   # (N,) float32
    colors: np.ndarray       # (N, 3) uint8
    enrollment_metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    def __post_init__(self):
        """Validate template data after initialization."""
        # Ensure arrays have correct dtypes
        if self.point_cloud.dtype != np.float32:
            self.point_cloud = self.point_cloud.astype(np.float32)
        if self.descriptors.dtype != np.float32:
            self.descriptors = self.descriptors.astype(np.float32)
        if self.confidence.dtype != np.float32:
            self.confidence = self.confidence.astype(np.float32)
        if self.colors.dtype != np.uint8:
            self.colors = self.colors.astype(np.uint8)

        # Validate shapes
        n_points = len(self.point_cloud)
        assert self.point_cloud.shape == (n_points, 3), \
            f"point_cloud must be (N, 3), got {self.point_cloud.shape}"
        assert len(self.descriptors) == n_points, \
            f"descriptors must have {n_points} points, got {len(self.descriptors)}"
        assert len(self.confidence) == n_points, \
            f"confidence must have {n_points} points, got {len(self.confidence)}"
        assert self.colors.shape == (n_points, 3), \
            f"colors must be (N, 3), got {self.colors.shape}"

    @property
    def n_points(self) -> int:
        """Return the number of 3D points in the template."""
        return len(self.point_cloud)

    @property
    def descriptor_dim(self) -> int:
        """Return the dimension of descriptors."""
        return self.descriptors.shape[1] if len(self.descriptors) > 0 else 0


def generate_user_id() -> str:
    """
    Generate a unique user ID.

    Format: "usr_" followed by 8 random hex characters.

    Returns:
        A unique user ID string (e.g., "usr_a1b2c3d4").
    """
    return f"usr_{uuid.uuid4().hex[:8]}"


class TemplateManager:
    """
    Manages persistence and retrieval of face templates.

    Templates are stored in two places:
    1. Filesystem (.npz files): Contains the actual template data
    2. SQLite database: Contains user metadata for efficient lookup

    The manager handles:
    - Creating storage directories if they don't exist
    - Initializing the SQLite schema on first use
    - CRUD operations for templates
    - Logging authentication attempts (optional)

    Attributes:
        storage_dir: Directory where .npz template files are stored.
        db_path: Path to the SQLite database file.
        conn: SQLite connection (lazy-initialized).
    """

    def __init__(self, storage_dir: str, db_path: str):
        """
        Initialize the TemplateManager.

        Creates the storage directory and database if they don't exist.

        Args:
            storage_dir: Path to directory for storing .npz files.
            db_path: Path to SQLite database file.

        Example:
            manager = TemplateManager(
                storage_dir="storage/templates",
                db_path="storage/db.sqlite"
            )
        """
        self.storage_dir = Path(storage_dir)
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database schema
        self._init_database()

        logger.info(f"TemplateManager initialized: storage={self.storage_dir}, db={self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get or create the SQLite connection.

        Uses a lazy initialization pattern to avoid creating
        the connection until it's actually needed.

        Returns:
            SQLite connection with Row factory for dict-like access.
        """
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_database(self) -> None:
        """
        Initialize the SQLite database schema.

        Creates tables if they don't exist:
        - users: Stores user metadata and template paths
        - auth_logs: Stores authentication attempt history (for analysis)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                user_name TEXT NOT NULL,
                template_path TEXT NOT NULL,
                enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                n_points INTEGER,
                n_frames_used INTEGER
            )
        """)

        # Create auth_logs table for tracking authentication attempts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS auth_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                final_score REAL,
                geometric_score REAL,
                descriptor_score REAL,
                is_match BOOLEAN,
                anti_spoof_passed BOOLEAN,
                processing_time_ms INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)

        conn.commit()
        logger.debug("Database schema initialized")

    def _get_template_path(self, user_id: str) -> Path:
        """
        Get the filesystem path for a user's template file.

        Args:
            user_id: The user's unique identifier.

        Returns:
            Path to the .npz template file.
        """
        return self.storage_dir / f"{user_id}.npz"

    def save_template(self, template: FaceTemplate) -> str:
        """
        Save a face template to disk and register in the database.

        The template is saved as a compressed .npz file containing:
        - point_cloud: (N, 3) float32
        - descriptors: (N, D) float32
        - confidence: (N,) float32
        - colors: (N, 3) uint8
        - metadata: JSON string with enrollment details

        Args:
            template: FaceTemplate object to save.

        Returns:
            Path to the saved template file (as string).

        Raises:
            ValueError: If a template with this user_id already exists.

        Example:
            template = FaceTemplate(
                user_id="usr_abc123",
                user_name="Alice",
                point_cloud=np.random.randn(10000, 3).astype(np.float32),
                descriptors=np.random.randn(10000, 24).astype(np.float32),
                confidence=np.random.rand(10000).astype(np.float32),
                colors=np.random.randint(0, 255, (10000, 3), dtype=np.uint8),
                enrollment_metadata={"n_frames": 12}
            )
            path = manager.save_template(template)
        """
        # Check if user already exists
        if self.load_template(template.user_id) is not None:
            raise ValueError(f"Template already exists for user_id: {template.user_id}")

        # Prepare metadata for storage
        metadata = template.enrollment_metadata.copy()
        metadata.update({
            "user_id": template.user_id,
            "user_name": template.user_name,
            "enrolled_at": datetime.now().isoformat(),
            "template_version": template.version,
        })

        # Ensure yaw_range and pitch_range are serializable
        if "yaw_range" in metadata and isinstance(metadata["yaw_range"], np.ndarray):
            metadata["yaw_range"] = metadata["yaw_range"].tolist()
        if "pitch_range" in metadata and isinstance(metadata["pitch_range"], np.ndarray):
            metadata["pitch_range"] = metadata["pitch_range"].tolist()

        # Save to .npz file
        template_path = self._get_template_path(template.user_id)
        np.savez_compressed(
            str(template_path),
            point_cloud=template.point_cloud,
            descriptors=template.descriptors,
            confidence=template.confidence,
            colors=template.colors,
            metadata=json.dumps(metadata)
        )

        # Register in database
        conn = self._get_connection()
        cursor = conn.cursor()

        n_frames = metadata.get("n_frames", 0)

        cursor.execute("""
            INSERT INTO users (user_id, user_name, template_path, n_points, n_frames_used)
            VALUES (?, ?, ?, ?, ?)
        """, (
            template.user_id,
            template.user_name,
            str(template_path),
            template.n_points,
            n_frames
        ))

        conn.commit()

        logger.info(f"Saved template for {template.user_name} (id={template.user_id}, "
                   f"points={template.n_points})")

        return str(template_path)

    def load_template(self, user_id: str) -> Optional[FaceTemplate]:
        """
        Load a single user's template.

        Args:
            user_id: The user's unique identifier.

        Returns:
            FaceTemplate object, or None if user not found.

        Example:
            template = manager.load_template("usr_abc123")
            if template:
                print(f"Loaded {template.n_points} points for {template.user_name}")
        """
        # Check database first
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT template_path FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        template_path = Path(row["template_path"])

        # Check if file exists
        if not template_path.exists():
            logger.warning(f"Template file missing for user {user_id}: {template_path}")
            return None

        # Load from .npz file
        try:
            data = np.load(str(template_path), allow_pickle=True)
            metadata = json.loads(str(data["metadata"]))

            template = FaceTemplate(
                user_id=metadata.get("user_id", user_id),
                user_name=metadata.get("user_name", "Unknown"),
                point_cloud=data["point_cloud"],
                descriptors=data["descriptors"],
                confidence=data["confidence"],
                colors=data["colors"],
                enrollment_metadata=metadata,
                version=metadata.get("template_version", "1.0")
            )

            logger.debug(f"Loaded template for {template.user_name} ({template.n_points} points)")
            return template

        except Exception as e:
            logger.error(f"Failed to load template {user_id}: {e}")
            return None

    def load_all_templates(self) -> List[FaceTemplate]:
        """
        Load all enrolled face templates.

        Used for 1:N identification mode where the probe is compared
        against all enrolled users.

        Returns:
            List of FaceTemplate objects for all enrolled users.

        Example:
            templates = manager.load_all_templates()
            print(f"Loaded {len(templates)} enrolled users")
            for t in templates:
                print(f"  - {t.user_name}: {t.n_points} points")
        """
        templates = []

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT user_id FROM users")
        rows = cursor.fetchall()

        for row in rows:
            template = self.load_template(row["user_id"])
            if template is not None:
                templates.append(template)

        logger.info(f"Loaded {len(templates)} templates")
        return templates

    def delete_template(self, user_id: str) -> bool:
        """
        Delete a user's template from both filesystem and database.

        Args:
            user_id: The user's unique identifier.

        Returns:
            True if deletion was successful, False if user not found.

        Example:
            if manager.delete_template("usr_abc123"):
                print("Template deleted successfully")
            else:
                print("User not found")
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Check if user exists and get template path
        cursor.execute("SELECT template_path FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()

        if row is None:
            logger.warning(f"Cannot delete: user {user_id} not found")
            return False

        template_path = Path(row["template_path"])

        # Delete from database
        cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

        # Also delete related auth logs
        cursor.execute("DELETE FROM auth_logs WHERE user_id = ?", (user_id,))

        conn.commit()

        # Delete file from filesystem
        if template_path.exists():
            try:
                template_path.unlink()
                logger.info(f"Deleted template file: {template_path}")
            except Exception as e:
                logger.warning(f"Failed to delete template file {template_path}: {e}")

        logger.info(f"Deleted template for user {user_id}")
        return True

    def list_users(self) -> List[Dict[str, Any]]:
        """
        List all enrolled users with their metadata.

        Returns:
            List of dictionaries, each containing:
            - user_id: Unique identifier
            - user_name: Human-readable name
            - enrolled_at: Enrollment timestamp
            - n_points: Number of 3D points in template
            - n_frames_used: Number of keyframes used for enrollment

        Example:
            users = manager.list_users()
            for user in users:
                print(f"{user['user_name']} ({user['user_id']}): "
                      f"{user['n_points']} points, enrolled {user['enrolled_at']}")
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id, user_name, enrolled_at, n_points, n_frames_used
            FROM users
            ORDER BY enrolled_at DESC
        """)

        rows = cursor.fetchall()

        users = []
        for row in rows:
            users.append({
                "user_id": row["user_id"],
                "user_name": row["user_name"],
                "enrolled_at": row["enrolled_at"],
                "n_points": row["n_points"],
                "n_frames_used": row["n_frames_used"]
            })

        return users

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific user.

        Args:
            user_id: The user's unique identifier.

        Returns:
            Dictionary with user details, or None if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id, user_name, template_path, enrolled_at, n_points, n_frames_used
            FROM users
            WHERE user_id = ?
        """, (user_id,))

        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "user_id": row["user_id"],
            "user_name": row["user_name"],
            "template_path": row["template_path"],
            "enrolled_at": row["enrolled_at"],
            "n_points": row["n_points"],
            "n_frames_used": row["n_frames_used"]
        }

    def user_exists(self, user_id: str) -> bool:
        """
        Check if a user with the given ID exists.

        Args:
            user_id: The user's unique identifier.

        Returns:
            True if user exists, False otherwise.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
        return cursor.fetchone() is not None

    def user_exists_by_name(self, user_name: str) -> bool:
        """
        Check if a user with the given name exists.

        Args:
            user_name: The user's name.

        Returns:
            True if user exists, False otherwise.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT 1 FROM users WHERE user_name = ?", (user_name,))
        return cursor.fetchone() is not None

    def get_user_by_name(self, user_name: str) -> Optional[Dict[str, Any]]:
        """
        Get user information by name.

        Args:
            user_name: The user's name.

        Returns:
            Dictionary with user details, or None if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id, user_name, template_path, enrolled_at, n_points, n_frames_used
            FROM users
            WHERE user_name = ?
        """, (user_name,))

        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "user_id": row["user_id"],
            "user_name": row["user_name"],
            "template_path": row["template_path"],
            "enrolled_at": row["enrolled_at"],
            "n_points": row["n_points"],
            "n_frames_used": row["n_frames_used"]
        }

    def log_authentication(
        self,
        user_id: Optional[str],
        final_score: float,
        geometric_score: float,
        descriptor_score: float,
        is_match: bool,
        anti_spoof_passed: bool,
        processing_time_ms: int
    ) -> int:
        """
        Log an authentication attempt for analytics and auditing.

        Args:
            user_id: User ID that was matched (or attempted).
            final_score: Final fused match score.
            geometric_score: Geometric matching score.
            descriptor_score: Descriptor matching score.
            is_match: Whether authentication succeeded.
            anti_spoof_passed: Whether anti-spoofing check passed.
            processing_time_ms: Total processing time in milliseconds.

        Returns:
            The log entry ID.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO auth_logs
            (user_id, final_score, geometric_score, descriptor_score,
             is_match, anti_spoof_passed, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            final_score,
            geometric_score,
            descriptor_score,
            is_match,
            anti_spoof_passed,
            processing_time_ms
        ))

        conn.commit()

        log_id = cursor.lastrowid
        logger.debug(f"Logged authentication attempt: id={log_id}, user={user_id}, "
                    f"match={is_match}, score={final_score:.3f}")

        return log_id

    def get_auth_logs(
        self,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get authentication attempt logs.

        Args:
            user_id: Filter by user ID (optional).
            limit: Maximum number of entries to return.

        Returns:
            List of log entries as dictionaries.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if user_id:
            cursor.execute("""
                SELECT * FROM auth_logs
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM auth_logs
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

        rows = cursor.fetchall()

        logs = []
        for row in rows:
            logs.append({
                "id": row["id"],
                "user_id": row["user_id"],
                "timestamp": row["timestamp"],
                "final_score": row["final_score"],
                "geometric_score": row["geometric_score"],
                "descriptor_score": row["descriptor_score"],
                "is_match": bool(row["is_match"]),
                "anti_spoof_passed": bool(row["anti_spoof_passed"]),
                "processing_time_ms": row["processing_time_ms"]
            })

        return logs

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the template database.

        Returns:
            Dictionary with:
            - total_users: Number of enrolled users
            - total_points: Sum of points across all templates
            - total_auth_attempts: Number of authentication attempts
            - successful_auths: Number of successful authentications
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as count, SUM(n_points) as total_points FROM users")
        user_stats = cursor.fetchone()

        cursor.execute("SELECT COUNT(*) as total, SUM(is_match) as successes FROM auth_logs")
        auth_stats = cursor.fetchone()

        return {
            "total_users": user_stats["count"] or 0,
            "total_points": user_stats["total_points"] or 0,
            "total_auth_attempts": auth_stats["total"] or 0,
            "successful_auths": int(auth_stats["successes"] or 0)
        }

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")

    def __del__(self):
        """Clean up resources on deletion."""
        self.close()


# Singleton instance for the manager
_manager_instance: Optional[TemplateManager] = None


def get_template_manager(
    storage_dir: Optional[str] = None,
    db_path: Optional[str] = None
) -> TemplateManager:
    """
    Get or create the singleton TemplateManager instance.

    This function ensures only one manager exists in the application,
    avoiding multiple database connections.

    Args:
        storage_dir: Path to template storage directory.
                     If None, uses value from config.
        db_path: Path to SQLite database.
                 If None, uses value from config.

    Returns:
        The shared TemplateManager instance.

    Example:
        # First call initializes with config values
        manager = get_template_manager()

        # Subsequent calls return the same instance
        manager = get_template_manager()
    """
    global _manager_instance

    if _manager_instance is None:
        if storage_dir is None or db_path is None:
            # Import config module to get default values
            from core.config import get_storage_config, get_project_root

            storage_config = get_storage_config()
            project_root = get_project_root()

            if storage_dir is None:
                storage_dir = str(project_root / storage_config["templates_dir"])
            if db_path is None:
                db_path = str(project_root / storage_config["db_path"])

        _manager_instance = TemplateManager(storage_dir, db_path)

    return _manager_instance


if __name__ == "__main__":
    # Quick test of the TemplateManager
    import tempfile
    import shutil

    print("=" * 60)
    print(" TemplateManager Test")
    print("=" * 60)

    # Create temporary storage for testing
    test_dir = tempfile.mkdtemp(prefix="template_manager_test_")
    storage_dir = os.path.join(test_dir, "templates")
    db_path = os.path.join(test_dir, "test.sqlite")

    try:
        # Initialize manager
        print("\n1. Initializing TemplateManager...")
        manager = TemplateManager(storage_dir=storage_dir, db_path=db_path)
        print(f"   Storage: {storage_dir}")
        print(f"   Database: {db_path}")

        # Create test template
        print("\n2. Creating test template...")
        n_points = 10000
        descriptor_dim = 24

        template = FaceTemplate(
            user_id=generate_user_id(),
            user_name="Test User",
            point_cloud=np.random.randn(n_points, 3).astype(np.float32),
            descriptors=np.random.randn(n_points, descriptor_dim).astype(np.float32),
            confidence=np.random.rand(n_points).astype(np.float32),
            colors=np.random.randint(0, 255, (n_points, 3), dtype=np.uint8),
            enrollment_metadata={
                "n_frames": 12,
                "yaw_range": [-25.0, 28.0],
                "pitch_range": [-10.0, 15.0],
                "mast3r_version": "ViTLarge_metric"
            }
        )
        print(f"   User ID: {template.user_id}")
        print(f"   User Name: {template.user_name}")
        print(f"   Points: {template.n_points}")
        print(f"   Descriptor dim: {template.descriptor_dim}")

        # Save template
        print("\n3. Saving template...")
        path = manager.save_template(template)
        print(f"   Saved to: {path}")

        # List users
        print("\n4. Listing users...")
        users = manager.list_users()
        print(f"   Found {len(users)} user(s)")
        for user in users:
            print(f"   - {user['user_name']} ({user['user_id']}): {user['n_points']} points")

        # Load template
        print("\n5. Loading template...")
        loaded = manager.load_template(template.user_id)
        if loaded:
            print(f"   Loaded {loaded.n_points} points")
            print(f"   Point cloud shape: {loaded.point_cloud.shape}")
            print(f"   Descriptors shape: {loaded.descriptors.shape}")
            assert np.allclose(loaded.point_cloud, template.point_cloud)
            print("   Data integrity verified!")
        else:
            print("   ERROR: Failed to load template")

        # Get stats
        print("\n6. Getting stats...")
        stats = manager.get_stats()
        print(f"   Total users: {stats['total_users']}")
        print(f"   Total points: {stats['total_points']}")

        # Log authentication attempt
        print("\n7. Logging authentication attempt...")
        log_id = manager.log_authentication(
            user_id=template.user_id,
            final_score=0.85,
            geometric_score=0.80,
            descriptor_score=0.88,
            is_match=True,
            anti_spoof_passed=True,
            processing_time_ms=1500
        )
        print(f"   Log ID: {log_id}")

        # Get auth logs
        logs = manager.get_auth_logs()
        print(f"   Found {len(logs)} auth log(s)")

        # Delete template
        print("\n8. Deleting template...")
        success = manager.delete_template(template.user_id)
        print(f"   Delete successful: {success}")

        # Verify deletion
        users = manager.list_users()
        print(f"   Users after deletion: {len(users)}")

        # Close manager
        manager.close()

        print("\n" + "=" * 60)
        print(" All tests passed!")
        print("=" * 60)

    finally:
        # Clean up test directory
        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory: {test_dir}")
