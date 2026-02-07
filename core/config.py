"""
Configuration Management Module

This module provides a centralized way to load and access configuration settings
from the config.yaml file. It uses the Singleton pattern to ensure only one
configuration instance exists throughout the application.

Usage:
    from core.config import get_config
    config = get_config()
    face_detection_config = config["face_detection"]
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


# Store the singleton instance (module-level variable)
_config_instance: Optional[Dict[str, Any]] = None


def get_project_root() -> Path:
    """
    Find the project root directory.

    The project root is identified by the presence of config.yaml file.
    This function walks up the directory tree from this file's location
    until it finds config.yaml.

    Returns:
        Path: The absolute path to the project root directory.

    Raises:
        FileNotFoundError: If config.yaml cannot be found in any parent directory.
    """
    # Start from the directory containing this file
    current_dir = Path(__file__).resolve().parent

    # Walk up the directory tree to find config.yaml
    while current_dir != current_dir.parent:
        config_path = current_dir / "config.yaml"
        if config_path.exists():
            return current_dir
        current_dir = current_dir.parent

    # If we reach here, config.yaml was not found
    raise FileNotFoundError(
        "Could not find config.yaml in any parent directory. "
        "Make sure you're running from within the project directory."
    )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Optional path to the config file.
                     If not provided, uses the default config.yaml in project root.

    Returns:
        Dict containing all configuration values.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the config file contains invalid YAML.
    """
    if config_path is None:
        project_root = get_project_root()
        config_path = project_root / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_config(reload: bool = False) -> Dict[str, Any]:
    """
    Get the configuration singleton.

    This is the main function you should use to access configuration.
    It ensures only one configuration instance exists in the application,
    which is more efficient and prevents inconsistencies.

    Args:
        reload: If True, forces reloading the configuration from disk.
                Useful for testing or if the config file has changed.

    Returns:
        Dict containing all configuration values.

    Example:
        # Get face detection settings
        config = get_config()
        min_confidence = config["face_detection"]["min_detection_confidence"]

        # Get MASt3R settings
        image_size = config["mast3r"]["image_size"]
    """
    global _config_instance

    if _config_instance is None or reload:
        _config_instance = load_config()

    return _config_instance


def get_section(section_name: str) -> Dict[str, Any]:
    """
    Get a specific section from the configuration.

    A convenience function for accessing top-level configuration sections.

    Args:
        section_name: Name of the configuration section
                      (e.g., "face_detection", "mast3r", "keyframe")

    Returns:
        Dict containing the section's configuration values.

    Raises:
        KeyError: If the section doesn't exist in the configuration.

    Example:
        face_config = get_section("face_detection")
        min_conf = face_config["min_detection_confidence"]
    """
    config = get_config()

    if section_name not in config:
        raise KeyError(
            f"Configuration section '{section_name}' not found. "
            f"Available sections: {list(config.keys())}"
        )

    return config[section_name]


# Convenience functions for commonly used configuration sections
def get_face_detection_config() -> Dict[str, Any]:
    """Get face detection configuration."""
    return get_section("face_detection")


def get_keyframe_config() -> Dict[str, Any]:
    """Get keyframe selection configuration."""
    return get_section("keyframe")


def get_mast3r_config() -> Dict[str, Any]:
    """Get MASt3R engine configuration."""
    return get_section("mast3r")


def get_matching_config() -> Dict[str, Any]:
    """Get matching algorithm configuration."""
    return get_section("matching")


def get_storage_config() -> Dict[str, Any]:
    """Get storage configuration."""
    return get_section("storage")


def get_api_config() -> Dict[str, Any]:
    """Get API configuration."""
    return get_section("api")


def get_server_config() -> Dict[str, Any]:
    """
    Get server configuration for the API.

    Returns:
        Dict with host and port for the API server.
    """
    api_config = get_api_config()
    base_url = api_config.get("base_url", "http://localhost:8000")

    # Parse host and port from base_url
    # Format: http://host:port
    host = "0.0.0.0"
    port = 8000

    try:
        url_part = base_url.split("//")[-1]  # Remove http:// or https://
        if ":" in url_part:
            host_part, port_str = url_part.rsplit(":", 1)
            port = int(port_str.rstrip("/"))
            if host_part != "localhost":
                host = host_part
    except (ValueError, IndexError):
        pass

    return {"host": host, "port": port}


if __name__ == "__main__":
    # Quick test of the config loading
    print("Testing configuration loader...")

    config = get_config()
    print(f"Successfully loaded config with sections: {list(config.keys())}")

    # Test section access
    face_config = get_face_detection_config()
    print(f"Face detection model: {face_config['model']}")
    print(f"Min detection confidence: {face_config['min_detection_confidence']}")
