#!/usr/bin/env python
"""
Smoke test for CS-2 frontend components.
Verifies that all components can be imported and basic functionality works.

Run with: python scripts/smoke_test_frontend.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all frontend modules can be imported."""
    print("Testing imports...")
    
    try:
        from frontend.components.webcam_capture import WebcamCapture, CaptureConfig
        print("  ✓ webcam_capture")
    except ImportError as e:
        print(f"  ✗ webcam_capture: {e}")
        return False
    
    try:
        from frontend.components.enrollment_guide import EnrollmentGuide, HeadPose, EnrollmentConfig
        print("  ✓ enrollment_guide")
    except ImportError as e:
        print(f"  ✗ enrollment_guide: {e}")
        return False
    
    try:
        from frontend.components.auth_panel import AuthPanel, AuthResult, AuthConfig
        print("  ✓ auth_panel")
    except ImportError as e:
        print(f"  ✗ auth_panel: {e}")
        return False
    
    try:
        from frontend.components.visualization import PointCloudVisualizer, PointCloudData
        print("  ✓ visualization")
    except ImportError as e:
        print(f"  ✗ visualization: {e}")
        return False
    
    return True


def test_enrollment_guide():
    """Test EnrollmentGuide functionality."""
    print("\nTesting EnrollmentGuide...")
    
    from frontend.components.enrollment_guide import EnrollmentGuide, HeadPose, EnrollmentConfig
    
    guide = EnrollmentGuide(EnrollmentConfig(target_frames=12))
    
    # Test initial state
    assert guide.get_progress_percent() == 0.0, "Initial progress should be 0"
    print("  ✓ Initial state correct")
    
    # Simulate capturing poses
    poses = [
        HeadPose(yaw=-20, pitch=5),
        HeadPose(yaw=0, pitch=0),
        HeadPose(yaw=15, pitch=-5),
        HeadPose(yaw=-10, pitch=10),
    ]
    
    coverage = guide.update_coverage(poses)
    assert coverage.total_captured == 4, "Should have 4 captured"
    assert guide.get_progress_percent() > 0, "Progress should be > 0"
    print(f"  ✓ Coverage updated: {coverage.total_captured} frames, {guide.get_progress_percent():.1f}%")
    
    # Test direction guidance
    direction = guide.get_next_direction()
    print(f"  ✓ Next direction: {direction}")
    
    # Test status message
    status = guide.format_status_message()
    assert len(status) > 0, "Status message should not be empty"
    print(f"  ✓ Status: {status}")
    
    return True


def test_visualization():
    """Test PointCloudVisualizer functionality."""
    print("\nTesting PointCloudVisualizer...")
    
    import numpy as np
    from frontend.components.visualization import PointCloudVisualizer, PointCloudData
    
    viz = PointCloudVisualizer(max_points=1000)
    
    # Create dummy point cloud
    points = np.random.randn(500, 3).astype(np.float32)
    colors = (np.random.rand(500, 3) * 255).astype(np.uint8)
    
    data = PointCloudData(points=points, colors=colors)
    
    # Test figure creation
    fig = viz.create_plotly_figure(data, "Test Cloud")
    assert "data" in fig, "Figure should have data"
    assert "layout" in fig, "Figure should have layout"
    print("  ✓ Plotly figure created")
    
    # Test subsampling
    large_data = PointCloudData(points=np.random.randn(5000, 3).astype(np.float32))
    subsampled = viz.subsample(large_data)
    assert len(subsampled.points) <= 1000, "Should subsample to max_points"
    print(f"  ✓ Subsampling: {len(large_data.points)} -> {len(subsampled.points)} points")
    
    # Test empty figure
    empty_fig = viz.create_empty_figure("No data")
    assert "layout" in empty_fig, "Empty figure should have layout"
    print("  ✓ Empty figure created")
    
    return True


def test_auth_panel():
    """Test AuthPanel functionality."""
    print("\nTesting AuthPanel...")
    
    from frontend.components.auth_panel import AuthPanel, AuthResult, AuthConfig
    
    panel = AuthPanel(AuthConfig())
    
    # Test result formatting
    result = AuthResult(
        is_match=True,
        matched_user_name="Alice",
        final_score=0.85,
        geometric_score=0.80,
        descriptor_score=0.90,
    )
    
    msg = panel.format_result_message(result)
    assert "Alice" in msg, "Message should contain user name"
    assert "0.85" in msg, "Message should contain score"
    print(f"  ✓ Result message formatted")
    
    # Test score color
    color = panel.get_score_color(0.85)
    assert color.startswith("#"), "Color should be hex"
    print(f"  ✓ Score color: {color}")
    
    return True


def test_webcam_capture_static():
    """Test WebcamCapture static methods (no actual camera needed)."""
    print("\nTesting WebcamCapture (static methods)...")
    
    import numpy as np
    from frontend.components.webcam_capture import WebcamCapture
    
    # Create a dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test base64 encoding
    b64 = WebcamCapture.frame_to_base64(dummy_frame)
    assert len(b64) > 0, "Base64 string should not be empty"
    print(f"  ✓ Frame encoded to base64 ({len(b64)} chars)")
    
    # Test base64 decoding
    decoded = WebcamCapture.base64_to_frame(b64)
    assert decoded.shape == dummy_frame.shape, "Decoded shape should match"
    print(f"  ✓ Frame decoded from base64")
    
    return True


def test_config_loading():
    """Test that config.yaml can be loaded."""
    print("\nTesting config.yaml...")
    
    import yaml
    
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config.yaml"
    )
    
    if not os.path.exists(config_path):
        print(f"  ⚠ config.yaml not found at {config_path}")
        return True  # Not a failure, just skip
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert "frontend" in config, "Config should have frontend section"
    assert "api" in config, "Config should have api section"
    print(f"  ✓ Config loaded with sections: {list(config.keys())}")
    
    return True


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("CS-2 Frontend Smoke Tests")
    print("=" * 60)
    
    all_passed = True
    
    tests = [
        test_imports,
        test_enrollment_guide,
        test_visualization,
        test_auth_panel,
        test_webcam_capture_static,
        test_config_loading,
    ]
    
    for test_fn in tests:
        try:
            if not test_fn():
                all_passed = False
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All smoke tests PASSED")
    else:
        print("❌ Some tests FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
