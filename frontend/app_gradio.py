"""
Gradio-based demo UI for MASt3R Face Authentication System.
CS-2 Primary Ownership.

This is the main entry point for the frontend application.
Run with: python -m frontend.app_gradio
"""

import gradio as gr
import numpy as np
import cv2
from typing import Optional, Tuple
import time

from frontend.components.webcam_capture import WebcamCapture, CaptureConfig
from frontend.components.enrollment_guide import EnrollmentGuide, HeadPose, EnrollmentConfig
from frontend.components.auth_panel import AuthPanel, AuthResult, AuthConfig
from frontend.components.visualization import PointCloudVisualizer, PointCloudData


# ============================================================
# Global State (will be replaced with proper state management)
# ============================================================
webcam = WebcamCapture(CaptureConfig(width=640, height=480))
enrollment_guide = EnrollmentGuide(EnrollmentConfig(target_frames=12))
auth_panel = AuthPanel(AuthConfig())
visualizer = PointCloudVisualizer(max_points=10000)

# Simulated captured frames for enrollment (placeholder until API connected)
_enrollment_frames = []
_enrollment_poses = []


# ============================================================
# Enrollment Tab Functions
# ============================================================

def start_enrollment(user_name: str):
    """Initialize enrollment session."""
    if not user_name or not user_name.strip():
        return None, "⚠️ Please enter a user name", gr.update()
    
    enrollment_guide.reset()
    _enrollment_frames.clear()
    _enrollment_poses.clear()
    
    return None, f"📷 Starting enrollment for **{user_name}**\nLook at the camera and slowly turn your head", gr.update(interactive=True)


def process_enrollment_frame(frame: Optional[np.ndarray], user_name: str):
    """
    Process a frame during enrollment.
    
    In the full implementation, this will:
    1. Send frame to backend via WebSocket
    2. Receive face detection + pose info
    3. Update enrollment guide
    
    For now, we simulate with placeholder logic.
    """
    if frame is None:
        return None, enrollment_guide.format_status_message(), create_coverage_plot()
    
    # Simulate face detection and pose estimation
    # In production, this comes from the backend API
    simulated_pose = HeadPose(
        yaw=np.random.uniform(-30, 30),
        pitch=np.random.uniform(-15, 15),
        roll=np.random.uniform(-5, 5),
    )
    
    # Simulate keyframe capture (every ~10 frames when face detected)
    should_capture = len(_enrollment_frames) < 12 and np.random.random() < 0.1
    
    if should_capture:
        _enrollment_frames.append(frame.copy())
        _enrollment_poses.append(simulated_pose)
    
    # Update coverage
    enrollment_guide.update_coverage(_enrollment_poses)
    
    # Draw overlay on frame
    annotated_frame = draw_enrollment_overlay(frame, simulated_pose, should_capture)
    
    status_msg = enrollment_guide.format_status_message()
    coverage_plot = create_coverage_plot()
    
    return annotated_frame, status_msg, coverage_plot


def draw_enrollment_overlay(
    frame: np.ndarray,
    pose: HeadPose,
    was_captured: bool
) -> np.ndarray:
    """Draw enrollment guidance overlay on the frame."""
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw pose info
    pose_text = f"Yaw: {pose.yaw:.1f}  Pitch: {pose.pitch:.1f}"
    cv2.putText(frame, pose_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)
    
    # Draw capture count
    count_text = f"Captured: {len(_enrollment_frames)}/12"
    cv2.putText(frame, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    
    # Flash green when frame captured
    if was_captured:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 10)
    
    # Draw direction arrow
    direction = enrollment_guide.get_next_direction()
    if direction and direction != "center":
        draw_direction_arrow(frame, direction)
    
    return frame


def draw_direction_arrow(frame: np.ndarray, direction: str):
    """Draw an arrow indicating which direction to turn."""
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    
    arrow_length = 80
    arrow_offsets = {
        "left": (-arrow_length, 0),
        "right": (arrow_length, 0),
        "up": (0, -arrow_length),
        "down": (0, arrow_length),
    }
    
    if direction in arrow_offsets:
        dx, dy = arrow_offsets[direction]
        end_point = (center[0] + dx, center[1] + dy)
        cv2.arrowedLine(frame, center, end_point, (0, 255, 255), 3, tipLength=0.3)
        
        # Draw instruction text
        text = f"Turn {direction}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(frame, text, (text_x, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 255), 2)


def create_coverage_plot():
    """Create a simple coverage visualization."""
    coverage = enrollment_guide.get_coverage_visualization_data()
    
    # Create a simple scatter plot of captured poses
    if not coverage["poses"]:
        return visualizer.create_empty_figure("No frames captured yet")
    
    yaws = [p["yaw"] for p in coverage["poses"]]
    pitches = [p["pitch"] for p in coverage["poses"]]
    
    figure = {
        "data": [{
            "type": "scatter",
            "x": yaws,
            "y": pitches,
            "mode": "markers",
            "marker": {"size": 10, "color": "blue"},
            "name": "Captured",
        }],
        "layout": {
            "title": f"Head Pose Coverage ({coverage['progress_percent']:.0f}%)",
            "xaxis": {"title": "Yaw (°)", "range": [-40, 40]},
            "yaxis": {"title": "Pitch (°)", "range": [-20, 20]},
            "width": 350,
            "height": 250,
            "shapes": [
                # Target coverage rectangle
                {
                    "type": "rect",
                    "x0": -30, "x1": 30,
                    "y0": -15, "y1": 15,
                    "line": {"color": "green", "dash": "dash"},
                    "fillcolor": "rgba(0,255,0,0.1)",
                }
            ],
        }
    }
    return figure


def complete_enrollment(user_name: str):
    """
    Complete the enrollment process.
    
    In full implementation, this finalizes with the backend.
    For now, we just show a success message.
    """
    if len(_enrollment_frames) < 3:
        return "⚠️ Not enough frames captured. Please try again.", None
    
    # Placeholder: In production, backend processes frames and returns point cloud
    status = f"✅ Enrollment complete for **{user_name}**!\n"
    status += f"Captured {len(_enrollment_frames)} frames.\n"
    status += f"Yaw range: {enrollment_guide.coverage.captured_yaw}\n"
    status += f"Pitch range: {enrollment_guide.coverage.captured_pitch}"
    
    # Create a placeholder point cloud visualization
    dummy_cloud = PointCloudData(
        points=np.random.randn(1000, 3).astype(np.float32) * 0.1,
        colors=(np.random.rand(1000, 3) * 255).astype(np.uint8),
    )
    point_cloud_fig = visualizer.create_plotly_figure(dummy_cloud, f"{user_name}'s Face Template")
    
    return status, point_cloud_fig


# ============================================================
# Authentication Tab Functions
# ============================================================

def authenticate(frame: Optional[np.ndarray], selected_user: Optional[str]):
    """
    Run authentication on current frame.
    
    In full implementation:
    1. Capture 2-4 frames
    2. Send to POST /authenticate endpoint
    3. Display results
    """
    if frame is None:
        return "⚠️ No camera feed available", None
    
    if not selected_user:
        return "⚠️ Please select a user to authenticate against", None
    
    # Placeholder: Simulate authentication result
    time.sleep(0.5)  # Simulate processing
    
    simulated_result = AuthResult(
        is_match=np.random.random() > 0.3,
        matched_user_id="usr_001",
        matched_user_name=selected_user,
        final_score=np.random.uniform(0.5, 0.95),
        geometric_score=np.random.uniform(0.4, 0.9),
        descriptor_score=np.random.uniform(0.5, 0.95),
        anti_spoof_passed=True,
        processing_time_sec=np.random.uniform(0.5, 2.0),
    )
    
    result_msg = auth_panel.format_result_message(simulated_result)
    
    # Create score visualization
    score_fig = create_score_visualization(simulated_result)
    
    return result_msg, score_fig


def create_score_visualization(result: AuthResult):
    """Create a gauge/bar chart for authentication scores."""
    scores = [
        ("Final", result.final_score),
        ("Geometric", result.geometric_score),
        ("Descriptor", result.descriptor_score),
    ]
    
    colors = [auth_panel.get_score_color(s) for _, s in scores]
    
    figure = {
        "data": [{
            "type": "bar",
            "x": [name for name, _ in scores],
            "y": [score for _, score in scores],
            "marker": {"color": colors},
            "text": [f"{score:.2f}" for _, score in scores],
            "textposition": "auto",
        }],
        "layout": {
            "title": "Authentication Scores",
            "yaxis": {"title": "Score", "range": [0, 1]},
            "width": 350,
            "height": 250,
            "shapes": [{
                "type": "line",
                "x0": -0.5, "x1": 2.5,
                "y0": 0.65, "y1": 0.65,
                "line": {"color": "red", "dash": "dash"},
            }],
            "annotations": [{
                "x": 2.5, "y": 0.65,
                "text": "Threshold",
                "showarrow": False,
                "font": {"size": 10},
            }],
        }
    }
    return figure


# ============================================================
# Build Gradio Interface
# ============================================================

def create_demo():
    """Create the Gradio demo interface."""
    
    with gr.Blocks(
        title="MASt3R Face Authentication",
    ) as demo:
        
        gr.Markdown("""
        # 🔐 MASt3R Face Authentication System
        
        A Face ID-like prototype using MASt3R 3D reconstruction for secure face authentication.
        
        **Status**: Demo mode (backend API not connected)
        """)
        
        with gr.Tabs():
            # ==================== ENROLLMENT TAB ====================
            with gr.TabItem("📝 Enrollment"):
                gr.Markdown("### Enroll a New User")
                gr.Markdown("Enter your name and follow the on-screen instructions to capture your face from multiple angles.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        user_name_input = gr.Textbox(
                            label="User Name",
                            placeholder="Enter your name...",
                            max_lines=1,
                        )
                        start_btn = gr.Button("▶️ Start Enrollment", variant="primary")
                        
                        webcam_feed = gr.Image(
                            label="Webcam Feed",
                            sources=["webcam"],
                            streaming=True,
                        )
                        
                        complete_btn = gr.Button("✅ Complete Enrollment", variant="secondary")
                    
                    with gr.Column(scale=1):
                        enrollment_status = gr.Markdown("Enter a name and click Start to begin enrollment.")
                        coverage_plot = gr.Plot(label="Pose Coverage")
                        result_cloud = gr.Plot(label="3D Face Preview")
                
                # Event handlers
                start_btn.click(
                    fn=start_enrollment,
                    inputs=[user_name_input],
                    outputs=[webcam_feed, enrollment_status, coverage_plot],
                )
                
                webcam_feed.stream(
                    fn=process_enrollment_frame,
                    inputs=[webcam_feed, user_name_input],
                    outputs=[webcam_feed, enrollment_status, coverage_plot],
                )
                
                complete_btn.click(
                    fn=complete_enrollment,
                    inputs=[user_name_input],
                    outputs=[enrollment_status, result_cloud],
                )
            
            # ==================== AUTHENTICATION TAB ====================
            with gr.TabItem("🔓 Authentication"):
                gr.Markdown("### Authenticate")
                gr.Markdown("Select a user and click Authenticate to verify your identity.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        # Placeholder user list (will be populated from API)
                        user_dropdown = gr.Dropdown(
                            label="Select User",
                            choices=["Alice", "Bob", "Charlie"],  # Placeholder
                            value=None,
                        )
                        
                        auth_webcam = gr.Image(
                            label="Camera Feed",
                            sources=["webcam"],
                            streaming=False,
                        )
                        
                        auth_btn = gr.Button("🔐 Authenticate", variant="primary")
                    
                    with gr.Column(scale=1):
                        auth_result = gr.Markdown("Select a user and capture your face to authenticate.")
                        score_plot = gr.Plot(label="Match Scores")
                
                auth_btn.click(
                    fn=authenticate,
                    inputs=[auth_webcam, user_dropdown],
                    outputs=[auth_result, score_plot],
                )
            
            # ==================== USERS TAB ====================
            with gr.TabItem("👥 Users"):
                gr.Markdown("### Enrolled Users")
                gr.Markdown("View and manage enrolled users.")
                
                refresh_btn = gr.Button("🔄 Refresh User List")
                user_table = gr.Dataframe(
                    headers=["User ID", "Name", "Enrolled At"],
                    datatype=["str", "str", "str"],
                    value=[
                        ["usr_001", "Alice", "2026-02-01 10:00"],
                        ["usr_002", "Bob", "2026-02-02 14:30"],
                    ],  # Placeholder data
                    interactive=False,
                )
                
                with gr.Row():
                    selected_user_id = gr.Textbox(label="User ID to Delete")
                    delete_btn = gr.Button("🗑️ Delete User", variant="stop")
                
                delete_status = gr.Markdown("")
            
            # ==================== VISUALIZATION TAB ====================
            with gr.TabItem("📊 3D Viewer"):
                gr.Markdown("### Point Cloud Visualization")
                gr.Markdown("View enrolled face templates as 3D point clouds.")
                
                viz_user_dropdown = gr.Dropdown(
                    label="Select User",
                    choices=["Alice", "Bob"],  # Placeholder
                )
                load_cloud_btn = gr.Button("Load Point Cloud")
                
                cloud_viewer = gr.Plot(label="3D Point Cloud")
                
                # Placeholder: load demo point cloud
                def load_demo_cloud(user_name):
                    if not user_name:
                        return visualizer.create_empty_figure("Select a user")
                    dummy = PointCloudData(
                        points=np.random.randn(2000, 3).astype(np.float32) * 0.1,
                        colors=(np.random.rand(2000, 3) * 255).astype(np.uint8),
                    )
                    return visualizer.create_plotly_figure(dummy, f"{user_name}'s Face")
                
                load_cloud_btn.click(
                    fn=load_demo_cloud,
                    inputs=[viz_user_dropdown],
                    outputs=[cloud_viewer],
                )
        
        gr.Markdown("""
        ---
        **Note**: This is a demo UI. Connect to the FastAPI backend for full functionality.
        
        CS-2 | MASt3R Face Authentication PBL Project
        """)
    
    return demo


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
