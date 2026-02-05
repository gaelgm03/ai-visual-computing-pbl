"""
Gradio-based demo UI for MASt3R Face Authentication System.
CS-2 Primary Ownership.

This is the main entry point for the frontend application.
Run with: python -m frontend.app_gradio
"""

import gradio as gr
import numpy as np
import cv2
from typing import Optional, Tuple, List
import time

from frontend.components.webcam_capture import WebcamCapture, CaptureConfig
from frontend.components.enrollment_guide import EnrollmentGuide, HeadPose, EnrollmentConfig
from frontend.components.auth_panel import AuthPanel, AuthResult, AuthConfig
from frontend.components.visualization import PointCloudVisualizer, PointCloudData
from frontend.api_client import get_api_client, ConnectionMode


# ============================================================
# Global State
# ============================================================
webcam = WebcamCapture(CaptureConfig(width=640, height=480))
enrollment_guide = EnrollmentGuide(EnrollmentConfig(target_frames=12))
auth_panel = AuthPanel(AuthConfig())
visualizer = PointCloudVisualizer(max_points=10000)
api_client = get_api_client()

# Enrollment state
_enrollment_active = False
_enrollment_user = ""
_enrollment_poses: List[HeadPose] = []
_last_frame_result = None


# ============================================================
# Enrollment Tab Functions
# ============================================================

def start_enrollment(user_name: str):
    """Initialize enrollment session."""
    global _enrollment_active, _enrollment_user, _enrollment_poses, _last_frame_result
    
    if not user_name or not user_name.strip():
        return None, "⚠️ Please enter a user name", create_coverage_plot(), create_live_cloud_plot()
    
    # Reset state
    enrollment_guide.reset()
    _enrollment_poses.clear()
    _enrollment_active = True
    _enrollment_user = user_name.strip()
    _last_frame_result = None
    
    # Start API session
    api_client.start_enrollment(_enrollment_user)
    
    status = f"""## 📷 Enrollment Started for **{_enrollment_user}**
    
**Instructions:**
1. Look at the camera
2. Slowly turn your head following the arrows
3. Cover all directions (left, right, up, down)
4. Wait for 12 keyframes to be captured
"""
    
    return None, status, create_coverage_plot(), create_live_cloud_plot()


def process_enrollment_frame(frame: Optional[np.ndarray], user_name: str):
    """
    Process a frame during enrollment via API client.
    """
    global _last_frame_result
    
    if frame is None or not _enrollment_active:
        return None, enrollment_guide.format_status_message(), create_coverage_plot()
    
    # Send frame to API (mock or live)
    result = api_client.process_enrollment_frame(frame)
    _last_frame_result = result
    
    # Extract pose from result
    pose = HeadPose(
        yaw=result.head_pose.get("yaw", 0) if result.head_pose else 0,
        pitch=result.head_pose.get("pitch", 0) if result.head_pose else 0,
        roll=result.head_pose.get("roll", 0) if result.head_pose else 0,
    )
    
    # Track poses for keyframes
    if result.is_keyframe:
        _enrollment_poses.append(pose)
    
    # Update coverage
    enrollment_guide.update_coverage(_enrollment_poses)
    
    # Draw overlay on frame
    annotated_frame = draw_enrollment_overlay(
        frame, 
        pose, 
        result.face_detected,
        result.face_bbox,
        result.is_keyframe,
        result.quality_score,
    )
    
    status_msg = format_enrollment_status(result)
    coverage_plot = create_coverage_plot()
    
    return annotated_frame, status_msg, coverage_plot


def format_enrollment_status(result) -> str:
    """Format enrollment status message based on API result."""
    keyframes = len(_enrollment_poses)
    
    if not result.face_detected:
        return f"""## ⚠️ No Face Detected

**Keyframes:** {keyframes}/12

Please position your face in the camera view."""
    
    direction = enrollment_guide.get_next_direction()
    direction_text = f"Turn **{direction}**" if direction and direction != "center" else "Hold steady"
    
    quality_bar = "█" * int(result.quality_score * 10) + "░" * (10 - int(result.quality_score * 10))
    
    status = f"""## 📷 Enrollment in Progress

**Keyframes:** {keyframes}/12 {'✅' if keyframes >= 12 else ''}
**Quality:** [{quality_bar}] {result.quality_score:.0%}
**Direction:** {direction_text}
"""
    
    if result.is_keyframe:
        status += "\n✨ **Keyframe captured!**"
    
    return status


def draw_enrollment_overlay(
    frame: np.ndarray,
    pose: HeadPose,
    face_detected: bool,
    face_bbox: Optional[tuple],
    is_keyframe: bool,
    quality_score: float,
) -> np.ndarray:
    """Draw enrollment guidance overlay on the frame with improved visuals."""
    frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw face bounding box if detected
    if face_detected and face_bbox:
        x, y, bw, bh = face_bbox
        color = (0, 255, 0) if quality_score > 0.8 else (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        
        # Draw quality indicator bar above bbox
        bar_width = int(bw * quality_score)
        cv2.rectangle(frame, (x, y - 10), (x + bar_width, y - 5), color, -1)
        cv2.rectangle(frame, (x, y - 10), (x + bw, y - 5), color, 1)
    
    # Draw semi-transparent info panel at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Draw pose info
    pose_text = f"Yaw: {pose.yaw:+6.1f}  Pitch: {pose.pitch:+5.1f}"
    cv2.putText(frame, pose_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Draw capture count with progress bar
    keyframes = len(_enrollment_poses)
    count_text = f"Keyframes: {keyframes}/12"
    cv2.putText(frame, count_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Progress bar
    progress_width = int((keyframes / 12) * 150)
    cv2.rectangle(frame, (150, 40), (300, 55), (100, 100, 100), -1)
    cv2.rectangle(frame, (150, 40), (150 + progress_width, 55), (0, 255, 0), -1)
    
    # Flash green border when keyframe captured
    if is_keyframe:
        cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 255, 0), 8)
        cv2.putText(frame, "CAPTURED!", (w//2 - 80, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3, cv2.LINE_AA)
    
    # Draw direction arrow if needed
    direction = enrollment_guide.get_next_direction()
    if face_detected and direction and direction != "center":
        draw_direction_arrow(frame, direction)
    elif not face_detected:
        # Draw "position face" message
        cv2.putText(frame, "Position face in frame", (w//2 - 120, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
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


def create_live_cloud_plot():
    """Create live 3D point cloud visualization during enrollment."""
    points, colors = api_client.get_enrollment_cloud()
    
    if points is None or len(points) == 0:
        return visualizer.create_empty_figure("Point cloud will appear here")
    
    data = PointCloudData(points=points, colors=colors)
    return visualizer.create_plotly_figure(data, f"Live Preview ({len(points)} pts)")


def complete_enrollment(user_name: str):
    """
    Complete the enrollment process via API client.
    """
    global _enrollment_active
    
    keyframes = len(_enrollment_poses)
    if keyframes < 3:
        return "⚠️ Not enough keyframes captured. Please try again.", None
    
    # Finalize with API
    result = api_client.complete_enrollment()
    _enrollment_active = False
    
    if not result.get("success"):
        return f"❌ Enrollment failed: {result.get('error', 'Unknown error')}", None
    
    # Format success message
    status = f"""## ✅ Enrollment Complete!

**User:** {result.get('user_name', user_name)}
**User ID:** `{result.get('user_id', 'N/A')}`
**Keyframes:** {result.get('keyframes_count', keyframes)}
**Points:** {result.get('point_count', 'N/A'):,}

Template saved to: `{result.get('template_path', 'storage/templates/')}`
"""
    
    # Get final point cloud
    points, colors = api_client.get_enrollment_cloud()
    if points is not None:
        data = PointCloudData(points=points, colors=colors)
        point_cloud_fig = visualizer.create_plotly_figure(data, f"{user_name}'s Face Template")
    else:
        point_cloud_fig = visualizer.create_empty_figure("Template created")
    
    return status, point_cloud_fig


# ============================================================
# Authentication Tab Functions
# ============================================================

def authenticate(frame: Optional[np.ndarray], selected_user: Optional[str]):
    """
    Run authentication via API client.
    """
    if frame is None:
        return "⚠️ No camera feed available", None
    
    if not selected_user:
        return "⚠️ Please select a user to authenticate against", None
    
    # Send to API (mock or live)
    api_result = api_client.authenticate([frame], target_user=selected_user)
    
    # Convert API response to AuthResult for display
    result = AuthResult(
        is_match=api_result.is_match,
        matched_user_id=api_result.matched_user_id,
        matched_user_name=api_result.matched_user_name,
        final_score=api_result.final_score,
        geometric_score=api_result.geometric_score,
        descriptor_score=api_result.descriptor_score,
        anti_spoof_passed=api_result.anti_spoof_passed,
        processing_time_sec=api_result.processing_time_ms / 1000,
    )
    
    result_msg = auth_panel.format_result_message(result)
    
    # Create score visualization
    score_fig = create_score_visualization(result)
    
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
                    outputs=[webcam_feed, enrollment_status, coverage_plot, result_cloud],
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
