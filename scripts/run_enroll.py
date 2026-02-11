"""
One-Command Enrollment Pipeline: CMD -> WSL2 Orchestrator

Automates the two-terminal enrollment workflow:
  1. Keyframe capture with webcam (Windows CMD)
  2. MASt3R 3D reconstruction + ArcFace + template registration (WSL2 or API)
  3. Open 3D visualization in browser

Usage:
    # Activate Windows venv first
    mast3r-face-auth\\Scripts\\activate

    # API mode (fast -- requires server running on WSL2)
    python scripts/run_enroll.py --api --user-name "Alice"

    # Legacy WSL spawn (slower, no server needed)
    python scripts/run_enroll.py --user-name "Alice"

    # Custom keyframe directory
    python scripts/run_enroll.py --user-name "Alice" --keyframe-dir storage/demo_keyframes_v01

    # Skip capture (reuse existing keyframes)
    python scripts/run_enroll.py --api --user-name "Alice" --skip-capture

Author: CS-1
"""

import argparse
import base64
import glob
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
WSL_VENV_ACTIVATE = "source ~/mast3r-face-auth/bin/activate"
DEFAULT_API_URL = "http://localhost:8000"


def win_to_wsl(win_path: str) -> str:
    """Convert a Windows path to WSL /mnt/c/... equivalent."""
    p = Path(win_path).resolve()
    drive = p.drive  # e.g. "C:"
    if not drive:
        raise ValueError(f"Cannot convert path without drive letter: {win_path}")
    drive_letter = drive[0].lower()
    remainder = str(p)[len(drive):]  # e.g. "\\Users\\sekit\\..."
    return f"/mnt/{drive_letter}{remainder.replace(os.sep, '/')}"


def run_wsl(script_rel_path: str, args: list) -> subprocess.CompletedProcess:
    """Run a Python script inside WSL2 with venv activation.

    Args:
        script_rel_path: Script path relative to project root (e.g. "scripts/demo_enrollment.py").
        args: List of CLI arguments for the script.

    Returns:
        CompletedProcess with the WSL exit code.
    """
    wsl_project = win_to_wsl(str(PROJECT_ROOT))
    args_str = " ".join(shlex.quote(str(a)) for a in args)
    bash_cmd = (
        f"cd {shlex.quote(wsl_project)} && "
        f"{WSL_VENV_ACTIVATE} && "
        f"python {script_rel_path} {args_str}"
    )
    try:
        return subprocess.run(["wsl.exe", "bash", "-c", bash_cmd])
    except FileNotFoundError:
        print("\nERROR: wsl.exe not found. Is WSL2 installed?")
        sys.exit(1)


def validate_keyframes(keyframe_dir: Path, min_count: int = 2) -> tuple:
    """Validate that keyframes were exported successfully.

    Returns:
        (success: bool, message: str)
    """
    if not keyframe_dir.exists():
        return False, f"Directory does not exist: {keyframe_dir}"

    meta_path = keyframe_dir / "metadata.json"
    if not meta_path.exists():
        return False, f"metadata.json not found in {keyframe_dir}"

    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return False, f"Failed to read metadata.json: {e}"

    jpg_files = list(keyframe_dir.glob("keyframe_*.jpg"))
    if len(jpg_files) < min_count:
        return False, (
            f"Found {len(jpg_files)} keyframe(s), need at least {min_count}. "
            f"Capture may have been cancelled."
        )

    return True, f"Found {len(jpg_files)} keyframes (metadata count: {meta.get('count', '?')})"


def open_latest_html(output_dir: Path) -> str | None:
    """Find and open the latest face_3d_*.html file."""
    pattern = str(output_dir / "face_3d_*.html")
    html_files = glob.glob(pattern)
    if not html_files:
        return None
    html_files.sort(key=os.path.getmtime, reverse=True)
    latest = html_files[0]
    os.startfile(latest)
    return latest


def print_banner(text: str, char: str = "="):
    line = char * 60
    print(f"\n{line}")
    print(text)
    print(line)


def load_keyframes_for_api(keyframe_dir: Path):
    """Read keyframe JPEGs and metadata for API submission.

    Returns:
        (frames_b64, head_poses) where frames_b64 is list of base64 strings
        and head_poses is list of [yaw, pitch, roll].
    """
    meta_path = keyframe_dir / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    frames_b64 = []
    head_poses = []
    for i in range(meta["count"]):
        jpg_path = keyframe_dir / f"keyframe_{i:02d}.jpg"
        with open(jpg_path, "rb") as f:
            frames_b64.append(base64.b64encode(f.read()).decode("ascii"))
        head_poses.append(meta["poses"][i])

    return frames_b64, head_poses


def generate_visualization_html(point_cloud_preview_b64: str, n_points: int,
                                 output_dir: Path) -> str:
    """Decode point cloud preview from API and generate a Plotly HTML visualization.

    Args:
        point_cloud_preview_b64: Base64-encoded JSON with 'points' and 'colors'.
        n_points: Total point count (for title).
        output_dir: Directory to save the HTML file.

    Returns:
        Path to the generated HTML file.
    """
    preview = json.loads(base64.b64decode(point_cloud_preview_b64).decode())
    points = preview["points"]
    colors = preview.get("colors")

    # Generate color strings
    if colors:
        color_strs = [f'rgb({int(r)},{int(g)},{int(b)})' for r, g, b in colors]
    else:
        # Fallback: color by depth
        zs = [p[2] for p in points]
        z_min, z_max = min(zs), max(zs)
        z_range = z_max - z_min if z_max > z_min else 1e-6
        color_strs = [
            f'rgb({int(255*(p[2]-z_min)/z_range)},{int(100*(1-(p[2]-z_min)/z_range))},'
            f'{int(255*(1-(p[2]-z_min)/z_range))})'
            for p in points
        ]

    title = f"3D Face Reconstruction ({n_points:,} points)"
    figure = {
        "data": [{
            "type": "scatter3d",
            "x": [p[0] for p in points],
            "y": [p[1] for p in points],
            "z": [p[2] for p in points],
            "mode": "markers",
            "marker": {"size": 1.5, "color": color_strs},
            "hoverinfo": "skip",
        }],
        "layout": {
            "title": {"text": title, "font": {"size": 20}},
            "scene": {
                "aspectmode": "data",
                "xaxis": {"title": "X", "showgrid": True},
                "yaxis": {"title": "Y", "showgrid": True},
                "zaxis": {"title": "Z", "showgrid": True},
                "camera": {
                    "eye": {"x": 0, "y": 0, "z": -1.5},
                    "up": {"x": 0, "y": -1, "z": 0},
                },
            },
            "margin": {"l": 0, "r": 0, "t": 50, "b": 0},
            "width": 900, "height": 700,
            "paper_bgcolor": "#1a1a1a",
            "plot_bgcolor": "#1a1a1a",
            "font": {"color": "white"},
        }
    }

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            margin: 0;
            background-color: #1a1a1a;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }}
        #plot {{ width: 100%; max-width: 900px; }}
        .info {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            color: #888;
            font-family: sans-serif;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div id="plot"></div>
    <div class="info">
        Points: {len(points):,} (preview) | Total: {n_points:,} | Drag to rotate, scroll to zoom
    </div>
    <script>
        var figure = {json.dumps(figure)};
        Plotly.newPlot('plot', figure.data, figure.layout, {{responsive: true}});
    </script>
</body>
</html>"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    html_path = output_dir / f"face_3d_{timestamp}.html"
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    return str(html_path)


def run_api_enroll(api_url: str, user_name: str, keyframe_dir: Path,
                   output_dir: Path) -> int:
    """Send keyframes to the API server for enrollment and display results.

    Returns:
        0 for success, 1 for enrollment error, 2 for connection error.
    """
    print("  Encoding keyframes...")
    frames_b64, head_poses = load_keyframes_for_api(keyframe_dir)
    print(f"  Sending {len(frames_b64)} frames to {api_url}/enroll ...")

    payload = {
        "user_name": user_name,
        "frames": frames_b64,
        "head_poses": head_poses,
        "pre_cropped": True,
    }

    body = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{api_url}/enroll",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.time()
    try:
        with urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        reason = getattr(e, "reason", e)
        # Check for HTTP error (e.g. 409 Conflict for duplicate user)
        if hasattr(e, "code"):
            try:
                err_body = json.loads(e.read().decode("utf-8"))
                detail = err_body.get("detail", str(e))
            except Exception:
                detail = str(e)
            print(f"\n  ERROR: {detail}")
            return 1
        print(f"\n  ERROR: Cannot reach API server at {api_url}")
        print(f"  Reason: {reason}")
        print(f"\n  Start the server first on WSL2:")
        print(f"    source ~/mast3r-face-auth/bin/activate")
        print(f"    cd /mnt/c/Users/sekit/ai-visual-computing-pbl")
        print(f"    python -m api.app")
        return 2
    except Exception as e:
        print(f"\n  ERROR: API request failed: {e}")
        return 2

    elapsed = time.time() - start

    # Extract response fields
    user_id = result.get("user_id", "unknown")
    n_points = result.get("n_points", 0)
    n_frames_used = result.get("n_frames_used", 0)
    recon_time = result.get("reconstruction_time_sec", 0.0)
    point_cloud_preview = result.get("point_cloud_preview")

    # Display results matching demo_enrollment.py output format
    print(f"\n" + "=" * 60)
    print("PHASE 2: 3D Reconstruction with MASt3R")
    print("=" * 60)
    print(f"Running reconstruction with {n_frames_used} frames...")
    print(f"Reconstruction complete in {recon_time:.1f} seconds!")
    print(f"  Points: {n_points:,}")
    print(f"  ArcFace embedding: extracted")
    print(f"\nTemplate saved on server")
    print(f"  User: {user_name} ({user_id})")

    # Phase 3: Generate and open visualization from preview
    html_path = None
    if point_cloud_preview:
        print(f"\n" + "=" * 60)
        print("PHASE 3: Visualization")
        print("=" * 60)
        try:
            html_path = generate_visualization_html(
                point_cloud_preview, n_points, output_dir
            )
            print(f"Opening visualization in browser...")
            print(f"  File: {html_path}")
            os.startfile(html_path)
            print(f"  Browser opened!")
        except Exception as e:
            print(f"  WARNING: Failed to generate visualization: {e}")

    # Final summary
    print(f"\n" + "=" * 60)
    print("ENROLLMENT COMPLETE!")
    print("=" * 60)
    print(f"Total keyframes: {n_frames_used}")
    print(f"Total 3D points: {n_points:,}")
    print(f"ArcFace embedding: extracted")
    print(f"User: {user_name} ({user_id})")
    if html_path:
        print(f"Visualization: {html_path}")
    print(f"Total time: {elapsed:.1f}s (including network)")
    print()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-command enrollment: capture (Windows) + reconstruct (WSL2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--user-name", type=str, required=True,
        help="Name for the enrolled user (required)",
    )
    parser.add_argument(
        "--keyframe-dir", type=str, default="storage/demo_keyframes",
        help="Directory for keyframe export (default: storage/demo_keyframes)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for HTML visualization (default: ~/Desktop)",
    )
    parser.add_argument(
        "--resolution", type=str, default="1280x720",
        help="Webcam resolution WxH (default: 1280x720)",
    )
    parser.add_argument(
        "--skip-capture", action="store_true",
        help="Skip webcam capture, use existing keyframes in --keyframe-dir",
    )
    parser.add_argument(
        "--api", action="store_true",
        help="Use API server for inference (fast, requires server running on WSL2)",
    )
    parser.add_argument(
        "--api-url", type=str, default=DEFAULT_API_URL,
        help=f"API server URL (default: {DEFAULT_API_URL})",
    )
    args = parser.parse_args()

    # Resolve paths
    keyframe_dir = (PROJECT_ROOT / args.keyframe_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path.home() / "Desktop"

    # ── Phase 1: Capture (Windows) ──────────────────────────────
    if not args.skip_capture:
        print_banner("PHASE 1: Keyframe Capture (Windows)")
        print(f"  Export dir:  {keyframe_dir}")
        print(f"  Resolution:  {args.resolution}")
        print()

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_face_detector.py"),
             "--export-dir", str(keyframe_dir),
             "--resolution", args.resolution],
            cwd=str(PROJECT_ROOT),
        )
        # demo_face_detector.py always exits 0, so validate output
        ok, msg = validate_keyframes(keyframe_dir)
        if not ok:
            print(f"\nERROR: {msg}")
            return 1
        print(f"\nCapture OK: {msg}")
    else:
        print_banner("PHASE 1: Using existing keyframes (--skip-capture)")
        ok, msg = validate_keyframes(keyframe_dir)
        if not ok:
            print(f"\nERROR: {msg}")
            return 1
        print(f"  {msg}")

    # ── Phase 2 + 3: Enrollment ─────────────────────────────────
    backend = "API server" if args.api else "WSL2"

    if args.api:
        # Fast path: send to running API server
        print_banner("PHASE 2: 3D Reconstruction + Registration (API Server)")
        print(f"  Server:      {args.api_url}")
        print(f"  Keyframes:   {keyframe_dir}")
        print(f"  User name:   {args.user_name}")
        print(f"  Output dir:  {output_dir}")
        print()

        rc = run_api_enroll(args.api_url, args.user_name, keyframe_dir, output_dir)
    else:
        # Legacy path: spawn WSL2 process
        print_banner("PHASE 2: 3D Reconstruction + Registration (WSL2)")
        print(f"  Keyframes:   {keyframe_dir}")
        print(f"  User name:   {args.user_name}")
        print(f"  Output dir:  {output_dir}")
        print()

        wsl_args = [
            "--skip-capture",
            "--keyframe-dir", win_to_wsl(str(keyframe_dir)),
            "--user-name", args.user_name,
            "--output-dir", win_to_wsl(str(output_dir)),
        ]
        result = run_wsl("scripts/demo_enrollment.py", wsl_args)

        if result.returncode != 0:
            print(f"\nERROR: Enrollment failed (exit code {result.returncode}).")
            return 1

        # Open visualization (WSL mode saves HTML to output_dir)
        print_banner("PHASE 3: Opening 3D Visualization")
        html_path = open_latest_html(output_dir)
        if html_path:
            print(f"  Opened: {html_path}")
        else:
            print(f"  WARNING: No face_3d_*.html found in {output_dir}")

        rc = 0

    if rc == 0:
        print_banner("ENROLLMENT COMPLETE")
    else:
        print_banner(f"ENROLLMENT FAILED (exit code {rc})")

    return rc


if __name__ == "__main__":
    sys.exit(main())
