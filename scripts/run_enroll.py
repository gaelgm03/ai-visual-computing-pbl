"""
One-Command Enrollment Pipeline: CMD -> WSL2 Orchestrator

Automates the two-terminal enrollment workflow:
  1. Keyframe capture with webcam (Windows CMD)
  2. MASt3R 3D reconstruction + ArcFace + template registration (WSL2)
  3. Open 3D visualization in browser

Usage:
    # Activate Windows venv first
    mast3r-face-auth\\Scripts\\activate

    # Full pipeline
    python scripts/run_enroll.py --user-name "Alice"

    # Custom keyframe directory
    python scripts/run_enroll.py --user-name "Alice" --keyframe-dir storage/demo_keyframes_v01

    # Skip capture (reuse existing keyframes)
    python scripts/run_enroll.py --user-name "Alice" --keyframe-dir storage/demo_keyframes --skip-capture

Author: CS-1
"""

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
WSL_VENV_ACTIVATE = "source ~/mast3r-face-auth/bin/activate"


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

    # ── Phase 2: Enrollment (WSL2) ──────────────────────────────
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

    # ── Phase 3: Open visualization ─────────────────────────────
    print_banner("PHASE 3: Opening 3D Visualization")
    html_path = open_latest_html(output_dir)
    if html_path:
        print(f"  Opened: {html_path}")
    else:
        print(f"  WARNING: No face_3d_*.html found in {output_dir}")

    print_banner("ENROLLMENT COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
