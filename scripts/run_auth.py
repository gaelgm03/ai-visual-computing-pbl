"""
One-Command Authentication Pipeline: CMD -> WSL2 Orchestrator

Automates the two-terminal authentication workflow:
  1. Auth frame capture with pose guidance (Windows CMD)
  2. MASt3R 3D reconstruction + ArcFace + template matching (WSL2)

Usage:
    # Activate Windows venv first
    mast3r-face-auth\\Scripts\\activate

    # 1:N identification (match against all enrolled users)
    python scripts/run_auth.py

    # 1:1 verification against a specific user
    python scripts/run_auth.py --user-id usr_abc123

    # Custom keyframe directory
    python scripts/run_auth.py --keyframe-dir storage/auth_keyframes_v02

    # Skip capture (reuse existing frames)
    python scripts/run_auth.py --skip-capture --keyframe-dir storage/auth_keyframes

Author: CS-1
"""

import argparse
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
        script_rel_path: Script path relative to project root (e.g. "scripts/demo_auth.py").
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


def print_banner(text: str, char: str = "="):
    line = char * 60
    print(f"\n{line}")
    print(text)
    print(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-command authentication: capture (Windows) + match (WSL2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--keyframe-dir", type=str, default="storage/auth_keyframes",
        help="Directory for auth frame capture/load (default: storage/auth_keyframes)",
    )
    parser.add_argument(
        "--user-id", type=str, default=None,
        help="User ID for 1:1 verification (omit for 1:N identification)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=5,
        help="Number of target poses to capture (default: 5, max: 5)",
    )
    parser.add_argument(
        "--resolution", type=str, default="1280x720",
        help="Webcam resolution WxH (default: 1280x720)",
    )
    parser.add_argument(
        "--timed-capture", action="store_true",
        help="Capture every 2s instead of pose-guided targeting (for anti-spoofing demo)",
    )
    parser.add_argument(
        "--skip-capture", action="store_true",
        help="Skip webcam capture, use existing keyframes in --keyframe-dir",
    )
    args = parser.parse_args()

    # Resolve paths
    keyframe_dir = (PROJECT_ROOT / args.keyframe_dir).resolve()

    mode = "1:1 verification" if args.user_id else "1:N identification"

    # ── Phase 1: Capture (Windows) ──────────────────────────────
    if not args.skip_capture:
        print_banner("PHASE 1: Auth Frame Capture (Windows)")
        print(f"  Output dir:   {keyframe_dir}")
        print(f"  Target poses: {args.num_frames}")
        print(f"  Mode:         {mode}")
        print()

        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "demo_auth.py"),
             "--capture-only",
             "--output-dir", str(keyframe_dir),
             "--num-frames", str(args.num_frames),
             "--resolution", args.resolution]
            + (["--timed-capture"] if args.timed_capture else []),
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print(f"\nERROR: Capture failed (exit code {result.returncode}).")
            return 1

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

    # ── Phase 2: Authentication (WSL2) ──────────────────────────
    print_banner("PHASE 2: Authentication Pipeline (WSL2)")
    print(f"  Keyframes: {keyframe_dir}")
    print(f"  Mode:      {mode}")
    if args.user_id:
        print(f"  Target:    {args.user_id}")
    print()

    wsl_args = [
        "--skip-capture",
        "--keyframe-dir", win_to_wsl(str(keyframe_dir)),
    ]
    if args.user_id:
        wsl_args.extend(["--user-id", args.user_id])

    result = run_wsl("scripts/demo_auth.py", wsl_args)

    # ── Summary ─────────────────────────────────────────────────
    if result.returncode == 0:
        print_banner("AUTHENTICATION: MATCH")
    elif result.returncode == 1:
        print_banner("AUTHENTICATION: NO MATCH")
    else:
        print_banner(f"AUTHENTICATION: ERROR (exit code {result.returncode})")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
