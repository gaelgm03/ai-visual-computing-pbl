"""
One-Command Authentication Pipeline: CMD -> WSL2 Orchestrator

Automates the two-terminal authentication workflow:
  1. Auth frame capture with pose guidance (Windows CMD)
  2. MASt3R 3D reconstruction + ArcFace + template matching (WSL2 or API)

Usage:
    # Activate Windows venv first
    mast3r-face-auth\\Scripts\\activate

    # API mode (fast -- requires server running on WSL2)
    python scripts/run_auth.py --api

    # Legacy WSL spawn (slower, no server needed)
    python scripts/run_auth.py

    # 1:1 verification
    python scripts/run_auth.py --api --user-id usr_abc123

    # Skip capture (reuse existing frames)
    python scripts/run_auth.py --api --skip-capture

Author: CS-1
"""

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import time
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
    """Run a Python script inside WSL2 with venv activation."""
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
    """Validate that keyframes were exported successfully."""
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


def run_api_auth(api_url: str, keyframe_dir: Path, user_id: str = None) -> int:
    """Send keyframes to the API server and display results.

    Returns:
        0 for MATCH, 1 for NO MATCH, 2 for error.
    """
    print("  Encoding keyframes...")
    frames_b64, head_poses = load_keyframes_for_api(keyframe_dir)
    print(f"  Sending {len(frames_b64)} frames to {api_url}/authenticate ...")

    payload = {
        "frames": frames_b64,
        "head_poses": head_poses,
        "pre_cropped": True,
    }
    if user_id:
        payload["user_id"] = user_id

    body = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{api_url}/authenticate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.time()
    try:
        with urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        reason = getattr(e, "reason", e)
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

    # Display results in same format as demo_auth.py
    is_match = result.get("is_match", False)
    matched_name = result.get("matched_user_name")
    matched_id = result.get("matched_user_id")
    final_score = result.get("final_score", 0.0)
    emb_score = result.get("embedding_score", 0.0)
    geo_score = result.get("geometric_score", 0.0)
    desc_score = result.get("descriptor_score", 0.0)
    anti_spoof = result.get("anti_spoof", {})
    n_probe_points = result.get("n_probe_points", 0)
    recon_time = result.get("reconstruction_time_sec", 0.0)
    n_templates = result.get("n_templates", 0)
    all_scores = result.get("all_scores", [])

    # Step-by-step summary (matching demo_auth.py output)
    print(f"\n[1/5] MASt3R 3D Reconstruction...")
    print(f"  Reconstruction: {n_probe_points:,} points in {recon_time:.1f}s")

    print(f"\n[2/5] ArcFace Embedding Extraction...")
    print(f"  Embedding: extracted")

    print(f"\n[3/5] Anti-Spoofing Check...")
    print(f"  Passed: {anti_spoof.get('passed', False)}")
    if anti_spoof.get("depth_variance") is not None:
        print(f"  Depth variance: {anti_spoof['depth_variance']:.6f}")
    if anti_spoof.get("planarity_ratio") is not None:
        print(f"  Planarity ratio: {anti_spoof['planarity_ratio']:.6f}")
    if anti_spoof.get("confidence_mean") is not None:
        print(f"  Confidence mean: {anti_spoof['confidence_mean']:.4f}")

    print(f"\n[4/5] Loading Templates...")
    print(f"  Loaded {n_templates} template(s)")

    print(f"\n[5/5] Running Matchers...")
    for s in all_scores:
        label = "MATCH" if s["is_match"] else "NO MATCH"
        print(f"  vs {s['user_name']} (id={s['user_id']}): "
              f"emb={s['embedding_score']:.3f}, geo={s['geometric_score']:.3f}, "
              f"desc={s['descriptor_score']:.3f}, fused={s['fused_score']:.3f} {label}")

    # Final results table
    print()
    print("=" * 60)
    print("AUTHENTICATION RESULTS")
    print("=" * 60)
    if is_match:
        print(f"  Decision:     MATCH")
        print(f"  Matched User: {matched_name} ({matched_id})")
    else:
        print(f"  Decision:     NO MATCH")
    print()
    print(f"  Final Score:      {final_score:.4f}  {'>' if is_match else '<'} 0.65 (threshold)")
    print(f"  Embedding Score:  {emb_score:.4f}  (weight: 0.4)")
    print(f"  Geometric Score:  {geo_score:.4f}  (weight: 0.1)")
    print(f"  Descriptor Score: {desc_score:.4f}  (weight: 0.5)")
    print()
    print(f"  Anti-Spoof:       {'PASSED' if anti_spoof.get('passed') else 'FAILED'}")
    if anti_spoof.get("depth_variance") is not None:
        print(f"  Depth Variance:   {anti_spoof['depth_variance']:.6f}")
    if anti_spoof.get("planarity_ratio") is not None:
        print(f"  Planarity Ratio:  {anti_spoof['planarity_ratio']:.6f}")
    print()
    print(f"  Probe Points:     {n_probe_points:,}")
    print(f"  Recon Time:       {recon_time:.1f}s")
    print("=" * 60)

    return 0 if is_match else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-command authentication: capture (Windows) + match (WSL2 or API)",
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

    mode = "1:1 verification" if args.user_id else "1:N identification"
    backend = "API server" if args.api else "WSL2"

    # ── Phase 1: Capture (Windows) ──────────────────────────────
    if not args.skip_capture:
        print_banner("PHASE 1: Auth Frame Capture (Windows)")
        print(f"  Output dir:   {keyframe_dir}")
        print(f"  Target poses: {args.num_frames}")
        print(f"  Mode:         {mode}")
        print(f"  Backend:      {backend}")
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

    # ── Phase 2: Authentication ─────────────────────────────────
    if args.api:
        # Fast path: send to running API server
        print_banner("PHASE 2: Authentication via API Server")
        print(f"  Server:    {args.api_url}")
        print(f"  Keyframes: {keyframe_dir}")
        print(f"  Mode:      {mode}")
        if args.user_id:
            print(f"  Target:    {args.user_id}")
        print()

        rc = run_api_auth(args.api_url, keyframe_dir, args.user_id)
    else:
        # Legacy path: spawn WSL2 process
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
        rc = result.returncode

    # ── Summary ─────────────────────────────────────────────────
    if rc == 0:
        print_banner("AUTHENTICATION: MATCH")
    elif rc == 1:
        print_banner("AUTHENTICATION: NO MATCH")
    else:
        print_banner(f"AUTHENTICATION: ERROR (exit code {rc})")

    return rc


if __name__ == "__main__":
    sys.exit(main())
