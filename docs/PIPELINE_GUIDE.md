# End-to-End Pipeline Guide

This document describes the complete face authentication pipeline, from keyframe capture through 3D reconstruction, ArcFace embedding, enrollment, and authentication.

## Environment Overview

The pipeline spans two environments due to hardware constraints:

| Environment | Purpose | Key Dependency |
|-------------|---------|----------------|
| **Windows (cmd)** | Webcam capture (MediaPipe face detection) | Webcam access, `mast3r-face-auth` venv |
| **WSL2 (bash)** | GPU-accelerated inference (MASt3R, ArcFace, matching) | NVIDIA GPU, deadsnakes Python 3.11 venv |

> WSL2 cannot access the webcam. All capture steps run on Windows; all GPU inference steps run on WSL2. The shared filesystem (`/mnt/c/...`) bridges the two.

### Activating the Virtual Environment

```bash
# Windows (cmd)
mast3r-face-auth\Scripts\activate

# WSL2 (bash)
source ~/mast3r-face-auth/bin/activate
```

---

## Stage 1: Keyframe Capture (Windows)

Captures face keyframes with real-time pose guidance and angular coverage tracking.

```cmd
:: Windows cmd (with venv activated)
python scripts/demo_face_detector.py
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--export-dir` | from `config.yaml` | Directory to save exported keyframes |
| `--resolution` | `640x480` | Webcam resolution `WxH` (e.g. `1280x720`) |

> **Recommendation**: On CS-1's laptop, `--resolution 1280x720` produced the best 3D reconstruction quality.

### Controls

| Key | Action |
|-----|--------|
| `e` | Export captured keyframes to disk |
| `SPACE` | Manually trigger keyframe capture |
| `q` | Quit |

### What Happens

1. Opens webcam at the specified resolution.
2. MediaPipe detects faces and extracts 468 landmarks per frame.
3. Computes head pose (yaw, pitch, roll) and blur score.
4. Auto-captures keyframes when a novel pose is detected (angular distance from previous captures exceeds threshold).
5. Displays pose guidance overlay showing which directions still need coverage.
6. On `e`, saves keyframes as `keyframe_00.jpg`, `keyframe_01.jpg`, ... plus `head_poses.json` and `metadata.json`.

### Output

```
storage/demo_keyframes/
  keyframe_00.jpg
  keyframe_01.jpg
  ...
  head_poses.json     # [{"yaw": ..., "pitch": ..., "roll": ...}, ...]
  metadata.json       # capture parameters and face detection config
```

---

## Stage 2: Enrollment (WSL2)

Runs MASt3R 3D reconstruction on captured keyframes, extracts ArcFace embeddings, and registers the user as a face template.

```bash
# WSL2 bash (with venv activated)
python scripts/demo_enrollment.py --skip-capture \
  --keyframe-dir /mnt/c/Users/sekit/ai-visual-computing-pbl/storage/demo_keyframes \
  --user-name "Alice" \
  --output-dir /mnt/c/Users/sekit/Desktop
```

> `--user-name` triggers template registration. Without it, the script only performs reconstruction + visualization (useful for checking 3D quality without registering).

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--min-keyframes` | `8` | Minimum keyframes required to start reconstruction |
| `--skip-capture` | off | Load keyframes from disk instead of opening webcam |
| `--keyframe-dir` | `storage/demo_keyframes` | Directory to load keyframes from (with `--skip-capture`) |
| `--save-keyframes` | off | Save captured keyframes for later use |
| `--output-dir` | (temp dir) | Directory to save visualization HTML (timestamped) |
| `--user-name` | (none) | User name for template registration. If provided, saves `.npz` to `storage/templates/` |

### What Happens

1. **Load keyframes**: Reads `keyframe_*.jpg` and `metadata.json` from `--keyframe-dir`.
2. **Face detection & cropping**: Detects faces, crops with `face_padding=0.3` and mean-color padding.
3. **Pose-aware pairing**: Pairs each frame with K angularly-nearest neighbors (K=3..8) based on head pose, avoiding useless far-angle pairs.
4. **MASt3R inference**: Runs sparse global alignment to produce a 3D point cloud with per-point descriptors and confidence.
5. **Post-processing**: Confidence filtering (conf >= 1.5), voxel deduplication (10mm), largest-cluster extraction, outlier removal.
6. **ArcFace embedding**: Extracts a 512-dim L2-normalized identity embedding from the face crops (insightface `buffalo_l`).
7. **Template save** (if `--user-name` given): Writes `.npz` file to `storage/templates/` and registers in the database.
8. **Visualization**: Generates an interactive 3D Plotly HTML and opens it in the browser.

### Output

When `--user-name` is provided:

```
storage/templates/
  usr_<id>.npz              # face template (v2, with face_embedding)
```

The `.npz` template contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `point_cloud` | `(N, 3)` | 3D point positions (meters) |
| `descriptors` | `(N, 24)` | MASt3R per-point descriptors |
| `colors` | `(N, 3)` | RGB colors |
| `confidence` | `(N,)` | MASt3R confidence values |
| `face_embedding` | `(512,)` | ArcFace L2-normalized embedding |
| `enrollment_metadata` | dict | Frame count, yaw range, processing time, etc. |

---

## Stage 3: Authentication

Authentication is a two-step process across Windows and WSL2, mirroring the enrollment workflow.

### Step 3a: Capture Auth Frames (Windows)

```cmd
:: Windows cmd (with venv activated)
python scripts/demo_auth.py --capture-only --output-dir storage/auth_keyframes
```

The `--capture-only` flag captures frames and saves them, then exits without running matching (since Windows lacks the GPU environment for MASt3R).

#### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--user-id` | (none) | User ID for 1:1 verification; omit for 1:N identification |
| `--num-frames` | `4` | Number of frames to capture |
| `--skip-capture` | off | Load saved keyframes instead of opening webcam |
| `--keyframe-dir` | `storage/demo_keyframes` | Directory for loading keyframes (with `--skip-capture`) |
| `--output-dir` | (none) | Directory to save captured frames |
| `--capture-only` | off | Capture and save frames, then exit (no GPU inference) |

#### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Manually capture a frame |
| `ENTER` | Finish capture (proceed to auth or exit) |
| `q` | Cancel |

Auto-capture triggers approximately every 1 second when a face is detected.

### Step 3b: Run Authentication (WSL2)

```bash
# WSL2 bash (with venv activated)
# 1:N identification (match against all enrolled users)
python scripts/demo_auth.py --skip-capture \
  --keyframe-dir /mnt/c/Users/sekit/ai-visual-computing-pbl/storage/auth_keyframes

# 1:1 verification (match against a specific user)
python scripts/demo_auth.py --skip-capture \
  --keyframe-dir /mnt/c/Users/sekit/ai-visual-computing-pbl/storage/auth_keyframes \
  --user-id usr_abc123
```

### What Happens

1. **Load frames**: Reads `keyframe_*.jpg` from `--keyframe-dir`.
2. **Face detection & cropping**: Same as enrollment (padding, BGR crops retained for ArcFace).
3. **MASt3R reconstruction**: Builds a probe 3D point cloud from the auth frames.
4. **ArcFace embedding**: Extracts a probe embedding from the face crops.
5. **Anti-spoofing check**: Verifies depth variance and planarity ratio of the reconstructed face. Rejects flat (printed/screen) inputs.
6. **Template matching**: For each enrolled template, computes three scores:
   - **Embedding score**: ArcFace cosine similarity (512-dim identity vector)
   - **Geometric score**: ICP alignment + Chamfer distance on 3D point clouds
   - **Descriptor score**: Reciprocal nearest-neighbor matching on MASt3R descriptors
7. **Score fusion**: Weighted combination of the three scores (see below).
8. **Decision**: Accept if fused score >= `accept_threshold` (0.65).

### Fusion Weights

Current defaults (tuned on public dataset evaluation):

| Channel | Weight | Description |
|---------|--------|-------------|
| Embedding (ArcFace) | **0.40** | Primary identity signal (512-dim) |
| Descriptor (MASt3R) | **0.50** | View-correspondence features (24-dim per point) |
| Geometric (ICP) | **0.10** | 3D shape similarity |

These weights are configured in `config.yaml` under `matching:` and can be re-tuned using the evaluation notebook.

### Output

The script prints a formatted results table:

```
============================================================
AUTHENTICATION RESULT
============================================================
  Decision:         MATCH
  Matched user:     Alice (usr_abc123)

  Final score:      0.782  (threshold: 0.65)
  Embedding score:  0.891  (weight: 0.40)
  Geometric score:  0.634  (weight: 0.10)
  Descriptor score: 0.705  (weight: 0.50)

  Anti-spoofing:    PASSED
    Depth variance: 0.0042
    Planarity:      0.23

  Processing time:  6.2s
============================================================
```

Exit code: `0` = match, `1` = no match.

---

## Stage 4: Evaluation (Google Colab)

The evaluation notebook runs all-vs-all matching on pre-computed `.npz` templates and computes biometric metrics.

### Prerequisites

1. Upload enrollment `.npz` files to Google Drive: `face-auth-data/mast3r_outputs/archive/mast3r_outputs/`
2. Upload probe `.npz` files to: `face-auth-data/mast3r_outputs/archive/auth_probes/`
3. Ensure templates are v2 (contain `face_embedding`). If not, run `scripts/augment_npz_with_embeddings.py` first.

### Running

Open `notebooks/ds_evaluation_pipeline.ipynb` on Google Colab and run all cells in order.

### What the Notebook Does

| Cell | Purpose |
|------|---------|
| 1 | Install deps, mount Drive |
| 2 | Clone repo |
| 3 | Discover enrollment/probe `.npz` pairs by subject name |
| 4 | Inspect data format and check for ArcFace embeddings |
| 5 | Visualize enrollment and probe point clouds |
| 6 | Initialize matchers with default weights |
| 7-8 | Test single genuine pair and single impostor pair |
| 9 | All-vs-all matching (N x N score matrix) |
| 10 | Evaluate: accuracy, precision, recall, F1, FAR, TAR, EER, AUC |
| 11 | Per-channel score distributions (histogram) |
| 12 | Score matrix heatmap |
| 13 | Grid search for optimal fusion weights |
| 14 | Re-evaluate with optimal weights |
| 15 | Print recommended `config.yaml` values |

---

## Quick Reference: Command Sequence

```
[Windows cmd]
1. mast3r-face-auth\Scripts\activate
2. python scripts/demo_face_detector.py --resolution 1280x720
   → Press 'e' to export keyframes

[WSL2 bash]
3. source ~/mast3r-face-auth/bin/activate
4. python scripts/demo_enrollment.py --skip-capture \
     --keyframe-dir /mnt/c/Users/sekit/ai-visual-computing-pbl/storage/demo_keyframes \
     --user-name "Alice" \
     --output-dir /mnt/c/Users/sekit/Desktop

[Windows cmd]
5. python scripts/demo_auth.py --capture-only --output-dir storage/auth_keyframes

[WSL2 bash]
6. python scripts/demo_auth.py --skip-capture \
     --keyframe-dir /mnt/c/Users/sekit/ai-visual-computing-pbl/storage/auth_keyframes

[Google Colab]
7. Run notebooks/ds_evaluation_pipeline.ipynb
```

---

## Configuration Reference

Key parameters in `config.yaml`:

```yaml
face_detection:
  face_padding: 0.3               # padding around face crop (fraction of bbox)
  keyframe_blur_threshold: 15.0    # reject frames with blur score below this
  target_frames: 12                # target keyframe count for enrollment

matching:
  embedding_weight: 0.40           # ArcFace embedding weight
  geometric_weight: 0.10           # ICP + Chamfer weight
  descriptor_weight: 0.50          # MASt3R descriptor weight
  accept_threshold: 0.65           # accept/reject threshold

frontend:
  auth:
    capture_frames: 4              # number of auth frames to capture

face_embedding:
  model: "buffalo_l"               # insightface model bundle
  embedding_dim: 512
```
