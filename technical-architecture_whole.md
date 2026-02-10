# MASt3R-Based Face Authentication System — Technical Architecture

> **Audience**: CS-1 (Infrastructure & Integration Lead) and CS-2 (UI/UX & Demo Pipeline Lead)
> **Version**: 2.0 | Draft
> **Last Updated**: 2026-02-10
>
> **Changelog**:
> - v2.0 (2026-02-10): Added ArcFace face recognition integration (§2, §3, §4.4–4.6, §5, §6.2, §8.2, §9.1, §13). MASt3R descriptors replaced by ArcFace embeddings as primary identity signal; MASt3R retained for 3D reconstruction and anti-spoofing.
> - v1.0 (2026-02-05): Initial architecture document.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Technology Stack](#2-technology-stack)
3. [Repository Structure](#3-repository-structure)
4. [Module Architecture](#4-module-architecture)
5. [Data Flow & Pipeline Design](#5-data-flow--pipeline-design)
6. [Interface Contracts Between CS-1 and CS-2](#6-interface-contracts-between-cs-1-and-cs-2)
7. [Enrollment Pipeline (Detailed)](#7-enrollment-pipeline-detailed)
8. [Authentication Pipeline (Detailed)](#8-authentication-pipeline-detailed)
9. [Template Storage Schema](#9-template-storage-schema)
10. [Anti-Spoofing Module](#10-anti-spoofing-module)
11. [GPU & Memory Management](#11-gpu--memory-management)
12. [API Specification](#12-api-specification)
13. [Configuration Management](#13-configuration-management)
14. [Development Workflow & Task Ownership](#14-development-workflow--task-ownership)
15. [Testing Strategy](#15-testing-strategy)
16. [Known Constraints & Risks](#16-known-constraints--risks)

---

## 1. System Overview

### 1.1 What We Are Building

A Face ID-like prototype that performs real-time face enrollment and authentication using **MASt3R** (Matching And Stereo 3D Reconstruction) for 3D face reconstruction and matching — all from a standard RGB webcam, with no depth sensor required.

### 1.2 High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Frontend (CS-2)                               │
│  ┌─────────────┐  ┌──────────────────┐  ┌────────────────────────┐  │
│  │ Webcam Feed  │  │ Enrollment Guide │  │ Result Visualization   │  │
│  │ (live view)  │  │ (head rotation   │  │ (3D pointcloud, match  │  │
│  │              │  │  prompts + face   │  │  overlay, score gauge) │  │
│  │              │  │  coverage meter)  │  │                        │  │
│  └──────┬───────┘  └────────┬─────────┘  └───────────▲────────────┘  │
│         │                   │                        │               │
│─────────┼───────────────────┼────────────────────────┼───────────────│
│         ▼                   ▼                        │               │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                    Backend API Layer                          │    │
│  │              (FastAPI — WebSocket + REST)                     │    │
│  └──────┬──────────────────┬───────────────────────┬────────────┘    │
│─────────┼──────────────────┼───────────────────────┼─────────────────│
│                        Core Engine (CS-1)                            │
│         ▼                  ▼                       ▼                 │
│  ┌─────────────┐  ┌───────────────┐  ┌──────────────────────┐       │
│  │ Face        │  │ MASt3R        │  │ Template Manager     │       │
│  │ Detector    │  │ Inference     │  │ (save/load/delete)   │       │
│  │ + Keyframe  │  │ Engine        │  │                      │       │
│  │ Selector    │  │               │  │                      │       │
│  └─────────────┘  └───────────────┘  └──────────────────────┘       │
│                           │                                          │
│                   ┌───────▼────────┐                                 │
│                   │ Matching &     │  ← DS team implements internals │
│                   │ Scoring Engine │  ← CS-1 provides the interface  │
│                   └────────────────┘                                 │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.3 Responsibility Boundary (CS-1 vs. CS-2)

```
           CS-1 Owns                    CS-2 Owns
    ┌───────────────────┐        ┌───────────────────┐
    │ core/              │        │ frontend/          │
    │   face_detector    │        │   webcam_capture   │
    │   mast3r_engine    │        │   enrollment_ui    │
    │   template_manager │        │   auth_ui          │
    │   anti_spoof       │        │   visualization    │
    │ api/               │        │   result_display   │
    │   routes & schemas │        │                    │
    └───────────────────┘        └───────────────────┘

    Shared: api/ layer (CS-1 defines endpoints, CS-2 consumes them)
```

---

## 2. Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| **3D Reconstruction** | MASt3R (ViTLarge, metric checkpoint) | Core model — dense pointmaps + descriptors from RGB pairs |
| **Face Recognition** | ArcFace (insightface buffalo_l) | 512-dim identity embeddings — primary identity signal. Added 2026-02-10 |
| **Face Detection** | MediaPipe Face Mesh (or dlib 68-landmark) | Lightweight, real-time, provides 468 3D landmarks for face region cropping |
| **Deep Learning Runtime** | PyTorch 2.x + CUDA 12.x | MASt3R dependency; works on both RTX 5070 Laptop and Colab T4 |
| **Backend API** | FastAPI + uvicorn | Async support, WebSocket for streaming, auto-generated OpenAPI docs |
| **Frontend** | Gradio (for rapid prototyping) or React (if polish needed) | Gradio recommended for demo speed; runs on CS-1's laptop for demo |
| **3D Visualization** | Open3D (backend) + Three.js or Plotly (frontend) | Point cloud rendering; Plotly also works in Colab notebooks |
| **Point Cloud Ops** | Open3D, scipy.spatial | ICP, KDTree, Chamfer distance computation |
| **Template Storage** | SQLite + filesystem (`.npz` files) | Simple, portable, no external DB dependency |
| **Code Management** | GitHub | Single source of truth for all source code, configs, and tests |
| **Large Binary Data** | Google Drive (shared folder) | MASt3R checkpoints (~1 GB), `.npz` templates, raw captures — too large for GitHub's 100 MB file limit |
| **Config** | YAML (`config.yaml`) | Single source of truth for all tunable parameters; tracked in GitHub |
| **Environment** | Conda + pip (local) / Colab runtime (remote) | Match MASt3R's official environment setup |

---

## 3. Repository Structure

```
face-auth-mast3r/
├── README.md
├── config.yaml                     # All tunable parameters
├── requirements.txt
├── environment.yml                 # Conda environment
│
├── core/                           # ─── CS-1 Primary Ownership ───
│   ├── __init__.py
│   ├── face_detector.py            # Face detection + landmark extraction
│   ├── keyframe_selector.py        # Select optimal frames from video stream
│   ├── mast3r_engine.py            # MASt3R model loading + inference wrapper
│   ├── global_alignment.py         # Multi-view point cloud fusion
│   ├── template_manager.py         # Template CRUD operations
│   ├── face_embedder.py            # ArcFace identity embedding extraction (added 2026-02-10)
│   ├── anti_spoof.py               # Liveness / planarity detection
│   └── matching/                   # ─── DS Team Fills Internals ───
│       ├── __init__.py
│       ├── geometric_matcher.py    # ICP + Chamfer distance
│       ├── descriptor_matcher.py   # MASt3R descriptor comparison (supplementary)
│       ├── embedding_matcher.py    # ArcFace cosine similarity (primary identity, added 2026-02-10)
│       ├── score_fusion.py         # Weighted combination → decision (MultiModalFusion added 2026-02-10)
│       └── interfaces.py           # Abstract base classes (EmbeddingMatcher added 2026-02-10)
│
├── api/                            # ─── CS-1 Defines, CS-2 Consumes ───
│   ├── __init__.py
│   ├── app.py                      # FastAPI application entry point
│   ├── routes/
│   │   ├── enrollment.py           # POST /enroll, WebSocket /ws/enroll
│   │   ├── authentication.py       # POST /authenticate
│   │   └── management.py           # GET/DELETE /users
│   └── schemas.py                  # Pydantic request/response models
│
├── frontend/                       # ─── CS-2 Primary Ownership ───
│   ├── app_gradio.py               # Gradio-based demo UI
│   ├── components/
│   │   ├── webcam_capture.py       # Webcam access + frame dispatch
│   │   ├── enrollment_guide.py     # Guided head rotation UI
│   │   ├── auth_panel.py           # Authentication trigger + result
│   │   └── visualization.py        # 3D point cloud + match overlay
│   └── assets/
│       └── guide_arrows.png        # UI assets for enrollment guidance
│
├── storage/                        # Runtime data (gitignored)
│   ├── templates/                  # Enrolled face templates (.npz)
│   └── db.sqlite                   # User metadata
│
├── scripts/
│   ├── setup_mast3r.sh             # Clone MASt3R repo + download weights
│   ├── run_demo.sh                 # Launch full system
│   ├── test_inference.py           # Smoke test for MASt3R inference
│   ├── prepare_public_dataset.py   # Dataset → MASt3R + ArcFace → .npz templates
│   └── augment_npz_with_embeddings.py  # Retrofit ArcFace embeddings into existing .npz (added 2026-02-10)
│
├── tests/
│   ├── test_face_detector.py
│   ├── test_mast3r_engine.py
│   ├── test_template_manager.py
│   └── test_api_endpoints.py
│
└── third_party/
    └── mast3r/                     # Git submodule → github.com/naver/mast3r
```

---

## 4. Module Architecture

### 4.1 `core/face_detector.py`

**Owner**: CS-1

```python
class FaceDetector:
    """Detect faces and extract landmarks from RGB frames."""

    def __init__(self, config: dict):
        # Initialize MediaPipe Face Mesh
        ...

    def detect(self, frame: np.ndarray) -> Optional[FaceDetection]:
        """
        Args:
            frame: BGR image (H, W, 3), uint8

        Returns:
            FaceDetection with:
              - bbox: (x1, y1, x2, y2) in pixel coords
              - landmarks_2d: np.ndarray (468, 2) — pixel coordinates
              - landmarks_3d: np.ndarray (468, 3) — normalized 3D coords
              - head_pose: (yaw, pitch, roll) in degrees
            or None if no face detected.
        """
        ...

    def crop_face_region(self, frame: np.ndarray, detection: FaceDetection,
                         padding: float = 0.3) -> np.ndarray:
        """Crop frame to face bounding box with padding."""
        ...
```

### 4.2 `core/keyframe_selector.py`

**Owner**: CS-1

The keyframe selector decides which frames from the webcam stream are worth sending to MASt3R. During enrollment, we want frames that collectively cover diverse viewpoints.

```python
@dataclass
class KeyframeCandidate:
    frame: np.ndarray           # Cropped face image
    head_pose: Tuple[float, float, float]  # (yaw, pitch, roll)
    timestamp: float
    quality_score: float        # Blur detection, face confidence

class KeyframeSelector:
    """Select diverse, high-quality keyframes for enrollment."""

    def __init__(self, config: dict):
        self.target_count: int          # e.g., 12
        self.min_yaw_spread: float      # e.g., 40 degrees total
        self.min_pitch_spread: float    # e.g., 20 degrees total
        self.blur_threshold: float
        ...

    def should_capture(self, detection: FaceDetection,
                       existing: List[KeyframeCandidate]) -> bool:
        """Determine if this frame adds enough novelty to the set."""
        ...

    def get_coverage_status(self, candidates: List[KeyframeCandidate]) -> CoverageStatus:
        """
        Returns:
            CoverageStatus with:
              - yaw_range: (min, max) of captured yaw angles
              - pitch_range: (min, max) of captured pitch angles
              - total_frames: int
              - is_sufficient: bool
              - missing_directions: List[str]  # e.g., ["left", "up"]
        """
        ...
```

### 4.3 `core/mast3r_engine.py`

**Owner**: CS-1 — This is the most critical module.

```python
class MASt3REngine:
    """Wrapper around MASt3R model for face-specific inference."""

    def __init__(self, config: dict):
        self.device: torch.device
        self.model: AsymmetricMASt3R  # Loaded from checkpoint
        self.image_size: int = 512    # MASt3R's max input dimension
        self.force_fp16: bool = config.get("force_fp16", True)  # Mandatory on 8GB laptop
        ...

    def load_model(self) -> None:
        """Load MASt3R checkpoint. Call once at startup."""
        ...

    def infer_pair(self, img1: np.ndarray, img2: np.ndarray) -> PairwiseResult:
        """
        Run MASt3R on a single image pair.

        Args:
            img1, img2: RGB images (H, W, 3), uint8.
                        Will be resized to fit 512px max dimension.

        Returns:
            PairwiseResult:
              - pointmap1: np.ndarray (H', W', 3)  — 3D coords for img1 pixels
              - pointmap2: np.ndarray (H', W', 3)  — 3D coords for img2 pixels
              - confidence1: np.ndarray (H', W')
              - confidence2: np.ndarray (H', W')
              - descriptors1: np.ndarray (H', W', D) — dense local features
              - descriptors2: np.ndarray (H', W', D)
        """
        ...

    def reconstruct_multiview(
        self,
        frames: List[np.ndarray],
        pairs: Optional[List[Tuple[int, int]]] = None,
        head_poses: Optional[List[Tuple[float, float, float]]] = None,
        confidence_threshold: Optional[float] = None,
        use_global_alignment: bool = True,
        global_alignment_config: Optional[Dict[str, Any]] = None,
    ) -> MultiViewResult:
        """
        Reconstruct a unified 3D point cloud from multiple frames.
        Uses MASt3R-SfM sparse_global_alignment for globally consistent
        3D reconstruction with optimized camera poses.

        Args:
            frames: List of RGB face crops.
            pairs: Explicit pairing strategy. If None, auto-generated
                   using pose-aware K-nearest neighbor pairing (when
                   head_poses provided) or all-pairs for small N.
            head_poses: Optional (yaw, pitch, roll) per frame for
                        angular proximity pairing.
            confidence_threshold: Min confidence to keep a point.
                                  If None, reads from config.yaml.

        Returns:
            MultiViewResult:
              - point_cloud: np.ndarray (N, 3) — fused 3D points
              - colors: np.ndarray (N, 3)      — RGB per point
              - descriptors: np.ndarray (N, D)  — aggregated descriptors
              - confidence: np.ndarray (N,)
              - per_frame_poses: List[np.ndarray]  — estimated camera poses
        """
        ...

    @staticmethod
    def generate_pair_indices(
        n_frames: int,
        head_poses: Optional[List[Tuple[float, float, float]]] = None,
        min_neighbors: int = 3,
        max_neighbors: int = 8,
    ) -> List[Tuple[int, int]]:
        """
        Generate image pairs for multi-view reconstruction.

        When head_poses are provided, uses angular proximity to pair
        each frame with its K nearest neighbors (K between min_neighbors
        and max_neighbors). This avoids pairing frames with very
        different viewing angles which produce poor matches.

        Fallback: all-pairs for small N, sequential+skip for large N.
        """
        ...
```

### 4.4 `core/template_manager.py`

**Owner**: CS-1

```python
@dataclass
class FaceTemplate:
    user_id: str
    user_name: str
    point_cloud: np.ndarray     # (N, 3) — 3D geometry
    descriptors: np.ndarray     # (N, D) — dense features (supplementary)
    confidence: np.ndarray      # (N,)
    colors: np.ndarray          # (N, 3) — for visualization
    face_embedding: Optional[np.ndarray]  # (512,) — ArcFace identity embedding (added 2026-02-10)
    enrollment_metadata: dict   # Timestamp, n_frames, coverage stats
    version: str = "2.0"        # "2.0" when face_embedding is present (updated 2026-02-10)

class TemplateManager:
    """Persist and retrieve face templates."""

    def __init__(self, storage_dir: str, db_path: str):
        ...

    def save_template(self, template: FaceTemplate) -> str:
        """
        Save template to disk (.npz) and register in SQLite.
        Returns: file path of saved template.
        """
        ...

    def load_template(self, user_id: str) -> Optional[FaceTemplate]:
        """Load a single user's template."""
        ...

    def load_all_templates(self) -> List[FaceTemplate]:
        """Load all enrolled templates (for 1:N identification)."""
        ...

    def delete_template(self, user_id: str) -> bool:
        ...

    def list_users(self) -> List[dict]:
        """Return list of enrolled user metadata."""
        ...
```

### 4.5 `core/matching/interfaces.py`

**Owner**: CS-1 defines these; DS team implements the concrete logic.

```python
from abc import ABC, abstractmethod

@dataclass
class MatchResult:
    score: float                # 0.0 (no match) to 1.0 (perfect match)
    details: dict               # Algorithm-specific details for visualization
    is_match: bool              # Thresholded decision

class GeometricMatcher(ABC):
    @abstractmethod
    def compare(self, probe_cloud: np.ndarray,
                template_cloud: np.ndarray) -> MatchResult:
        """Compare two 3D point clouds geometrically."""
        ...

class DescriptorMatcher(ABC):
    @abstractmethod
    def compare(self, probe_desc: np.ndarray,
                template_desc: np.ndarray,
                probe_cloud: np.ndarray,
                template_cloud: np.ndarray) -> MatchResult:
        """Compare descriptor sets, optionally using 3D positions."""
        ...

# Added 2026-02-10: ArcFace embedding comparison interface
class EmbeddingMatcher(ABC):
    @abstractmethod
    def compare(self, probe_embedding: np.ndarray,
                template_embedding: np.ndarray) -> MatchResult:
        """Compare face identity embeddings (e.g., ArcFace 512-dim)."""
        ...

class ScoreFusion(ABC):
    @abstractmethod
    def fuse(self, geometric_result: MatchResult,
             descriptor_result: MatchResult) -> MatchResult:
        """Combine geometric and descriptor scores into a final decision."""
        ...
```

**Stub implementations for CS team to use before DS team delivers:**

```python
class StubGeometricMatcher(GeometricMatcher):
    """Placeholder: always returns 0.5 score. Replace with real ICP + Chamfer."""
    def compare(self, probe_cloud, template_cloud):
        return MatchResult(score=0.5, details={"method": "stub"}, is_match=True)
```

### 4.6 `core/face_embedder.py` (Added 2026-02-10)

**Owner**: CS-1

ArcFace identity embedding extractor. This is the **primary identity signal** — MASt3R descriptors are retained for 3D reconstruction but have no identity discrimination power (see §8.2).

```python
class FaceEmbedder:
    """Extract ArcFace face identity embeddings from face images."""

    def __init__(self, config: dict):
        # config keys: model ("buffalo_l"), device ("cuda"/"cpu"), backend ("auto")
        # Backend priority: insightface (ONNX) > facenet-pytorch > error
        ...

    def load_model(self) -> None:
        """Load ArcFace model. Call once at startup."""
        ...

    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract identity embedding from a single face image.

        Args:
            face_image: BGR image containing a face.

        Returns:
            (512,) float32 L2-normalized embedding, or None if no face detected.
        """
        ...

    def extract_multi_frame(
        self, face_images: List[np.ndarray],
        quality_scores: Optional[List[float]] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract and aggregate embeddings across multiple frames.
        Uses quality-weighted mean aggregation.

        Returns:
            (512,) float32 L2-normalized aggregated embedding.
        """
        ...
```

---

## 5. Data Flow & Pipeline Design

### 5.1 Enrollment Data Flow

> Updated 2026-02-10: Added ArcFace embedding extraction branch.

```
Webcam → Frame(BGR)
  │
  ├─[FaceDetector.detect()]─→ FaceDetection (bbox, landmarks, head_pose)
  │                              │
  │                              ├─→ [KeyframeSelector.should_capture()]
  │                              │         │
  │                              │    (if True) → append to keyframe_buffer
  │                              │
  │                              └─→ [CS-2: update coverage UI]
  │
  │  ... repeat until KeyframeSelector.is_sufficient() == True ...
  │
  ├─[KeyframeSelector] produces final List[KeyframeCandidate]
  │
  ├─[MASt3REngine.reconstruct_multiview(frames)]
  │     │
  │     ├─ For each pair: infer_pair() → pointmaps + descriptors
  │     └─ global_alignment() → fused point cloud + aggregated descriptors
  │
  ├─[FaceEmbedder.extract_multi_frame(face_crops)]     ← Added 2026-02-10
  │     └─ quality-weighted mean → 512-dim identity embedding
  │
  ├─[Face Region Cropping in 3D]
  │     Use 2D face landmarks → project to 3D → filter points outside face mask
  │
  ├─[TemplateManager.save_template()]
  │     Template v2.0: point_cloud + descriptors + face_embedding(512,) + metadata
  │
  └─→ Response: EnrollmentResult (success, user_id, preview_point_cloud)
```

### 5.2 Authentication Data Flow

> Updated 2026-02-10: Added ArcFace embedding matching as primary identity signal.

```
Webcam → Capture 2-4 frames (with slight head movement prompt)
  │
  ├─[FaceDetector.detect()] per frame
  │
  ├─[MASt3REngine.infer_pair() or reconstruct_multiview()]
  │     → probe_cloud, probe_descriptors
  │
  ├─[FaceEmbedder.extract_multi_frame(face_crops)]     ← Added 2026-02-10
  │     → probe_embedding (512-dim)
  │
  ├─[AntiSpoof.check(probe_cloud, confidence)]
  │     → liveness_result (pass/fail + planarity_score)  ← gate: blocks if fail
  │
  ├─[TemplateManager.load_template(claimed_user_id)]  # 1:1 verification
  │   OR
  │  [TemplateManager.load_all_templates()]             # 1:N identification
  │
  ├─[EmbeddingMatcher.compare(probe_emb, template_emb)]  → embedding_score  (PRIMARY)
  ├─[GeometricMatcher.compare(probe_cloud, template_cloud)]  → geometric_score  (supplementary)
  ├─[MultiModalFusion.fuse({embedding, geometric})]       → final_decision
  │     final = 0.7 × embedding + 0.3 × geometric
  │
  └─→ Response: AuthResult (is_match, score, matched_user, details)
```

---

## 6. Interface Contracts Between CS-1 and CS-2

This section defines the exact data shapes exchanged between backend (CS-1) and frontend (CS-2), so both can develop in parallel.

### 6.1 WebSocket: Enrollment Stream

**Endpoint**: `ws://localhost:8000/ws/enroll/{user_name}`

```
Client → Server (per frame):
{
    "type": "frame",
    "data": "<base64-encoded JPEG>"
}

Server → Client (per frame response):
{
    "type": "frame_status",
    "face_detected": true,
    "head_pose": {"yaw": -12.3, "pitch": 5.1, "roll": 0.8},
    "captured": true,              // Was this frame selected as keyframe?
    "total_captured": 7,
    "target_count": 12,
    "coverage": {
        "yaw_range": [-25.0, 18.0],
        "pitch_range": [-8.0, 12.0],
        "is_sufficient": false,
        "missing_directions": ["left"]
    }
}

Server → Client (enrollment complete):
{
    "type": "enrollment_complete",
    "user_id": "usr_a1b2c3",
    "point_cloud_preview": "<base64-encoded PLY or JSON points>",
    "n_frames_used": 12,
    "reconstruction_time_sec": 8.3
}
```

### 6.2 REST: Authentication

**Endpoint**: `POST /authenticate`

```
Request:
{
    "user_id": "usr_a1b2c3",          // null for 1:N identification mode
    "frames": ["<base64 JPEG>", ...], // 2-4 frames
}

Response:
{
    "is_match": true,
    "matched_user_id": "usr_a1b2c3",
    "matched_user_name": "Alice",
    "final_score": 0.87,
    "embedding_score": 0.92,          // ArcFace cosine similarity (added 2026-02-10)
    "geometric_score": 0.82,
    "descriptor_score": 0.91,         // Legacy — weight 0.0 by default
    "anti_spoof": {
        "passed": true,
        "depth_variance": 0.043,
        "planarity_ratio": 0.12
    },
    "processing_time_sec": 2.1,
    "visualization_data": {
        "probe_cloud": "<base64 encoded>",
        "matched_points": [[x1,y1,x2,y2], ...]
    }
}
```

### 6.3 REST: User Management

```
GET  /users                          → List[{user_id, user_name, enrolled_at}]
GET  /users/{user_id}                → User detail + template metadata
DELETE /users/{user_id}              → Delete enrollment
GET  /users/{user_id}/point_cloud    → 3D point cloud data for visualization
```

---

## 7. Enrollment Pipeline (Detailed)

### 7.1 Frame Capture Strategy

During enrollment, the user slowly rotates their head. The system captures keyframes that maximize angular coverage.

**Target coverage:**
- Yaw: -30° to +30° (at least 40° spread required)
- Pitch: -15° to +15° (at least 20° spread required)
- Roll: keep near 0° (reject frames with |roll| > 15°)

**Keyframe acceptance criteria:**
1. Face is detected with confidence > 0.9
2. Laplacian blur score > threshold (reject blurry frames)
3. Head pose adds novelty (>5° yaw or pitch delta from nearest existing keyframe)
4. Frame count < target (default: 12)

### 7.2 Pair Generation for MASt3R

Given N keyframes with head pose information, pairs are generated using **pose-aware K-nearest neighbor pairing**:

```python
# For each frame, compute angular distance to all other frames
# using (yaw, pitch) Euclidean distance, then pair with K nearest.
# K = min(max_neighbors, N-1), clamped to [min_neighbors, max_neighbors].
# Default: min_neighbors=3, max_neighbors=8.

# Example for N=12, K=8: produces ~56 pairs
# (compared to 66 all-pairs or ~24 sequential+skip)

pairs = generate_pair_indices(n_frames, head_poses=head_poses)
```

This avoids pairing frames with very different viewing angles (e.g., left profile vs right profile) which produce poor MASt3R matches, while ensuring sufficient connectivity for global alignment. Fallback: all-pairs for small N when head poses are unavailable.

### 7.3 MASt3R Inference Flow (per pair)

```python
# Pseudocode for a single pair (i, j)
img_i = load_and_resize(frames[i], max_dim=512)
img_j = load_and_resize(frames[j], max_dim=512)

with torch.no_grad():
    result = model(img_i, img_j)  # Forward pass

# result contains:
#   pred1['pts3d']       → (1, H, W, 3)  pointmap for img_i in img_i's frame
#   pred2['pts3d_in_other_view'] → (1, H, W, 3) pointmap for img_j in img_i's frame
#   pred1['conf']        → (1, H, W)     confidence
#   pred2['conf']        → (1, H, W)
#   pred1['desc']        → (1, H, W, D)  descriptors (from matching head)
#   pred2['desc']        → (1, H, W, D)
```

### 7.4 Global Alignment

Uses MASt3R-SfM's `sparse_global_alignment` (implemented in `core/global_alignment.py`):
- Input: RGB frames, pair indices, and the loaded MASt3R model
- Process: two-stage optimization — coarse 3D alignment (niter1=300, lr1=0.07) followed by fine 2D reprojection refinement (niter2=500, lr2=0.014)
- Output: globally consistent 3D point cloud with optimized camera poses and per-point confidence
- Parameters are configurable via `config.yaml` `global_alignment` section

### 7.5 Post-Processing Pipeline

After global alignment, the raw point cloud undergoes a multi-stage noise removal pipeline (all parameters configurable via `config.yaml` `post_processing` section):

1. **Confidence filtering**: Remove points with MASt3R confidence below threshold (default ≥1.5; values >1 are considered reliable by MASt3R)
2. **Voxel deduplication**: Quantize points to a 3D grid (default 10mm voxel size), keeping the highest-confidence point per voxel. Merges double-layer artifacts from overlapping views
3. **Statistical outlier removal**: For each point, compute mean distance to K nearest neighbors (default K=30). Remove points whose mean distance exceeds `global_mean + std_ratio × global_std` (default std_ratio=1.5). Eliminates isolated floating noise
4. **Largest-cluster filter**: Build a K-NN graph (default K=10) and connect points within distance `eps` (default 15mm). Keep only the largest connected component, discarding small scattered debris clusters

---

## 8. Authentication Pipeline (Detailed)

### 8.1 Probe Construction

Capture 2-4 frames with slight movement. Even 2 frames suffice for MASt3R (it's natively a stereo model). More frames improve robustness.

**Minimum viable**: 2 frames → 1 MASt3R pair → 1 pointmap + descriptors.

**Recommended**: 3-4 frames → 3-6 pairs → global alignment → cleaner probe.

### 8.2 Matching Pipeline

> Updated 2026-02-10: ArcFace embedding matching is now the primary identity signal.
> MASt3R descriptors have been experimentally shown to have **zero identity discrimination**
> (EER ≈ 0.46, AUC ≈ 0.53 — near random chance). The descriptor weight is set to 0.0 by default.

```
probe_cloud (N_p, 3) + probe_embedding (512,)
template_cloud (N_t, 3) + template_embedding (512,)

Step 1: Embedding Matching (PRIMARY — added 2026-02-10)
  - ArcFace 512-dim L2-normalized identity embeddings
  - cosine_similarity = dot(probe_emb, template_emb)
  - score = (cosine_similarity + 1) / 2  →  [0, 1]

Step 2: Pre-alignment (for geometric matching)
  - Coarse alignment using PCA on both point clouds
    (align principal axes to canonical frame)
  - Or: use centroid alignment + uniform scaling

Step 3: Geometric Matching (supplementary — DS team refines this)
  - ICP refinement after pre-alignment
  - Compute bidirectional Chamfer distance
  - Normalize to [0, 1] score

Step 4: Descriptor Matching (disabled by default — weight 0.0)
  - MASt3R descriptors lack identity discrimination (designed for view correspondence)
  - Retained for backward compatibility; can be re-enabled if fine-tuned

Step 5: Multi-Modal Score Fusion (→ DS team tunes weights)
  - final_score = w_emb × embedding_score + w_geo × geometric_score + w_desc × descriptor_score
  - Default: w_emb=0.7, w_geo=0.3, w_desc=0.0
  - is_match = final_score ≥ threshold (default: 0.55)
  - Missing channels are dynamically re-normalized (for v1 templates without embeddings)
```

---

## 9. Template Storage Schema

### 9.1 File Format (`.npz`)

> Updated 2026-02-10: Template v2.0 adds `face_embedding` field for ArcFace identity.

Each enrolled user gets one `.npz` file:

```python
np.savez_compressed(
    filepath,
    point_cloud=cloud,          # (N, 3) float32
    descriptors=descriptors,    # (N, D) float32 — supplementary (no identity discrimination)
    confidence=confidence,      # (N,)   float32
    colors=colors,              # (N, 3) uint8
    face_embedding=embedding,   # (512,) float32 — L2-normalized ArcFace identity embedding (v2.0)
    metadata=json.dumps({       # serialized as string
        "user_id": "usr_a1b2c3",
        "user_name": "Alice",
        "enrolled_at": "2026-02-05T14:30:00Z",
        "n_frames": 12,
        "yaw_range": [-28.0, 25.0],
        "pitch_range": [-12.0, 14.0],
        "mast3r_version": "ViTLarge_metric",
        "template_version": "2.0",      # "2.0" when face_embedding present
        "embedding_model": "buffalo_l", # ArcFace model used
        "embedding_dim": 512
    })
)
```

### 9.2 SQLite Schema

```sql
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    user_name TEXT NOT NULL,
    template_path TEXT NOT NULL,
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    n_points INTEGER,
    n_frames_used INTEGER
);

CREATE TABLE auth_logs (
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
);
```

---

## 10. Anti-Spoofing Module

**Owner**: CS-1 builds the infrastructure; DS-2 tunes thresholds.

### 10.1 Planarity Check (primary defense)

A real face has ~5-10cm of depth variation (nose to ears). A printed photo or screen presents a near-flat surface.

```python
class AntiSpoof:
    def check(self, point_cloud: np.ndarray,
              confidence: np.ndarray) -> AntiSpoofResult:
        # 1. Compute PCA on the point cloud
        # 2. Ratio of smallest eigenvalue to largest eigenvalue
        #    - Real face: ratio > 0.05 (significant depth)
        #    - Flat photo: ratio ≈ 0.0
        # 3. Z-axis (depth) standard deviation
        #    - Real face: std > 3mm (in metric MASt3R output)
        #    - Flat photo: std < 1mm
        # 4. Confidence distribution analysis
        #    - Real face: mostly high confidence
        #    - Photo of photo: lower avg confidence, more variance
        ...
```

### 10.2 Temporal Consistency (secondary)

If capturing multiple frames, check that the 3D reconstruction changes consistently with head movement (a flat photo won't produce consistent 3D changes).

---

## 11. GPU & Memory Management

### 11.1 Compute Environment

The team operates in a **dual-environment** setup:

| Environment | GPU | VRAM | Owner | Use Case |
|---|---|---|---|---|
| **Local Laptop** | GeForce RTX 5070 Laptop | 8 GB GDDR7 | CS-1 | Final demo, real-time webcam pipeline, integration testing |
| **Google Colab (Free)** | Tesla T4 | 16 GB GDDR6 | CS-2, DS-1, DS-2 | Model experimentation, offline batch evaluation, algorithm dev |

**The RTX 5070 Laptop is the demo machine.** All final demos run here. Colab is the development/experimentation platform for everyone else.

### 11.2 Memory Budget & Feasibility

MASt3R ViT-Large at 512px input:
- Model weights: ~2.5 GB VRAM
- Per-pair inference: ~3-4 GB additional VRAM
- **Total per-pair peak**: ~6 GB

| | RTX 5070 Laptop (8 GB) | Colab T4 (16 GB) |
|---|---|---|
| Single pair inference | ✅ Fits (~6 GB peak) | ✅ Comfortable |
| Headroom | ~2 GB (tight) | ~10 GB (plenty) |
| Must use FP16 | **Yes — mandatory** | Optional but recommended for speed |
| Batch multiple pairs | ❌ Never | ❌ Still sequential (safer) |

### 11.3 Optimization Strategies (Critical for 8 GB Laptop)

```python
# 1. Keep model loaded as singleton (avoid reload per request)
_engine_instance: Optional[MASt3REngine] = None

def get_engine() -> MASt3REngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = MASt3REngine(config)
        _engine_instance.load_model()
    return _engine_instance

# 2. MANDATORY on RTX 5070 Laptop: use FP16 mixed precision
#    This roughly halves activation memory, making 8GB viable.
with torch.autocast(device_type='cuda', dtype=torch.float16):
    with torch.no_grad():
        result = model(img1, img2)

# 3. Aggressively free GPU memory after each pair
with torch.no_grad():
    result = model(img1, img2)
    pointmap = result['pts3d'].cpu().numpy()
    descriptors = result['desc'].cpu().numpy()
    # Immediately free ALL GPU tensors
    del result
    torch.cuda.empty_cache()

# 4. Process pairs strictly sequentially (never batch on 8GB)

# 5. Reduce keyframe count if memory issues arise
#    Fallback: 8 keyframes instead of 12 (fewer pairs to process)
```

### 11.4 Colab-Specific Constraints

```
⚠️  Colab free tier limitations to design around:
  - Session timeout: ~90 min idle, ~12 hr max runtime
  - No persistent disk: files lost on session restart
  - No direct webcam access from Python kernel
  - T4 may be unavailable during peak hours (falls back to CPU)
  - RAM: 12.7 GB system RAM
```

**Workarounds for Colab development (see also Appendix C):**

1. **No webcam on Colab** — CS-2 develops the webcam/UI components locally (even without GPU, the UI logic doesn't need one). For testing MASt3R on Colab, upload pre-captured face images instead.
2. **Session persistence** — Save checkpoints and templates to Google Drive (`/content/drive/`), not `/content/`.
3. **GPU availability** — Add a runtime check at notebook top:
   ```python
   import torch
   assert torch.cuda.is_available(), "No GPU! Go to Runtime > Change runtime type > T4 GPU"
   print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
   ```
4. **Install MASt3R in Colab** — Takes ~3-5 min per session (see Appendix C for the cell).

### 11.5 Inference Time Budget

| Operation | RTX 5070 Laptop (FP16) | Colab T4 (FP16) | Notes |
|---|---|---|---|
| Face detection (MediaPipe) | ~10 ms/frame | ~15 ms/frame | CPU-bound |
| ArcFace embedding (per face) | ~5-10 ms | ~10-15 ms | ONNX Runtime (added 2026-02-10) |
| MASt3R pair inference | ~0.3-0.5 s/pair | ~0.8-1.2 s/pair | T4 is ~2x slower than RTX 5070 |
| Global alignment (12 frames, 24 pairs) | ~5-8 s | ~12-20 s | Mixed CPU/GPU |
| ICP alignment | ~0.1-0.3 s | ~0.2-0.4 s | CPU (Open3D) |
| Descriptor matching | ~0.05-0.1 s | ~0.05-0.1 s | CPU, KDTree |
| **Enrollment total** | **~10-15 s** | **~20-35 s** | After keyframe capture |
| **Authentication total** | **~1-3 s** | **~3-6 s** | 2-4 frames |

For the live demo, the RTX 5070 Laptop timings are what matters. Colab latency is acceptable for offline development and evaluation.

---

## 12. API Specification

### 12.1 FastAPI Application Structure

```python
# api/app.py
from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load MASt3R model once
    engine = get_engine()
    engine.load_model()
    yield
    # Shutdown: cleanup

app = FastAPI(
    title="MASt3R Face Auth API",
    version="0.1.0",
    lifespan=lifespan
)
```

### 12.2 Endpoint Summary

| Method | Path | Purpose | Owner |
|---|---|---|---|
| `WS` | `/ws/enroll/{user_name}` | Stream frames, get real-time feedback, complete enrollment | CS-1 |
| `POST` | `/enroll` | Non-streaming enrollment (upload all frames at once) | CS-1 |
| `POST` | `/authenticate` | Run authentication against enrolled template(s) | CS-1 |
| `GET` | `/users` | List all enrolled users | CS-1 |
| `GET` | `/users/{user_id}` | Get user details + enrollment metadata | CS-1 |
| `DELETE` | `/users/{user_id}` | Remove enrolled user | CS-1 |
| `GET` | `/health` | System health check (model loaded, GPU available) | CS-1 |

---

## 13. Configuration Management

All tunable parameters live in a single `config.yaml`:

```yaml
# config.yaml

# ── Environment: set to "laptop" on CS-1's machine, "colab" elsewhere ──
environment: "laptop"             # "laptop" (RTX 5070) or "colab" (T4)

mast3r:
  checkpoint: "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
  image_size: 512
  device: "cuda"                  # "cuda" or "cpu"
  force_fp16: true                # Always true on laptop (8GB); optional on Colab (16GB)

face_detection:
  backend: "mediapipe"        # "mediapipe" or "dlib"
  min_confidence: 0.9
  face_padding: 0.3           # Fractional padding around face bbox

enrollment:
  target_keyframes: 12
  min_keyframes: 8
  min_yaw_spread_deg: 40.0
  min_pitch_spread_deg: 20.0
  max_roll_deg: 15.0
  blur_threshold: 100.0       # Laplacian variance threshold
  pose_novelty_deg: 5.0       # Min angular delta to accept new keyframe

authentication:
  min_frames: 2
  max_frames: 4
  mode: "verification"        # "verification" (1:1) or "identification" (1:N)

# Added 2026-02-10: ArcFace face recognition embeddings
face_embedding:
  model: "buffalo_l"            # insightface model bundle (ArcFace R100)
  embedding_dim: 512
  device: "cuda"                # "cuda" or "cpu"
  backend: "auto"               # "insightface", "facenet", or "auto"

matching:
  embedding_weight: 0.40        # ArcFace (primary identity signal) — added 2026-02-10
  geometric_weight: 0.10        # ICP + Chamfer (supplementary 3D shape)
  descriptor_weight: 0.50       # MASt3R descriptors (view-correspondence features)
  accept_threshold: 0.65        # tuned via public dataset evaluation
  icp_max_iterations: 50
  icp_convergence_threshold: 1e-6
  chamfer_normalization: "mean"

anti_spoof:
  enabled: true
  min_depth_variance: 0.003   # meters (3mm)
  min_eigenvalue_ratio: 0.05
  min_confidence_mean: 0.5

storage:
  template_dir: "storage/templates"
  db_path: "storage/db.sqlite"

server:
  host: "0.0.0.0"
  port: 8000
  reload: true                # Dev mode
```

---

## 14. Development Workflow & Task Ownership

### 14.0 Who Works Where

```
┌─────────────────────────────────────────────────────────────────┐
│                   Development Environment Map                    │
├──────────┬──────────────────┬───────────────────────────────────┤
│  Member  │  Primary Env     │  What They Run There              │
├──────────┼──────────────────┼───────────────────────────────────┤
│  CS-1    │  RTX 5070 Laptop │  Full local stack: webcam →       │
│          │                  │  MASt3R → API → demo              │
│          │                  │  (This is the demo machine)       │
├──────────┼──────────────────┼───────────────────────────────────┤
│  CS-2    │  Local (CPU) +   │  UI/frontend: develop locally     │
│          │  Colab (T4)      │  without GPU. Use Colab only to   │
│          │                  │  test visualization with real     │
│          │                  │  point cloud data.                │
├──────────┼──────────────────┼───────────────────────────────────┤
│  DS-1    │  Colab (T4)      │  Matching algorithm experiments,  │
│          │                  │  threshold tuning with offline    │
│          │                  │  pre-captured face image sets.    │
├──────────┼──────────────────┼───────────────────────────────────┤
│  DS-2    │  Colab (T4)      │  Evaluation experiments, anti-    │
│          │                  │  spoofing analysis, statistical   │
│          │                  │  analysis on pre-computed data.   │
└──────────┴──────────────────┴───────────────────────────────────┘
```

**Where code and data live**:

```
GitHub Repository (source of truth for ALL code)
└── face-auth-mast3r/
    ├── core/                  # All source code
    ├── api/
    ├── frontend/
    ├── scripts/
    ├── tests/
    ├── config.yaml
    ├── requirements.txt
    ├── .gitignore             # Excludes checkpoints/, storage/, *.npz, third_party/
    └── README.md

Google Drive (shared folder — large binary files only, NOT code)
└── face-auth-data/
    ├── checkpoints/               # MASt3R weights (~1 GB, too large for GitHub)
    │   └── MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
    ├── raw_captures/              # Webcam frames captured by CS-1
    │   ├── alice/
    │   │   ├── frame_001.jpg
    │   │   └── ...
    │   └── bob/
    ├── datasets/                  # ← Added on 202602026: DS team uploads public datasets here
    │   └── <dataset_name>/        #   e.g., "multi_pie" or "feret"
    │       ├── subject_001/
    │       │   ├── view_01.jpg
    │       │   ├── view_02.jpg
    │       │   └── ...
    │       └── subject_002/
    │           └── ...
    ├── mast3r_outputs/            # Pre-computed by CS-1 on RTX 5070
    │   ├── alice_enrollment.npz   # {point_cloud, descriptors, confidence, face_embedding}
    │   ├── alice_probe_01.npz     #   face_embedding: (512,) ArcFace (v2.0, added 2026-02-10)
    │   └── bob_enrollment.npz
    └── evaluation_results/        # DS team writes here
```

CS-1 captures face images on the laptop, runs MASt3R, and exports intermediate data to the shared Drive folder. DS members and CS-2 can load these on Colab without running MASt3R themselves. This keeps the GitHub repo lightweight while ensuring everyone can access GPU-generated artifacts.

### 14.1 Phase 1: Foundation (Week 1-2)

| Task | Owner | Environment | Depends On | Deliverable |
|---|---|---|---|---|
| MASt3R env setup + smoke test | CS-1 | **RTX 5070 Laptop** | — | `scripts/setup_mast3r.sh`, passing `test_inference.py` |
| Colab MASt3R notebook template | CS-1 | Colab | Laptop setup done | Shared notebook that others can copy (see Appendix C) |
| `face_detector.py` complete | CS-1 | Laptop | — | Unit test with sample images |
| `keyframe_selector.py` complete | CS-1 | Laptop | `face_detector.py` | Unit test with simulated head poses |
| Gradio skeleton (webcam + placeholders) | CS-2 | **Local CPU** (no GPU needed) | — | Live webcam feed in browser |
| Enrollment guide UI component | CS-2 | Local CPU | — | Directional arrows + progress meter |
| Config system (`config.yaml` loader) | CS-1 | Laptop | — | Shared config singleton |
| **Capture test face datasets** | **CS-1** | **Laptop** | Face detector | **12+ images per team member, export to shared Drive** |

### 14.2 Phase 2: Core Pipeline (Week 3-4)

| Task | Owner | Environment | Depends On | Deliverable |
|---|---|---|---|---|
| `mast3r_engine.py` — `infer_pair()` | CS-1 | RTX 5070 Laptop | MASt3R env | Pointmaps + descriptors from 2 images |
| `mast3r_engine.py` — `reconstruct_multiview()` | CS-1 | Laptop | `infer_pair()` | Fused point cloud from N frames |
| **Export `.npz` of all team members** | **CS-1** | **Laptop** | `reconstruct_multiview()` | **Upload to shared Drive — unblocks DS team** |
| `template_manager.py` | CS-1 | Laptop | — | Save/load/delete templates |
| Enrollment WebSocket endpoint | CS-1 | Laptop | All core modules | E2E enrollment via API |
| Connect enrollment UI to WebSocket | CS-2 | Local CPU → Laptop | WebSocket endpoint | Real-time feedback during enrollment |
| 3D point cloud viewer component | CS-2 | Local CPU + Colab | — | Render point clouds in browser |

### 14.3 Phase 3: Auth + Demo (Week 5-6)

| Task | Owner | Environment | Depends On | Deliverable |
|---|---|---|---|---|
| Auth endpoint (`POST /authenticate`) | CS-1 | RTX 5070 Laptop | `mast3r_engine`, `template_manager` | Working auth API |
| `anti_spoof.py` | CS-1 | Laptop | Pointmap outputs | Planarity check functional |
| `matching/interfaces.py` + stubs | CS-1 | Laptop | — | DS team can plug in implementations |
| Auth UI panel | CS-2 | Local CPU → Laptop | Auth endpoint | Click-to-authenticate flow |
| Match result visualization | CS-2 | Local CPU | Auth response data | Score gauge, match overlay |
| Demo run-through (end-to-end) | Both | **RTX 5070 Laptop** | Everything | Enroll → Auth works live |

### 14.4 Git Branching Strategy

```
main
 ├── develop                    ← Integration branch
 │    ├── feature/cs1-mast3r-engine
 │    ├── feature/cs1-face-detector
 │    ├── feature/cs1-template-manager
 │    ├── feature/cs1-api-endpoints
 │    ├── feature/cs2-enrollment-ui
 │    ├── feature/cs2-auth-ui
 │    ├── feature/cs2-visualization
 │    └── feature/ds-matching-algorithms
 │
 └── release/v0.1               ← Tagged demo versions
```

**Merge protocol**: Feature branch → PR to `develop` → at least 1 review → merge. Integrate to `main` for milestones.

### 14.5 Git Workflow on Each Environment

**CS-1 (RTX 5070 Laptop) — standard local development:**
```bash
git clone git@github.com:your-team/face-auth-mast3r.git
cd face-auth-mast3r
git checkout -b feature/cs1-mast3r-engine
# ... develop ...
git add -A && git commit -m "feat: implement infer_pair()"
git push origin feature/cs1-mast3r-engine
# → Open PR on GitHub
```

**CS-2 / DS members (Google Colab) — clone into Colab runtime, push back to GitHub:**
```python
# ── Cell 1: Mount Drive (for large data files) ──
from google.colab import drive
drive.mount('/content/drive')

# ── Cell 2: Clone repo from GitHub (every session) ──
# Option A: HTTPS (simpler — prompts for token on push)
!git clone https://github.com/your-team/face-auth-mast3r.git /content/repo

# Option B: SSH (if you add your SSH key to Colab — more seamless)
# !git clone git@github.com:your-team/face-auth-mast3r.git /content/repo

%cd /content/repo
!git checkout develop
!git pull origin develop
!git checkout -b feature/ds1-matching-algorithm
```

```python
# ── Cell 3: Symlink large files from Drive into repo ──
import os
# Model checkpoint: lives on Drive, symlinked into repo's checkpoints/
os.makedirs("/content/repo/checkpoints", exist_ok=True)
!ln -sf /content/drive/MyDrive/face-auth-data/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
        /content/repo/checkpoints/

# Pre-computed data: symlink into a local path the code can reference
!ln -sf /content/drive/MyDrive/face-auth-data/mast3r_outputs /content/repo/data_shared
```

```python
# ── After making changes: commit and push ──
!git config user.email "member@example.com"
!git config user.name "Member Name"
!git add -A
!git commit -m "feat: implement ICP-based geometric matcher"
!git push origin feature/ds1-matching-algorithm
# → Open PR on GitHub
```

> **Note for Colab users**: The repo is cloned into `/content/repo/` which is **ephemeral** — it disappears when the session ends. Always push your changes to GitHub before the session times out. Do NOT rely on Colab's filesystem for code persistence. Google Drive is only for large binary data, not code.

### 14.6 `.gitignore`

```gitignore
# Model weights (stored on Google Drive, not GitHub)
checkpoints/
*.pth

# Runtime data
storage/
*.npz
*.sqlite

# Symlinks to Drive data
data_shared

# MASt3R submodule build artifacts
third_party/mast3r/dust3r/croco/models/curope/build/
third_party/mast3r/dust3r/croco/models/curope/*.so

# Python
__pycache__/
*.pyc
*.egg-info/
.venv/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

---

## 15. Testing Strategy

### 15.1 Unit Tests

```
tests/
├── test_face_detector.py       # Test with known face images
├── test_keyframe_selector.py   # Test pose novelty logic with synthetic data
├── test_mast3r_engine.py       # Smoke test: 2 images → pointmap shapes correct
├── test_template_manager.py    # Save/load roundtrip, delete, list
├── test_anti_spoof.py          # Flat plane vs. curved surface detection
└── test_api_endpoints.py       # FastAPI TestClient
```

### 15.2 Integration Tests

- **Enrollment smoke test**: 5 images of a face → template saved → template loadable → point cloud has >1000 points
- **Auth smoke test**: Enroll user → authenticate same user → score > threshold
- **Rejection test**: Enroll user A → authenticate with user B frames → score < threshold

### 15.3 Test Fixtures

Place test images in `tests/fixtures/`:
- `face_frontal.jpg`, `face_left30.jpg`, `face_right30.jpg` — for detector tests
- `flat_photo.jpg` — printed face photo for anti-spoof test
- These can be collected from team members early on

---

## 16. Known Constraints & Risks

| Risk | Impact | Mitigation |
|---|---|---|
| RTX 5070 Laptop has only 8 GB VRAM | OOM on large enrollments | FP16 mandatory; limit to 12 keyframes; aggressive `torch.cuda.empty_cache()` after each pair |
| Colab T4 not always available on free tier | DS/CS-2 blocked from GPU experiments | Pre-compute `.npz` outputs on RTX 5070 Laptop and share via Drive; DS can work on CPU with pre-computed data |
| Colab session timeout / data loss | Lose experiment progress | Always save to Google Drive, not `/content/`; use Colab notebook checkpointing |
| Only CS-1's laptop can run the full real-time pipeline | Bus factor = 1 for demo | Document setup thoroughly; rehearse demo well in advance; have pre-recorded video as backup |
| 512px input limit may lose fine facial detail | Reduced discriminative power | Crop tightly to face before sending to MASt3R (maximize face pixel coverage) |
| MASt3R not trained on faces specifically | Descriptors have zero identity discrimination (EER≈0.46) | ArcFace added as primary identity signal (2026-02-10); MASt3R retained for 3D reconstruction + anti-spoofing only |
| CC BY-NC-SA 4.0 license on MASt3R | Cannot use commercially | Acceptable for academic project; document in README |
| Global alignment is slow for many pairs | Enrollment > 15s may frustrate users | Limit to 12 keyframes; optimize pair selection |
| Two-person face confusion under poor lighting | False accept | Ensure demo venue has adequate lighting; mention as limitation |

---

## Appendix A: MASt3R Setup Script

```bash
#!/bin/bash
# scripts/setup_mast3r.sh

set -e

echo "=== Setting up MASt3R ==="

# Clone MASt3R as submodule
git submodule add https://github.com/naver/mast3r.git third_party/mast3r
git submodule update --init --recursive

# Download checkpoint
mkdir -p checkpoints/
wget -nc https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    -P checkpoints/

# Install MASt3R dependencies (run inside conda env)
pip install -r third_party/mast3r/requirements.txt
pip install -r third_party/mast3r/dust3r/requirements.txt

# Optional: compile RoPE CUDA kernels for speed
cd third_party/mast3r/dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../../../

echo "=== MASt3R setup complete ==="
```

## Appendix B: Minimal Inference Test

```python
# scripts/test_inference.py
"""Smoke test: verify MASt3R loads and produces expected output shapes."""

import torch
import numpy as np
from PIL import Image

def test_mast3r_inference():
    from mast3r.model import AsymmetricMASt3R
    from mast3r.fast_nn import fast_reciprocal_NNs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images

    model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create dummy test images (or use real face photos)
    dummy1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Save as temp files (MASt3R's load_images expects file paths)
    Image.fromarray(dummy1).save("/tmp/test1.jpg")
    Image.fromarray(dummy2).save("/tmp/test2.jpg")

    images = load_images(["/tmp/test1.jpg", "/tmp/test2.jpg"], size=512)
    output = inference([tuple(images)], model, device, batch_size=1)

    view1, view2 = output["view1"], output["view2"]

    print(f"Pointmap 1 shape: {view1['pts3d'].shape}")     # (1, H, W, 3)
    print(f"Pointmap 2 shape: {view2['pts3d_in_other_view'].shape}")
    print(f"Descriptor 1 shape: {view1['desc'].shape}")     # (1, H, W, D)
    print(f"Confidence 1 shape: {view1['conf'].shape}")     # (1, H, W)

    assert view1['pts3d'].dim() == 4
    assert view1['desc'].dim() == 4
    print("All shape checks passed.")

if __name__ == "__main__":
    test_mast3r_inference()
```

## Appendix C: Google Colab Session Setup

Run these cells at the **start of every Colab session**. The Colab runtime is ephemeral — code must be pushed to GitHub before the session ends.

```python
# ============================================================
# Cell 1: GPU Check + Google Drive Mount
# ============================================================
# Check GPU availability first
!nvidia-smi

import os

from google.colab import drive
drive.mount('/content/drive')

SHARED_DIR = "/content/drive/MyDrive/face-auth-data"
os.makedirs(f"{SHARED_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{SHARED_DIR}/raw_captures", exist_ok=True)
os.makedirs(f"{SHARED_DIR}/mast3r_outputs", exist_ok=True)
os.makedirs(f"{SHARED_DIR}/evaluation_results", exist_ok=True)
print(f"✅ Shared data dir: {SHARED_DIR}")
```

```python
# ============================================================
# Cell 2: Install PyTorch with CUDA FIRST (before other dependencies)
# ============================================================
# This must come before pip install -r requirements.txt to ensure
# proper CUDA support is installed.
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

import torch
if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected. Go to Runtime > Change runtime type > T4 GPU")

gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"✅ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
print(f"✅ PyTorch CUDA: {torch.version.cuda}")
```

```python
# ============================================================
# Cell 3: Clone GitHub repo (clean slate every session)
# ============================================================
REPO_URL = "https://github.com/gaelgm03/ai-visual-computing-pbl.git"
BRANCH   = "main"  # or "develop" or your feature branch

%cd /content
!rm -rf ai-visual-computing-pbl 2>/dev/null
!git clone --depth 1 {REPO_URL}
%cd ai-visual-computing-pbl

!git fetch origin
!git checkout {BRANCH}

# Configure git identity (needed for commits)
!git config user.email "your-email@example.com"   # ← UPDATE THIS
!git config user.name "Your Name"                  # ← UPDATE THIS

print(f"✅ Repo ready at /content/ai-visual-computing-pbl on branch: {BRANCH}")
```

```python
# ============================================================
# Cell 4: Install project dependencies
# ============================================================
!pip install -r requirements.txt
print("✅ Project dependencies installed")
```

```python
# ============================================================
# Cell 5: Symlink large binary data from Drive into repo
#          (These files are gitignored — they live on Drive, not GitHub)
# ============================================================
import os

REPO_DIR = "/content/ai-visual-computing-pbl"
SHARED_DIR = "/content/drive/MyDrive/face-auth-data"

# Create directories
os.makedirs(f"{REPO_DIR}/checkpoints/mast3r", exist_ok=True)
os.makedirs(f"{REPO_DIR}/storage", exist_ok=True)

# Symlink model checkpoints
!ln -sf {SHARED_DIR}/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
        {REPO_DIR}/checkpoints/mast3r/

# Symlink raw captures storage
!ln -sf {SHARED_DIR}/raw_captures {REPO_DIR}/storage/raw_captures

# Symlink pre-computed MASt3R outputs (for DS team)
!ln -sf {SHARED_DIR}/mast3r_outputs {REPO_DIR}/data_shared

print("✅ Drive data symlinked into repo")
```

```python
# ============================================================
# Cell 6: (Optional) Install MASt3R dependencies for full pipeline
#          Skip this cell if only running frontend UI
# ============================================================
%%bash
cd /content
if [ ! -d "mast3r" ]; then
    echo "Cloning MASt3R repository..."
    git clone --recursive https://github.com/naver/mast3r.git
fi
cd mast3r
pip install -q -r requirements.txt
pip install -q -r dust3r/requirements.txt

# Optional: compile RoPE CUDA kernels
cd dust3r/croco/models/curope/
python setup.py build_ext --inplace 2>/dev/null || echo "RoPE compile skipped (non-critical)"
echo "✅ MASt3R dependencies installed"
```

```python
# ============================================================
# Cell 7: Configure Python path + Load MASt3R model
#          Skip this cell if only running frontend UI
# ============================================================
import sys
import os
import torch

REPO_DIR = "/content/ai-visual-computing-pbl"

# Add MASt3R to path
sys.path.insert(0, "/content/mast3r")
sys.path.insert(0, "/content/mast3r/dust3r")

# Add our project to path
sys.path.insert(0, REPO_DIR)

from mast3r.model import AsymmetricMASt3R

model = AsymmetricMASt3R.from_pretrained(
    "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
)
device = torch.device("cuda")
model = model.to(device)
print("✅ MASt3R model loaded")
```

```python
# ============================================================
# Cell 8: Run Frontend UI (share=True for Colab public URL)
#          This launches the Gradio demo with a shareable link
# ============================================================
%cd /content/ai-visual-computing-pbl
!python -m frontend.app_gradio
# The app will output a public URL like: https://xxxxx.gradio.live
```

```python
# ============================================================
# Cell 9: Load Pre-computed Data (for DS team — no GPU needed)
#          DS members can skip Cells 6-7 and work directly with
#          .npz files exported by CS-1.
# ============================================================
import numpy as np

def load_template(user_name: str) -> dict:
    """Load a pre-computed enrollment template from shared Drive."""
    path = f"{SHARED_DIR}/mast3r_outputs/{user_name}_enrollment.npz"
    data = np.load(path, allow_pickle=True)
    return {
        "point_cloud": data["point_cloud"],     # (N, 3)
        "descriptors": data["descriptors"],     # (N, D)
        "confidence": data["confidence"],       # (N,)
        "colors": data["colors"],               # (N, 3)
    }

# Example usage:
# alice = load_template("alice")
# print(f"Alice template: {alice['point_cloud'].shape[0]} points")
```

```python
# ============================================================
# ⚠️  BEFORE SESSION ENDS: Push your changes to GitHub!
# ============================================================
# %cd /content/ai-visual-computing-pbl
# !git add -A
# !git commit -m "feat: your commit message here"
# !git push origin {BRANCH}
```

## Appendix D: CS-1 Data Export Script (for Shared Drive)

CS-1 runs this on the RTX 5070 Laptop after enrollment to produce `.npz` files that the rest of the team can consume on Colab without needing a local GPU.

```python
# scripts/export_for_team.py
"""
Run on CS-1's RTX 5070 Laptop after enrollment.
Exports MASt3R outputs to a format the team can use on Colab.
"""
import numpy as np
from core.template_manager import TemplateManager

def export_template(user_id: str, output_dir: str):
    tm = TemplateManager(storage_dir="storage/templates", db_path="storage/db.sqlite")
    template = tm.load_template(user_id)

    output_path = f"{output_dir}/{template.user_name}_enrollment.npz"
    np.savez_compressed(
        output_path,
        point_cloud=template.point_cloud,
        descriptors=template.descriptors,
        confidence=template.confidence,
        colors=template.colors,
    )
    print(f"Exported {output_path} ({template.point_cloud.shape[0]} points)")

def export_probe(frames_dir: str, user_name: str, output_dir: str):
    """Run MASt3R on a set of probe frames and export the result."""
    from core.mast3r_engine import MASt3REngine
    import yaml, glob, cv2

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    engine = MASt3REngine(config["mast3r"])
    engine.load_model()

    frames = []
    for p in sorted(glob.glob(f"{frames_dir}/*.jpg")):
        frames.append(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))

    result = engine.reconstruct_multiview(frames)

    output_path = f"{output_dir}/{user_name}_probe.npz"
    np.savez_compressed(
        output_path,
        point_cloud=result.point_cloud,
        descriptors=result.descriptors,
        confidence=result.confidence,
    )
    print(f"Exported {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--output_dir", default="./export")
    args = parser.parse_args()
    export_template(args.user_id, args.output_dir)
```