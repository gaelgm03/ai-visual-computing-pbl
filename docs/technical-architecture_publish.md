# MASt3R Face Authentication System — Technical Architecture

## 1. System Overview

### 1.1 What We Built

A **Face ID-like authentication prototype** that performs face enrollment and authentication using **3D face reconstruction** and **multi-modal matching** — all from a standard RGB webcam, with no depth sensor required.

The system combines three complementary technologies:

- **MediaPipe Face Mesh** — Real-time face detection and head pose estimation
- **MASt3R** — Dense 3D face reconstruction from multi-view RGB images
- **ArcFace** — Deep face recognition via identity-discriminative embeddings

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        User-Facing Application                         │
│   Webcam Feed  ·  Enrollment Guide (pose grid)  ·  Result Display      │
├────────────────────────────────────────┬────────────────────────────────┤
│            Backend API (FastAPI)       │     Frontend (Gradio)          │
│       WebSocket + REST endpoints      │     Webcam capture + UI        │
├────────────────────────────────────────┴────────────────────────────────┤
│                           Core Engine                                   │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  Face        │  │  MASt3R      │  │  ArcFace     │  │  Template  │  │
│  │  Detector    │→ │  3D Recon    │→ │  Embedding   │→ │  Manager   │  │
│  │  (MediaPipe) │  │  Engine      │  │  Extractor   │  │  (.npz+DB) │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  Anti-Spoof  │  │  Multi-Modal │  │  Score       │                  │
│  │  (3D Shape)  │→ │  Matchers    │→ │  Fusion      │→ Accept/Reject  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Face Detection** | MediaPipe Face Landmarker | 478 facial landmarks + real-time head pose estimation via PnP |
| **3D Reconstruction** | MASt3R (ViT-Large, 512px) | Dense stereo 3D pointmaps + per-pixel confidence from RGB image pairs |
| **Face Recognition** | ArcFace (insightface buffalo_l) | 512-dim identity embeddings — primary identity discrimination signal |
| **Anti-Spoofing** | 3D Shape Analysis (PCA + depth) | Detects flat media (photos/screens) using MASt3R 3D output |
| **3D Alignment** | MASt3R-SfM sparse_global_alignment | Multi-view coordinate system unification |
| **Point Cloud Ops** | Open3D, scipy.spatial | ICP registration, KDTree queries, statistical outlier removal |
| **Deep Learning** | PyTorch 2.x + CUDA 12.x | MASt3R inference runtime (FP16) |
| **Backend API** | FastAPI + uvicorn | Async WebSocket (enrollment) + REST (auth, user management) |
| **Frontend** | Gradio | Rapid-prototype UI: webcam, pose guide, score visualization |
| **Storage** | SQLite + NumPy .npz | User metadata DB + template archives (point cloud, descriptors, embedding) |

---

## 3. Pipeline Architecture

### 3.1 Enrollment Pipeline

The enrollment process captures the user's face from multiple angles and builds a multi-modal identity template.

```
 Webcam Capture                    3D Reconstruction                 Template Creation
┌───────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────────┐
│ MediaPipe │    │ Keyframe │    │   MASt3R     │    │  Global  │    │ Post-Process │
│ Face Mesh │ →  │ Selector │ →  │  Pairwise   │ →  │ Alignment│ →  │ (conf→voxel  │
│ Detection │    │ (12 pose │    │  Inference   │    │ (SfM)    │    │  →outlier    │
│ + PnP     │    │  targets)│    │  (FP16)     │    │          │    │  →cluster)   │
└───────────┘    └──────────┘    └──────────────┘    └──────────┘    └──────┬───────┘
                                                                           │
                                        ┌──────────────┐                   │
                                        │   ArcFace    │                   │
                                        │  Embedding   │ ←─── face crops ──┘
                                        │  (512-dim)   │
                                        └──────┬───────┘
                                               │
                                        ┌──────▼───────┐
                                        │   Template   │  point_cloud (N,3)
                                        │   .npz File  │  descriptors (N,D)
                                        │   + SQLite   │  face_embedding (512,)
                                        └──────────────┘  confidence, colors
```

**Keyframe Capture — 12 Frames on a 4x3 Pose Grid:**

The system guides the user to look in 12 target directions, covering a grid of 4 yaw angles and 3 pitch angles:

| | Yaw -25° | Yaw -8.3° | Yaw +8.3° | Yaw +25° |
|---|---|---|---|---|
| **Pitch +15°** | Upper-Left | Up-Left | Up-Right | Upper-Right |
| **Pitch 0°** | Left | Center-Left | Center-Right | Right |
| **Pitch -15°** | Lower-Left | Down-Left | Down-Right | Lower-Right |

- **Capture tolerance**: ±7° from each target center
- **Quality filter**: Laplacian blur score ≥ 15.0, detection confidence ≥ 0.9
- The grid factorization prefers more yaw variety (horizontal) than pitch, as horizontal parallax contributes more to 3D reconstruction quality.

**3D Reconstruction Steps:**

1. **Pose-aware pair generation**: Each frame is paired with its K=8 angularly-nearest neighbors (not all-pairs) to avoid useless cross-profile matching
2. **MASt3R pairwise inference**: Each pair → dense 3D pointmaps + confidence + descriptors (FP16, 512px input)
3. **Global alignment**: MASt3R-SfM's sparse_global_alignment unifies all pairwise results into a single coordinate system
4. **Post-processing pipeline**:
   - Confidence filter: keep points with confidence ≥ 1.5 (MASt3R uses positive reals, >1 = reliable)
   - Voxel deduplication: 10mm grid, keep highest-confidence point per voxel
   - Statistical outlier removal: k=30, std_ratio=1.2
   - Cluster filter: k=10, eps=11mm, keep largest connected component

**ArcFace Embedding**: Extract 512-dim L2-normalized identity embedding from face crops, aggregated across frames via quality-weighted mean.

### 3.2 Authentication Pipeline

Authentication captures a smaller set of frames and matches against stored templates.

```
 Probe Capture           3D Reconstruction        Anti-Spoof    Multi-Modal Matching
┌───────────┐          ┌──────────────┐          ┌──────────┐  ┌──────────────────────┐
│ 5 frames  │          │   MASt3R     │          │ 3D Shape │  │ Embedding (ArcFace)  │
│ diamond   │ →  3D →  │ Reconstruct  │ → cloud →│ Analysis │→ │ Geometric (ICP)      │→ Fuse
│ pattern   │          │ + Embedding  │          │ Pass?    │  │ Descriptor (NN)      │
└───────────┘          └──────────────┘          └──────────┘  └──────────┬───────────┘
                                                                          │
                                                   ┌──────────────────────▼──────┐
                                                   │        Score Fusion         │
                                                   │  0.40 × emb + 0.10 × geo   │
                                                   │  + 0.50 × desc              │
                                                   │  Accept if score ≥ 0.65     │
                                                   └────────────────────────────────┘
```

**Probe Capture — 5 Frames in a Diamond Pattern:**

| Frame | Yaw | Pitch | Direction |
|---|---|---|---|
| 1 | 0° | 0° | Center (straight ahead) |
| 2 | -15° | +5° | Upper-Right |
| 3 | +15° | +5° | Upper-Left |
| 4 | -15° | -5° | Lower-Right |
| 5 | +15° | -5° | Lower-Left |

- **Capture tolerance**: ±10° (more lenient than enrollment)
- The diamond pattern maximizes parallax for 3D reconstruction while keeping the capture process fast.

**Matching Pipeline (per template):**

1. **Anti-spoofing gate**: Must pass depth variance and planarity checks (see Section 7)
2. **Embedding matching**: Cosine similarity of ArcFace embeddings → score ∈ [0, 1]
3. **Geometric matching**: ICP alignment + Chamfer distance on 3D point clouds → score ∈ [0, 1]
4. **Descriptor matching**: Reciprocal nearest-neighbor matching on MASt3R descriptors → score ∈ [0, 1]
5. **Score fusion**: Weighted sum → final decision

**Fusion Weights** (tuned on public dataset evaluation):

| Channel | Weight | Signal |
|---|---|---|
| **ArcFace Embedding** | 0.40 | Primary identity discrimination |
| **MASt3R Descriptor** | 0.50 | View-correspondence features |
| **Geometric (ICP)** | 0.10 | 3D shape similarity |
| **Accept Threshold** | 0.65 | Decision boundary |

---

## 4. Key Technology: MediaPipe Face Mesh

### 4.1 Overview

**MediaPipe Face Mesh** is Google's real-time face analysis solution that detects facial landmarks — 478 3D points that form a dense mesh covering the entire face surface, including the iris.

Unlike bounding-box-only detectors, Face Mesh provides **geometric understanding** of the face: where each facial feature is located in both 2D image coordinates and estimated 3D space. This enables our system to perform head pose estimation and guided keyframe capture.

### 4.2 Architecture

The pipeline consists of two stages:

1. **Face Detection (BlazeFace)**: A lightweight single-shot detector locates the face region in the image. BlazeFace is optimized for mobile inference and produces a bounding box with 6 key points (eyes, nose, mouth, ears) in under 5ms.

2. **Face Landmark Regression (Face Mesh Network)**: Given the cropped face region, a regression network predicts 478 landmark positions. Each landmark has:
   - **(x, y)** — 2D pixel coordinates in the image
   - **(x, y, z)** — Normalized 3D coordinates where z represents relative depth

   The model uses a **convolutional attention architecture** trained on a large annotated dataset of face images with ground-truth 3D landmark positions.

### 4.3 Head Pose Estimation via PnP

Our system uses 6 key landmarks from the 478-point mesh to estimate head orientation:

- **Nose tip** (landmark 1)
- **Chin** (landmark 152)
- **Left eye outer corner** (landmark 263)
- **Right eye outer corner** (landmark 33)
- **Left mouth corner** (landmark 287)
- **Right mouth corner** (landmark 57)

These 6 detected 2D points are matched against a **canonical 3D face model** (a set of known 3D coordinates representing an "average" face). Using OpenCV's `solvePnP` (Perspective-n-Point algorithm), the system solves for the rotation and translation that best maps the 3D model onto the observed 2D points:

```
solvePnP(3D_model_points, 2D_image_points, camera_matrix) → (rotation, translation)
```

The rotation is then decomposed into **yaw** (left-right turn), **pitch** (up-down nod), and **roll** (head tilt) angles in degrees. These angles drive the pose-guided keyframe selection system, ensuring the user captures faces from the required viewing directions.

### 4.4 Role in Our System

- **Face detection & cropping**: Locate the face and crop with 30% padding (optimal for MASt3R — enough texture context without background corruption)
- **Head pose estimation**: Continuously track yaw/pitch/roll to guide users toward target poses
- **Quality gating**: Reject frames with excessive blur (Laplacian variance < 15.0) or roll (> 15°)

---

## 5. Key Technology: MASt3R

### 5.1 Overview

**MASt3R** (Matching And Stereo 3D Reconstruction) is a state-of-the-art model from Naver Labs that performs **dense 3D reconstruction from pairs of RGB images**. Given two photographs of the same scene taken from different viewpoints, MASt3R predicts a 3D point for every pixel — producing dense pointmaps, per-pixel confidence scores, and local feature descriptors.

Unlike traditional stereo methods that require calibrated cameras, MASt3R works with **uncalibrated images** and predicts geometry in **real-world metric scale** (meters).

### 5.2 Architecture: From DUSt3R to MASt3R

MASt3R builds upon **DUSt3R** (Dense and Universal Stereo 3D Reconstruction), extending it with local feature matching capabilities.

**DUSt3R Base Architecture:**

```
Image A ──→ ┌───────────────┐     ┌──────────────────────┐     ┌─────────────┐
            │   ViT Encoder │ ──→ │  Cross-Attention     │ ──→ │  Regression │ → Pointmap A (H,W,3)
Image B ──→ │  (ViT-Large)  │ ──→ │  Decoder             │ ──→ │  Heads      │ → Pointmap B (H,W,3)
            └───────────────┘     └──────────────────────┘     │             │ → Confidence (H,W)
                                                               └─────────────┘
```

1. **ViT Encoder**: Each image is independently encoded by a Vision Transformer (ViT-Large) into a sequence of patch tokens. The image is divided into non-overlapping 16×16 patches, each projected into a high-dimensional embedding, then processed through transformer layers with self-attention.

2. **Cross-Attention Decoder**: The two sets of patch tokens are fed into a decoder that performs **cross-attention** — allowing tokens from Image A to attend to tokens from Image B, and vice versa. This enables the network to establish correspondences between the two views implicitly, without explicit feature matching.

3. **Regression Heads**: Per-pixel outputs:
   - **Pointmap**: (H, W, 3) — 3D coordinates for every pixel, expressed in the coordinate frame of Image A
   - **Confidence**: (H, W) — reliability score per pixel (positive reals; values > 1 indicate reliable predictions)

**MASt3R Extension:**

MASt3R adds a **local feature head** that outputs dense descriptors alongside the pointmaps. These descriptors enable explicit matching between views, improving reconstruction accuracy for multi-view scenarios.

### 5.3 Multi-View Reconstruction via Global Alignment

For enrollment (12 frames), we need to combine multiple pairwise reconstructions into a unified 3D model:

1. **Pose-aware pair generation**: Pairs are selected by angular proximity of head poses (K=8 nearest neighbors per frame), avoiding useless pairs between very different viewing angles.

2. **Pairwise inference**: MASt3R processes each pair independently, producing pointmaps in local coordinate frames.

3. **Sparse global alignment** (MASt3R-SfM): A joint optimization that:
   - Estimates camera poses for all frames simultaneously
   - Unifies all pairwise pointmaps into a single global coordinate system
   - Optimizes via gradient descent (coarse: lr=0.07, 300 iters → fine: lr=0.014, 500 iters)

4. **Post-processing**: The raw point cloud is cleaned through a multi-stage pipeline:
   - **Confidence filter** (≥ 1.5): Removes uncertain predictions
   - **Voxel deduplication** (10mm): Merges duplicate points from overlapping views
   - **Statistical outlier removal** (k=30, std_ratio=1.2): Removes isolated noise
   - **Cluster filter** (k=10, eps=11mm): Keeps only the largest connected component

   This produces a clean point cloud of ~35K points representing the 3D face surface.

### 5.4 Limitations and Complementary Approach

A critical finding from our evaluation: **MASt3R descriptors have zero identity discrimination**. They are designed for view correspondence (matching the same physical point across different images), not for distinguishing one person from another. Our weight optimization experiments yielded EER ≈ 0.46 (near random chance) when using descriptors alone. This motivated the integration of ArcFace as the primary identity signal.

---

## 6. Key Technology: ArcFace

### 6.1 The Problem: Why Not Just Use Softmax?

Standard face recognition models are trained with softmax classification: given a face image, predict which person it belongs to. However, standard softmax only requires features to be *separable* — it does not enforce any *margin* between classes. This means the learned feature space may not generalize well to unseen identities.

### 6.2 The ArcFace Solution: Additive Angular Margin

**ArcFace** (Additive Angular Margin Loss) reformulates the classification objective in angular space. Instead of measuring Euclidean distance between feature vectors, it operates on the **angle** between the feature vector and class weight vectors on a hypersphere.

**Key idea**: After L2-normalizing both the feature vector **x** and the class weights **W**, the logit for class *i* becomes the cosine of the angle between them:

```
logit_i = cos(θ_i)    where θ_i = angle(x, W_i)
```

ArcFace adds an **angular margin penalty** *m* to the target class angle:

```
L = -log( e^(s · cos(θ_y + m)) / (e^(s · cos(θ_y + m)) + Σ_{j≠y} e^(s · cos(θ_j))) )
```

Where:
- **θ_y** = angle between the feature and the correct class weight
- **m** = additive angular margin (typically 0.5 radians)
- **s** = scale factor (typically 64)
- The summation is over all incorrect classes

**Effect**: During training, the model must learn features where the angle to the correct class is at least *m* radians smaller than to any incorrect class. This forces:
- **Tight intra-class clustering**: Features of the same person are pulled closer together on the hypersphere
- **Wide inter-class separation**: Features of different people are pushed further apart

The result is a feature space where identity similarity can be reliably measured by **cosine similarity** of the normalized embeddings.

### 6.3 Our Implementation

- **Model**: insightface `buffalo_l` bundle — SCRFD face detector + ArcFace R100 recognition backbone
- **Output**: 512-dimensional L2-normalized embedding per face image
- **Multi-frame aggregation**: Embeddings from multiple frames are averaged and re-normalized
- **Matching**: Cosine similarity mapped to [0, 1] via `score = (cosine + 1) / 2`
- **Weight in fusion**: 0.40 — the primary identity signal

### 6.4 Why ArcFace Was Essential

MASt3R produces excellent 3D reconstructions but its descriptors carry **no identity information** (EER ≈ 0.46, equivalent to random guessing). ArcFace provides the complementary capability: strong identity discrimination from 2D face appearance, independent of 3D geometry. Together, the two technologies cover both structural (3D shape) and identity (appearance) aspects of face authentication.

---

## 7. Key Technology: Anti-Spoofing via 3D Shape Analysis

### 7.1 The Threat

Presentation attacks — holding up a photo, displaying a face on a screen, or using a printed mask — can fool 2D-only face recognition systems. These attacks share a common trait: **the presented face is geometrically flat**.

### 7.2 Our Approach: Exploiting 3D Reconstruction

Since MASt3R already produces a 3D point cloud of the face during reconstruction, we can analyze its geometric properties to detect flat presentations at no additional computational cost.

**Two complementary checks:**

1. **Depth Variance Test**: Compute the standard deviation of z-coordinates across all reconstructed points. Real faces exhibit significant depth variation (nose protrudes, cheeks curve, eye sockets are recessed). Flat media produce near-zero z-variance.
   - **Threshold**: σ_z ≥ 3mm (0.003 meters)

2. **PCA Planarity Test**: Perform Principal Component Analysis on the 3D point cloud and compute the ratio of the smallest eigenvalue to the largest. A perfectly flat surface has a near-zero smallest eigenvalue (all variance lies in 2 dimensions). A real 3D face distributes variance across all three axes.
   - **Threshold**: λ_min / λ_max ≥ 0.05

Both checks must pass, along with a minimum MASt3R confidence mean (≥ 0.7), for the input to be accepted as genuine. If anti-spoofing fails, the system immediately rejects the authentication attempt without proceeding to identity matching.

### 7.3 Advantage

This approach requires **no additional neural network** — it is a pure geometric analysis of the 3D output that MASt3R already produces. This makes it lightweight, interpretable, and inherently tied to the physical geometry of the input rather than learned artifacts.
