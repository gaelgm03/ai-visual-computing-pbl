# Face ID Project: Technical Implementation Detail

## Table of Contents

- [Library Selection](#library-selection)
- [Repository Structure](#repository-structure)
- [Module Interfaces](#module-interfaces)
  - [config.py](#configpy--shared-by-all)
  - [detector.py](#detectorpy--cs-a)
  - [recognizer.py](#recognizerpy--cs-b)
  - [enrollment.py](#enrollmentpy--cs-a)
  - [liveness.py](#livenesspy--cs-b-design--ds-a-training)
  - [challenge.py](#challengepy--cs-b)
  - [training/train_liveness.py](#trainingtrain_livenesspy--ds-a)
  - [evaluation/metrics.py](#evaluationmetricspy--ds-b)
- [Collaboration Flow](#collaboration-flow)
  - [Git Branching Strategy](#git-branching-strategy)
  - [Critical Handoff Points](#critical-handoff-points)

---

## Library Selection

| Purpose | Library | Reason |
|---|---|---|
| Camera & Image Processing | **OpenCV (`cv2`)** | Webcam capture, image manipulation, drawing overlays |
| Face Detection + Alignment | **insightface** | Unified API for RetinaFace detection + ArcFace embedding extraction |
| Liveness Model | **PyTorch + torchvision** | Pretrained MobileNetV3-Small weights available out of the box |
| Landmark / Blink Detection | **mediapipe** | 468-point face mesh enables easy EAR (Eye Aspect Ratio) computation; lightweight |
| Demo UI | **Streamlit** | Easy webcam integration, Python-only, low learning curve |
| Evaluation & Visualization | **scikit-learn + matplotlib** | Standard tools for ROC curves, confusion matrices, etc. |
| Data Handling | **numpy, pandas, pickle** | Embedding storage, evaluation result aggregation |

---

## Repository Structure

```
face-id/
├── README.md
├── requirements.txt
├── config.py                        # Shared configuration values
│
├── modules/                         # Each member develops their assigned module independently
│   ├── __init__.py
│   ├── detector.py                  # [CS-A] Face detection + alignment
│   ├── enrollment.py                # [CS-A] Enrollment pipeline
│   ├── recognizer.py                # [CS-B] ArcFace embedding + verification
│   ├── liveness.py                  # [CS-B design → DS-A training] Liveness module
│   └── challenge.py                 # [CS-B] Challenge-response (blink / head turn)
│
├── training/                        # DS-A owned
│   ├── train_liveness.py            # Liveness training script
│   ├── dataset.py                   # Data loader
│   └── augmentation.py              # Augmentation definitions
│
├── evaluation/                      # DS-B owned
│   ├── evaluate_recognition.py      # ArcFace threshold optimization + recognition accuracy
│   ├── evaluate_liveness.py         # Liveness detection rate evaluation
│   ├── evaluate_e2e.py              # End-to-end evaluation
│   ├── metrics.py                   # FAR, FRR, ROC, etc.
│   └── visualize.py                 # Graphs, confusion matrix rendering
│
├── app/                             # CS-B owned (Phase 3)
│   └── demo.py                      # Streamlit demo UI
│
├── data/                            # Add to .gitignore
│   ├── raw/                         # Original datasets
│   ├── processed/                   # Preprocessed data
│   └── custom/                      # Team member captured data
│
├── models/                          # Saved models
│   ├── liveness_model.pth
│   └── templates/                   # Enrolled identity templates
│
└── notebooks/                       # Exploration / experiments (optional)
    └── exploration.ipynb
```

---

## Module Interfaces

> **The key to parallel team development is agreeing on interfaces upfront.**
> As long as each module respects the defined Inputs/Outputs below, members can develop implementations independently and integration will be seamless.

### config.py — Shared by All

```python
# ============================================================
# Shared project configuration
# ============================================================

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Detection & Alignment ---
ALIGNED_FACE_SIZE = (112, 112)

# --- ArcFace ---
EMBEDDING_DIM = 512
SIMILARITY_THRESHOLD = 0.45          # DS-B will optimize this value

# --- Liveness ---
LIVENESS_THRESHOLD = 0.5
LIVENESS_MODEL_PATH = "models/liveness_model.pth"

# --- Enrollment ---
ENROLLMENT_DURATION = 30             # seconds
ENROLLMENT_NUM_KEYFRAMES = 8
TEMPLATE_DIR = "models/templates/"

# --- Challenge-Response ---
BLINK_EAR_THRESHOLD = 0.21
HEAD_TURN_YAW_THRESHOLD = 25        # degrees
CHALLENGE_TIMEOUT = 5                # seconds per challenge
```

---

### detector.py — CS-A

This is the foundational module that **all other modules depend on**. It should be built and tested first.

```python
"""
Face detection + alignment module.

Pipeline: Raw frame → RetinaFace detection → 5-point landmark extraction
          → Affine alignment → 112x112 normalized face crop

Dependencies: insightface, numpy, opencv-python
"""
import numpy as np


class FaceDetector:
    def __init__(self):
        """Load RetinaFace model via insightface."""
        ...

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Detect faces in a BGR frame.

        Args:
            frame: BGR image of shape (H, W, 3), dtype uint8

        Returns:
            List of dicts, each containing:
            {
                "bbox": np.ndarray,        # shape (4,) — [x1, y1, x2, y2]
                "landmarks": np.ndarray,   # shape (5, 2) — left_eye, right_eye,
                                           #   nose, mouth_left, mouth_right
                "confidence": float        # detection confidence score
            }
            Empty list if no face detected.
        """
        ...

    def align(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Align and crop a face using 5-point landmarks.

        Applies a similarity transform (rotation + scale + translation)
        to map the detected landmarks to canonical positions, then crops
        and resizes to (112, 112).

        Args:
            frame: Original BGR image, shape (H, W, 3), dtype uint8
            landmarks: shape (5, 2), landmark coordinates in pixel space

        Returns:
            Aligned face image, shape (112, 112, 3), dtype uint8, BGR
        """
        ...
```

**Key implementation notes for CS-A:**

- `insightface.app.FaceAnalysis` provides both detection and landmark extraction in a single call.
- The alignment step uses a standard similarity transform based on 5 reference points. The `insightface` library includes `insightface.utils.face_align.norm_crop()` which handles this, but implementing it manually (using `cv2.estimateAffinePartial2D`) is also straightforward and gives more control.
- Always return results sorted by confidence (highest first) so callers can simply take `results[0]` for the primary face.

---

### recognizer.py — CS-B

```python
"""
ArcFace embedding extraction + cosine similarity verification.

Pipeline: Aligned face (112x112) → ArcFace backbone (ResNet-100)
          → 512-d embedding → L2 normalization → cosine similarity

Dependencies: insightface, numpy
Depends on: detector.py (for aligned face images)
"""
import numpy as np


class FaceRecognizer:
    def __init__(self):
        """
        Load ArcFace pretrained model via insightface.

        The model used is 'buffalo_l' which includes a ResNet-100
        backbone trained with ArcFace loss on a large-scale face dataset.
        """
        ...

    def extract_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Extract a 512-dimensional L2-normalized embedding.

        Args:
            aligned_face: shape (112, 112, 3), dtype uint8, BGR

        Returns:
            np.ndarray of shape (512,), L2-normalized
            (i.e., np.linalg.norm(result) ≈ 1.0)
        """
        ...

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two L2-normalized embeddings.

        Since both vectors are L2-normalized, this is simply their dot product:
            similarity = emb1 · emb2

        Returns:
            float in range [-1.0, 1.0] (practically [0.0, 1.0] for face embeddings)
        """
        ...

    def verify(
        self,
        embedding: np.ndarray,
        template: np.ndarray,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> tuple[bool, float]:
        """
        Verify whether an embedding matches a stored template.

        Args:
            embedding: Query embedding, shape (512,)
            template: Stored identity template, shape (512,)
            threshold: Decision threshold (optimized by DS-B)

        Returns:
            (is_match, similarity)
            - is_match: True if similarity > threshold
            - similarity: The computed cosine similarity score
        """
        ...
```

**Key implementation notes for CS-B:**

- The `insightface` model object exposes a `.get_embedding()` method, but wrapping it gives us control over preprocessing and normalization.
- The threshold in config is a placeholder. DS-B will determine the optimal value via ROC analysis on LFW pairs.

---

### enrollment.py — CS-A

```python
"""
Enrollment pipeline: guided face capture + template generation.

Workflow:
  1. Open camera and display real-time guidance overlays
  2. Prompt user to show face from multiple angles (front, left, right, up, down)
  3. Auto-select keyframes based on pose diversity and image quality
  4. Extract embeddings from keyframes, average them, L2-normalize → template
  5. Save template to disk

Dependencies: opencv-python, numpy, pickle
Depends on: detector.py (FaceDetector), recognizer.py (FaceRecognizer)
"""
import numpy as np


class EnrollmentPipeline:
    def __init__(self, detector: "FaceDetector", recognizer: "FaceRecognizer"):
        """
        Args:
            detector: Instance of modules.detector.FaceDetector
            recognizer: Instance of modules.recognizer.FaceRecognizer
        """
        ...

    def capture_keyframes(
        self,
        camera,
        duration: int = 30,
        num_keyframes: int = 8,
    ) -> list[np.ndarray]:
        """
        Guide user through face registration, selecting diverse-angle keyframes.

        During the capture window:
          - Display text prompts ("Look straight", "Turn left", etc.)
          - Continuously detect face and estimate head pose (yaw/pitch)
          - Score each frame by: detection confidence, blur score (Laplacian
            variance), and pose diversity relative to already-selected frames
          - Select top-N frames that maximize angular coverage

        Args:
            camera: cv2.VideoCapture object (already opened)
            duration: Maximum capture time in seconds
            num_keyframes: Target number of keyframes to select

        Returns:
            List of aligned face images, each shape (112, 112, 3)
        """
        ...

    def create_template(self, keyframes: list[np.ndarray]) -> np.ndarray:
        """
        Generate an identity template from selected keyframes.

        Process:
          1. Extract embedding for each keyframe via recognizer
          2. Compute element-wise mean of all embeddings
          3. L2-normalize the mean vector

        Args:
            keyframes: List of aligned face images (112, 112, 3)

        Returns:
            np.ndarray of shape (512,), L2-normalized identity template
        """
        ...

    def save_template(self, user_id: str, template: np.ndarray) -> None:
        """
        Save template to disk as a .pkl file.

        Path: {TEMPLATE_DIR}/{user_id}.pkl
        """
        ...

    def load_template(self, user_id: str) -> np.ndarray:
        """
        Load a previously saved template from disk.

        Returns:
            np.ndarray of shape (512,)

        Raises:
            FileNotFoundError: If no template exists for user_id
        """
        ...
```

---

### liveness.py — CS-B (design) → DS-A (training)

This module has **split ownership**: CS-B defines the architecture and inference API; DS-A trains the model and delivers the weights.

```python
"""
2D Liveness detection — binary classification: Real vs Spoof.

Architecture: MobileNetV3-Small (pretrained on ImageNet) with the final
classifier replaced by nn.Linear(576, 2).

- CS-B: Defines build_model() and predict() interface
- DS-A: Uses build_model() for training, saves weights to LIVENESS_MODEL_PATH

Dependencies: torch, torchvision, numpy, opencv-python
"""
import numpy as np
import torch
import torch.nn as nn


class LivenessDetector:
    def __init__(self, model_path: str = LIVENESS_MODEL_PATH):
        """
        Load the trained liveness model from disk.

        The model file is produced by DS-A's training pipeline.
        """
        ...

    def predict(self, aligned_face: np.ndarray) -> tuple[bool, float]:
        """
        Predict whether a face is real or spoofed.

        Preprocessing:
          1. Resize to 224x224 (MobileNetV3 input size)
          2. Normalize with ImageNet mean/std
          3. Convert to tensor, add batch dimension

        Args:
            aligned_face: shape (112, 112, 3), dtype uint8, BGR

        Returns:
            (is_real, confidence)
            - is_real: True if classified as real face
            - confidence: Softmax probability of the predicted class
        """
        ...

    @staticmethod
    def build_model() -> nn.Module:
        """
        Construct the liveness detection model architecture.

        Architecture details:
          - Base: torchvision.models.mobilenet_v3_small(pretrained=True)
          - Replace classifier[-1]: nn.Linear(576, 2)
          - Output: 2-class logits (index 0 = spoof, index 1 = real)

        This method is used by DS-A in the training script.

        Returns:
            nn.Module ready for training
        """
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        model.classifier[-1] = nn.Linear(576, 2)
        return model
```

---

### challenge.py — CS-B

```python
"""
Challenge-response liveness detection.

Complements the static liveness model with active checks that require
real-time human responses. This defeats replay attacks where a pre-recorded
video of the target person is played to the camera.

Techniques:
  - Blink detection via Eye Aspect Ratio (EAR)
  - Head turn detection via landmark-based yaw estimation

Dependencies: mediapipe, opencv-python, numpy
"""


class ChallengeResponse:
    def __init__(self):
        """
        Initialize MediaPipe FaceMesh for 468-point landmark tracking.
        """
        ...

    def compute_ear(self, landmarks, eye_indices: list[int]) -> float:
        """
        Compute Eye Aspect Ratio (EAR) for blink detection.

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        When the eye is open, EAR ≈ 0.25–0.35.
        When the eye is closed, EAR drops below ~0.21.

        Args:
            landmarks: MediaPipe face landmarks
            eye_indices: Indices for the 6 eye contour points

        Returns:
            float: EAR value
        """
        ...

    def estimate_yaw(self, landmarks) -> float:
        """
        Estimate head yaw angle from face landmarks.

        Uses the horizontal displacement ratio between nose tip and
        face edges as a proxy for yaw rotation.

        Returns:
            float: Estimated yaw angle in degrees
                   Negative = looking left, Positive = looking right
        """
        ...

    def request_blink(self, camera, timeout: int = 5) -> bool:
        """
        Prompt user to blink and detect it via EAR.

        Detection logic:
          1. Track EAR over consecutive frames
          2. A blink is registered when EAR drops below threshold
             for at least 2 consecutive frames, then rises back above

        Args:
            camera: cv2.VideoCapture object
            timeout: Maximum wait time in seconds

        Returns:
            True if a blink was detected within the timeout
        """
        ...

    def request_head_turn(
        self, camera, direction: str, timeout: int = 5
    ) -> bool:
        """
        Prompt user to turn their head in a specified direction.

        Detection logic:
          1. Record initial yaw angle
          2. Monitor for sufficient yaw change (> HEAD_TURN_YAW_THRESHOLD)
             in the requested direction

        Args:
            camera: cv2.VideoCapture object
            direction: "left" or "right"
            timeout: Maximum wait time in seconds

        Returns:
            True if sufficient head turn was detected
        """
        ...

    def run_random_challenge(self, camera) -> bool:
        """
        Run 1–2 randomly selected challenges. All must pass.

        Randomly picks from: [blink, turn_left, turn_right].
        Displays on-screen instructions for each challenge.

        Returns:
            True if all challenges passed
        """
        ...
```

---

### training/train_liveness.py — DS-A

```python
"""
Training script for the liveness binary classifier.

Uses the architecture defined in modules.liveness.LivenessDetector.build_model().

Expected data directory structure:
    data_dir/
    ├── train/
    │   ├── real/       # Real face images
    │   └── spoof/      # Print attack, replay attack, mask images
    └── val/
        ├── real/
        └── spoof/

Training details:
  - Loss: CrossEntropyLoss
  - Optimizer: Adam
  - LR schedule: CosineAnnealingLR
  - Augmentation: Random horizontal flip, color jitter, random rotation
  - Early stopping based on validation accuracy

Outputs:
  - Best model checkpoint → models/liveness_model.pth
  - Training log (loss/accuracy per epoch) → training/training_log.csv

Dependencies: torch, torchvision, PIL, pandas
"""


def train(data_dir: str, epochs: int = 20, lr: float = 1e-4, batch_size: int = 32):
    """
    Train the liveness detection model.

    Steps:
      1. Load model via LivenessDetector.build_model()
      2. Freeze base layers (optional — experiment with fine-tuning depth)
      3. Build train/val DataLoaders with augmentation
      4. Train loop with validation after each epoch
      5. Save best model (highest val accuracy) to LIVENESS_MODEL_PATH

    Args:
        data_dir: Root directory containing train/ and val/ subdirectories
        epochs: Maximum number of training epochs
        lr: Initial learning rate
        batch_size: Batch size for DataLoader
    """
    ...
```

---

### evaluation/metrics.py — DS-B

```python
"""
Evaluation metrics for face recognition and liveness detection.

Provides reusable functions for computing standard biometric metrics
and generating publication-quality visualizations.

Dependencies: numpy, scikit-learn, matplotlib
"""
import numpy as np


def compute_far_frr(
    similarities: list[float],
    labels: list[bool],
    threshold: float,
) -> tuple[float, float]:
    """
    Compute False Accept Rate and False Reject Rate at a given threshold.

    FAR = (false accepts) / (total impostor attempts)
    FRR = (false rejects) / (total genuine attempts)

    Args:
        similarities: List of cosine similarity scores
        labels: True = genuine pair, False = impostor pair
        threshold: Decision threshold

    Returns:
        (FAR, FRR) as floats in [0.0, 1.0]
    """
    ...


def find_optimal_threshold(
    similarities: list[float],
    labels: list[bool],
) -> float:
    """
    Find the threshold where FAR ≈ FRR (Equal Error Rate point).

    Sweeps thresholds from 0.0 to 1.0 and finds the crossing point
    where FAR and FRR are minimally different.

    Returns:
        Optimal threshold (float)
    """
    ...


def plot_roc_curve(
    similarities: list[float],
    labels: list[bool],
    save_path: str | None = None,
) -> float:
    """
    Plot ROC curve (TAR vs FAR) and compute AUC.

    Args:
        similarities: Cosine similarity scores
        labels: Ground truth (True = genuine, False = impostor)
        save_path: If provided, save the figure to this path

    Returns:
        AUC (Area Under Curve) value
    """
    ...


def plot_similarity_distribution(
    pos_sims: list[float],
    neg_sims: list[float],
    threshold: float | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot overlapping histograms of genuine vs impostor similarity scores.

    This visualization makes it intuitive to see the separation between
    the two distributions and where the threshold sits.

    Args:
        pos_sims: Similarity scores for genuine (same-identity) pairs
        neg_sims: Similarity scores for impostor (different-identity) pairs
        threshold: If provided, draw a vertical line at this threshold
        save_path: If provided, save the figure to this path
    """
    ...


def confusion_matrix_report(
    y_true: list[int],
    y_pred: list[int],
    labels: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """
    Generate and plot a confusion matrix with precision/recall/F1.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Class label names (e.g., ["Impostor", "Genuine"])
        save_path: If provided, save the figure to this path
    """
    ...
```

---

## Collaboration Flow

### Git Branching Strategy

```
main                            ← Production-ready releases only
 └── dev                        ← Integration branch (merge target for all features)
      ├── feature/detector           # CS-A
      ├── feature/enrollment         # CS-A
      ├── feature/recognizer         # CS-B
      ├── feature/liveness           # CS-B (architecture) → DS-A (training)
      ├── feature/challenge          # CS-B
      ├── feature/training           # DS-A
      ├── feature/evaluation         # DS-B
      └── feature/demo               # CS-B
```

**Workflow:**

1. Each member works on their own `feature/` branch
2. When a feature is ready, open a Pull Request to `dev`
3. At least 1 other team member reviews before merging
4. `dev` → `main` merge only at phase milestones after team-wide testing

### Critical Handoff Points

```
[Phase 0 — Kickoff]
  All  →  All:   Liveness architecture jointly decided
                  → DS-A can begin training in Phase 1

[Phase 1 — End]
  CS-A →  All:   detector.py complete and merged to dev
                  → All modules can now use detect() and align()

[Phase 2 — Mid]
  CS-B →  DS-B:  recognizer.py complete and merged to dev
                  → DS-B begins threshold optimization on LFW pairs

[Phase 2 — End]
  DS-A →  CS-B:  liveness_model.pth delivered
                  → CS-B integrates into liveness.py predict()

  DS-B →  All:   Optimal SIMILARITY_THRESHOLD determined
                  → config.py updated

[Phase 3]
  CS-A + CS-B:   Full pipeline integration
  DS-A + DS-B:   End-to-end system evaluation
```

---

## End-to-End Pipeline Summary

```
                         ┌─────────────────────────┐
                         │   Webcam Input (frame)   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   FaceDetector.detect()  │
                         │   → bbox + 5 landmarks   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   FaceDetector.align()   │
                         │   → 112×112 aligned face │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                     │
              [ENROLLMENT]                         [VERIFICATION]
                    │                                     │
                    ▼                                     ▼
        ┌───────────────────┐              ┌──────────────────────┐
        │ Capture keyframes │              │  ChallengeResponse   │
        │ (multi-angle)     │              │  .run_random_challenge│
        └────────┬──────────┘              └──────────┬───────────┘
                 │                                    │
                 ▼                                    │ Pass?
        ┌───────────────────┐                    ┌────┴────┐
        │ FaceRecognizer    │                No ←┤         ├→ Yes
        │ .extract_embedding│                    └────┬────┘
        │ (× N keyframes)  │                         │
        └────────┬──────────┘                         ▼
                 │                         ┌──────────────────────┐
                 ▼                         │  LivenessDetector    │
        ┌───────────────────┐              │  .predict()          │
        │ Average + L2 norm │              └──────────┬───────────┘
        │ → template (512d) │                         │
        └────────┬──────────┘                    ┌────┴────┐
                 │                           No ←┤ Real?   ├→ Yes
                 ▼                               └────┬────┘
        ┌───────────────────┐                         │
        │ Save template     │                         ▼
        │ (.pkl to disk)    │              ┌──────────────────────┐
        └───────────────────┘              │  FaceRecognizer      │
                                           │  .extract_embedding()│
                                           └──────────┬───────────┘
                                                      │
                                                      ▼
                                           ┌──────────────────────┐
                                           │  FaceRecognizer      │
                                           │  .verify()           │
                                           │  (cosine similarity) │
                                           └──────────┬───────────┘
                                                      │
                                                 ┌────┴────┐
                                                 │ sim > θ │
                                                 └────┬────┘
                                                 │         │
                                              Yes │         │ No
                                                 ▼         ▼
                                           ✅ ACCEPT   ❌ REJECT
```