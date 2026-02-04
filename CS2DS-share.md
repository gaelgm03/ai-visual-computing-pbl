# MASt3R Face Authentication — Data Science Team

> **Audience**: DS-1 (Matching Algorithm & Threshold Tuning) and DS-2 (Evaluation & Anti-Spoofing Analysis)
> **Companion Document**: `technical_architecture.md` (CS-focused; referenced as **[TA §X]** throughout)
> **Version**: 1.0 | Draft
> **Last Updated**: 2026-02-05

---

## 1. Project at a Glance

We are building a **Face ID–like prototype** that enrolls and authenticates users using only an RGB webcam — no depth sensor. The core innovation: using **MASt3R**, a state-of-the-art 3D reconstruction model, to build a dense 3D face representation from ordinary photos, and then matching it for identity verification.

### 1.1 How It Works (Simplified)

```
┌─ ENROLLMENT ──────────────────────────────────────────────────────┐
│                                                                    │
│  User rotates head in front of webcam                             │
│       ↓                                                           │
│  System captures ~12 keyframes at different angles                │
│       ↓                                                           │
│  MASt3R reconstructs a 3D face (point cloud + descriptors)       │
│       ↓                                                           │
│  Saved as "template" (the enrolled identity)                      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

┌─ AUTHENTICATION ──────────────────────────────────────────────────┐
│                                                                    │
│  User looks at webcam (2-4 frames captured)                       │
│       ↓                                                           │
│  MASt3R reconstructs a "probe" point cloud + descriptors          │
│       ↓                                                           │
│  ┌──────────────────────────────────────────────────────┐         │
│  │  ★ DS TEAM'S DOMAIN ★                               │         │
│  │                                                      │         │
│  │  Compare probe vs. template:                         │         │
│  │    Path A — 3D geometric shape similarity            │         │
│  │    Path B — Descriptor-based identity matching       │         │
│  │    Fusion — Combine scores → Accept / Reject         │         │
│  │                                                      │         │
│  │  Also: anti-spoofing analysis, evaluation metrics    │         │
│  └──────────────────────────────────────────────────────┘         │
│       ↓                                                           │
│  Result: "Match" or "No Match" (with confidence score)            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 1.2 What MASt3R Gives You (Your Input Data)

| Output | Shape | What It Is | Analogy |
|---|---|---|---|
| **Point cloud** | `(N, 3)` — N points × (x, y, z) | 3D coordinates of every visible surface point | A 3D sculpture of the face |
| **Descriptors** | `(N, D)` — N points × D-dim feature vector | A learned "fingerprint" per surface point, encoding local appearance + geometry | A unique barcode stamped on each spot of the face |
| **Confidence** | `(N,)` — one value per point | How sure MASt3R is about each point's 3D position | Quality indicator — higher = more reliable |
| **Colors** | `(N, 3)` — RGB per point | Original pixel color at each point | For visualization |

**Point cloud** tells you *shape* (is this geometrically the same face?).
**Descriptors** tell you *identity* (do the local features match, even from different angles?).

Need to design how these two signals are compared and combined into a reliable authentication decision.

---

## 2. DS Team Roles & Deliverables

### DS-1: Matching Algorithm & Threshold Tuning

| Deliverable | Format | Deadline |
|---|---|---|
| Geometric matching implementation | Python file: `core/matching/geometric_matcher.py` | Week 5 |
| Descriptor matching implementation | Python file: `core/matching/descriptor_matcher.py` | Week 5 |
| Score fusion implementation | Python file: `core/matching/score_fusion.py` | Week 5 |
| Optimized fusion weights (α, β) | Values in `config.yaml` with justification | Week 6 |
| Accept/reject threshold | A float value with statistical justification | Week 6 |

### DS-2: Evaluation & Anti-Spoofing Analysis

| Deliverable | Format | Deadline |
|---|---|---|
| Evaluation protocol & metrics | Documented in notebook / report | Week 5 |
| Anti-spoofing analysis | Notebook with statistical evidence | Week 6 |
| Anti-spoofing threshold(s) | Values with justification for CS integration | Week 6 |
| Result figures for final presentation | Exported PNG/SVG in `notebooks/figures/` | Week 7 |

### Overlap

DS-1 and DS-2 work closely: DS-1 builds the matching logic, DS-2 evaluates it and feeds results back for tuning. A build → measure → adjust loop.

---

## 3. CS ↔ DS Interface Map

### 3.1 What CS Builds

| Component | Owner | What It Does |
|---|---|---|
| Webcam capture + face detection | CS-1 | Captures frames, detects and crops face region |
| MASt3R 3D reconstruction | CS-1 | Converts images → point cloud + descriptors |
| Keyframe selection | CS-1 | Picks the best ~12 frames during enrollment |
| Template storage (save/load) | CS-1 | Saves `.npz` files, manages SQLite metadata |
| API server (FastAPI) | CS-1 | Exposes `/enroll` and `/authenticate` endpoints |
| Demo UI (Gradio) | CS-2 | Frontend that calls the API |
| Abstract matching interfaces | CS-1 | Defines the code contracts you implement |

### 3.2 What CS Gives You (Your Starting Materials)

**1. Pre-computed `.npz` files** on Google Drive, from two sources:

- **Team captures**: CS-1 records team members via webcam on the RTX 5070 Laptop, runs MASt3R, and exports `.npz`. Used for demo, anti-spoofing experiments, and initial development.
- **Public dataset**: DS team provides multi-view face dataset images (e.g., datasets containing multiple angles per subject) to CS-1, who batch-processes them through MASt3R → `.npz`. Used for large-scale evaluation, threshold tuning, and statistical validation.

Both sources produce `.npz` files in the identical format:

```python
import numpy as np
data = np.load("subject_001_enrollment.npz", allow_pickle=True)

data["point_cloud"]   # (N, 3)  — 3D face surface
data["descriptors"]   # (N, D)  — identity features per point
data["confidence"]    # (N,)    — per-point reliability
data["colors"]        # (N, 3)  — RGB for visualization
```

**2. Abstract interface classes** in `core/matching/interfaces.py`:

```python
# Already defined by CS-1. You implement concrete subclasses.

@dataclass
class MatchResult:
    score: float       # 0.0 = completely different, 1.0 = perfect match
    details: dict      # Any extra info for debugging / visualization
    is_match: bool     # Thresholded decision

class GeometricMatcher(ABC):
    @abstractmethod
    def compare(self, probe_cloud, template_cloud) -> MatchResult: ...

class DescriptorMatcher(ABC):
    @abstractmethod
    def compare(self, probe_desc, template_desc,
                probe_cloud, template_cloud) -> MatchResult: ...

class ScoreFusion(ABC):
    @abstractmethod
    def fuse(self, geometric_result, descriptor_result) -> MatchResult: ...
```

Full interface definitions: **[TA §4.5]**

### 3.3 Dependency Timeline

```
Week 1-2                  Week 3-4                     Week 5-6              Week 7-8
────────────────────────────────────────────────────────────────────────────────────────

CS-1: MASt3R setup        Enrollment pipeline           Auth endpoint          Demo polish
      Face detector        Template manager              Integration
      Batch-process     ──→ exports .npz (team + dataset)      ▲
      dataset images        │                                  │
                            ↓                                  │
DS-1: Read papers        ✅ Implement matchers           Tune thresholds ──────┘
      Design on paper       on real .npz data            Optimize weights
      ⬆ Provide dataset                                       │
        images to CS-1                                         │
                                                               │
DS-2: Read papers        ✅ Design eval protocol         Run experiments       Figures &
      Plan metrics          Explore .npz data            Anti-spoof study      presentation
      ⬆ Provide dataset     (team + dataset)             (dataset = large N)
        images to CS-1

         ✋ Waiting            ✅ Unblocked                ✅ Full speed
         for data              (data available)            (iterating)
```

### 3.4 Handoff Summary

| When | Direction | What |
|---|---|---|
| Week 1-2 | DS → CS-1 | Public dataset images (multi-view face images) uploaded to Google Drive |
| Week 2-3 | CS-1 → DS | Batch-processed dataset `.npz` files on Google Drive |
| Week 2+ | CS-1 → DS | Team capture `.npz` files uploaded to Google Drive |
| Week 3 | CS-1 → DS | `matching/interfaces.py` skeleton pushed to GitHub |
| Week 5 | DS-1 → CS | Implemented matching classes (PR to `develop`) |
| Week 6 | DS-1 → CS | Tuned `config.yaml` values (weights + threshold) |
| Week 6 | DS-2 → CS | Anti-spoofing thresholds |
| Week 7 | DS-2 → CS-2 | Evaluation figures for presentation slides |

---

## 4. Your Development Environment

### 4.1 Quick-Start: Loading Data on Colab

For full Colab environment setup (git, Drive mount, symlinks): **[TA Appendix C]**.

**Minimal setup when you only need `.npz` files** (no MASt3R installation required):

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Clone repo (your matching code lives here)
!git clone https://github.com/<your-team>/face-auth-mast3r.git /content/repo
%cd /content/repo
!git checkout develop && git pull origin develop

# Cell 3: Paths
import sys, numpy as np
sys.path.insert(0, "/content/repo")

SHARED = "/content/drive/MyDrive/face-auth-data/mast3r_outputs"
```

### 4.2 Visualizing a Point Cloud (First Thing to Do)

Before writing any algorithm, see what the data looks like:

```python
import plotly.graph_objects as go

data = np.load(f"{SHARED}/alice_enrollment.npz")
pts = data["point_cloud"]
colors = data["colors"]

# Subsample for performance
idx = np.random.choice(len(pts), min(10000, len(pts)), replace=False)

fig = go.Figure(data=[go.Scatter3d(
    x=pts[idx, 0], y=pts[idx, 1], z=pts[idx, 2],
    mode='markers',
    marker=dict(
        size=1.5,
        color=['rgb({},{},{})'.format(r,g,b) for r,g,b in colors[idx]],
    )
)])
fig.update_layout(scene=dict(aspectmode='data'), width=800, height=600)
fig.show()
```

This produces a rotatable 3D face. Explore it to build intuition about point density, noise patterns, and spatial structure.

### 4.3 Useful Libraries

| Library | Install | Use For |
|---|---|---|
| `numpy` | Pre-installed | Core array operations |
| `scipy` | Pre-installed | KDTree (nearest-neighbor), statistical tests |
| `open3d` | `pip install -q open3d` | ICP registration, point cloud normals |
| `plotly` | `pip install -q plotly` | Interactive 3D visualization |
| `sklearn` | Pre-installed | NearestNeighbors, ROC/AUC metrics |
| `matplotlib` / `seaborn` | Pre-installed | 2D plotting, distributions |

---

## 5. Coding Guide: Implementing the Matching Pipeline (DS-1)

### 5.1 File 1: `core/matching/geometric_matcher.py`

**Goal**: Compare two 3D face point clouds by shape.

```python
"""Geometric matching: compare 3D face shapes."""

import numpy as np
from .interfaces import GeometricMatcher, MatchResult

# import open3d as o3d
# from scipy.spatial import cKDTree


class ICPGeometricMatcher(GeometricMatcher):
    """
    Compare two face point clouds by aligning them with ICP and
    measuring remaining distance.

    General approach:
      1. Pre-align (center at origin, optionally PCA-align principal axes)
      2. Run ICP to refine alignment
      3. Compute distance metric (e.g., Chamfer distance) after alignment
      4. Convert distance → similarity score in [0, 1]
    """

    def __init__(self, config: dict):
        self.icp_max_iterations = config.get("icp_max_iterations", 50)
        self.icp_threshold = config.get("icp_convergence_threshold", 1e-6)
        # Add your own parameters as needed

    def compare(self, probe_cloud: np.ndarray,
                template_cloud: np.ndarray) -> MatchResult:
        """
        Args:
            probe_cloud:    (N, 3) — the NEW face capture (authentication)
            template_cloud: (M, 3) — the ENROLLED face (template)

        Returns:
            MatchResult with score in [0, 1] (higher = more similar)
        """

        # ──────────────────────────────────────────────────────
        # DS-1: IMPLEMENT YOUR ALGORITHM HERE
        #
        # Suggested steps (feel free to modify):
        #
        # 1. Center both clouds:
        #    probe_c = probe_cloud - probe_cloud.mean(axis=0)
        #    template_c = template_cloud - template_cloud.mean(axis=0)
        #
        # 2. (Optional) PCA pre-alignment for better ICP convergence
        #
        # 3. ICP alignment (see Quick Reference in Section 10)
        #
        # 4. Chamfer distance on aligned clouds:
        #    tree = cKDTree(template_c)
        #    d_p2t, _ = tree.query(aligned_probe)
        #    tree2 = cKDTree(aligned_probe)
        #    d_t2p, _ = tree2.query(template_c)
        #    chamfer = (d_p2t.mean() + d_t2p.mean()) / 2
        #
        # 5. Distance → score:
        #    score = 1.0 / (1.0 + alpha * chamfer)
        #    (alpha is a scaling parameter you can tune)
        #
        # ──────────────────────────────────────────────────────

        raise NotImplementedError("DS-1: implement this method")
```

### 5.2 File 2: `core/matching/descriptor_matcher.py`

**Goal**: Compare the learned feature vectors between two face captures.

```python
"""Descriptor matching: compare identity feature vectors."""

import numpy as np
from .interfaces import DescriptorMatcher, MatchResult

# from scipy.spatial import cKDTree


class NNDescriptorMatcher(DescriptorMatcher):
    """
    Compare descriptor sets using mutual nearest neighbor matching.

    General approach:
      1. For each descriptor in probe, find nearest neighbor in template
      2. For each descriptor in template, find nearest neighbor in probe
      3. Keep only reciprocal matches (A→B and B→A agree)
      4. Score based on match ratio and/or average similarity
    """

    def __init__(self, config: dict):
        pass  # Add parameters as needed

    def compare(self, probe_desc: np.ndarray,
                template_desc: np.ndarray,
                probe_cloud: np.ndarray,
                template_cloud: np.ndarray) -> MatchResult:
        """
        Args:
            probe_desc:     (N, D) — descriptor per probe point
            template_desc:  (M, D) — descriptor per template point
            probe_cloud:    (N, 3) — 3D positions (useful for spatial filtering)
            template_cloud: (M, 3)

        Returns:
            MatchResult with score in [0, 1]

        Hints:
            - Descriptors are unit-normalized, so:
              cosine_similarity(a, b) = dot(a, b)
            - Nearest neighbor search in D-dimensional descriptor space
              (NOT in 3D xyz space)
            - Reciprocal matching pseudocode:
                nn_p2t = for each probe[i], nearest template[j]
                nn_t2p = for each template[j], nearest probe[i]
                reciprocal if nn_p2t[i] == j AND nn_t2p[j] == i
            - Two useful metrics:
                match_ratio = n_reciprocal / min(N, M)
                avg_similarity = mean cosine sim of matched pairs
        """

        # ──────────────────────────────────────────────────────
        # DS-1: IMPLEMENT YOUR ALGORITHM HERE
        # ──────────────────────────────────────────────────────

        raise NotImplementedError("DS-1: implement this method")
```

### 5.3 File 3: `core/matching/score_fusion.py`

**Goal**: Combine geometric and descriptor scores into a single accept/reject decision.

```python
"""Score fusion: combine matching signals into a final decision."""

import numpy as np
from .interfaces import ScoreFusion, MatchResult


class WeightedFusion(ScoreFusion):
    """
    Weighted linear combination. This is the simplest approach;
    you may also explore non-linear fusion, per-path minimums,
    geometric mean, or other strategies.
    """

    def __init__(self, config: dict):
        self.geo_weight = config.get("geometric_weight", 0.4)
        self.desc_weight = config.get("descriptor_weight", 0.6)
        self.threshold = config.get("accept_threshold", 0.65)

    def fuse(self, geometric_result: MatchResult,
             descriptor_result: MatchResult) -> MatchResult:

        # ──────────────────────────────────────────────────────
        # DS-1: IMPLEMENT
        #
        # Basic version:
        #   score = geo_weight * geometric_result.score
        #        + desc_weight * descriptor_result.score
        #   is_match = score >= threshold
        #
        # Return MatchResult with details dict containing
        # the individual scores for transparency.
        # ──────────────────────────────────────────────────────

        raise NotImplementedError("DS-1: implement this method")
```

### 5.4 Testing Your Matchers on Colab

```python
# ── No full system needed — works entirely with .npz files ──

import sys, numpy as np
sys.path.insert(0, "/content/repo")

from core.matching.geometric_matcher import ICPGeometricMatcher
from core.matching.descriptor_matcher import NNDescriptorMatcher
from core.matching.score_fusion import WeightedFusion

config = {"icp_max_iterations": 50, "geometric_weight": 0.4,
          "descriptor_weight": 0.6, "accept_threshold": 0.65}

geo  = ICPGeometricMatcher(config)
desc = NNDescriptorMatcher(config)
fuse = WeightedFusion(config)

# Load data
alice_t = np.load(f"{SHARED}/alice_enrollment.npz")
alice_p = np.load(f"{SHARED}/alice_probe_01.npz")
bob_t   = np.load(f"{SHARED}/bob_enrollment.npz")

# ── Genuine pair: Alice vs Alice (should give HIGH score) ──
g = geo.compare(alice_p["point_cloud"], alice_t["point_cloud"])
d = desc.compare(alice_p["descriptors"], alice_t["descriptors"],
                 alice_p["point_cloud"], alice_t["point_cloud"])
result = fuse.fuse(g, d)
print(f"Alice vs Alice: score={result.score:.3f}  match={result.is_match}")

# ── Impostor pair: Bob vs Alice (should give LOW score) ──
g = geo.compare(bob_t["point_cloud"], alice_t["point_cloud"])
d = desc.compare(bob_t["descriptors"], alice_t["descriptors"],
                 bob_t["point_cloud"], alice_t["point_cloud"])
result = fuse.fuse(g, d)
print(f"Bob vs Alice:   score={result.score:.3f}  match={result.is_match}")

# ✓ Sanity check: genuine score should be clearly higher than impostor score
```

---

## 6. Anti-Spoofing Analysis (DS-2 Focus)

### 6.1 The Core Insight

MASt3R reconstructs *actual 3D geometry*. A real face has significant depth variation (nose protrudes ~2-4 cm from cheeks). A photo of a face presented to the camera is essentially **flat**. This geometric difference is primary signal.

### 6.2 Data You'll Receive

CS-1 will capture and process:
- **Real faces**: Team members captured naturally (multiple sessions)
- **Photo attacks**: Photos of team members displayed on a phone/tablet, captured by webcam

Both produce `.npz` files. Task: find features that reliably separate real from fake.

> **Note**: Anti-spoofing experiments require **team member captures only**. Public datasets cannot substitute here because spoofing analysis depends on the physical act of presenting a photo/screen to the camera and observing how MASt3R reconstructs it. The public dataset is valuable for matching evaluation (§7) but not for this task.

### 6.3 Starting Point for Analysis

CS-1's `anti_spoof.py` implements basic checks (see **[TA §10]**). Here's the analytical foundation:

```python
def analyze_3d_properties(point_cloud, confidence):
    """
    Starting-point features for real vs. spoof discrimination.
    DS-2: extend this with your own features.
    """
    z = point_cloud[:, 2]
    centered = point_cloud - point_cloud.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]

    return {
        "depth_std": np.std(z),                              # Real: high, Photo: ~0
        "depth_range": np.ptp(z),                            # Real: ~5-10cm, Photo: ~0
        "flatness_ratio": eigenvalues[2] / eigenvalues[0],   # Real: >0.05, Photo: ≈0
        "confidence_mean": confidence.mean(),
        "confidence_std": confidence.std(),
    }
```

**Encouraged to explore freely**: surface normal distributions, curvature estimation, local descriptor variance, spatial confidence patterns, or any other features you hypothesize might be discriminative. The specific approach and statistical methods are your creative contribution.

### 6.4 Attack Scenarios to Request from CS-1

Coordinate with CS-1 to capture these:
- Printed photo at various distances
- Phone/tablet screen display
- Photo held at an angle or slowly rotated
- (Optional) Video playback on a screen

---

## 7. Evaluation Protocol (DS-1 + DS-2 Collaboration)

This section provides a **framework**, not prescriptive instructions. The specific metrics, analysis methods, and visualization choices are where your analytical expertise shines.

### 7.1 The Core Task

You have a set of match scores. Some are **genuine pairs** (same person) and some are **impostor pairs** (different people). Characterize how well the system separates them.

**Data sources for evaluation**:

| Source | Subjects | Strength | Use For |
|---|---|---|---|
| Team captures | 4 members | Realistic end-to-end validation | Sanity checks, demo rehearsal, anti-spoofing |
| Public dataset | Tens to hundreds of subjects | Statistically significant sample size | Threshold tuning, FAR/FRR/EER, final evaluation |

With team captures alone (4 people × N probes), the number of genuine/impostor pairs is limited. The public dataset dramatically increases statistical power:

```python
# Team only (4 members × N probes):
#   Genuine pairs:  4 × N
#   Impostor pairs: 4 × 3 × N

# With public dataset (S subjects × N probes):
#   Genuine pairs:  S × N
#   Impostor pairs: S × (S-1) × N
#   e.g., S=50, N=3 → 150 genuine + 7,350 impostor pairs
```

### 7.2 Standard Biometric Metrics (Reference, Not Prescription)

| Metric | What It Measures |
|---|---|
| FAR (False Accept Rate) | Fraction of impostors incorrectly accepted |
| FRR (False Reject Rate) | Fraction of genuine users incorrectly rejected |
| EER (Equal Error Rate) | The threshold where FAR = FRR |
| ROC curve | FAR vs FRR trade-off across all thresholds |
| DET curve | Alternative to ROC, common in biometrics |

These are starting points. If your background suggests other evaluation frameworks, apply them.

### 7.3 Weight & Threshold Optimization

For the fusion weights (α for geometric, β for descriptor, α + β = 1):
- Grid search over (α, β) combinations
- Evaluate your chosen metric at each point
- Select the best combination

The final output: three numbers for `config.yaml`:

```yaml
matching:
  geometric_weight: ???    # Optimized α
  descriptor_weight: ???   # Optimized β
  accept_threshold: ???    # Optimized threshold
```

### 7.4 Optional Directions to Explore

These are ideas, not assignments:
- How does enrollment quality (frame count, angle coverage) affect accuracy?
- Which path (geometric vs. descriptor) is more discriminative, and under what conditions?
- Per-region analysis: are certain facial areas more reliable?
- Failure mode characterization: when the system errs, what patterns emerge?
- Sensitivity analysis on conversion functions (distance → score)

---

## 8. Git Workflow for DS

For full details: **[TA §14.5]**.

### The Essential Loop

```python
# ── Start of every Colab session ──
!git clone https://github.com/<team>/face-auth-mast3r.git /content/repo
%cd /content/repo
!git checkout develop && git pull origin develop

# Create or switch to your feature branch
!git checkout -b feature/ds1-matching    # DS-1 (first time)
!git checkout feature/ds1-matching       # DS-1 (subsequent)

# ... do your work ...

# ── Before closing the tab (CRITICAL) ──
%cd /content/repo
!git add -A
!git commit -m "feat(matching): implement Chamfer distance scoring"
!git push origin feature/ds1-matching
```

> ⚠️ **Colab is ephemeral.** Push before you close. Data is safe on Drive; code is safe only on GitHub.

---

## 9. File Ownership Summary

```
core/matching/
├── interfaces.py             ← CS-1 defines (DO NOT EDIT)
├── geometric_matcher.py      ← ★ DS-1 implements
├── descriptor_matcher.py     ← ★ DS-1 implements
└── score_fusion.py           ← ★ DS-1 implements

core/anti_spoof.py            ← CS-1 builds, DS-2 tunes thresholds
config.yaml (matching section)← DS-1 + DS-2 propose tuned values
notebooks/                    ← DS-2 creates evaluation notebooks
```

Everything else (`core/`, `api/`, `frontend/`) — CS owns.

### Recommended Notebook Organization

```
notebooks/
├── ds1_matching_experiments.ipynb
├── ds1_weight_optimization.ipynb
├── ds2_evaluation_protocol.ipynb
├── ds2_anti_spoof_analysis.ipynb
├── ds2_final_results.ipynb
└── figures/
    ├── roc_curve.png
    ├── score_distributions.png
    └── ...
```

---

## 10. Quick Reference: Common Operations

### Centering a Point Cloud

```python
centered = cloud - cloud.mean(axis=0)
```

### Chamfer Distance

```python
from scipy.spatial import cKDTree

def chamfer_distance(a, b):
    d_a2b, _ = cKDTree(b).query(a)
    d_b2a, _ = cKDTree(a).query(b)
    return (d_a2b.mean() + d_b2a.mean()) / 2
```

### Reciprocal Nearest Neighbor Matching

```python
from scipy.spatial import cKDTree

def reciprocal_match(desc_a, desc_b):
    _, idx_a2b = cKDTree(desc_b).query(desc_a)
    _, idx_b2a = cKDTree(desc_a).query(desc_b)

    pairs = []
    for i, j in enumerate(idx_a2b):
        if idx_b2a[j] == i:          # Mutual match confirmed
            sim = np.dot(desc_a[i], desc_b[j])  # Cosine sim (unit vectors)
            pairs.append((i, j, sim))
    return pairs
```

### ICP with Open3D

```python
import open3d as o3d
import numpy as np

def run_icp(source_pts, target_pts, max_dist=0.05, max_iter=50):
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(source_pts)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(target_pts)

    result = o3d.pipelines.registration.registration_icp(
        src, tgt,
        max_correspondence_distance=max_dist,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter
        )
    )
    aligned = np.asarray(src.transform(result.transformation).points)
    return aligned, result.fitness, result.inlier_rmse
```

---

## 11. Communication with CS

### What to Request from CS-1

| Need | What to Ask | Turnaround |
|---|---|---|
| Dataset batch processing | "We've uploaded [dataset] images to Drive, can you batch-process through MASt3R?" | 2-3 days (one-time) |
| More capture data | "Capture [person] under [conditions], export `.npz`" | 1-2 days |
| Photo attack data | "Do a photo attack capture of [person]" | 1-2 days |
| New field in `.npz` | "Also export [field] in the `.npz`" | Same day |
| Interface question | "Can I add a parameter to the config?" | Same day |

### What to Deliver to CS

| What | Recipient | Format |
|---|---|---|
| Implemented matching code | CS-1 (PR to `develop`) | Python files in `core/matching/` |
| Tuned threshold + weights | CS-1 | `config.yaml` values |
| Anti-spoof thresholds | CS-1 | Values + brief justification |
| Evaluation figures | CS-2 (for presentation) | PNG/SVG in `notebooks/figures/` |

---

## 12. Integration Checklist

### DS-1 — Before Handing Off Matching Code

- [ ] All three classes implement the interfaces from `interfaces.py`
- [ ] Every `compare()` / `fuse()` returns a valid `MatchResult`
- [ ] No hardcoded file paths (use config dict or function arguments)
- [ ] Genuine scores consistently higher than impostor scores
- [ ] All scores in [0, 1]; `is_match` respects the threshold
- [ ] Tunable parameters read from `config` dict
- [ ] Code runs on **CPU** (matching does NOT require GPU)
- [ ] No Colab-specific paths (`/content/...`) in committed code
- [ ] PR opened to `develop` with clear description

### DS-2 — Before Presentation

- [ ] Evaluation notebooks are reproducible (re-run from top works)
- [ ] Anti-spoofing analysis includes statistical evidence
- [ ] All threshold recommendations include brief justification
- [ ] Figures exported to `notebooks/figures/`
- [ ] Notebooks pushed to GitHub

---

## Appendix A: Conceptual Refresher

### Point Clouds

A point cloud is a list of (x, y, z) coordinates representing positions on a surface. Imagine spraying thousands of tiny dots onto someone's face — the collection of dot positions forms the point cloud.

### Chamfer Distance

Measures average proximity between two point clouds. For each point in cloud A, find its nearest point in cloud B; do the same from B → A; average both directions. Same face → small distance; different face → larger.

### ICP (Iterative Closest Point)

An algorithm that finds the best rigid transformation (rotation + translation) to align one point cloud onto another. After alignment, the remaining mismatch tells you how different the shapes are.

### Cosine Similarity

MASt3R descriptors are unit-length vectors. Cosine similarity between two unit vectors equals their dot product: `sim = np.dot(a, b)`. Value of 1.0 = identical direction; 0.0 = unrelated.

---

## Appendix B: References

| Resource | Focus | Priority |
|---|---|---|
| [MASt3R paper (arXiv:2406.09756)](https://arxiv.org/abs/2406.09756) — Sections 1-3 | Understanding pointmap and descriptor outputs | High |
| [Open3D ICP tutorial](http://www.open3d.org/docs/latest/tutorial/pipelines/icp_registration.html) | Practical ICP usage | High (DS-1) |
| [DUSt3R paper (arXiv:2312.14132)](https://arxiv.org/abs/2312.14132) | Background on pointmap regression | Medium |
| NIST FRVT evaluation reports | Real-world biometric evaluation methodology | Medium (DS-2) |