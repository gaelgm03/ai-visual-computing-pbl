# MASt3R Face Authentication System

A Face ID-like prototype using **MASt3R** (Matching And Stereo 3D Reconstruction) for 3D face reconstruction and authentication — all from a standard RGB webcam.

## Project Structure

```
face-auth-mast3r/
├── frontend/                    # CS-2 Primary Ownership
│   ├── app_gradio.py           # Main Gradio UI entry point
│   └── components/
│       ├── webcam_capture.py   # Webcam access + frame dispatch
│       ├── enrollment_guide.py # Guided head rotation UI
│       ├── auth_panel.py       # Authentication trigger + result
│       └── visualization.py    # 3D point cloud viewer
├── core/                        # CS-1 Primary Ownership (TODO)
├── api/                         # CS-1 Defines, CS-2 Consumes (TODO)
├── scripts/
│   └── smoke_test_frontend.py  # Frontend component tests
├── config.yaml                  # All tunable parameters
├── requirements.txt             # Python dependencies
└── .gitignore
```

## Quick Start (CS-2 Frontend)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Gradio UI

```bash
python -m frontend.app_gradio
```

The UI will be available at: **http://localhost:7860**

### 3. Run Smoke Tests

```bash
python scripts/smoke_test_frontend.py
```

## Features (Current Status)

### ✅ Implemented (CS-2 Milestone 1)
- **Gradio skeleton UI** with tabs for Enrollment, Authentication, Users, 3D Viewer
- **Webcam capture component** with base64 encoding for API transmission
- **Enrollment guide** with head pose coverage tracking and direction prompts
- **Authentication panel** with result formatting and score visualization
- **Point cloud visualizer** using Plotly for 3D rendering
- **Configuration system** via `config.yaml`

### 🔄 Pending (Future Milestones)
- Connect enrollment UI to WebSocket endpoint (`ws://localhost:8000/ws/enroll`)
- Connect authentication to REST endpoint (`POST /authenticate`)
- Real face detection overlay (requires CS-1 backend)
- Live 3D point cloud preview during enrollment
- User management (list, delete) via API

## Team Responsibilities

| Role | Primary Ownership |
|------|-------------------|
| **CS-1** | `core/`, `api/`, MASt3R integration, face detection |
| **CS-2** | `frontend/`, Gradio UI, visualization components |
| **DS-1** | `core/matching/` algorithm implementations |
| **DS-2** | Evaluation, anti-spoofing analysis |

## Branch Conventions

```
main
 └── develop                    ← Integration branch
      ├── feature/cs1-*         ← CS-1 feature branches
      ├── feature/cs2-*         ← CS-2 feature branches
      └── feature/ds-*          ← DS feature branches
```

## Documentation

- `technical-architecure_whole.md` — Full system architecture
- `CS2DS-share.md` — DS team onboarding and interface specs

## License

Academic use only. MASt3R is licensed under CC BY-NC-SA 4.0.
