# MASt3R Face Authentication System

## Project Overview

This is a **Face ID-like prototype** that uses MASt3R (Matching And Stereo 3D Reconstruction) for 3D face reconstruction and authentication — all from a standard RGB webcam, no depth sensor required.

**What problem does it solve?**  
Traditional face authentication relies on 2D images, which are vulnerable to spoofing (photos, screens). By reconstructing a 3D face model during enrollment and comparing geometry + learned descriptors during authentication, we get better security and accuracy.

---

## Quick Start (TL;DR)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the demo UI (mock mode — no backend needed)
python -m frontend.app_gradio

# 3. Open in browser
# → http://localhost:7860
```

That's it! The UI runs in **mock mode** by default, simulating backend responses so you can explore the full flow.

---

## Running the Demo (Mock Mode)

Mock mode lets you test the complete UI without spinning up the backend API. The frontend simulates face detection, keyframe capture, and authentication results.

**Steps:**
1. Install dependencies: `pip install -r requirements.txt`
2. Launch the UI: `python -m frontend.app_gradio`
3. Open **http://localhost:7860** in your browser
4. Try the **Enrollment** tab: enter a name, click Start, and move your head slowly
5. Try the **Authentication** tab: select a user and click Authenticate

**What you'll see:**
- Simulated face detection with bounding boxes
- Head pose coverage tracking (yaw/pitch)
- Keyframe capture progress
- Mock authentication scores and match results

---

## Running with Real Backend (Live Mode)

> ⚠️ **Work in Progress** — The backend API is not yet complete.

When the FastAPI backend is ready:
1. Start the backend: `uvicorn api.app:app --host 0.0.0.0 --port 8000`
2. The frontend will auto-detect and switch to live mode (or manually set `ConnectionMode.LIVE` in code)
3. Real MASt3R inference will run on GPU

**Endpoints (defined but not yet implemented):**
- WebSocket: `ws://localhost:8000/ws/enroll/{user_name}` — streaming enrollment
- REST: `POST /authenticate` — authentication
- REST: `GET /users` — list enrolled users

---

## Repository Structure

```
ai-visual-computing-pbl/
├── frontend/                   # UI components (Gradio)
│   ├── app_gradio.py          # Main entry point
│   ├── api_client.py          # Backend communication (mock/live)
│   └── components/            # Webcam, enrollment guide, auth panel, visualization
├── scripts/
│   └── smoke_test_frontend.py # Automated tests for frontend
├── config.yaml                # All tunable parameters
├── requirements.txt           # Python dependencies
├── technical-architecure_whole.md  # Full system design (read this for deep details)
└── CS2DS-share.md             # DS team onboarding doc
```

**Folders to be added:**
- `core/` — MASt3R engine, face detection, template manager (CS-1)
- `api/` — FastAPI routes and schemas (CS-1)
- `storage/` — Enrolled templates and SQLite database (gitignored)

---

## Enrollment & Authentication Flow (High-Level)

### Enrollment
1. User enters their name and starts enrollment
2. System captures ~12 keyframes as user rotates head (left, right, up, down)
3. Frames are processed through MASt3R to build a 3D point cloud + descriptors
4. Template is saved to disk

### Authentication
1. User selects their name and captures 2-4 frames
2. System reconstructs a probe point cloud
3. Probe is compared against stored template (geometric + descriptor matching)
4. Match score determines pass/fail

For full pipeline details, see `technical-architecure_whole.md`.

---

## Team Responsibilities

| Role | Owns | Focus |
|------|------|-------|
| **CS-1** | `core/`, `api/` | MASt3R integration, face detection, backend API |
| **CS-2** | `frontend/` | Gradio UI, visualization, API client |
| **DS-1** | `core/matching/` | Geometric + descriptor matching algorithms |
| **DS-2** | Evaluation | Anti-spoofing, accuracy metrics |

---

## Development Notes

### Mock Backend
The `MockBackend` class in `frontend/api_client.py` simulates:
- Face detection (90% success rate)
- Head pose estimation (temporal consistency)
- Keyframe selection (every ~8 frames)
- Point cloud generation (random face-like shape)
- Authentication (70% match rate for demo)

This lets CS-2 develop the UI independently of CS-1's backend.

### Smoke Tests
Run automated tests to verify components work:
```bash
python scripts/smoke_test_frontend.py
```

Tests cover: webcam capture, enrollment guide, auth panel, visualization, and API client (mock mode).

### Configuration
All tunable parameters live in `config.yaml`:
- Frontend server settings (port 7860)
- Webcam resolution (640×480)
- Enrollment targets (12 keyframes, yaw/pitch coverage)
- API endpoints

---

## Common Issues / Tips

| Issue | Solution |
|-------|----------|
| **Port 7860 already in use** | Kill the existing process or change `frontend.server.port` in `config.yaml` |
| **Webcam not detected** | Check browser permissions; try a different `device_id` in config |
| **Import errors** | Make sure you're running from the repo root: `python -m frontend.app_gradio` |
| **PyTorch not found** | Install separately: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| **Slow/no GPU** | Mock mode doesn't need GPU; live mode requires CUDA-compatible GPU with ≥8GB VRAM |

---

## Documentation

- **`technical-architecure_whole.md`** — Full architecture, data flows, API contracts, module specs
- **`CS2DS-share.md`** — DS team onboarding and interface specifications

---

## License

Academic use only. MASt3R is licensed under CC BY-NC-SA 4.0.
