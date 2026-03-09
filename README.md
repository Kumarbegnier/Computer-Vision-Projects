# Basketball Dribble Analysis

Computer vision project for basketball dribble analytics with a reusable core engine, API service, Streamlit dashboard, and Tkinter desktop UI.

## What This Project Includes

- Unified analysis engine: `analyze_video(...)` in `Basketball Dribble Analysis/Code/Code.py`
- Detector backends: `classic`, `yolo`, `triton`
- Tracker backends: `kalman`, `bytetrack`, `botsort`, `deepsort` (with compatibility fallbacks)
- Interfaces:
  - CLI
  - FastAPI (`/analyze`, `/analyze-input`)
  - Streamlit app
  - Tkinter desktop app
- Deployment assets:
  - YOLO training + TensorRT export scripts
  - Triton model repository template + HTTP client
  - DeepStream config templates
- Benchmark utility: `Basketball Dribble Analysis/Code/benchmark_models.py`
- Runner script for one-command setup/run: `run_full_stack.ps1`

## Repository Layout

```text
Basketball Dribble Analysis/
+-- Code/
|   +-- Code.py
|   +-- api.py
|   +-- streamlit_app.py
|   +-- ui.py
|   +-- io_utils.py
|   +-- benchmark_models.py
+-- Deployment/
|   +-- TRAINING/
|   +-- TRITON/
|   +-- DEEPSTREAM/
+-- dataset/
|   +-- WHATSAAP ASSIGNMENT.mp4
+-- benchmark_results.json
+-- benchmark_tuning_results.json
+-- Internship _Assignment.pdf

ARCHITECTURE.md
FLOW.md
CHANGELOG.md
README.md
requirements.txt
run_full_stack.ps1
```

## Documentation

- Architecture: `ARCHITECTURE.md`
- Runtime/data flow: `FLOW.md`
- Production/deployment notes: `Basketball Dribble Analysis/Deployment/README_PRODUCTION.md`
- Change history: `CHANGELOG.md`

## Setup

### Option A: Recommended (PowerShell helper)

```powershell
.\run_full_stack.ps1 -Target setup
```

### Option B: Manual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Modes

### CLI

```powershell
python "Basketball Dribble Analysis/Code/Code.py" --max-frames 1000
```

### FastAPI

```powershell
python -m uvicorn api:app --app-dir "Basketball Dribble Analysis/Code" --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs`.

### Streamlit

```powershell
python -m streamlit run "Basketball Dribble Analysis/Code/streamlit_app.py"
```

Open `http://127.0.0.1:8501`.

### Tkinter Desktop UI

```powershell
python "Basketball Dribble Analysis/Code/ui.py"
```

### Benchmark Backends

```powershell
python "Basketball Dribble Analysis/Code/benchmark_models.py" --max-frames 1520
```

### One-command Full Stack (API + Streamlit in new windows)

```powershell
.\run_full_stack.ps1 -Target full
```

## API Summary

### Endpoints

- `GET /` service info
- `GET /health` liveness
- `GET /favicon.ico` no-content
- `POST /analyze` JSON request body
- `POST /analyze-input` form-data (`video_path` or `video_url` or `video_file`)

### Input constraints

- `max_frames` must be between `1000` and `5000`
- `detector_backend`: `classic | yolo | triton`
- `tracker_backend`: `kalman | bytetrack | botsort | deepsort`
- `/analyze-input`: exactly one input source among path/url/upload

### Output highlights

`analyze_video(...)` returns:

- `processed_frames`
- `estimated_bounces`
- `missed_detections`
- `average_smoothed_speed_m_s`
- `detector_backend`, `tracker_backend`
- `warnings` (fallback/dependency/runtime notes)

## Optional Dependency Notes

- `ultralytics` needed for YOLO backend
- `tritonclient[http]` needed for Triton backend
- `deep-sort-realtime` needed for DeepSORT option
- `supervision` may be used for external ByteTrack path

If optional packages are missing, the pipeline degrades to safe defaults (`classic + kalman`) with warning messages.

## Notes

- `.env` and `.venv/` are ignored via `.gitignore`.
- Generated `.mp4` outputs are git-ignored.
- For quick API testing, use Swagger UI (`/docs`) and `POST /analyze-input`.
