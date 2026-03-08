# Architecture

## 1) System Overview

This project is built around a single reusable analysis engine and multiple interfaces.

- Core CV engine: `Basketball Dribble Analysis/Code/Code.py`
- REST API interface: `Basketball Dribble Analysis/Code/api.py`
- Streamlit web UI: `Basketball Dribble Analysis/Code/streamlit_app.py`
- Tkinter desktop UI: `Basketball Dribble Analysis/Code/ui.py`

All interfaces call the same function: `analyze_video(...)`.

## 2) Module Responsibilities

### Core Engine (`Code.py`)

- Ball candidate generation
  - HSV color segmentation for orange basketball
  - contour-based shape filtering
  - Hough circle fallback
- Tracking
  - Kalman filter (`BallTracker`) for smooth center estimation
- Event detection
  - bounce detection using 3-point trajectory peak logic
- Metrics
  - processed frames
  - estimated bounces
  - missed detections
  - smoothed speed (px/s, m/s approx)
- Optional outputs
  - annotated output video

### API (`api.py`)

- `GET /` : service metadata
- `GET /health` : liveness check
- `POST /analyze` : JSON input (`video_path`, `max_frames`, `save_video_path`)
- `POST /analyze-input` : multipart input (`video_path` or `video_url` or `video_file`)
- Handles temporary files for URL/upload mode and cleanup in `finally`

### Streamlit UI (`streamlit_app.py`)

- Input modes: Path / URL / Upload
- Run settings in sidebar
- Metric cards + detection quality bar
- Raw JSON + download button
- Recent run history in session state

### Tkinter UI (`ui.py`)

- Desktop form-based runner
- Uses worker thread for non-blocking analysis
- Displays summarized results in text area

## 3) Data Contracts

`analyze_video(...)` returns:

```json
{
  "video_path": "...",
  "processed_frames": 1000,
  "estimated_bounces": 6,
  "missed_detections": 1,
  "average_smoothed_speed_m_s": 0.1894,
  "elapsed_seconds": 6.656
}
```

## 4) Runtime Dependencies

- Python 3.8+
- `opencv-python`
- `numpy`
- `fastapi`
- `uvicorn`
- `python-multipart` (required for file upload/form endpoints)
- `streamlit` (for Streamlit UI)

## 5) Architectural Decisions

- Single processing core avoids duplication across CLI/API/UIs.
- Heuristic CV + Kalman approach keeps runtime lightweight (no heavy model dependency).
- API supports local path, URL, and upload to simplify integration.
- Temporary files are cleaned up after processing.

## 6) Future Advanced Architecture

Recommended next step:

- Split into packages:
  - `core/` (detection, tracking, events)
  - `service/` (analysis orchestration, config)
  - `interfaces/` (api/ui/cli)
  - `tests/` (unit + regression videos)
- Add async job queue for long video analysis (`job_id` pattern)
- Add frame-level CSV/event timeline outputs
- Replace heuristic detector with trained detector (YOLO/RT-DETR) for robustness
