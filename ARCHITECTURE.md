# Architecture

## 1) System Goal

Production-ready basketball dribble analytics with:
- robust detection (YOLO / Triton / classic fallback),
- robust tracking (BoT-SORT / ByteTrack / DeepSORT / Kalman),
- custom bounce logic,
- multiple runtime interfaces (CLI, API, Streamlit),
- deployment paths (TensorRT, Triton, DeepStream 8).

## 2) High-Level Layout

Core code:
- `Basketball Dribble Analysis/Code/Code.py`
- `Basketball Dribble Analysis/Code/api.py`
- `Basketball Dribble Analysis/Code/streamlit_app.py`
- `Basketball Dribble Analysis/Code/io_utils.py`

Deployment code:
- `Basketball Dribble Analysis/Deployment/TRAINING/*`
- `Basketball Dribble Analysis/Deployment/TRITON/*`
- `Basketball Dribble Analysis/Deployment/DEEPSTREAM/*`

## 3) Runtime Architecture

### 3.1 Inference Pipeline

Per frame flow in `analyze_video(...)`:
1. Frame read
2. Detection backend
- `classic`: HSV + contour + Hough circle
- `yolo`: local Ultralytics model (`YOLO.predict/track`)
- `triton`: Triton HTTP inference client
3. Tracking backend
- `kalman` (always available)
- `bytetrack` / `botsort` (via YOLO track mode)
- `deepsort` (optional external tracker path)
4. Kalman smoothing (`BallTracker`)
5. Metrics update (speed, misses, radius stats)
6. Bounce estimation from `y` trajectory peaks
7. Optional annotated video write

### 3.2 Backend Selection Rules

- If requested backend dependency is missing, system falls back to `classic + kalman`.
- Fallback reasons are returned in `warnings` field.
- This keeps API/UI stable in partially provisioned environments.

## 4) Interface Layer

### CLI

`Code.py` CLI options expose:
- `detector_backend`
- `tracker_backend`
- `yolo_model`
- `triton_url`
- `triton_model_name`

### API (FastAPI)

`api.py` endpoints:
- `GET /health`
- `POST /analyze`
- `POST /analyze-input`

API supports:
- local path / upload / URL input,
- secure URL handling via `io_utils.py`,
- same backend knobs as CLI,
- validation + cleanup + error mapping.

### Streamlit

`streamlit_app.py` provides:
- Path/URL/Upload input,
- runtime backend selectors,
- model/server config inputs,
- metric dashboard + warnings + run history.

## 5) Data Contracts

Primary analysis output:

```json
{
  "video_path": "...",
  "processed_frames": 1520,
  "estimated_bounces": 102,
  "missed_detections": 2,
  "average_smoothed_speed_m_s": 2.69,
  "elapsed_seconds": 63.36,
  "processing_seconds": 8.9,
  "detector_backend": "classic",
  "tracker_backend": "kalman",
  "triton_url": "",
  "triton_model_name": "",
  "warnings": []
}
```

## 6) Training and Optimization Architecture

### 6.1 Fine-tuning

- Script: `Deployment/TRAINING/train_finetune.py`
- Dataset contract: `Deployment/TRAINING/data.yaml`
- Output: trained YOLO checkpoint (`best.pt`)

### 6.2 TensorRT

- Script: `Deployment/TRAINING/export_tensorrt.py`
- Output: TensorRT plan (`model.plan`)

## 7) Serving Architecture

### 7.1 Triton

- Model repo template:
`Deployment/TRITON/model_repository/basketball_yolo_trt`
- Config:
`config.pbtxt` with `images -> detections` contract
- Client example:
`Deployment/TRITON/client/infer_http.py`

### 7.2 DeepStream 8

Templates:
- `Deployment/DEEPSTREAM/deepstream_app_config.txt`
- `Deployment/DEEPSTREAM/config_infer_primary_yolo.txt`

DeepStream pipeline path is for GPU-native, low-latency production streaming.

## 8) Reliability and Security Controls

- URL ingestion validation blocks unsafe/private hosts.
- Size limits and timeout on remote downloads.
- Temp file lifecycle cleanup.
- Graceful degradation when optional dependencies are unavailable.

## 9) Recommended Production Modes

1. Best practical local mode:
- `detector_backend=yolo`
- `tracker_backend=botsort`

2. Scalable service mode:
- `detector_backend=triton`
- `tracker_backend=kalman` (or DS pipeline if using DeepStream end-to-end)

3. Guaranteed fallback mode:
- `detector_backend=classic`
- `tracker_backend=kalman`

## 10) Next Engineering Steps

- Add unit tests for backend selection and bounce logic.
- Add regression suite with labeled clips.
- Add structured logging and request IDs in API.
- Add async job queue for long videos (`job_id` + status polling).
