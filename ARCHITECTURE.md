# Architecture

## 1) Goal

Provide a single dribble-analysis engine that can run consistently across:

- local CLI workflows,
- API service usage,
- Streamlit and Tkinter UIs,
- production-style inference stacks (YOLO, Triton, DeepStream templates).

## 2) Component Map

### Core analysis

- `Basketball Dribble Analysis/Code/Code.py`
  - `analyze_video(...)` main orchestration
  - classic detector (`HSV + contour + Hough`)
  - optional YOLO detector
  - optional Triton detector
  - Kalman state tracking
  - bounce and speed estimation

### Interface layer

- `Basketball Dribble Analysis/Code/api.py` (FastAPI)
- `Basketball Dribble Analysis/Code/streamlit_app.py` (web dashboard)
- `Basketball Dribble Analysis/Code/ui.py` (Tkinter desktop app)

### IO/security helpers

- `Basketball Dribble Analysis/Code/io_utils.py`
  - remote URL validation
  - safe temporary download handling

### Ops and deployment assets

- `Basketball Dribble Analysis/Code/benchmark_models.py`
- `Basketball Dribble Analysis/Deployment/TRAINING/*`
- `Basketball Dribble Analysis/Deployment/TRITON/*`
- `Basketball Dribble Analysis/Deployment/DEEPSTREAM/*`
- `run_full_stack.ps1`

## 3) Runtime Pipeline (`analyze_video`)

Per frame:

1. Read frame from OpenCV capture.
2. Predict next position using Kalman filter.
3. Run detector backend:
   - `classic`: color + shape + motion scoring
   - `yolo`: Ultralytics model (`predict`/`track`)
   - `triton`: HTTP inference client (`images -> detections`)
4. Optional external tracker handling (when compatible and available).
5. Select measurement with distance/score/motion cost.
6. Update Kalman state.
7. Update metrics:
   - missed detections,
   - speed (px/s and approx m/s),
   - y-history for bounce estimation.
8. Draw overlays and optionally write output video frame.
9. Finalize summary payload and warnings.

## 4) Backend Compatibility Rules

- If YOLO/Triton dependency is unavailable, detector falls back to `classic`.
- If requested tracker is incompatible with active detector, tracker falls back to `kalman`.
- Fallback causes are returned in `warnings`.

Current practical rules:

- Non-YOLO detector with `bytetrack`/`botsort`/`deepsort` -> fallback to `kalman`.
- Triton with advanced trackers -> fallback to `kalman`.
- YOLO + DeepSORT attempts external tracker; on failure -> fallback to `kalman`.

## 5) API Design

### Endpoints

- `GET /` service metadata
- `GET /health` liveness
- `GET /favicon.ico` no-content
- `POST /analyze` JSON body contract
- `POST /analyze-input` form input for path/url/upload

### Input contracts

- `max_frames` range: `1000..5000`
- exactly one source for `/analyze-input`: `video_path` OR `video_url` OR `video_file`

### Error mapping

- 400 for validation/value errors,
- 404 for missing files,
- 500 for runtime failures.

## 6) Data Contract (Output)

Typical response fields:

```json
{
  "video_path": "C:/.../video.mp4",
  "processed_frames": 1520,
  "estimated_bounces": 101,
  "missed_detections": 4,
  "average_smoothed_speed_m_s": 2.51,
  "elapsed_seconds": 63.333,
  "processing_seconds": 8.41,
  "detector_backend": "yolo",
  "tracker_backend": "kalman",
  "triton_url": "",
  "triton_model_name": "",
  "yolo_conf": 0.08,
  "yolo_iou": 0.5,
  "yolo_imgsz": 960,
  "yolo_class_id": 32,
  "yolo_class_fallback": true,
  "warnings": []
}
```

## 7) Security and Reliability Controls

- URL scheme restriction (`http/https`) for remote inputs.
- DNS/IP validation to avoid private/unsafe hosts.
- Remote size cap (`200 MB`) and timeout-based download guard.
- Temporary file cleanup in API/Streamlit upload and URL flows.
- Degraded but stable execution when optional dependencies are missing.

## 8) Production Paths

1. Training and export:
   - `Deployment/TRAINING/train_finetune.py`
   - `Deployment/TRAINING/export_tensorrt.py`
2. Triton serving template:
   - `Deployment/TRITON/model_repository/basketball_yolo_trt`
3. DeepStream configuration templates:
   - `Deployment/DEEPSTREAM/deepstream_app_config.txt`
   - `Deployment/DEEPSTREAM/config_infer_primary_yolo.txt`

## 9) Recommended Modes

1. Baseline reliability: `classic + kalman`
2. Local quality/performance: `yolo + botsort` (or `yolo + kalman`)
3. Service inference at scale: `triton + kalman`
