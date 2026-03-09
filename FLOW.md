# Flow

## 1) End-to-End Runtime Flow

```text
Caller (CLI / API / Streamlit / Tkinter)
  -> resolve input source (path / url / upload / default)
  -> analyze_video(...)
  -> detector + tracker loop per frame
  -> metric aggregation (bounces, misses, speed)
  -> optional annotated output video
  -> result payload + warnings
  -> caller-specific rendering (JSON / cards / text)
```

## 2) Input Resolution Flow

### CLI (`Code.py`)

- Input comes from `--video` (default dataset path if omitted).

### API `POST /analyze`

- JSON body only.
- Path resolution:
  - absolute path used as-is,
  - relative path resolved under project root.

### API `POST /analyze-input`

1. Validate backend names.
2. Count selected sources among `video_path`, `video_url`, `video_file`.
3. If more than one source is provided -> `400`.
4. Resolve source:
   - `video_file`: write temporary file
   - `video_url`: validate URL then download temporary file
   - `video_path`: resolve relative/absolute path
   - none: use default dataset video
5. Always cleanup temp file in `finally`.

### Streamlit

- Same source logic as API form flow:
  - Path tab,
  - URL tab (with validation/download),
  - Upload tab (temp file).

## 3) Frame Processing Flow

1. Read frame from OpenCV capture.
2. Kalman predict (`BallTracker.predict`).
3. Choose detector backend:
   - `classic`: MOG2 + HSV/contour/Hough candidate generation
   - `yolo`: Ultralytics detection/tracking
   - `triton`: Triton HTTP inference call
4. Optional tracker refinement:
   - YOLO-based tracker modes or external tracker path where applicable.
5. Candidate-to-measurement association (`choose_measurement`).
6. Kalman correction (`update_with_prediction`).
7. Update metrics:
   - missed detections,
   - y-history,
   - speed estimate (px/s and m/s),
   - smoothed EMA speed.
8. Draw overlays.
9. Write output frame when video saving is enabled.
10. Continue until `max_frames` or stream end.

## 4) Backend Fallback Flow

1. Requested backend unavailable (missing dependency/runtime issue) -> warn + fallback.
2. Unsupported detector/tracker combination -> warn + fallback to `kalman`.
3. Execution continues with stable baseline path (`classic + kalman` when needed).

## 5) Post-Processing Flow

1. Estimate bounce count from y-trajectory local peaks (`estimate_bounces_from_y`).
2. Compute:
   - `elapsed_seconds` (video time),
   - `processing_seconds` (wall clock).
3. Construct response payload.
4. Return payload to caller.

## 6) Caller Output Flow

- CLI:
  - prints summary lines and warnings.
- API:
  - returns JSON payload,
  - adds `input_mode` for `/analyze-input`.
- Streamlit:
  - renders cards, quality score, warnings, raw JSON, run history.
- Tkinter:
  - displays textual summary in result box.
