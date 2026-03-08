# Flow

## End-to-End Execution Flow

```text
User (CLI/API/Streamlit/Tkinter)
    -> input resolution (path/url/upload)
    -> analyze_video(...)
    -> OpenCV frame loop
    -> candidate detection (HSV + contour + Hough)
    -> Kalman predict/update tracking
    -> bounce event detection from trajectory
    -> speed estimation + smoothing
    -> optional frame annotation/output video
    -> summary metrics dict
    -> returned to caller (JSON/cards/text)
```

## Detailed Frame Flow

1. Read frame from `VideoCapture`.
2. Build foreground mask using MOG2 background subtractor.
3. Detect candidate ball locations:
   - HSV orange segmentation
   - contour circularity filtering
   - Hough circle fallback
4. Score candidates with motion patch score.
5. Predict position with Kalman filter.
6. Choose best measurement using distance + appearance + motion cost.
7. Update tracker state.
8. Push `y` coordinate to short window and detect bounce peak.
9. Compute speed from center displacement over `dt`.
10. Convert px/s to approximate m/s using estimated pixel-to-meter ratio.
11. Smooth speed via EMA.
12. Draw overlays and write output frame if enabled.

## API Input Flow (`/analyze-input`)

- If `video_file` provided:
  - save to temp file -> analyze -> delete temp file
- Else if `video_url` provided:
  - download to temp file -> analyze -> delete temp file
- Else if `video_path` provided:
  - resolve absolute/project-relative path -> analyze
- Else:
  - use default dataset video

Validation rule: only one of `video_path`, `video_url`, `video_file` should be provided.

## Output Flow

All interfaces consume the same result payload from `analyze_video(...)`.

- CLI: prints summary
- API: returns JSON response
- Streamlit: shows metrics cards + JSON + history
- Tkinter: writes metrics in result text box
