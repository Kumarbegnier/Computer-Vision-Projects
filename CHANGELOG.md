# Changelog

All notable changes to this project are documented in this file.

## [2026-03-08] - Advanced Dribble Analysis + API + UIs

### Added
- FastAPI service in `Basketball Dribble Analysis/Code/api.py`
  - `GET /`
  - `GET /health`
  - `GET /favicon.ico`
  - `POST /analyze` (JSON input)
  - `POST /analyze-input` (path/url/upload input)
- Streamlit interface in `Basketball Dribble Analysis/Code/streamlit_app.py`
  - Path/URL/Upload input support
  - Sidebar run settings
  - Metrics cards, quality indicator, raw JSON download, history
- Tkinter desktop UI in `Basketball Dribble Analysis/Code/ui.py`
- `.gitignore` for Python cache and generated video artifacts

### Changed
- Replaced notebook-style `Code.py` with runnable Python pipeline.
- Refactored processing into reusable `analyze_video(...)` in `Basketball Dribble Analysis/Code/Code.py`.
- Upgraded CV pipeline:
  - Ball candidate fusion (HSV + contours + Hough circles)
  - Motion scoring via background subtraction
  - Kalman filter tracking
  - Trajectory-based bounce detection
  - Speed estimation (px/s and approx m/s)
  - Optional annotated video output
- Updated `README.md` with API and UI run instructions.

### Fixed
- Removed invalid notebook JSON execution issue.
- Resolved undefined `start_time` usage from old implementation.
- Replaced hardcoded machine-specific video path with project-relative defaults.
- Fixed Kalman array indexing runtime bug.

### Notes
- Output video files are intentionally ignored by git.
- Use `/docs` on FastAPI server for interactive API testing.
