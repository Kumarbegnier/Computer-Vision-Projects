# Changelog

All notable changes to this project are documented in this file.

## [2026-03-09] - Documentation Refresh + API Error Mapping Fix

### Changed
- Refreshed `README.md` with:
  - complete repository layout,
  - setup paths (`requirements.txt` and `run_full_stack.ps1`),
  - run commands for CLI/API/Streamlit/Tkinter/benchmark,
  - API constraints and output summary,
  - optional dependency and fallback notes.
- Reworked `ARCHITECTURE.md` to align with current implementation:
  - component map,
  - backend compatibility rules,
  - security/reliability controls,
  - production path summary.
- Reworked `FLOW.md` to reflect current source-resolution, processing, fallback, and output behavior.

### Fixed
- `POST /analyze` now preserves intentional `HTTPException` responses (for example invalid backend input stays `400` instead of being wrapped as `500`).

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
