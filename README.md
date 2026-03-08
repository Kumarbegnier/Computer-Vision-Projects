# Basketball Dribble Analysis

Computer vision project for analyzing basketball dribbling video with a reusable analysis core, API, and Python UIs.

## Project Structure

```text
Basketball Dribble Analysis/
+-- Code/
¦   +-- Code.py
¦   +-- api.py
¦   +-- streamlit_app.py
¦   +-- ui.py
+-- dataset/
¦   +-- WHATSAAP ASSIGNMENT.mp4
+-- Internship _Assignment.pdf

ARCHITECTURE.md
FLOW.md
CHANGELOG.md
README.md
```

## Documentation

- Architecture: `ARCHITECTURE.md`
- Flow: `FLOW.md`
- Change history: `CHANGELOG.md`

## Core Engine

Main reusable function:

- `analyze_video(...)` in `Basketball Dribble Analysis/Code/Code.py`

Pipeline highlights:

- candidate detection (HSV + contours + Hough circles)
- Kalman tracking
- trajectory-based bounce detection
- speed estimation and smoothing
- optional annotated video output

## Dependencies

- Python 3.8+
- opencv-python
- numpy
- fastapi
- uvicorn
- python-multipart
- streamlit

Install all:

```bash
pip install opencv-python numpy fastapi uvicorn python-multipart streamlit
```

## CLI Run

```bash
python "Basketball Dribble Analysis/Code/Code.py" --max-frames 1000
```

## FastAPI Run

```bash
python -m uvicorn api:app --app-dir "Basketball Dribble Analysis/Code" --reload
```

Open:

- `http://127.0.0.1:8000/docs`

Key endpoints:

- `GET /`
- `GET /health`
- `POST /analyze`
- `POST /analyze-input` (path/url/upload)

## Streamlit UI Run

```bash
python -m streamlit run "Basketball Dribble Analysis/Code/streamlit_app.py"
```

Open:

- `http://127.0.0.1:8501`

## Tkinter UI Run

```bash
python "Basketball Dribble Analysis/Code/ui.py"
```

## Notes

- Generated `.mp4` outputs and Python cache are ignored via `.gitignore`.
- Use `POST /analyze-input` in Swagger for easiest upload-based testing.
