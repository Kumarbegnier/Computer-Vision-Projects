# Basketball Dribble Analysis

Computer vision assignment project for analyzing a basketball dribbling video.

## Project Structure

```text
Basketball Dribble Analysis/
├── Code/
│   └── Code.py
├── dataset/
│   └── WHATSAAP ASSIGNMENT.mp4
└── Internship _Assignment.pdf
```

## Objective

Extract measurable dribbling insights from video frames using computer vision techniques.

## Current Implementation (Code/Code.py)

The current pipeline follows this flow:

1. Load video with OpenCV `VideoCapture`
2. Read frames in a loop (up to `max_frames = 1000`)
3. Detect dribble-like events using:
   - grayscale conversion
   - Gaussian blur
   - Canny edge detection
   - Hough line transform
4. Compute frame-level speed estimate
5. Print frame-wise logs and total dribble count

## Dependencies

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- FastAPI (`fastapi`)
- Uvicorn (`uvicorn`)

Install:

```bash
pip install opencv-python numpy fastapi uvicorn
```

## FastAPI Server

API file:

- `Basketball Dribble Analysis/Code/api.py`

Run server:

```bash
python -m uvicorn api:app --app-dir "Basketball Dribble Analysis/Code" --reload
```

Endpoints:

- `GET /health`
- `POST /analyze`
- `POST /analyze-input` (video path, video URL, or upload file)

Sample request:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" ^
  -H "Content-Type: application/json" ^
  -d "{\"video_path\":\"Basketball Dribble Analysis/dataset/WHATSAAP ASSIGNMENT.mp4\",\"max_frames\":1000}"
```

For link or file upload (open `/docs` and use `POST /analyze-input`), or use curl with multipart:

```bash
curl -X POST "http://127.0.0.1:8000/analyze-input" ^
  -F "max_frames=1000" ^
  -F "video_url=https://example.com/sample.mp4"
```

## Python UI (Tkinter)

UI file:

- `Basketball Dribble Analysis/Code/ui.py`

Run UI:

```bash
python "Basketball Dribble Analysis/Code/ui.py"
```

Features:

- Select input video from file picker
- Set max frames
- Optionally save annotated output video
- Run analysis and view result metrics in app window

## Streamlit UI

UI file:

- `Basketball Dribble Analysis/Code/streamlit_app.py`

Install:

```bash
pip install streamlit
```

Run:

```bash
python -m streamlit run "Basketball Dribble Analysis/Code/streamlit_app.py"
```

Supports:

- Video path input
- Video URL input
- Video file upload

## How To Run

### Important Note

`Code/Code.py` is currently a Jupyter notebook export in JSON format (not a clean Python script). Running it directly with `python Code.py` can fail.

Recommended options:

1. Open it in Jupyter/VS Code Notebook and run cells.
2. Convert/refactor it into a standard `.py` script before CLI execution.

## Known Issues

- Hardcoded absolute video path inside code
- `start_time` usage is not initialized in the execution block
- Dribble detection logic is line-based and may overcount
- Ball speed formula uses a fixed scale factor without calibration

## Suggested Improvements

- Use project-relative paths
- Add robust ball detection + tracking
- Detect bounce events from trajectory instead of edge lines
- Calibrate pixel-to-meter ratio for realistic speed
- Save outputs to CSV/JSON and annotated video
- Add modular project files and tests

## Evaluation Alignment

This project is documented against typical internship criteria:

- analysis accuracy/effectiveness
- code quality and optimization
- documentation clarity
- creativity in additional metrics

## License

For educational and internship evaluation use.
