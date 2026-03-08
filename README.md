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

Install:

```bash
pip install opencv-python numpy
```

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
