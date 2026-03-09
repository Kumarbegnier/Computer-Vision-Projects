from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from Code import analyze_video
from io_utils import download_video_to_temp_file, validate_remote_video_url


app = FastAPI(title="Basketball Dribble Analysis API", version="1.0.0")
ALLOWED_DETECTORS = {"classic", "yolo", "triton"}
ALLOWED_TRACKERS = {"kalman", "bytetrack", "botsort", "deepsort"}


class AnalyzeRequest(BaseModel):
    video_path: str = Field(
        default="",
        description="Absolute or project-relative path to video. Empty uses default dataset video.",
    )
    max_frames: int = Field(default=1000, ge=1000, le=5000)
    save_video_path: str = Field(
        default="",
        description="Optional output path for annotated video.",
    )
    detector_backend: str = Field(default="classic", pattern="^(classic|yolo|triton)$")
    tracker_backend: str = Field(default="kalman", pattern="^(kalman|bytetrack|botsort|deepsort)$")
    yolo_model: str = Field(default="yolov8n.pt")
    yolo_conf: float = Field(default=0.08, ge=0.001, le=1.0)
    yolo_iou: float = Field(default=0.5, ge=0.001, le=1.0)
    yolo_imgsz: int = Field(default=960, ge=320, le=1920)
    yolo_class_id: int = Field(default=32, ge=0, le=999)
    yolo_class_fallback: bool = Field(default=True)
    triton_url: str = Field(default="localhost:8000")
    triton_model_name: str = Field(default="basketball_yolo_trt")


@app.get("/")
def root() -> dict:
    return {
        "message": "Basketball Dribble Analysis API is running",
        "docs": "/docs",
        "health": "/health",
        "analyze": "/analyze",
        "analyze_input": "/analyze-input (video_path or video_url or video_file)",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    default_video = project_root / "dataset" / "WHATSAAP ASSIGNMENT.mp4"

    video_path = Path(request.video_path).expanduser() if request.video_path else default_video
    if not video_path.is_absolute():
        video_path = project_root / video_path

    save_video = None
    if request.save_video_path:
        save_video = Path(request.save_video_path).expanduser()
        if not save_video.is_absolute():
            save_video = project_root / save_video
        save_video.parent.mkdir(parents=True, exist_ok=True)

    try:
        if request.detector_backend not in ALLOWED_DETECTORS:
            raise HTTPException(status_code=400, detail="Invalid detector_backend.")
        if request.tracker_backend not in ALLOWED_TRACKERS:
            raise HTTPException(status_code=400, detail="Invalid tracker_backend.")
        result = analyze_video(
            video_path=video_path,
            max_frames=request.max_frames,
            quiet=True,
            show=False,
            save_video=save_video,
            detector_backend=request.detector_backend,
            tracker_backend=request.tracker_backend,
            yolo_model=request.yolo_model,
            yolo_conf=request.yolo_conf,
            yolo_iou=request.yolo_iou,
            yolo_imgsz=request.yolo_imgsz,
            yolo_class_id=request.yolo_class_id,
            yolo_class_fallback=request.yolo_class_fallback,
            triton_url=request.triton_url,
            triton_model_name=request.triton_model_name,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@app.post("/analyze-input")
def analyze_input(
    max_frames: int = Form(default=1000, ge=1000, le=5000),
    save_video_path: str = Form(default=""),
    video_path: str = Form(default=""),
    video_url: str = Form(default=""),
    video_file: UploadFile = File(default=None),
    detector_backend: str = Form(default="classic"),
    tracker_backend: str = Form(default="kalman"),
    yolo_model: str = Form(default="yolov8n.pt"),
    yolo_conf: float = Form(default=0.08),
    yolo_iou: float = Form(default=0.5),
    yolo_imgsz: int = Form(default=960),
    yolo_class_id: int = Form(default=32),
    yolo_class_fallback: bool = Form(default=True),
    triton_url: str = Form(default="localhost:8000"),
    triton_model_name: str = Form(default="basketball_yolo_trt"),
) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    default_video = project_root / "dataset" / "WHATSAAP ASSIGNMENT.mp4"
    temp_path = None

    save_video = None
    if save_video_path:
        save_video = Path(save_video_path).expanduser()
        if not save_video.is_absolute():
            save_video = project_root / save_video
        save_video.parent.mkdir(parents=True, exist_ok=True)

    try:
        if detector_backend not in ALLOWED_DETECTORS:
            raise HTTPException(status_code=400, detail="Invalid detector_backend.")
        if tracker_backend not in ALLOWED_TRACKERS:
            raise HTTPException(status_code=400, detail="Invalid tracker_backend.")
        chosen_inputs = int(bool(video_path.strip())) + int(bool(video_url.strip())) + int(video_file is not None)
        if chosen_inputs > 1:
            raise HTTPException(
                status_code=400,
                detail="Provide only one input: video_path or video_url or video_file.",
            )

        if video_file is not None:
            suffix = Path(video_file.filename or "upload.mp4").suffix or ".mp4"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_path = Path(temp_file.name)
            with temp_file as fp:
                shutil.copyfileobj(video_file.file, fp)
            resolved_video = temp_path
        elif video_url.strip():
            safe_url = validate_remote_video_url(video_url)
            temp_path = download_video_to_temp_file(safe_url)
            resolved_video = temp_path
        elif video_path.strip():
            resolved_video = Path(video_path.strip()).expanduser()
            if not resolved_video.is_absolute():
                resolved_video = project_root / resolved_video
        else:
            resolved_video = default_video

        result = analyze_video(
            video_path=resolved_video,
            max_frames=max_frames,
            quiet=True,
            show=False,
            save_video=save_video,
            detector_backend=detector_backend,
            tracker_backend=tracker_backend,
            yolo_model=yolo_model,
            yolo_conf=yolo_conf,
            yolo_iou=yolo_iou,
            yolo_imgsz=yolo_imgsz,
            yolo_class_id=yolo_class_id,
            yolo_class_fallback=yolo_class_fallback,
            triton_url=triton_url,
            triton_model_name=triton_model_name,
        )
        result["input_mode"] = (
            "upload" if video_file is not None else "url" if video_url.strip() else "path_or_default"
        )
        return result
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
