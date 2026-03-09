import logging
import os
import tempfile
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from Code import analyze_video
from io_utils import download_video_to_temp_file, validate_remote_video_url


APP_VERSION = os.getenv("APP_VERSION", "1.1.0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
UPLOAD_CHUNK_BYTES = 1024 * 1024
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
ENABLE_API_KEY_AUTH = os.getenv("ENABLE_API_KEY_AUTH", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
API_KEY = os.getenv("API_KEY", "").strip()
CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS", "*").strip()
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

ALLOWED_UPLOAD_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
ALLOWED_UPLOAD_CONTENT_TYPES = {
    "video/mp4",
    "video/x-msvideo",
    "video/quicktime",
    "video/x-matroska",
    "application/octet-stream",
}
ALLOWED_DETECTORS = {"classic", "yolo", "triton"}
ALLOWED_TRACKERS = {"kalman", "bytetrack", "botsort", "deepsort"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "dataset" / "WHATSAAP ASSIGNMENT.mp4"
START_TIME = time.time()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("dribble_api")

app = FastAPI(title="Basketball Dribble Analysis API", version=APP_VERSION)
_rate_limit_buckets: dict[str, deque] = defaultdict(deque)
_rate_limit_lock = threading.Lock()

if CORS_ORIGINS_RAW == "*":
    cors_origins = ["*"]
else:
    cors_origins = [item.strip() for item in CORS_ORIGINS_RAW.split(",") if item.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["*"],
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

if ENABLE_API_KEY_AUTH and not API_KEY:
    logger.warning("ENABLE_API_KEY_AUTH=true but API_KEY is empty; all secured routes will reject requests.")


class AnalyzeRequest(BaseModel):
    """JSON contract for /analyze endpoint.

    The API accepts either an absolute path or a project-relative path.
    If `video_path` is empty, a default dataset video is used.
    """

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


def _resolve_input_video_path(raw_path: str) -> Path:
    if not raw_path.strip():
        return DEFAULT_VIDEO
    candidate = Path(raw_path.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def _resolve_output_path(raw_path: str) -> Optional[Path]:
    if not raw_path.strip():
        return None
    out = Path(raw_path.strip()).expanduser()
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _validate_backends(detector_backend: str, tracker_backend: str) -> None:
    if detector_backend not in ALLOWED_DETECTORS:
        raise HTTPException(status_code=400, detail="Invalid detector_backend.")
    if tracker_backend not in ALLOWED_TRACKERS:
        raise HTTPException(status_code=400, detail="Invalid tracker_backend.")


def _save_upload_to_temp_file(video_file: UploadFile) -> Path:
    suffix = Path(video_file.filename or "upload.mp4").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {suffix}. Allowed: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}",
        )
    if video_file.content_type and video_file.content_type not in ALLOWED_UPLOAD_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {video_file.content_type}",
        )

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_path = Path(temp_file.name)
    total = 0
    try:
        with temp_file as fp:
            while True:
                chunk = video_file.file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Uploaded file too large (>{MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
                    )
                fp.write(chunk)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    if total == 0:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return temp_path


def _is_public_path(path: str) -> bool:
    # Public routes are intentionally left without API key/rate-limit checks for health and docs access.
    return path in {"/", "/health", "/ready", "/favicon.ico", "/docs", "/openapi.json", "/redoc"}


def _enforce_rate_limit(client_key: str) -> None:
    now = time.time()
    window_start = now - 60.0
    with _rate_limit_lock:
        bucket = _rate_limit_buckets[client_key]
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
        bucket.append(now)


def _enforce_api_key(request: Request) -> None:
    if not ENABLE_API_KEY_AUTH:
        return
    provided = request.headers.get("x-api-key", "")
    if not API_KEY or provided != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized. Missing or invalid x-api-key.")


def _error_response(request: Request, status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": detail,
            "request_id": getattr(request.state, "request_id", ""),
        },
    )


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex)
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        if not _is_public_path(request.url.path):
            client_host = request.client.host if request.client else "unknown"
            _enforce_rate_limit(client_host)
            _enforce_api_key(request)
        response = await call_next(request)
    except HTTPException as exc:
        response = _error_response(request, exc.status_code, str(exc.detail))
    except Exception:
        logger.exception("Unhandled exception for request_id=%s path=%s", request_id, request.url.path)
        response = _error_response(request, 500, "Internal Server Error")

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    response.headers["x-request-id"] = request_id
    response.headers["x-process-time-ms"] = f"{elapsed_ms:.2f}"
    logger.info(
        "request_id=%s method=%s path=%s status=%s process_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/")
def root() -> dict:
    """Service metadata endpoint used by browser and quick health smoke tests."""

    return {
        "message": "Basketball Dribble Analysis API is running",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "analyze": "/analyze",
        "analyze_input": "/analyze-input (video_path or video_url or video_file)",
    }


@app.get("/health")
def health() -> dict:
    """Lightweight liveness check for deployment probes."""

    return {
        "status": "ok",
        "version": APP_VERSION,
        "uptime_seconds": round(time.time() - START_TIME, 3),
    }


@app.get("/ready")
def ready() -> Response:
    """Readiness check for startup/deployment automation."""

    checks = {
        "default_video_exists": DEFAULT_VIDEO.exists(),
        "project_root_exists": PROJECT_ROOT.exists(),
    }
    ready_state = all(checks.values())
    payload = {
        "status": "ready" if ready_state else "not_ready",
        "checks": checks,
    }
    if not ready_state:
        return JSONResponse(status_code=503, content=payload)
    return JSONResponse(status_code=200, content=payload)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Avoid 404 noise from browsers requesting favicon automatically."""

    return Response(status_code=204)


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    """Analyze video from JSON body and return dribble metrics.

    Input path resolution:
    - empty path -> default dataset video
    - relative path -> resolved under project root
    """

    video_path = _resolve_input_video_path(request.video_path)
    save_video = _resolve_output_path(request.save_video_path)

    try:
        _validate_backends(request.detector_backend, request.tracker_backend)
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
    except HTTPException:
        # Preserve explicitly raised HTTP status codes.
        raise
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
    video_file: Optional[UploadFile] = File(default=None),
    detector_backend: str = Form(default="classic"),
    tracker_backend: str = Form(default="kalman"),
    yolo_model: str = Form(default="yolov8n.pt"),
    yolo_conf: float = Form(default=0.08, ge=0.001, le=1.0),
    yolo_iou: float = Form(default=0.5, ge=0.001, le=1.0),
    yolo_imgsz: int = Form(default=960, ge=320, le=1920),
    yolo_class_id: int = Form(default=32, ge=0, le=999),
    yolo_class_fallback: bool = Form(default=True),
    triton_url: str = Form(default="localhost:8000"),
    triton_model_name: str = Form(default="basketball_yolo_trt"),
) -> dict:
    """Analyze video from form-data input.

    Supports exactly one of:
    - `video_path` (local path)
    - `video_url` (remote URL download)
    - `video_file` (uploaded file)

    Any temporary file created during URL/upload flow is deleted in `finally`.
    """

    temp_path = None
    save_video = _resolve_output_path(save_video_path)

    try:
        _validate_backends(detector_backend, tracker_backend)

        # Ensure mutually exclusive input source to avoid ambiguous behavior.
        chosen_inputs = int(bool(video_path.strip())) + int(bool(video_url.strip())) + int(video_file is not None)
        if chosen_inputs > 1:
            raise HTTPException(
                status_code=400,
                detail="Provide only one input: video_path or video_url or video_file.",
            )

        if video_file is not None:
            # Persist upload to temp path so the core analyzer can read from filesystem path.
            temp_path = _save_upload_to_temp_file(video_file)
            resolved_video = temp_path
        elif video_url.strip():
            # URL validation blocks unsafe/private targets before download.
            safe_url = validate_remote_video_url(video_url)
            temp_path = download_video_to_temp_file(safe_url)
            resolved_video = temp_path
        elif video_path.strip():
            resolved_video = _resolve_input_video_path(video_path)
        else:
            resolved_video = DEFAULT_VIDEO

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
        if video_file is not None:
            video_file.file.close()
        # Always cleanup temporary artifacts from URL/upload flows.
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
