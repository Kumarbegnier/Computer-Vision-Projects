import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class Candidate:
    center: Tuple[float, float]
    radius: float
    score: float
    motion_score: float = 0.0
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None


class BallTracker:
    def __init__(self) -> None:
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 5e-1
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def _set_transition(self, dt: float) -> None:
        self.kalman.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )

    def predict(self, dt: float) -> Tuple[float, float]:
        dt = max(dt, 1e-3)
        self._set_transition(dt)
        pred = self.kalman.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def update_with_prediction(
        self,
        predicted: Tuple[float, float],
        measurement: Optional[Tuple[float, float]],
    ) -> Tuple[float, float]:
        if not self.initialized and measurement is None:
            return (0.0, 0.0)

        if not self.initialized and measurement is not None:
            self.kalman.statePost = np.array(
                [[measurement[0]], [measurement[1]], [0], [0]], dtype=np.float32
            )
            self.initialized = True
            return measurement

        estimate = predicted
        if measurement is not None:
            measure = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
            corrected = self.kalman.correct(measure)
            estimate = (float(corrected[0, 0]), float(corrected[1, 0]))
        return estimate


class YoloBallDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.08,
        iou_threshold: float = 0.5,
        image_size: int = 960,
        class_ids: Optional[Sequence[int]] = (32,),
        class_fallback: bool = True,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YOLO backend requested, but 'ultralytics' is not installed."
            ) from exc

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.class_ids = list(class_ids) if class_ids is not None else None
        self.class_fallback = class_fallback
        self._tracker_yaml = {
            "bytetrack": "bytetrack.yaml",
            "botsort": "botsort.yaml",
        }

    def _to_candidates(self, result: Any) -> List[Candidate]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        candidates: List[Candidate] = []
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            conf = float(box.conf[0].detach().cpu().item())
            x1, y1, x2, y2 = xyxy
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            radius = max(4.0, 0.25 * ((x2 - x1) + (y2 - y1)))
            candidates.append(
                Candidate(
                    center=(float(cx), float(cy)),
                    radius=float(radius),
                    score=conf * 1000.0,
                    motion_score=0.0,
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return candidates

    def _run_yolo(self, frame: np.ndarray, tracker_backend: str, classes: Optional[List[int]]) -> Any:
        if tracker_backend in self._tracker_yaml:
            return self.model.track(
                source=frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                persist=True,
                verbose=False,
                tracker=self._tracker_yaml[tracker_backend],
                classes=classes,
            )[0]
        return self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            verbose=False,
            classes=classes,
        )[0]

    def detect(self, frame: np.ndarray, tracker_backend: str = "kalman") -> List[Candidate]:
        result = self._run_yolo(frame, tracker_backend, self.class_ids)
        candidates = self._to_candidates(result)

        # Fallback for custom fine-tuned models where class index may differ from COCO sports-ball (32).
        if not candidates and self.class_fallback and self.class_ids is not None:
            result = self._run_yolo(frame, tracker_backend, None)
            candidates = self._to_candidates(result)
        return candidates


class TritonYoloDetector:
    def __init__(
        self,
        url: str = "localhost:8000",
        model_name: str = "basketball_yolo_trt",
        sports_ball_class_id: int = 32,
        conf_threshold: float = 0.2,
    ) -> None:
        try:
            import tritonclient.http as httpclient  # type: ignore
            from tritonclient.utils import np_to_triton_dtype  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Triton backend requested, but 'tritonclient[http]' is not installed."
            ) from exc

        self.client = httpclient.InferenceServerClient(url=url, verbose=False)
        self.httpclient = httpclient
        self.np_to_triton_dtype = np_to_triton_dtype
        self.model_name = model_name
        self.sports_ball_class_id = sports_ball_class_id
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> List[Candidate]:
        # Contract: Triton model should accept UINT8 [1,H,W,3] input named "images"
        # and return [N,6] rows named "detections" => [x1,y1,x2,y2,score,class_id].
        batched = np.expand_dims(frame, axis=0).astype(np.uint8)
        infer_input = self.httpclient.InferInput(
            "images",
            batched.shape,
            self.np_to_triton_dtype(batched.dtype),
        )
        infer_input.set_data_from_numpy(batched)

        infer_output = self.httpclient.InferRequestedOutput("detections")
        result = self.client.infer(
            model_name=self.model_name,
            inputs=[infer_input],
            outputs=[infer_output],
        )
        raw = result.as_numpy("detections")
        if raw is None or raw.size == 0:
            return []

        detections = raw.reshape(-1, 6)
        out: List[Candidate] = []
        for row in detections:
            x1, y1, x2, y2, score, cls_id = row.tolist()
            if int(cls_id) != self.sports_ball_class_id:
                continue
            if float(score) < self.conf_threshold:
                continue
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            radius = max(4.0, 0.25 * ((x2 - x1) + (y2 - y1)))
            out.append(
                Candidate(
                    center=(float(cx), float(cy)),
                    radius=float(radius),
                    score=float(score) * 1000.0,
                    bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return out


class ExternalBallTracker:
    def __init__(self, backend: str, fps: float) -> None:
        self.backend = backend
        self.fps = fps
        self._tracker: Any = None
        self._tracker_type: Optional[str] = None

        if backend == "bytetrack":
            try:
                import supervision as sv  # type: ignore

                self._tracker = sv.ByteTrack(frame_rate=max(1, int(round(fps))))
                self._tracker_type = "supervision_bytetrack"
                self._sv = sv
            except Exception as exc:
                raise RuntimeError(
                    "ByteTrack backend requested, but 'supervision' is not installed."
                ) from exc
        elif backend == "deepsort":
            try:
                from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore

                self._tracker = DeepSort(max_age=20, n_init=2)
                self._tracker_type = "deep_sort_realtime"
            except Exception as exc:
                raise RuntimeError(
                    "DeepSORT backend requested, but 'deep-sort-realtime' is not installed."
                ) from exc
        else:
            raise RuntimeError(f"Unsupported tracker backend: {backend}")

    def update(
        self,
        candidates: List[Candidate],
        frame: np.ndarray,
    ) -> List[Candidate]:
        if not candidates:
            return []

        if self._tracker_type == "supervision_bytetrack":
            valid = [c for c in candidates if c.bbox_xyxy is not None]
            if not valid:
                return []
            xyxy = np.array([c.bbox_xyxy for c in valid], dtype=np.float32)
            class_id = np.full((len(valid),), 32, dtype=np.int32)
            confidence = np.array([min(1.0, c.score / 1000.0) for c in valid], dtype=np.float32)
            detections = self._sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            tracked = self._tracker.update_with_detections(detections)

            out: List[Candidate] = []
            for i in range(len(tracked.xyxy)):
                x1, y1, x2, y2 = tracked.xyxy[i].tolist()
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.5
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                radius = max(4.0, 0.25 * ((x2 - x1) + (y2 - y1)))
                out.append(
                    Candidate(
                        center=(float(cx), float(cy)),
                        radius=float(radius),
                        score=conf * 1000.0,
                        bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )
            return out

        if self._tracker_type == "deep_sort_realtime":
            raw_dets = []
            for c in candidates:
                if c.bbox_xyxy is None:
                    continue
                x1, y1, x2, y2 = c.bbox_xyxy
                raw_dets.append(([x1, y1, x2 - x1, y2 - y1], min(1.0, c.score / 1000.0), "ball"))
            tracks = self._tracker.update_tracks(raw_dets, frame=frame)
            out = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                l, t, r, b = track.to_ltrb()
                cx = 0.5 * (l + r)
                cy = 0.5 * (t + b)
                radius = max(4.0, 0.25 * ((r - l) + (b - t)))
                out.append(
                    Candidate(
                        center=(float(cx), float(cy)),
                        radius=float(radius),
                        score=1000.0,
                        bbox_xyxy=(float(l), float(t), float(r), float(b)),
                    )
                )
            return out

        return []


def detect_ball_candidates(frame: np.ndarray, fg_mask: Optional[np.ndarray] = None) -> List[Candidate]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 80, 60], dtype=np.uint8)
    upper_orange = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    candidates: List[Candidate] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 80 or area > 30000:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < 5:
            continue

        score = area * circularity
        candidates.append(Candidate(center=(x, y), radius=float(radius), score=float(score)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=100,
        param2=25,
        minRadius=5,
        maxRadius=80,
    )

    if circles is not None:
        for c in np.round(circles[0]).astype(int):
            x, y, radius = int(c[0]), int(c[1]), int(c[2])
            candidates.append(
                Candidate(center=(float(x), float(y)), radius=float(radius), score=float(radius))
            )

    if fg_mask is not None and candidates:
        h, w = fg_mask.shape[:2]
        for candidate in candidates:
            cx = int(candidate.center[0])
            cy = int(candidate.center[1])
            r = int(max(4, candidate.radius))
            x0 = max(0, cx - r)
            x1 = min(w, cx + r + 1)
            y0 = max(0, cy - r)
            y1 = min(h, cy + r + 1)
            patch = fg_mask[y0:y1, x0:x1]
            if patch.size == 0:
                candidate.motion_score = 0.0
            else:
                candidate.motion_score = float(np.count_nonzero(patch)) / float(patch.size)

    return candidates


def choose_measurement(
    candidates: List[Candidate],
    predicted: Tuple[float, float],
    initialized: bool,
    frame_shape: Tuple[int, int, int],
    max_distance: float = 140.0,
) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    if not candidates:
        return None, None

    if not initialized:
        frame_h = frame_shape[0]
        best_init = max(
            candidates,
            key=lambda c: (2.5 * c.motion_score) + (0.001 * c.score) + (0.001 * (frame_h - c.center[1])),
        )
        return best_init.center, best_init.radius

    px, py = predicted
    best: Optional[Candidate] = None
    best_cost = float("inf")

    for c in candidates:
        d = np.hypot(c.center[0] - px, c.center[1] - py)
        if d > max_distance:
            continue
        cost = d - 0.002 * c.score - 80.0 * c.motion_score
        if cost < best_cost:
            best_cost = cost
            best = c

    if best is None:
        return None, None

    return best.center, best.radius


def estimate_bounces_from_y(
    y_positions: List[float],
    fps: float,
    median_radius: float,
) -> int:
    if len(y_positions) < 3:
        return 0

    arr = np.array(y_positions, dtype=np.float32)
    prominence = max(0.12, 0.011 * max(median_radius, 1.0))
    min_frame_gap = max(4, int(round(fps / 3.5)))

    bounce_count = 0
    last_bounce = -10**9
    for idx in range(1, len(arr) - 1):
        if (idx - last_bounce) < min_frame_gap:
            continue
        if (
            arr[idx] > arr[idx - 1]
            and arr[idx] > arr[idx + 1]
            and (arr[idx] - arr[idx - 1]) > prominence
            and (arr[idx] - arr[idx + 1]) > prominence
        ):
            bounce_count += 1
            last_bounce = idx

    return bounce_count


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_video = project_root / "dataset" / "WHATSAAP ASSIGNMENT.mp4"

    parser = argparse.ArgumentParser(
        description="Advanced basketball dribble analysis using tracking and trajectory events"
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=default_video,
        help=f"Path to input video (default: {default_video})",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1000,
        help="Maximum number of frames to process",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-frame logs",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live visualization window",
    )
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Path for annotated output video",
    )
    parser.add_argument(
        "--detector-backend",
        type=str,
        default="classic",
        choices=["classic", "yolo", "triton"],
        help="Detection backend",
    )
    parser.add_argument(
        "--tracker-backend",
        type=str,
        default="kalman",
        choices=["kalman", "bytetrack", "botsort", "deepsort"],
        help="Tracking backend",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path/name used when detector-backend=yolo",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.08,
        help="YOLO confidence threshold (0-1)",
    )
    parser.add_argument(
        "--yolo-iou",
        type=float,
        default=0.5,
        help="YOLO IoU threshold (0-1)",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=960,
        help="YOLO inference image size",
    )
    parser.add_argument(
        "--yolo-class-id",
        type=int,
        default=32,
        help="YOLO class id to prefer (COCO sports ball is 32)",
    )
    parser.add_argument(
        "--disable-yolo-class-fallback",
        action="store_true",
        help="Disable fallback run without class filter when no detections are found",
    )
    parser.add_argument(
        "--triton-url",
        type=str,
        default="localhost:8000",
        help="Triton inference server URL used when detector-backend=triton",
    )
    parser.add_argument(
        "--triton-model-name",
        type=str,
        default="basketball_yolo_trt",
        help="Triton model name used when detector-backend=triton",
    )
    return parser.parse_args()


def analyze_video(
    video_path: Path,
    max_frames: int = 1000,
    quiet: bool = True,
    show: bool = False,
    save_video: Optional[Path] = None,
    detector_backend: str = "classic",
    tracker_backend: str = "kalman",
    yolo_model: str = "yolov8n.pt",
    yolo_conf: float = 0.08,
    yolo_iou: float = 0.5,
    yolo_imgsz: int = 960,
    yolo_class_id: int = 32,
    yolo_class_fallback: bool = True,
    triton_url: str = "localhost:8000",
    triton_model_name: str = "basketball_yolo_trt",
) -> dict:
    if max_frames < 1000 or max_frames > 5000:
        raise ValueError("max_frames must be between 1000 and 5000.")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_count = 0
    missed_detections = 0
    speed_px_s_ema = 0.0
    speed_m_s_ema = 0.0

    start_time = time.time()
    prev_center: Optional[Tuple[float, float]] = None
    radius_samples: deque = deque(maxlen=40)
    radius_all: List[float] = []
    y_history: List[float] = []
    tracker = BallTracker()
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=25, detectShadows=False
    )
    warnings: List[str] = []
    detector: Optional[YoloBallDetector] = None
    triton_detector: Optional[TritonYoloDetector] = None
    external_tracker: Optional[ExternalBallTracker] = None
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    dt = 1.0 / fps

    if detector_backend == "yolo":
        try:
            detector = YoloBallDetector(
                model_path=yolo_model,
                conf_threshold=max(0.001, min(1.0, float(yolo_conf))),
                iou_threshold=max(0.001, min(1.0, float(yolo_iou))),
                image_size=max(320, int(yolo_imgsz)),
                class_ids=[int(yolo_class_id)],
                class_fallback=bool(yolo_class_fallback),
            )
        except Exception as exc:
            warnings.append(f"YOLO unavailable, fallback to classic detector: {exc}")
            detector_backend = "classic"
    elif detector_backend == "triton":
        try:
            triton_detector = TritonYoloDetector(url=triton_url, model_name=triton_model_name)
        except Exception as exc:
            warnings.append(f"Triton unavailable, fallback to classic detector: {exc}")
            detector_backend = "classic"

    if detector_backend == "triton" and tracker_backend in {"bytetrack", "botsort", "deepsort"}:
        warnings.append("Triton detector currently supports Kalman/deepsort flow only. Falling back tracker to kalman.")
        tracker_backend = "kalman"

    if detector_backend != "yolo" and tracker_backend in {"bytetrack", "botsort", "deepsort"}:
        warnings.append(
            "External tracker requested without YOLO detector. Falling back to Kalman-only tracking."
        )
        tracker_backend = "kalman"

    if detector_backend == "yolo" and tracker_backend == "deepsort":
        try:
            external_tracker = ExternalBallTracker(backend=tracker_backend, fps=fps)
        except Exception as exc:
            warnings.append(f"{tracker_backend} unavailable, fallback to Kalman-only: {exc}")
            tracker_backend = "kalman"

    out = None
    if save_video is not None:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(
            str(save_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not out.isOpened():
            raise RuntimeError(f"Unable to create output video file: {save_video}")

    try:
        while frame_count < max_frames:
            ret, frame = video.read()
            if not ret:
                break

            predicted = tracker.predict(dt)
            if detector_backend == "yolo" and detector is not None:
                candidates = detector.detect(frame, tracker_backend=tracker_backend)
                if external_tracker is not None:
                    tracked = external_tracker.update(candidates, frame=frame)
                    if tracked:
                        candidates = tracked
            elif detector_backend == "triton" and triton_detector is not None:
                candidates = triton_detector.detect(frame)
            else:
                fg_mask = bg_subtractor.apply(frame)
                _, fg_mask = cv2.threshold(fg_mask, 210, 255, cv2.THRESH_BINARY)
                fg_mask = cv2.morphologyEx(
                    fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
                )
                fg_mask = cv2.morphologyEx(
                    fg_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
                )
                candidates = detect_ball_candidates(frame, fg_mask=fg_mask)
            measurement, radius = choose_measurement(
                candidates,
                predicted,
                initialized=tracker.initialized,
                frame_shape=frame.shape,
            )
            center = tracker.update_with_prediction(predicted, measurement)

            if measurement is None:
                missed_detections += 1
            else:
                radius_samples.append(float(radius))
                if radius is not None:
                    radius_all.append(float(radius))

            speed_px_s = 0.0
            speed_m_s = 0.0
            if prev_center is not None and center != (0.0, 0.0) and dt > 0:
                speed_px_s = np.hypot(center[0] - prev_center[0], center[1] - prev_center[1]) / dt
                if radius_samples:
                    # Single-camera videos overestimate radius at close range; a lower percentile is more stable.
                    ball_radius_px = float(np.percentile(np.array(radius_samples, dtype=np.float32), 20))
                    pixels_per_meter = (2.0 * ball_radius_px) / 0.24
                    if pixels_per_meter > 0:
                        # Empirical correction for perspective compression in monocular footage.
                        speed_m_s = (speed_px_s / pixels_per_meter) * 2.4

            alpha = 0.2
            speed_px_s_ema = alpha * speed_px_s + (1 - alpha) * speed_px_s_ema
            speed_m_s_ema = alpha * speed_m_s + (1 - alpha) * speed_m_s_ema

            if center != (0.0, 0.0):
                prev_center = center
                y_history.append(center[1])

            if not quiet:
                print(
                    f"Frame {frame_count + 1}: "
                    f"Bounces: pending, "
                    f"Speed: {speed_m_s_ema:.2f} m/s ({speed_px_s_ema:.1f} px/s), "
                    f"Detected: {'yes' if measurement is not None else 'no'}"
                )

            if center != (0.0, 0.0):
                cv2.circle(frame, (int(center[0]), int(center[1])), 10, (0, 255, 0), 2)
            if measurement is not None:
                cv2.circle(
                    frame,
                    (int(measurement[0]), int(measurement[1])),
                    max(8, int(radius or 0)),
                    (0, 165, 255),
                    2,
                )

            cv2.putText(
                frame,
                "Bounces: estimating...",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Speed: {speed_m_s_ema:.2f} m/s",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            if out is not None:
                out.write(frame)

            if show:
                cv2.imshow("Basketball Dribble Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1
    finally:
        video.release()
        if out is not None:
            out.release()
        if show:
            cv2.destroyAllWindows()

    median_radius = float(np.median(np.array(radius_all, dtype=np.float32))) if radius_all else 12.0
    bounce_count = estimate_bounces_from_y(y_history, fps=fps, median_radius=median_radius)
    elapsed_processing = time.time() - start_time
    elapsed_video = frame_count / fps if fps > 0 else 0.0
    result = {
        "video_path": str(video_path),
        "processed_frames": frame_count,
        "estimated_bounces": bounce_count,
        "missed_detections": missed_detections,
        "average_smoothed_speed_m_s": round(float(speed_m_s_ema), 4),
        "elapsed_seconds": round(float(elapsed_video), 3),
        "processing_seconds": round(float(elapsed_processing), 3),
        "detector_backend": detector_backend,
        "tracker_backend": tracker_backend,
        "triton_url": triton_url if detector_backend == "triton" else "",
        "triton_model_name": triton_model_name if detector_backend == "triton" else "",
        "yolo_conf": float(yolo_conf) if detector_backend == "yolo" else None,
        "yolo_iou": float(yolo_iou) if detector_backend == "yolo" else None,
        "yolo_imgsz": int(yolo_imgsz) if detector_backend == "yolo" else None,
        "yolo_class_id": int(yolo_class_id) if detector_backend == "yolo" else None,
        "yolo_class_fallback": bool(yolo_class_fallback) if detector_backend == "yolo" else None,
        "warnings": warnings,
    }
    return result


def main() -> None:
    args = parse_args()
    result = analyze_video(
        video_path=args.video,
        max_frames=args.max_frames,
        quiet=args.quiet,
        show=args.show,
        save_video=args.save_video,
        detector_backend=args.detector_backend,
        tracker_backend=args.tracker_backend,
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf,
        yolo_iou=args.yolo_iou,
        yolo_imgsz=args.yolo_imgsz,
        yolo_class_id=args.yolo_class_id,
        yolo_class_fallback=not args.disable_yolo_class_fallback,
        triton_url=args.triton_url,
        triton_model_name=args.triton_model_name,
    )
    print(f"Processed frames: {result['processed_frames']}")
    print(f"Estimated dribbles (bounces): {result['estimated_bounces']}")
    print(f"Missed detections: {result['missed_detections']}")
    print(f"Average smoothed speed: {result['average_smoothed_speed_m_s']:.2f} m/s")
    print(f"Elapsed time: {result['elapsed_seconds']:.2f} s")
    print(f"Detector: {result['detector_backend']} | Tracker: {result['tracker_backend']}")
    if result.get("warnings"):
        print("Warnings:")
        for item in result["warnings"]:
            print(f"- {item}")


if __name__ == "__main__":
    main()
