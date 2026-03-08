import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Candidate:
    center: Tuple[float, float]
    radius: float
    score: float
    motion_score: float = 0.0


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

    def update(
        self, dt: float, measurement: Optional[Tuple[float, float]]
    ) -> Tuple[float, float]:
        if not self.initialized and measurement is None:
            return (0.0, 0.0)

        if not self.initialized and measurement is not None:
            self.kalman.statePost = np.array(
                [[measurement[0]], [measurement[1]], [0], [0]], dtype=np.float32
            )
            self.initialized = True
            return measurement

        estimate = self.predict(dt)
        if measurement is not None:
            measure = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
            corrected = self.kalman.correct(measure)
            estimate = (float(corrected[0, 0]), float(corrected[1, 0]))
        return estimate


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
    return parser.parse_args()


def analyze_video(
    video_path: Path,
    max_frames: int = 1000,
    quiet: bool = True,
    show: bool = False,
    save_video: Optional[Path] = None,
) -> dict:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_count = 0
    bounce_count = 0
    missed_detections = 0
    speed_px_s_ema = 0.0
    speed_m_s_ema = 0.0

    start_time = time.time()
    previous_frame_time = start_time
    prev_center: Optional[Tuple[float, float]] = None
    y_window: deque = deque(maxlen=3)
    radius_samples: deque = deque(maxlen=40)
    last_bounce_frame = -10**9
    tracker = BallTracker()
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=25, detectShadows=False
    )

    out = None
    if save_video is not None:
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(
            str(save_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    try:
        while frame_count < max_frames:
            ret, frame = video.read()
            if not ret:
                break

            now = time.time()
            frame_dt = now - previous_frame_time
            previous_frame_time = now
            dt = max(frame_dt, 1 / 30)

            predicted = tracker.predict(dt)
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
            center = tracker.update(dt, measurement)

            if measurement is None:
                missed_detections += 1
            else:
                radius_samples.append(float(radius))

            if center != (0.0, 0.0):
                y_window.append(center[1])
                if len(y_window) == 3:
                    y1, y2, y3 = y_window[0], y_window[1], y_window[2]
                    prominence = 6.0
                    min_frame_gap = 5
                    is_bounce = (
                        y2 > y1
                        and y2 > y3
                        and (y2 - y1) > prominence
                        and (y2 - y3) > prominence
                        and (frame_count - last_bounce_frame) >= min_frame_gap
                    )
                    if is_bounce:
                        bounce_count += 1
                        last_bounce_frame = frame_count

            speed_px_s = 0.0
            speed_m_s = 0.0
            if prev_center is not None and center != (0.0, 0.0) and dt > 0:
                speed_px_s = np.hypot(center[0] - prev_center[0], center[1] - prev_center[1]) / dt
                if radius_samples:
                    # Approximate pixel-to-meter from basketball diameter (0.24 m).
                    pixels_per_meter = (2.0 * float(np.median(radius_samples))) / 0.24
                    if pixels_per_meter > 0:
                        speed_m_s = speed_px_s / pixels_per_meter

            alpha = 0.2
            speed_px_s_ema = alpha * speed_px_s + (1 - alpha) * speed_px_s_ema
            speed_m_s_ema = alpha * speed_m_s + (1 - alpha) * speed_m_s_ema

            if center != (0.0, 0.0):
                prev_center = center

            if not quiet:
                print(
                    f"Frame {frame_count + 1}: "
                    f"Bounces: {bounce_count}, "
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
                f"Bounces: {bounce_count}",
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

    elapsed = time.time() - start_time
    result = {
        "video_path": str(video_path),
        "processed_frames": frame_count,
        "estimated_bounces": bounce_count,
        "missed_detections": missed_detections,
        "average_smoothed_speed_m_s": round(float(speed_m_s_ema), 4),
        "elapsed_seconds": round(float(elapsed), 3),
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
    )
    print(f"Processed frames: {result['processed_frames']}")
    print(f"Estimated dribbles (bounces): {result['estimated_bounces']}")
    print(f"Missed detections: {result['missed_detections']}")
    print(f"Average smoothed speed: {result['average_smoothed_speed_m_s']:.2f} m/s")
    print(f"Elapsed time: {result['elapsed_seconds']:.2f} s")


if __name__ == "__main__":
    main()
