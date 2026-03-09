import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from Code import analyze_video


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_video = project_root / "dataset" / "WHATSAAP ASSIGNMENT.mp4"
    default_out = project_root / "benchmark_results.json"

    parser = argparse.ArgumentParser(description="Benchmark model backends on the same video")
    parser.add_argument("--video", type=Path, default=default_video, help="Input video path")
    parser.add_argument("--max-frames", type=int, default=2000, help="Max frames to process")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt", help="YOLO model path/name")
    parser.add_argument("--triton-url", type=str, default="localhost:8000", help="Triton URL")
    parser.add_argument(
        "--triton-model-name",
        type=str,
        default="basketball_yolo_trt",
        help="Triton model name",
    )
    parser.add_argument("--output", type=Path, default=default_out, help="Output JSON path")
    return parser.parse_args()


def run_case(
    name: str,
    video_path: Path,
    max_frames: int,
    detector_backend: str,
    tracker_backend: str,
    yolo_model: str,
    triton_url: str,
    triton_model_name: str,
) -> Dict[str, Any]:
    result = analyze_video(
        video_path=video_path,
        max_frames=max_frames,
        quiet=True,
        show=False,
        save_video=None,
        detector_backend=detector_backend,
        tracker_backend=tracker_backend,
        yolo_model=yolo_model,
        triton_url=triton_url,
        triton_model_name=triton_model_name,
    )
    result["case"] = name
    return result


def build_cases() -> List[Dict[str, str]]:
    return [
        {"name": "classic_kalman", "detector_backend": "classic", "tracker_backend": "kalman"},
        {"name": "yolo_botsort", "detector_backend": "yolo", "tracker_backend": "botsort"},
        {"name": "yolo_kalman", "detector_backend": "yolo", "tracker_backend": "kalman"},
        {"name": "triton_kalman", "detector_backend": "triton", "tracker_backend": "kalman"},
    ]


def print_summary(rows: List[Dict[str, Any]]) -> None:
    header = (
        f"{'case':<16} {'detector':<8} {'tracker':<8} {'frames':>7} "
        f"{'bounces':>8} {'missed':>7} {'speed(m/s)':>10} {'elapsed(s)':>10} {'proc(s)':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.get('case',''):<16} "
            f"{row.get('detector_backend',''):<8} "
            f"{row.get('tracker_backend',''):<8} "
            f"{int(row.get('processed_frames', 0)):>7} "
            f"{int(row.get('estimated_bounces', 0)):>8} "
            f"{int(row.get('missed_detections', 0)):>7} "
            f"{float(row.get('average_smoothed_speed_m_s', 0.0)):>10.3f} "
            f"{float(row.get('elapsed_seconds', 0.0)):>10.2f} "
            f"{float(row.get('processing_seconds', 0.0)):>8.2f}"
        )
        warnings = row.get("warnings", [])
        if warnings:
            for warn in warnings:
                print(f"  warning: {warn}")


def main() -> None:
    args = parse_args()
    video_path = args.video.expanduser()
    if not video_path.is_absolute():
        video_path = Path(__file__).resolve().parents[1] / video_path

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cases = build_cases()
    results: List[Dict[str, Any]] = []
    for case in cases:
        result = run_case(
            name=case["name"],
            video_path=video_path,
            max_frames=args.max_frames,
            detector_backend=case["detector_backend"],
            tracker_backend=case["tracker_backend"],
            yolo_model=args.yolo_model,
            triton_url=args.triton_url,
            triton_model_name=args.triton_model_name,
        )
        results.append(result)

    print_summary(results)

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "video_path": str(video_path),
        "max_frames": int(args.max_frames),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark JSON: {args.output}")


if __name__ == "__main__":
    main()
