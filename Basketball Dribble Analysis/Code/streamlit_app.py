import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st

from Code import analyze_video
from io_utils import download_video_to_temp_file, validate_remote_video_url


def resolve_video_path(
    input_mode: str,
    project_root: Path,
    path_value: str,
    url_value: str,
    upload_value,
) -> Tuple[Path, Optional[Path], str]:
    temp_file = None
    default_video = project_root / "dataset" / "WHATSAAP ASSIGNMENT.mp4"

    if input_mode == "Path":
        if not path_value.strip():
            return default_video, temp_file, "default"
        candidate = Path(path_value.strip()).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return candidate, temp_file, "path"

    if input_mode == "URL":
        safe_url = validate_remote_video_url(url_value)
        temp_file = download_video_to_temp_file(safe_url)
        return temp_file, temp_file, "url"

    if upload_value is None:
        raise ValueError("Please upload a video file.")

    suffix = Path(upload_value.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file = Path(tmp.name)
    with tmp as fp:
        fp.write(upload_value.read())
    return temp_file, temp_file, "upload"


def get_quality_label(result: Dict[str, Any]) -> Tuple[str, float]:
    frames = max(1, int(result["processed_frames"]))
    missed = int(result["missed_detections"])
    detected_ratio = 1.0 - (missed / frames)

    if detected_ratio > 0.95:
        return "Excellent", detected_ratio
    if detected_ratio > 0.85:
        return "Good", detected_ratio
    if detected_ratio > 0.70:
        return "Fair", detected_ratio
    return "Needs tuning", detected_ratio


def build_report_line(result: Dict[str, Any], quality_label: str, quality_score: float) -> str:
    return (
        f"Processed Frames: {result['processed_frames']}, "
        f"Estimated Bounces: {result['estimated_bounces']}, "
        f"Missed Detections: {result['missed_detections']}, "
        f"Avg Speed: {result['average_smoothed_speed_m_s']:.2f} m/s, "
        f"Elapsed: {result['elapsed_seconds']:.2f} s, "
        f"Tracking Quality: {quality_label} (~{quality_score * 100:.1f}%)"
    )


def main() -> None:
    st.set_page_config(page_title="Basketball Dribble Analysis", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;700&family=Nunito:wght@400;600;800&display=swap');

        .stApp {
            font-family: 'Nunito', sans-serif;
            background:
                radial-gradient(1200px 500px at 15% -10%, rgba(251, 113, 133, 0.35), transparent 60%),
                radial-gradient(1000px 500px at 95% 0%, rgba(167, 139, 250, 0.35), transparent 60%),
                linear-gradient(160deg, #fff7ed 0%, #fef2f2 40%, #fdf2f8 72%, #f5f3ff 100%);
        }
        .title-wrap {
            padding: 1rem 1.1rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #fb7185 0%, #f472b6 35%, #a78bfa 100%);
            color: #fff7fb;
            margin-bottom: 1.1rem;
            box-shadow: 0 14px 34px rgba(244, 114, 182, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.35);
        }
        .title-wrap h1 {
            margin: 0;
            font-family: 'Baloo 2', cursive;
            font-size: 2rem;
            letter-spacing: 0.2px;
        }
        .title-wrap p {
            margin: 0.4rem 0 0 0;
            color: #fff1f6;
        }
        .panel {
            background: #fff7fb;
            border: 1px solid #f9a8d4;
            border-radius: 16px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 6px 18px rgba(244, 114, 182, 0.12);
        }
        .report-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(254,242,242,0.95));
            border: 1px solid #fda4af;
            border-radius: 16px;
            padding: 0.8rem 0.95rem;
            box-shadow: 0 8px 20px rgba(251, 113, 133, 0.12);
            margin-top: 0.7rem;
        }
        .report-title {
            font-weight: 800;
            color: #be185d;
            margin-bottom: 0.25rem;
        }
        [data-testid="stMetricValue"] {
            color: #be185d;
            font-family: 'Baloo 2', cursive;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fff7ed 0%, #fef2f2 60%, #fdf2f8 100%);
            border-right: 1px solid #fed7e2;
        }
        [data-testid="stSidebar"] * {
            font-family: 'Nunito', sans-serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="title-wrap">
            <h1>Basketball Dribble Analysis</h1>
            <p>Upload, paste URL, or provide a local path to analyze dribbles and speed with a friendly dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    project_root = Path(__file__).resolve().parents[1]
    default_video = project_root / "dataset" / "WHATSAAP ASSIGNMENT.mp4"
    default_output = project_root / "output_streamlit.mp4"

    if "history" not in st.session_state:
        st.session_state["history"] = []

    with st.sidebar:
        st.header("Run Settings")
        max_frames = st.slider("Max Frames", min_value=1000, max_value=5000, value=1000, step=100)
        detector_backend = st.selectbox("Detector", options=["classic", "yolo", "triton"], index=0)
        tracker_options = ["kalman"]
        if detector_backend == "yolo":
            tracker_options = ["kalman", "bytetrack", "botsort", "deepsort"]
        tracker_backend = st.selectbox("Tracker", options=tracker_options, index=0)
        yolo_model = "yolov8n.pt"
        yolo_conf = 0.08
        yolo_iou = 0.5
        yolo_imgsz = 960
        yolo_class_id = 32
        yolo_class_fallback = True
        if detector_backend == "yolo":
            yolo_model = st.text_input("YOLO Model", value="yolov8n.pt")
            yolo_conf = st.slider("YOLO Conf", min_value=0.01, max_value=0.90, value=0.08, step=0.01)
            yolo_iou = st.slider("YOLO IoU", min_value=0.10, max_value=0.90, value=0.50, step=0.01)
            yolo_imgsz = st.select_slider("YOLO Img Size", options=[640, 768, 896, 960, 1024, 1280], value=960)
            yolo_class_id = st.number_input("YOLO Class ID", min_value=0, max_value=999, value=32, step=1)
            yolo_class_fallback = st.checkbox("YOLO Class Fallback", value=True)
        triton_url = "localhost:8000"
        triton_model_name = "basketball_yolo_trt"
        if detector_backend == "triton":
            triton_url = st.text_input("Triton URL", value="localhost:8000")
            triton_model_name = st.text_input("Triton Model", value="basketball_yolo_trt")
        save_output = st.checkbox("Save Annotated Video", value=False)
        output_path_value = str(default_output)
        if save_output:
            output_path_value = st.text_input("Output Path", value=str(default_output))
        st.caption("Tip: Keep max frames between 1000 and 5000 for consistent comparisons.")

    left, right = st.columns([1.2, 1.8], gap="large")
    path_value = str(default_video)
    url_value = ""
    upload_value = None

    with left:
        st.subheader("Input")
        tab_path, tab_url, tab_upload = st.tabs(["Path", "URL", "Upload"])

        with tab_path:
            path_value = st.text_input("Video Path", value=str(default_video))
        with tab_url:
            url_value = st.text_input("Video URL", value="", placeholder="https://example.com/video.mp4")
        with tab_upload:
            upload_value = st.file_uploader(
                "Upload Video",
                type=["mp4", "avi", "mov", "mkv"],
                accept_multiple_files=False,
            )

        input_mode = st.segmented_control(
            "Use input from",
            options=["Path", "URL", "Upload"],
            default="Path",
        )

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.caption("Selected Source")
        if input_mode == "Path":
            st.code(path_value or str(default_video), language="text")
        elif input_mode == "URL":
            st.code(url_value or "<empty URL>", language="text")
        else:
            st.code(upload_value.name if upload_value else "<no file uploaded>", language="text")
        st.markdown("</div>", unsafe_allow_html=True)

        run = st.button("Run Analysis", type="primary", use_container_width=True)

    with right:
        st.subheader("Results")
        result_placeholder = st.container()
        raw_placeholder = st.container()

    if run:
        temp_to_delete = None
        try:
            video_path, temp_to_delete, input_used = resolve_video_path(
                input_mode=input_mode,
                project_root=project_root,
                path_value=path_value,
                url_value=url_value,
                upload_value=upload_value,
            )

            save_video = None
            if save_output:
                save_video = Path(output_path_value).expanduser()
                if not save_video.is_absolute():
                    save_video = project_root / save_video
                save_video.parent.mkdir(parents=True, exist_ok=True)

            with st.spinner("Analyzing video..."):
                result = analyze_video(
                    video_path=video_path,
                    max_frames=int(max_frames),
                    quiet=True,
                    show=False,
                    save_video=save_video,
                    detector_backend=detector_backend,
                    tracker_backend=tracker_backend,
                    yolo_model=yolo_model,
                    yolo_conf=float(yolo_conf),
                    yolo_iou=float(yolo_iou),
                    yolo_imgsz=int(yolo_imgsz),
                    yolo_class_id=int(yolo_class_id),
                    yolo_class_fallback=bool(yolo_class_fallback),
                    triton_url=triton_url,
                    triton_model_name=triton_model_name,
                )

            result["input_mode"] = input_used
            result["run_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["history"] = [result] + st.session_state["history"][:9]

            with result_placeholder:
                quality_label, quality_score = get_quality_label(result)
                c1, c2, c3 = st.columns(3)
                c1.metric("Processed Frames", result["processed_frames"])
                c2.metric("Estimated Bounces", result["estimated_bounces"])
                c3.metric("Missed Detections", result["missed_detections"])

                c4, c5, c6 = st.columns(3)
                c4.metric("Avg Speed (m/s)", f"{result['average_smoothed_speed_m_s']:.3f}")
                c5.metric("Elapsed (s)", f"{result['elapsed_seconds']:.2f}")
                c6.metric("Tracking Quality", quality_label)

                st.progress(quality_score, text=f"Detection coverage: {quality_score * 100:.1f}%")
                st.success("Analysis completed.")
                st.caption(f"Video: {result['video_path']}")
                st.caption(f"Input mode: {result['input_mode']} | Run at: {result['run_at']}")
                st.caption(
                    f"Pipeline: detector={result.get('detector_backend', 'n/a')}, "
                    f"tracker={result.get('tracker_backend', 'n/a')}"
                )
                for warn in result.get("warnings", []):
                    st.warning(warn)

                report_line = build_report_line(result, quality_label, quality_score)
                st.markdown(
                    f"""
                    <div class="report-card">
                        <div class="report-title">Report Summary Line</div>
                        <div>{report_line}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.code(report_line, language="text")

            with raw_placeholder.expander("Raw JSON", expanded=False):
                st.json(result)
                st.download_button(
                    "Download Result JSON",
                    data=json.dumps(result, indent=2),
                    file_name="analysis_result.json",
                    mime="application/json",
                    use_container_width=True,
                )

            if save_video is not None:
                st.info(f"Annotated video saved at: {save_video}")
                if save_video.exists():
                    st.video(str(save_video))

        except Exception as exc:
            result_placeholder.error(f"Analysis failed: {exc}")
        finally:
            if temp_to_delete is not None and temp_to_delete.exists():
                temp_to_delete.unlink(missing_ok=True)

    if st.session_state["history"]:
        st.subheader("Recent Runs")
        for i, item in enumerate(st.session_state["history"], start=1):
            st.caption(
                f"{i}. {item['run_at']} | bounces={item['estimated_bounces']} | "
                f"frames={item['processed_frames']} | mode={item.get('input_mode', 'n/a')}"
            )


if __name__ == "__main__":
    main()
