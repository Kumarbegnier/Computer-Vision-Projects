import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlretrieve

import streamlit as st

from Code import analyze_video


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
        if not url_value.strip():
            raise ValueError("Video URL is empty.")
        parsed = urlparse(url_value.strip())
        suffix = Path(parsed.path).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()
        temp_file = Path(tmp.name)
        urlretrieve(url_value.strip(), str(temp_file))
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


def main() -> None:
    st.set_page_config(page_title="Basketball Dribble Analysis", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 20% 20%, #ffe4ec 0%, #fdf2f8 30%, #f5f3ff 65%, #eef2ff 100%);
        }
        .title-wrap {
            padding: 0.9rem 1rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #fb7185 0%, #f472b6 45%, #a78bfa 100%);
            color: #fff7fb;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(244, 114, 182, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.35);
        }
        .title-wrap h1 {
            margin: 0;
            font-size: 1.8rem;
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
        [data-testid="stMetricValue"] {
            color: #be185d;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fff7ed 0%, #fdf2f8 100%);
            border-right: 1px solid #fed7e2;
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
        max_frames = st.slider("Max Frames", min_value=1, max_value=50000, value=1000, step=50)
        save_output = st.checkbox("Save Annotated Video", value=False)
        output_path_value = str(default_output)
        if save_output:
            output_path_value = st.text_input("Output Path", value=str(default_output))
        st.caption("Tip: Start with 200-500 frames to test quickly.")

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
