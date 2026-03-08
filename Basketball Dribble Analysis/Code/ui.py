from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from Code import analyze_video


class DribbleAnalysisUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Basketball Dribble Analysis")
        self.root.geometry("760x500")

        project_root = Path(__file__).resolve().parents[1]
        default_video = project_root / "dataset" / "WHATSAAP ASSIGNMENT.mp4"
        default_output = project_root / "output_ui.mp4"

        self.video_path_var = tk.StringVar(value=str(default_video))
        self.max_frames_var = tk.StringVar(value="1000")
        self.save_video_var = tk.BooleanVar(value=False)
        self.output_path_var = tk.StringVar(value=str(default_output))
        self.status_var = tk.StringVar(value="Ready")

        self._build_layout()

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="Input Video").grid(row=0, column=0, sticky=tk.W, pady=(0, 8))
        ttk.Entry(main, textvariable=self.video_path_var, width=72).grid(
            row=1, column=0, sticky=tk.EW, padx=(0, 8)
        )
        ttk.Button(main, text="Browse", command=self._browse_video).grid(row=1, column=1, sticky=tk.E)

        ttk.Label(main, text="Max Frames").grid(row=2, column=0, sticky=tk.W, pady=(12, 8))
        ttk.Entry(main, textvariable=self.max_frames_var, width=12).grid(row=3, column=0, sticky=tk.W)

        ttk.Checkbutton(
            main,
            text="Save Annotated Video",
            variable=self.save_video_var,
            command=self._toggle_output_path,
        ).grid(row=4, column=0, sticky=tk.W, pady=(12, 8))

        self.output_entry = ttk.Entry(
            main,
            textvariable=self.output_path_var,
            width=72,
            state="disabled",
        )
        self.output_entry.grid(row=5, column=0, sticky=tk.EW, padx=(0, 8))
        self.output_browse_btn = ttk.Button(
            main, text="Browse", command=self._browse_output, state="disabled"
        )
        self.output_browse_btn.grid(row=5, column=1, sticky=tk.E)

        self.run_btn = ttk.Button(main, text="Run Analysis", command=self._run_analysis)
        self.run_btn.grid(row=6, column=0, sticky=tk.W, pady=(16, 10))

        ttk.Label(main, textvariable=self.status_var).grid(row=6, column=1, sticky=tk.E)

        ttk.Label(main, text="Result").grid(row=7, column=0, sticky=tk.W, pady=(10, 6))
        self.result_text = tk.Text(main, height=14, wrap=tk.WORD)
        self.result_text.grid(row=8, column=0, columnspan=2, sticky=tk.NSEW)

        main.columnconfigure(0, weight=1)
        main.rowconfigure(8, weight=1)

    def _browse_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if selected:
            self.video_path_var.set(selected)

    def _browse_output(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Save Annotated Video As",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("All files", "*.*")],
        )
        if selected:
            self.output_path_var.set(selected)

    def _toggle_output_path(self) -> None:
        state = "normal" if self.save_video_var.get() else "disabled"
        self.output_browse_btn.configure(state=state)
        self.output_entry.configure(state=state)

    def _run_analysis(self) -> None:
        video_path = Path(self.video_path_var.get().strip())
        if not video_path.exists():
            messagebox.showerror("Invalid Input", f"Video not found:\n{video_path}")
            return

        try:
            max_frames = int(self.max_frames_var.get().strip())
            if max_frames <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Max frames must be a positive integer.")
            return

        save_video = None
        if self.save_video_var.get():
            output = self.output_path_var.get().strip()
            if not output:
                messagebox.showerror("Invalid Input", "Output path is empty.")
                return
            save_video = Path(output)

        self.run_btn.configure(state="disabled")
        self.status_var.set("Running...")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Analysis started...\n")

        thread = threading.Thread(
            target=self._analysis_worker,
            args=(video_path, max_frames, save_video),
            daemon=True,
        )
        thread.start()

    def _analysis_worker(
        self,
        video_path: Path,
        max_frames: int,
        save_video: Optional[Path],
    ) -> None:
        try:
            result = analyze_video(
                video_path=video_path,
                max_frames=max_frames,
                quiet=True,
                show=False,
                save_video=save_video,
            )
            self.root.after(0, self._on_success, result)
        except Exception as exc:
            self.root.after(0, self._on_error, str(exc))

    def _on_success(self, result: dict) -> None:
        self.status_var.set("Done")
        self.run_btn.configure(state="normal")
        lines = [
            f"Video: {result['video_path']}",
            f"Processed frames: {result['processed_frames']}",
            f"Estimated bounces: {result['estimated_bounces']}",
            f"Missed detections: {result['missed_detections']}",
            f"Average smoothed speed (m/s): {result['average_smoothed_speed_m_s']}",
            f"Elapsed seconds: {result['elapsed_seconds']}",
        ]
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "\n".join(lines))

    def _on_error(self, error_message: str) -> None:
        self.status_var.set("Failed")
        self.run_btn.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"Analysis failed:\n{error_message}")


def main() -> None:
    root = tk.Tk()
    app = DribbleAnalysisUI(root)
    app._toggle_output_path()
    root.mainloop()


if __name__ == "__main__":
    main()
