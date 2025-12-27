from __future__ import annotations

import logging
import threading
from pathlib import Path
from tkinter import BOTH, END, Button, Entry, Frame, Label, StringVar, Text, Tk, filedialog

from downloader import download_video
from main import run_pipeline

logger = logging.getLogger(__name__)


def launch_gui() -> None:
    root = Tk()
    root.title("Skeleton Motion Analyzer")

    input_path_var = StringVar()
    output_dir_var = StringVar(value=str(Path.cwd() / "output"))
    fps_var = StringVar(value="")
    url_var = StringVar()
    cookies_var = StringVar()

    status = Text(root, height=10)

    def log(message: str) -> None:
        status.insert(END, message + "\n")
        status.see(END)

    def browse_input() -> None:
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov")])
        if path:
            input_path_var.set(path)

    def browse_output() -> None:
        path = filedialog.askdirectory()
        if path:
            output_dir_var.set(path)

    def run_task() -> None:
        def task() -> None:
            try:
                output_dir = Path(output_dir_var.get())
                target_fps = float(fps_var.get()) if fps_var.get().strip() else None
                input_path = input_path_var.get().strip()

                if url_var.get().strip():
                    log("Downloading video via yt-dlp...")
                    input_path = str(download_video(url_var.get().strip(), output_dir, cookies_var.get().strip() or None))
                    input_path_var.set(input_path)

                if not input_path:
                    log("ERROR: Provide input video path or download URL.")
                    return

                log("Processing video...")
                run_pipeline(Path(input_path), output_dir, target_fps)
                log("Done. Outputs saved in output directory.")
            except Exception as exc:  # noqa: BLE001 - show error in GUI
                logger.exception("Processing failed")
                log(f"ERROR: {exc}")

        threading.Thread(target=task, daemon=True).start()

    main_frame = Frame(root, padx=10, pady=10)
    main_frame.pack(fill=BOTH, expand=True)

    Label(main_frame, text="Input Video").grid(row=0, column=0, sticky="w")
    Entry(main_frame, textvariable=input_path_var, width=60).grid(row=0, column=1, padx=5)
    Button(main_frame, text="Browse", command=browse_input).grid(row=0, column=2)

    Label(main_frame, text="Download URL (yt-dlp)").grid(row=1, column=0, sticky="w")
    Entry(main_frame, textvariable=url_var, width=60).grid(row=1, column=1, padx=5)

    Label(main_frame, text="Cookies from browser").grid(row=2, column=0, sticky="w")
    Entry(main_frame, textvariable=cookies_var, width=60).grid(row=2, column=1, padx=5)
    Label(main_frame, text="e.g. chrome or chrome:Profile 1").grid(row=2, column=2, sticky="w")

    Label(main_frame, text="Output directory").grid(row=3, column=0, sticky="w")
    Entry(main_frame, textvariable=output_dir_var, width=60).grid(row=3, column=1, padx=5)
    Button(main_frame, text="Browse", command=browse_output).grid(row=3, column=2)

    Label(main_frame, text="Target FPS").grid(row=4, column=0, sticky="w")
    Entry(main_frame, textvariable=fps_var, width=10).grid(row=4, column=1, sticky="w")

    Button(main_frame, text="Run", command=run_task).grid(row=5, column=1, pady=10)

    status.pack(fill=BOTH, expand=True, padx=10, pady=10)

    root.mainloop()
