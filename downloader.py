from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yt_dlp

logger = logging.getLogger(__name__)


def download_video(url: str, output_dir: Path, cookies_from_browser: Optional[str] = None) -> Path:
    """Download a video using yt-dlp and return the file path.

    cookies_from_browser examples: "chrome", "firefox", "edge", "brave".
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(output_dir / "downloaded.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
    }
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = cookies_from_browser

    logger.info("Downloading video from %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    path = Path(filename)
    if path.suffix.lower() != ".mp4":
        mp4_candidate = path.with_suffix(".mp4")
        if mp4_candidate.exists():
            return mp4_candidate
    return path
