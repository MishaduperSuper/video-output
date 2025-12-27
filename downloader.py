from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import yt_dlp

logger = logging.getLogger(__name__)


def _parse_cookies_from_browser(spec: Optional[str]) -> Optional[str | Tuple[str, ...]]:
    """Parse cookies-from-browser spec for yt-dlp.

    Accepts formats:
    - "chrome"
    - "chrome:Default"
    - "chrome:Profile 1"
    - "chrome,Default"

    Returns a string or tuple compatible with yt-dlp's cookiesfrombrowser option.
    """

    if not spec:
        return None

    cleaned = spec.strip()
    if not cleaned:
        return None

    if "," in cleaned:
        parts = tuple(item.strip() for item in cleaned.split(",") if item.strip())
        if len(parts) > 4:
            raise ValueError("cookies-from-browser supports at most 4 fields: browser, profile, keyring, container")
        return parts

    return cleaned


def download_video(url: str, output_dir: Path, cookies_from_browser: Optional[str] = None) -> Path:
    """Download a video using yt-dlp and return the file path.

    cookies_from_browser examples: "chrome", "chrome:Default", "firefox".
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(output_dir / "downloaded.%(ext)s")
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
    }
    cookies_spec = _parse_cookies_from_browser(cookies_from_browser)
    if cookies_spec:
        ydl_opts["cookiesfrombrowser"] = cookies_spec

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
