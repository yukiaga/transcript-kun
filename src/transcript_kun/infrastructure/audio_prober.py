"""Audio file probing via ffprobe."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from transcript_kun.domain.entities import AudioFileInfo
from transcript_kun.domain.ports import AudioProber

logger = logging.getLogger(__name__)


class FfprobeAudioProber(AudioProber):
    """Probe audio files using ffprobe (from ffmpeg)."""

    def probe(self, path: str) -> AudioFileInfo:
        p = Path(path)
        size = p.stat().st_size
        duration = self._get_duration(path)
        return AudioFileInfo(path=path, size_bytes=size, duration_seconds=duration)

    def _get_duration(self, path: str) -> float | None:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            logger.debug("ffprobe not found; skipping duration detection")
            return None

        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data["format"]["duration"])
        except (subprocess.TimeoutExpired, KeyError, json.JSONDecodeError, ValueError):
            logger.debug("Failed to parse ffprobe output", exc_info=True)
        return None
