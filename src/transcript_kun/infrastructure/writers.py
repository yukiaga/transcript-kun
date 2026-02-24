"""Result writers for various output formats."""

from __future__ import annotations

import json
from pathlib import Path

from transcript_kun.domain.entities import TranscriptionResult
from transcript_kun.domain.ports import ResultWriter


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_simple(seconds: float) -> str:
    """Format seconds as HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class TxtWriter(ResultWriter):
    """Plain text output with speaker labels and timestamps."""

    def write(self, result: TranscriptionResult, output_path: str) -> None:
        lines: list[str] = []
        for seg in result.segments:
            ts = _format_timestamp_simple(seg.start)
            speaker = seg.speaker or "?"
            lines.append(f"[{ts}] [{speaker}] {seg.text}")
        Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class SrtWriter(ResultWriter):
    """SRT subtitle format output."""

    def write(self, result: TranscriptionResult, output_path: str) -> None:
        lines: list[str] = []
        for i, seg in enumerate(result.segments, 1):
            start = _format_timestamp_srt(seg.start)
            end = _format_timestamp_srt(seg.end)
            speaker = f"[{seg.speaker}] " if seg.speaker else ""
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(f"{speaker}{seg.text}")
            lines.append("")
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")


class JsonWriter(ResultWriter):
    """JSON output with full metadata."""

    def write(self, result: TranscriptionResult, output_path: str) -> None:
        data = {
            "audio_path": result.audio_path,
            "language": result.language,
            "duration_seconds": result.duration_seconds,
            "speakers": result.speakers,
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "speaker": seg.speaker,
                }
                for seg in result.segments
            ],
        }
        Path(output_path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class TsvWriter(ResultWriter):
    """TSV output for spreadsheet import."""

    def write(self, result: TranscriptionResult, output_path: str) -> None:
        lines = ["start\tend\tspeaker\ttext"]
        for seg in result.segments:
            speaker = seg.speaker or ""
            lines.append(f"{seg.start:.3f}\t{seg.end:.3f}\t{speaker}\t{seg.text}")
        Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


WRITERS: dict[str, type[ResultWriter]] = {
    "txt": TxtWriter,
    "srt": SrtWriter,
    "json": JsonWriter,
    "tsv": TsvWriter,
}


def get_writer(fmt: str) -> ResultWriter:
    """Get a writer instance for the given format."""
    cls = WRITERS.get(fmt)
    if cls is None:
        raise ValueError(f"Unknown format '{fmt}'. Available: {', '.join(sorted(WRITERS))}")
    return cls()
