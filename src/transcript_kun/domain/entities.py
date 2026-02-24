"""Domain entities for transcription."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Segment:
    """A single transcribed segment with optional speaker label."""

    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    """Complete transcription output."""

    segments: tuple[Segment, ...]
    language: str
    audio_path: str

    @property
    def speakers(self) -> list[str]:
        return sorted({s.speaker for s in self.segments if s.speaker})

    @property
    def full_text(self) -> str:
        return "\n".join(
            f"[{s.speaker or '?'}] {s.text}" for s in self.segments
        )

    @property
    def duration_seconds(self) -> float:
        if not self.segments:
            return 0.0
        return self.segments[-1].end


@dataclass(frozen=True)
class AudioFileInfo:
    """Validated audio file metadata."""

    path: str
    size_bytes: int
    duration_seconds: float | None = None


@dataclass
class TranscriptionConfig:
    """Configuration for a transcription job."""

    model_name: str = "large-v3"
    language: str = "ja"
    device: str = "cpu"
    compute_type: str = "int8"
    batch_size: int = 16
    enable_diarization: bool = True
    hf_token: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    output_format: str = "txt"
