"""Domain ports — interfaces that infrastructure must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from transcript_kun.domain.entities import (
    AudioFileInfo,
    TranscriptionConfig,
    TranscriptionResult,
)

ProgressCallback = Callable[[str, float], None]  # (stage, progress 0.0-1.0)


class TranscriptionService(ABC):
    """Port for transcription engines."""

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        config: TranscriptionConfig,
        on_progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        """Transcribe an audio file and return structured results."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the transcription engine is installed and usable."""


class AudioProber(ABC):
    """Port for reading audio file metadata."""

    @abstractmethod
    def probe(self, path: str) -> AudioFileInfo:
        """Return metadata about an audio file."""


class ResultWriter(ABC):
    """Port for writing transcription results to files."""

    @abstractmethod
    def write(self, result: TranscriptionResult, output_path: str) -> None:
        """Write transcription result to the given path."""
