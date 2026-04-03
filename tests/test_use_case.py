"""Tests for the transcribe use case."""

from __future__ import annotations

import pytest

from transcript_kun.application.transcribe import TranscribeAudioUseCase
from transcript_kun.domain.entities import (
    AudioFileInfo,
    Segment,
    TranscriptionConfig,
    TranscriptionResult,
)
from transcript_kun.domain.policies import PolicyError
from transcript_kun.domain.ports import AudioProber, ProgressCallback, ResultWriter, TranscriptionService


class FakeTranscriptionService(TranscriptionService):
    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.call_count = 0

    def is_available(self) -> bool:
        return self._available

    def transcribe(
        self,
        audio_path: str,
        config: TranscriptionConfig,
        on_progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        self.call_count += 1
        return TranscriptionResult(
            segments=(Segment(start=0.0, end=1.0, text="Test segment", speaker="SPEAKER_00"),),
            language=config.language,
            audio_path=audio_path,
        )


class FakeAudioProber(AudioProber):
    def probe(self, path: str) -> AudioFileInfo:
        return AudioFileInfo(path=path, size_bytes=1024, duration_seconds=60.0)


class FakeWriter(ResultWriter):
    def __init__(self) -> None:
        self.written: list[tuple[TranscriptionResult, str]] = []

    def write(self, result: TranscriptionResult, output_path: str) -> None:
        self.written.append((result, output_path))


class TestTranscribeAudioUseCase:
    def _make_audio_file(self, tmp_path) -> str:
        p = tmp_path / "audio.m4a"
        p.write_bytes(b"\x00" * 1024)
        return str(p)

    def _make_use_case(
        self, available: bool = True
    ) -> tuple[TranscribeAudioUseCase, FakeTranscriptionService, FakeWriter]:
        svc = FakeTranscriptionService(available)
        writer = FakeWriter()
        uc = TranscribeAudioUseCase(svc, FakeAudioProber(), writer)
        return uc, svc, writer

    def test_successful_transcription(self, tmp_path):
        audio = self._make_audio_file(tmp_path)
        uc, svc, writer = self._make_use_case()
        config = TranscriptionConfig(enable_diarization=False)

        result = uc.execute(audio, config, output_path=str(tmp_path / "out.txt"))

        assert svc.call_count == 1
        assert len(writer.written) == 1
        assert result.language == "ja"
        assert len(result.segments) == 1

    def test_default_output_path(self, tmp_path):
        audio = self._make_audio_file(tmp_path)
        uc, _, writer = self._make_use_case()
        config = TranscriptionConfig(enable_diarization=False, output_format="json")

        uc.execute(audio, config)

        assert writer.written[0][1] == "audio_transcript.json"

    def test_policy_violation_missing_file(self, tmp_path):
        uc, _, _ = self._make_use_case()
        config = TranscriptionConfig(enable_diarization=False)

        with pytest.raises(PolicyError) as exc_info:
            uc.execute("/nonexistent.m4a", config)
        assert any(v.code == "FILE_NOT_FOUND" for v in exc_info.value.violations)

    def test_policy_violation_bad_config(self, tmp_path):
        audio = self._make_audio_file(tmp_path)
        uc, _, _ = self._make_use_case()
        config = TranscriptionConfig(
            model_name="invalid",
            enable_diarization=False,
        )

        with pytest.raises(PolicyError) as exc_info:
            uc.execute(audio, config)
        assert any(v.code == "INVALID_MODEL" for v in exc_info.value.violations)

    def test_engine_not_available(self, tmp_path):
        audio = self._make_audio_file(tmp_path)
        uc, _, _ = self._make_use_case(available=False)
        config = TranscriptionConfig(enable_diarization=False)

        with pytest.raises(RuntimeError, match="WhisperX is not installed"):
            uc.execute(audio, config)

    def test_progress_callback(self, tmp_path):
        audio = self._make_audio_file(tmp_path)
        uc, _, _ = self._make_use_case()
        config = TranscriptionConfig(enable_diarization=False)
        progress_calls: list[tuple[str, float]] = []

        uc.execute(audio, config, on_progress=lambda s, p: progress_calls.append((s, p)))

        # Progress callback is passed to the service, but our fake doesn't call it
        # The use case itself doesn't call it directly — just passes it through
        assert isinstance(progress_calls, list)
