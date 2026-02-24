"""Tests for domain entities."""

from transcript_kun.domain.entities import Segment, TranscriptionConfig, TranscriptionResult


class TestSegment:
    def test_frozen(self):
        seg = Segment(start=0.0, end=1.0, text="hello")
        assert seg.speaker is None

    def test_with_speaker(self):
        seg = Segment(start=0.0, end=1.0, text="hello", speaker="SPEAKER_00")
        assert seg.speaker == "SPEAKER_00"


class TestTranscriptionResult:
    def _make_result(self) -> TranscriptionResult:
        return TranscriptionResult(
            segments=(
                Segment(start=0.0, end=1.5, text="Hello", speaker="SPEAKER_00"),
                Segment(start=1.5, end=3.0, text="World", speaker="SPEAKER_01"),
                Segment(start=3.0, end=5.0, text="Bye", speaker="SPEAKER_00"),
            ),
            language="en",
            audio_path="test.m4a",
        )

    def test_speakers(self):
        result = self._make_result()
        assert result.speakers == ["SPEAKER_00", "SPEAKER_01"]

    def test_full_text(self):
        result = self._make_result()
        assert "[SPEAKER_00] Hello" in result.full_text
        assert "[SPEAKER_01] World" in result.full_text

    def test_duration(self):
        result = self._make_result()
        assert result.duration_seconds == 5.0

    def test_empty_result(self):
        result = TranscriptionResult(segments=(), language="ja", audio_path="x.m4a")
        assert result.speakers == []
        assert result.duration_seconds == 0.0
        assert result.full_text == ""


class TestTranscriptionConfig:
    def test_defaults(self):
        config = TranscriptionConfig()
        assert config.model_name == "large-v3"
        assert config.language == "ja"
        assert config.device == "cpu"
        assert config.compute_type == "int8"
        assert config.enable_diarization is True
        assert config.hf_token is None
