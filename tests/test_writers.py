"""Tests for output writers."""

import json

from transcript_kun.domain.entities import Segment, TranscriptionResult
from transcript_kun.infrastructure.writers import (
    JsonWriter,
    SrtWriter,
    TsvWriter,
    TxtWriter,
    get_writer,
)


def _sample_result() -> TranscriptionResult:
    return TranscriptionResult(
        segments=(
            Segment(start=0.0, end=2.5, text="Hello world", speaker="SPEAKER_00"),
            Segment(start=3.0, end=5.0, text="How are you?", speaker="SPEAKER_01"),
            Segment(start=5.5, end=8.0, text="I'm fine.", speaker="SPEAKER_00"),
        ),
        language="en",
        audio_path="test.m4a",
    )


class TestTxtWriter:
    def test_write(self, tmp_path):
        out = tmp_path / "out.txt"
        TxtWriter().write(_sample_result(), str(out))
        content = out.read_text()
        assert "[00:00:00] [SPEAKER_00] Hello world" in content
        assert "[00:00:03] [SPEAKER_01] How are you?" in content
        assert "[00:00:05] [SPEAKER_00] I'm fine." in content


class TestSrtWriter:
    def test_write(self, tmp_path):
        out = tmp_path / "out.srt"
        SrtWriter().write(_sample_result(), str(out))
        content = out.read_text()
        assert "00:00:00,000 --> 00:00:02,500" in content
        assert "[SPEAKER_00] Hello world" in content
        assert "1\n" in content
        assert "2\n" in content


class TestJsonWriter:
    def test_write(self, tmp_path):
        out = tmp_path / "out.json"
        JsonWriter().write(_sample_result(), str(out))
        data = json.loads(out.read_text())
        assert data["language"] == "en"
        assert data["audio_path"] == "test.m4a"
        assert len(data["segments"]) == 3
        assert data["segments"][0]["speaker"] == "SPEAKER_00"
        assert data["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
        assert data["duration_seconds"] == 8.0


class TestTsvWriter:
    def test_write(self, tmp_path):
        out = tmp_path / "out.tsv"
        TsvWriter().write(_sample_result(), str(out))
        content = out.read_text()
        lines = content.strip().split("\n")
        assert lines[0] == "start\tend\tspeaker\ttext"
        assert len(lines) == 4  # header + 3 segments
        parts = lines[1].split("\t")
        assert parts[2] == "SPEAKER_00"
        assert parts[3] == "Hello world"


class TestGetWriter:
    def test_known_formats(self):
        for fmt in ("txt", "srt", "json", "tsv"):
            writer = get_writer(fmt)
            assert writer is not None

    def test_unknown_format(self):
        try:
            get_writer("pdf")
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "pdf" in str(e)


class TestWriterNoSpeaker:
    """Test output when speaker is None."""

    def test_txt_no_speaker(self, tmp_path):
        result = TranscriptionResult(
            segments=(Segment(start=0.0, end=1.0, text="Hello", speaker=None),),
            language="en",
            audio_path="test.m4a",
        )
        out = tmp_path / "out.txt"
        TxtWriter().write(result, str(out))
        content = out.read_text()
        assert "[?]" in content

    def test_srt_no_speaker(self, tmp_path):
        result = TranscriptionResult(
            segments=(Segment(start=0.0, end=1.0, text="Hello", speaker=None),),
            language="en",
            audio_path="test.m4a",
        )
        out = tmp_path / "out.srt"
        SrtWriter().write(result, str(out))
        content = out.read_text()
        assert "[" not in content or "Hello" in content
