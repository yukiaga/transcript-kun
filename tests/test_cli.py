"""Tests for CLI argument parsing and error handling."""

from __future__ import annotations

from unittest.mock import patch

from transcript_kun.interface.cli import main


class TestCLIErrors:
    def test_missing_file(self):
        """Non-existent file should exit with code 1."""
        code = main(["/nonexistent/audio.m4a", "--no-diarize"])
        assert code == 1

    def test_unsupported_format_file(self, tmp_path):
        """Unsupported file extension should exit with code 1."""
        p = tmp_path / "file.xyz"
        p.write_text("data")
        code = main([str(p), "--no-diarize"])
        assert code == 1

    def test_empty_file(self, tmp_path):
        """Empty file should exit with code 1."""
        p = tmp_path / "empty.m4a"
        p.write_bytes(b"")
        code = main([str(p), "--no-diarize"])
        assert code == 1

    def test_diarize_without_token(self, tmp_path):
        """Diarization without HF token should exit with code 1."""
        p = tmp_path / "audio.m4a"
        p.write_bytes(b"\x00" * 100)
        with patch.dict("os.environ", {}, clear=True):
            # Ensure HF_TOKEN is not set
            import os

            old = os.environ.pop("HF_TOKEN", None)
            try:
                code = main([str(p)])
                assert code == 1
            finally:
                if old is not None:
                    os.environ["HF_TOKEN"] = old
