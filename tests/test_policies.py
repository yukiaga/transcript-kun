"""Tests for domain policies."""

from transcript_kun.domain.policies import (
    PolicyError,
    validate_audio_file,
    validate_config,
)


class TestValidateAudioFile:
    def test_file_not_found(self):
        violations = validate_audio_file("/nonexistent/file.m4a")
        assert len(violations) == 1
        assert violations[0].code == "FILE_NOT_FOUND"

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "file.xyz"
        p.write_text("data")
        violations = validate_audio_file(str(p))
        assert any(v.code == "UNSUPPORTED_FORMAT" for v in violations)

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.m4a"
        p.write_bytes(b"")
        violations = validate_audio_file(str(p))
        assert any(v.code == "EMPTY_FILE" for v in violations)

    def test_valid_file(self, tmp_path):
        p = tmp_path / "audio.m4a"
        p.write_bytes(b"\x00" * 1024)
        violations = validate_audio_file(str(p))
        assert violations == []

    def test_directory_not_file(self, tmp_path):
        violations = validate_audio_file(str(tmp_path))
        assert any(v.code == "NOT_A_FILE" for v in violations)

    def test_supported_extensions(self, tmp_path):
        for ext in [".m4a", ".mp3", ".wav", ".flac", ".ogg", ".webm", ".mp4"]:
            p = tmp_path / f"audio{ext}"
            p.write_bytes(b"\x00" * 100)
            violations = validate_audio_file(str(p))
            assert violations == [], f"Extension {ext} should be valid"


class TestValidateConfig:
    def test_valid_config(self):
        violations = validate_config(
            model_name="large-v3",
            device="cpu",
            compute_type="int8",
            output_format="txt",
            enable_diarization=True,
            hf_token="test_token",
        )
        assert violations == []

    def test_invalid_model(self):
        violations = validate_config(
            model_name="invalid",
            device="cpu",
            compute_type="int8",
            output_format="txt",
            enable_diarization=False,
            hf_token=None,
        )
        assert any(v.code == "INVALID_MODEL" for v in violations)

    def test_invalid_device(self):
        violations = validate_config(
            model_name="large-v3",
            device="tpu",
            compute_type="int8",
            output_format="txt",
            enable_diarization=False,
            hf_token=None,
        )
        assert any(v.code == "INVALID_DEVICE" for v in violations)

    def test_cpu_float16_incompatible(self):
        violations = validate_config(
            model_name="large-v3",
            device="cpu",
            compute_type="float16",
            output_format="txt",
            enable_diarization=False,
            hf_token=None,
        )
        assert any(v.code == "INCOMPATIBLE_COMPUTE" for v in violations)

    def test_diarization_requires_hf_token(self):
        violations = validate_config(
            model_name="large-v3",
            device="cpu",
            compute_type="int8",
            output_format="txt",
            enable_diarization=True,
            hf_token=None,
        )
        assert any(v.code == "MISSING_HF_TOKEN" for v in violations)

    def test_diarization_with_token_ok(self):
        violations = validate_config(
            model_name="large-v3",
            device="cpu",
            compute_type="int8",
            output_format="txt",
            enable_diarization=True,
            hf_token="hf_abc123",
        )
        assert violations == []

    def test_invalid_output_format(self):
        violations = validate_config(
            model_name="large-v3",
            device="cpu",
            compute_type="int8",
            output_format="pdf",
            enable_diarization=False,
            hf_token=None,
        )
        assert any(v.code == "INVALID_OUTPUT_FORMAT" for v in violations)

    def test_multiple_violations(self):
        violations = validate_config(
            model_name="invalid",
            device="tpu",
            compute_type="bfloat16",
            output_format="pdf",
            enable_diarization=True,
            hf_token=None,
        )
        codes = {v.code for v in violations}
        assert "INVALID_MODEL" in codes
        assert "INVALID_DEVICE" in codes
        assert "INVALID_COMPUTE_TYPE" in codes
        assert "INVALID_OUTPUT_FORMAT" in codes
        assert "MISSING_HF_TOKEN" in codes


class TestPolicyError:
    def test_error_message(self):
        from transcript_kun.domain.policies import PolicyViolation

        err = PolicyError([PolicyViolation("A", "msg_a"), PolicyViolation("B", "msg_b")])
        assert "msg_a" in str(err)
        assert "msg_b" in str(err)
        assert len(err.violations) == 2
