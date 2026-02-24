"""Domain policies — guard rails for transcription operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = frozenset({".m4a", ".mp3", ".wav", ".flac", ".ogg", ".webm", ".mp4"})
MAX_FILE_SIZE_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB
VALID_MODELS = frozenset({"tiny", "base", "small", "medium", "large-v2", "large-v3"})
VALID_DEVICES = frozenset({"cpu", "cuda"})
VALID_COMPUTE_TYPES = frozenset({"float16", "float32", "int8"})
VALID_OUTPUT_FORMATS = frozenset({"txt", "srt", "json", "tsv"})


@dataclass(frozen=True)
class PolicyViolation:
    """A single policy violation."""

    code: str
    message: str


class PolicyError(Exception):
    """Raised when a policy is violated."""

    def __init__(self, violations: list[PolicyViolation]) -> None:
        self.violations = violations
        messages = "; ".join(v.message for v in violations)
        super().__init__(f"Policy violations: {messages}")


def validate_audio_file(path: str) -> list[PolicyViolation]:
    """Validate that a file is acceptable for transcription."""
    violations: list[PolicyViolation] = []
    p = Path(path)

    if not p.exists():
        violations.append(PolicyViolation("FILE_NOT_FOUND", f"File not found: {path}"))
        return violations

    if not p.is_file():
        violations.append(PolicyViolation("NOT_A_FILE", f"Not a regular file: {path}"))
        return violations

    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        violations.append(
            PolicyViolation(
                "UNSUPPORTED_FORMAT",
                f"Unsupported format '{p.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )
        )

    size = p.stat().st_size
    if size == 0:
        violations.append(PolicyViolation("EMPTY_FILE", "File is empty"))
    elif size > MAX_FILE_SIZE_BYTES:
        mb = size / (1024 * 1024)
        violations.append(PolicyViolation("FILE_TOO_LARGE", f"File size {mb:.0f}MB exceeds limit of 4096MB"))

    return violations


def validate_config(
    model_name: str,
    device: str,
    compute_type: str,
    output_format: str,
    enable_diarization: bool,
    hf_token: str | None,
) -> list[PolicyViolation]:
    """Validate transcription configuration."""
    violations: list[PolicyViolation] = []

    if model_name not in VALID_MODELS:
        violations.append(
            PolicyViolation("INVALID_MODEL", f"Invalid model '{model_name}'. Valid: {', '.join(sorted(VALID_MODELS))}")
        )

    if device not in VALID_DEVICES:
        violations.append(
            PolicyViolation("INVALID_DEVICE", f"Invalid device '{device}'. Valid: {', '.join(sorted(VALID_DEVICES))}")
        )

    if compute_type not in VALID_COMPUTE_TYPES:
        violations.append(
            PolicyViolation(
                "INVALID_COMPUTE_TYPE",
                f"Invalid compute_type '{compute_type}'. Valid: {', '.join(sorted(VALID_COMPUTE_TYPES))}",
            )
        )

    if output_format not in VALID_OUTPUT_FORMATS:
        violations.append(
            PolicyViolation(
                "INVALID_OUTPUT_FORMAT",
                f"Invalid output_format '{output_format}'. Valid: {', '.join(sorted(VALID_OUTPUT_FORMATS))}",
            )
        )

    if device == "cpu" and compute_type == "float16":
        violations.append(
            PolicyViolation("INCOMPATIBLE_COMPUTE", "float16 is not supported on CPU. Use int8 or float32.")
        )

    if enable_diarization and not hf_token:
        violations.append(
            PolicyViolation(
                "MISSING_HF_TOKEN",
                "Hugging Face token is required for speaker diarization. Set HF_TOKEN env var or pass --hf-token.",
            )
        )

    return violations
