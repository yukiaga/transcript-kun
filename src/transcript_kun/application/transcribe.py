"""Use case: transcribe an audio file."""

from __future__ import annotations

import logging
from pathlib import Path

from transcript_kun.domain.entities import TranscriptionConfig, TranscriptionResult
from transcript_kun.domain.policies import PolicyError, validate_audio_file, validate_config
from transcript_kun.domain.ports import AudioProber, ProgressCallback, ResultWriter, TranscriptionService

logger = logging.getLogger(__name__)


class TranscribeAudioUseCase:
    """Orchestrates audio transcription with policy enforcement."""

    def __init__(
        self,
        transcription_service: TranscriptionService,
        audio_prober: AudioProber,
        result_writer: ResultWriter,
    ) -> None:
        self._transcription_service = transcription_service
        self._audio_prober = audio_prober
        self._result_writer = result_writer

    def execute(
        self,
        audio_path: str,
        config: TranscriptionConfig,
        output_path: str | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        """Run the full transcription pipeline.

        1. Validate policies (file, config)
        2. Probe audio metadata
        3. Transcribe
        4. Write output

        Raises PolicyError if any guard rail is violated.
        """
        # --- Policy checks ---
        file_violations = validate_audio_file(audio_path)
        if file_violations:
            raise PolicyError(file_violations)

        config_violations = validate_config(
            model_name=config.model_name,
            device=config.device,
            compute_type=config.compute_type,
            output_format=config.output_format,
            enable_diarization=config.enable_diarization,
            hf_token=config.hf_token,
        )
        if config_violations:
            raise PolicyError(config_violations)

        # --- Check engine availability ---
        if not self._transcription_service.is_available():
            raise RuntimeError(
                "WhisperX is not installed. Install it with: pip install whisperx"
            )

        # --- Probe audio ---
        info = self._audio_prober.probe(audio_path)
        logger.info(
            "Audio: %s (%.1f MB, duration=%s)",
            info.path,
            info.size_bytes / (1024 * 1024),
            f"{info.duration_seconds:.0f}s" if info.duration_seconds else "unknown",
        )

        # --- Transcribe ---
        result = self._transcription_service.transcribe(audio_path, config, on_progress)

        # --- Write output ---
        if output_path is None:
            stem = Path(audio_path).stem
            output_path = f"{stem}_transcript.{config.output_format}"

        self._result_writer.write(result, output_path)
        logger.info("Output written to %s", output_path)

        return result
