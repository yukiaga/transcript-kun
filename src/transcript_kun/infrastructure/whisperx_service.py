"""WhisperX-based transcription service."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from transcript_kun.domain.entities import Segment, TranscriptionConfig, TranscriptionResult
from transcript_kun.domain.ports import ProgressCallback, TranscriptionService

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class WhisperXTranscriptionService(TranscriptionService):
    """Transcription service backed by WhisperX."""

    def is_available(self) -> bool:
        try:
            import whisperx  # noqa: F401

            return True
        except ImportError:
            return False

    def transcribe(
        self,
        audio_path: str,
        config: TranscriptionConfig,
        on_progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        import whisperx

        def _progress(stage: str, value: float) -> None:
            if on_progress:
                on_progress(stage, value)

        # --- Step 1: Load model and transcribe ---
        _progress("loading_model", 0.0)
        logger.info("Loading model '%s' on %s (%s)", config.model_name, config.device, config.compute_type)
        model = whisperx.load_model(
            config.model_name,
            config.device,
            compute_type=config.compute_type,
        )

        _progress("transcribing", 0.1)
        logger.info("Transcribing %s (language=%s, batch_size=%d)", audio_path, config.language, config.batch_size)
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=config.batch_size, language=config.language)
        detected_language = result.get("language", config.language)
        _progress("transcribing", 0.5)

        # --- Step 2: Alignment ---
        _progress("aligning", 0.5)
        logger.info("Aligning timestamps (language=%s)", detected_language)
        try:
            model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=config.device)
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                config.device,
                return_char_alignments=False,
            )
        except Exception:
            logger.warning("Alignment failed; continuing without precise timestamps", exc_info=True)
        _progress("aligning", 0.7)

        # --- Step 3: Speaker diarization (optional) ---
        if config.enable_diarization and config.hf_token:
            _progress("diarizing", 0.7)
            logger.info("Running speaker diarization")
            try:
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=config.hf_token,
                    device=config.device,
                )
                diarize_kwargs = {}
                if config.min_speakers is not None:
                    diarize_kwargs["min_speakers"] = config.min_speakers
                if config.max_speakers is not None:
                    diarize_kwargs["max_speakers"] = config.max_speakers

                diarize_segments = diarize_model(audio, **diarize_kwargs)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception:
                logger.warning("Diarization failed; continuing without speaker labels", exc_info=True)
        _progress("diarizing", 0.9)

        # --- Build result ---
        segments = tuple(
            Segment(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", "").strip(),
                speaker=seg.get("speaker"),
            )
            for seg in result.get("segments", [])
            if seg.get("text", "").strip()
        )
        _progress("done", 1.0)
        logger.info("Transcription complete: %d segments", len(segments))

        return TranscriptionResult(
            segments=segments,
            language=detected_language,
            audio_path=audio_path,
        )
