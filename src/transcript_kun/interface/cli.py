"""CLI interface for transcript-kun."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from transcript_kun.application.transcribe import TranscribeAudioUseCase
from transcript_kun.domain.entities import TranscriptionConfig
from transcript_kun.domain.policies import (
    PolicyError,
    VALID_COMPUTE_TYPES,
    VALID_DEVICES,
    VALID_MODELS,
    VALID_OUTPUT_FORMATS,
)
from transcript_kun.infrastructure.audio_prober import FfprobeAudioProber
from transcript_kun.infrastructure.whisperx_service import WhisperXTranscriptionService
from transcript_kun.infrastructure.writers import get_writer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcript-kun",
        description="M4A audio transcription with speaker diarization.",
    )
    parser.add_argument("audio_file", help="Path to the audio file (m4a, mp3, wav, etc.)")
    parser.add_argument("-o", "--output", help="Output file path (default: <stem>_transcript.<fmt>)")
    parser.add_argument(
        "-f", "--format",
        choices=sorted(VALID_OUTPUT_FORMATS),
        default="txt",
        help="Output format (default: txt)",
    )
    parser.add_argument(
        "-m", "--model",
        choices=sorted(VALID_MODELS),
        default="large-v3",
        help="Whisper model (default: large-v3)",
    )
    parser.add_argument(
        "-l", "--language",
        default="ja",
        help="Language code (default: ja)",
    )
    parser.add_argument(
        "-d", "--device",
        choices=sorted(VALID_DEVICES),
        default="cpu",
        help="Compute device (default: cpu)",
    )
    parser.add_argument(
        "--compute-type",
        choices=sorted(VALID_COMPUTE_TYPES),
        default="int8",
        help="Compute type (default: int8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for transcription (default: 16)",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Disable speaker diarization",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers for diarization",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers for diarization",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def _make_progress_handler() -> callable:
    """Create a progress handler that prints status to stderr."""
    stages_seen: set[str] = set()

    def handler(stage: str, progress: float) -> None:
        if stage not in stages_seen:
            stages_seen.add(stage)
            labels = {
                "loading_model": "Loading model...",
                "transcribing": "Transcribing audio...",
                "aligning": "Aligning timestamps...",
                "diarizing": "Running speaker diarization...",
                "done": "Done!",
            }
            label = labels.get(stage, stage)
            print(f"  {label}", file=sys.stderr)

    return handler


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Resolve HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    enable_diarization = not args.no_diarize

    config = TranscriptionConfig(
        model_name=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        enable_diarization=enable_diarization,
        hf_token=hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        output_format=args.format,
    )

    # Wire up dependencies
    transcription_service = WhisperXTranscriptionService()
    audio_prober = FfprobeAudioProber()
    writer = get_writer(config.output_format)
    use_case = TranscribeAudioUseCase(transcription_service, audio_prober, writer)

    # Execute
    try:
        start_time = time.monotonic()
        print(f"transcript-kun: Processing {args.audio_file}", file=sys.stderr)

        result = use_case.execute(
            audio_path=args.audio_file,
            config=config,
            output_path=args.output,
            on_progress=_make_progress_handler(),
        )

        elapsed = time.monotonic() - start_time
        print(
            f"transcript-kun: Completed — {len(result.segments)} segments, "
            f"{len(result.speakers)} speakers, "
            f"{elapsed:.1f}s elapsed",
            file=sys.stderr,
        )
        return 0

    except PolicyError as e:
        for v in e.violations:
            print(f"Error [{v.code}]: {v.message}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
