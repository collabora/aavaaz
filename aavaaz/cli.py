"""Aavaaz CLI — entry point for the aavaaz command."""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="aavaaz",
        description="Aavaaz — production-grade speech-to-text platform",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- serve ---
    serve_parser = subparsers.add_parser("serve", help="Start the Aavaaz server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    serve_parser.add_argument(
        "--port", type=int, default=9090, help="WebSocket port (default: 9090)"
    )
    serve_parser.add_argument(
        "--rest-port", type=int, default=8000, help="REST API port (default: 8000)"
    )
    serve_parser.add_argument("--model", default="large-v3", help="Whisper model name or path")
    serve_parser.add_argument(
        "--backend",
        default="faster_whisper",
        choices=["faster_whisper", "tensorrt", "openvino"],
        help="Transcription backend",
    )
    serve_parser.add_argument("--no-rest", action="store_true", help="Disable REST API")
    serve_parser.add_argument("--api-key", default=None, help="API key for auth (REST + WebSocket)")
    serve_parser.add_argument(
        "--rate-limit-rpm",
        type=int,
        default=0,
        help="Max REST requests per minute per IP (0=unlimited)",
    )
    serve_parser.add_argument(
        "--metrics-port", type=int, default=0, help="Prometheus metrics port (0=disabled)"
    )
    serve_parser.add_argument(
        "--batch-inference", action="store_true", help="Enable cross-client GPU batching"
    )
    serve_parser.add_argument(
        "--batch-max-size", type=int, default=8, help="Max requests per GPU batch (default: 8)"
    )
    serve_parser.add_argument(
        "--batch-window-ms",
        type=int,
        default=50,
        help="Max ms to wait for batch to fill (default: 50)",
    )
    serve_parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Enable word-level timestamps and confidence scores",
    )
    serve_parser.add_argument(
        "--hotwords", default=None, help="Comma-separated list of terms to boost recognition"
    )
    serve_parser.add_argument(
        "--enable-diarization", action="store_true", help="Enable speaker diarization"
    )
    serve_parser.add_argument(
        "--max-speakers", type=int, default=10, help="Max speakers for diarization (default: 10)"
    )
    serve_parser.add_argument("--log-json", action="store_true", help="Use JSON structured logging")
    serve_parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    # --- transcribe ---
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe an audio file")
    transcribe_parser.add_argument("file", help="Path to audio file")
    transcribe_parser.add_argument("--model", default="large-v3", help="Whisper model name or path")
    transcribe_parser.add_argument(
        "--format",
        default="text",
        choices=["text", "json", "srt", "vtt"],
        help="Output format",
    )
    transcribe_parser.add_argument(
        "--language", default=None, help="Language code (auto-detect if omitted)"
    )

    # --- version ---
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(level=level)

    if args.command == "version":
        from aavaaz import __version__

        print(f"aavaaz {__version__}")

    elif args.command == "serve":
        try:
            from aavaaz.server import AavaazServer
        except ImportError as e:
            if "whisper_live" in str(e):
                print(
                    "Error: whisper-live is required for 'aavaaz serve'.\n\n"
                    "Install from PyPI (Python 3.12/3.13 recommended):\n"
                    "  pip install whisper-live\n\n"
                    "Or install from source (for development / Python 3.14+):\n"
                    "  pip install --no-deps -e /path/to/WhisperLive\n"
                    "  pip install faster-whisper websockets scipy",
                    file=sys.stderr,
                )
                sys.exit(1)
            raise

        server = AavaazServer(
            host=args.host,
            port=args.port,
            rest_port=args.rest_port,
            backend=args.backend,
            model=args.model,
            enable_rest_api=not args.no_rest,
            api_key=args.api_key,
            rate_limit_rpm=args.rate_limit_rpm,
            metrics_port=args.metrics_port,
            batch_inference=args.batch_inference,
            batch_max_size=args.batch_max_size,
            batch_window_ms=args.batch_window_ms,
            word_timestamps=args.word_timestamps,
            hotwords=args.hotwords,
            enable_diarization=args.enable_diarization,
            max_speakers=args.max_speakers,
        )
        server.run()

    elif args.command == "transcribe":
        from aavaaz.transcribe import transcribe_file

        transcribe_file(
            path=args.file,
            model=args.model,
            output_format=args.format,
            language=args.language,
        )


if __name__ == "__main__":
    main()
