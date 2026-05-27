"""
Aavaaz server — extends WhisperLive's TranscriptionServer with the full
plugin pipeline and extended REST API.

Uses WhisperLive's ``segment_post_processor`` hook to inject the Aavaaz
plugin pipeline without modifying WhisperLive core code.
"""

import logging

from whisper_live.server import TranscriptionServer

from aavaaz.features.plugins import PluginRegistry
from aavaaz.plugins import registry as default_registry

logger = logging.getLogger(__name__)


class AavaazServer:
    """High-level server that wires WhisperLive with Aavaaz plugins and API."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9090,
        backend: str = "faster_whisper",
        model: str = "large-v3",
        *,
        enable_rest_api: bool = True,
        rest_port: int = 8000,
        plugin_registry: PluginRegistry | None = None,
        api_key: str | None = None,
        rate_limit_rpm: int = 0,
        metrics_port: int = 0,
        batch_inference: bool = False,
        batch_max_size: int = 8,
        batch_window_ms: int = 50,
        word_timestamps: bool = False,
        hotwords: str | None = None,
        enable_diarization: bool = False,
        max_speakers: int = 10,
    ):
        self.host = host
        self.port = port
        self.backend = backend
        self.model = model
        self.enable_rest_api = enable_rest_api
        self.rest_port = rest_port
        self.plugin_registry = plugin_registry or default_registry
        self.api_key = api_key
        self.rate_limit_rpm = rate_limit_rpm
        self.metrics_port = metrics_port
        self.batch_inference = batch_inference
        self.batch_max_size = batch_max_size
        self.batch_window_ms = batch_window_ms
        self.word_timestamps = word_timestamps
        self.hotwords = hotwords
        self.enable_diarization = enable_diarization
        self.max_speakers = max_speakers

    def run(self):
        """Start the Aavaaz server (WhisperLive + plugins + REST API)."""
        server = TranscriptionServer()

        logger.info(
            "Starting Aavaaz server on %s:%d (backend=%s, model=%s)",
            self.host,
            self.port,
            self.backend,
            self.model,
        )

        plugins = self.plugin_registry.list_plugins()
        logger.info("Loaded %d plugins: %s", len(plugins), [p["name"] for p in plugins])

        # Use the registry's apply() as the WhisperLive segment_post_processor
        post_processor = self.plugin_registry.apply if len(self.plugin_registry) > 0 else None

        server.run(
            host=self.host,
            port=self.port,
            backend=self.backend,
            faster_whisper_custom_model_path=self.model,
            enable_rest=self.enable_rest_api,
            rest_port=self.rest_port,
            segment_post_processor=post_processor,
            batch_enabled=self.batch_inference,
            batch_max_size=self.batch_max_size,
            batch_window_ms=self.batch_window_ms,
            metrics_port=self.metrics_port,
            api_key=self.api_key,
            rate_limit_rpm=self.rate_limit_rpm,
            word_timestamps=self.word_timestamps,
            hotwords=self.hotwords,
            enable_diarization=self.enable_diarization,
            max_speakers=self.max_speakers,
        )
