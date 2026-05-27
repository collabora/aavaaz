"""Aavaaz — Modal GPU serverless LIVE transcription via WebSocket.

Extends the batch transcription deployment with real-time WebSocket streaming
using WhisperLive's faster_whisper backend.

Deploy with:
    modal deploy deploy/modal/app_live.py

Develop with live-reload:
    modal serve deploy/modal/app_live.py

Connect a WhisperLive client to wss://<modal-url>/ws

Environment variables (set via Modal Secrets):
    AAVAAZ_MODEL          Whisper model name (default: large-v3)
    AAVAAZ_LANGUAGE       Language code, or empty for auto-detect
    AAVAAZ_API_KEY        Optional API key for authentication
    AAVAAZ_MAX_CLIENTS    Max concurrent WebSocket clients (default: 4)
    AAVAAZ_MAX_TIME       Max connection time in seconds (default: 600)
"""

import asyncio
import logging

import fastapi
import modal

logger = logging.getLogger("aavaaz.modal.live")
logger.setLevel(logging.INFO)

WHISPER_MODEL = "large-v3"
WEB_DIR = "/web"

app = modal.App("aavaaz-live")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-runtime-ubuntu22.04", add_python="3.12"
    )
    .apt_install("ffmpeg")
    .pip_install(
        "faster-whisper>=1.0",
        "fastapi[standard]",
        "python-multipart",
        "tokenizers",
        "torch",
        "tqdm",
        "websockets",
    )
    .run_commands(
        f'python -c "from faster_whisper import WhisperModel; '
        f"WhisperModel('{WHISPER_MODEL}', device='cpu')\""
    )
    # Install local WhisperLive (with generator fix) BEFORE aavaaz so pip
    # doesn't pull the PyPI version when resolving aavaaz's dependencies.
    .add_local_dir(
        "/home/aaron/src/WhisperLive",
        remote_path="/root/WhisperLive",
        copy=True,
    )
    .run_commands("pip install --no-deps /root/WhisperLive")
    .add_local_dir("../../aavaaz", remote_path="/root/aavaaz_pkg/aavaaz", copy=True)
    .add_local_file(
        "../../pyproject.toml", remote_path="/root/aavaaz_pkg/pyproject.toml", copy=True
    )
    .run_commands("pip install --no-deps /root/aavaaz_pkg")
    .add_local_dir("../../aavaaz/web", remote_path=WEB_DIR)
)

_aavaaz_secret = modal.Secret.from_name("aavaaz-config")


class WebSocketAdapter:
    """Adapts FastAPI's async WebSocket to the sync send/recv interface
    that WhisperLive's TranscriptionServer expects."""

    def __init__(self, ws: fastapi.WebSocket, loop: asyncio.AbstractEventLoop):
        self._ws = ws
        self._loop = loop
        self._closed = False

    def send(self, data: str):
        """Send text data (JSON) to the client."""
        if self._closed:
            return
        logger.debug("Sending to client: %s", data[:200])
        future = asyncio.run_coroutine_threadsafe(self._ws.send_text(data), self._loop)
        future.result(timeout=10)

    def recv(self, timeout=None):
        """Receive data from the client (text or binary)."""
        future = asyncio.run_coroutine_threadsafe(self._recv_any(), self._loop)
        return future.result(timeout=timeout or 300)

    async def _recv_any(self):
        """Receive either text or binary message."""
        msg = await self._ws.receive()
        if msg["type"] == "websocket.receive":
            if "bytes" in msg and msg["bytes"]:
                return msg["bytes"]
            if "text" in msg and msg["text"]:
                return msg["text"]
        elif msg["type"] == "websocket.disconnect":
            self._closed = True
            raise ConnectionError("WebSocket disconnected")
        return b""

    def close(self):
        if not self._closed:
            self._closed = True
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)


@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
    secrets=[_aavaaz_secret],
    scaledown_window=120,
)
@modal.concurrent(max_inputs=4)
class LiveTranscriber:
    @modal.enter()
    def load_model(self):
        import os

        from whisper_live.server import TranscriptionServer

        model_name = os.environ.get("AAVAAZ_MODEL", WHISPER_MODEL)
        max_clients = int(os.environ.get("AAVAAZ_MAX_CLIENTS", "4"))
        max_time = int(os.environ.get("AAVAAZ_MAX_TIME", "600"))

        logger.info("Initializing TranscriptionServer (model=%s)", model_name)

        self.server = TranscriptionServer()
        self.server.client_manager = self._make_client_manager(max_clients, max_time)
        self.server.segment_post_processor = self._build_post_processor()
        self.model_name = model_name
        self.api_key = os.environ.get("AAVAAZ_API_KEY")
        self.server.use_vad = True
        self.server.single_model = True
        self.server.batch_config = None
        self.server.raw_pcm_input = False
        self.server.cache_path = "/tmp/whisper-cache"

        # Pre-load the model so first connection is fast
        from faster_whisper import WhisperModel
        from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper

        ServeClientFasterWhisper.SINGLE_MODEL = WhisperModel(
            model_name, device="cuda", compute_type="float16"
        )
        logger.info("Model loaded and ready for live transcription")

    def _make_client_manager(self, max_clients, max_time):
        from whisper_live.server import ClientManager

        return ClientManager(max_clients=max_clients, max_connection_time=max_time)

    def _build_post_processor(self, features=None):
        """Build a segment post-processing pipeline.

        If `features` dict is provided (from client options), build a
        per-connection pipeline based on the client's feature config.
        Otherwise, fall back to environment-variable-based defaults.
        """
        import os

        fns = []

        if features:
            # Per-client feature configuration from dashboard
            fmt = features.get("formatting", {})
            if fmt.get("enabled", False):
                from aavaaz.features.formatting import format_transcript

                capitalize = fmt.get("capitalize", True)
                numbers = fmt.get("numbers", True)
                smart = fmt.get("smart", False)
                fns.append(
                    lambda text: format_transcript(
                        text, capitalize=capitalize, numbers=numbers, smart=smart
                    )
                )

            pii = features.get("pii", {})
            if pii.get("enabled", False):
                from aavaaz.features.pii_redaction import redact_pii

                pii_types = set(
                    pii.get(
                        "types", ["ssn", "credit_card", "phone", "email", "ip_address"]
                    )
                )
                fns.append(lambda text, _t=pii_types: redact_pii(text, pii_types=_t))

            profanity = features.get("profanity", {})
            if profanity.get("enabled", False):
                from aavaaz.features.profanity_filter import filter_profanity

                mode = profanity.get("mode", "partial")
                extra = profanity.get("extraWords", [])
                fns.append(
                    lambda text, _m=mode, _e=extra: filter_profanity(
                        text, mode=_m, extra_words=_e or None
                    )
                )

            intel = features.get("intelligence", {})
            if intel.get("fillerRemoval", False):
                from aavaaz.features.audio_intelligence import remove_filler_words

                aggressive = intel.get("fillerAggressive", False)
                fns.append(
                    lambda text, _a=aggressive: remove_filler_words(text, aggressive=_a)
                )

            nr = features.get("noiseReduction", {})
            if nr.get("enabled", False):
                # Note: noise reduction operates on audio, not text segments.
                # It will be applied in the audio pre-processing step.
                pass

        else:
            # Legacy env-var-based config
            if os.environ.get("AAVAAZ_ENABLE_FORMAT", "1") == "1":
                from aavaaz.features.formatting import smart_format

                fns.append(smart_format)
            if os.environ.get("AAVAAZ_ENABLE_PII", "0") == "1":
                from aavaaz.features.pii_redaction import redact_pii

                fns.append(redact_pii)

        if not fns:
            return None

        def pipeline(segment):
            for fn in fns:
                if isinstance(segment, dict) and "text" in segment:
                    segment["text"] = fn(segment["text"])
                else:
                    segment = fn(segment)
            return segment

        return pipeline

    @modal.asgi_app()
    def web(self):
        import os

        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles

        web_app = fastapi.FastAPI(title="Aavaaz Live Transcription")

        web_app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @web_app.get("/", response_class=HTMLResponse)
        async def index():
            index_path = os.path.join(WEB_DIR, "live.html")
            if not os.path.exists(index_path):
                index_path = os.path.join(WEB_DIR, "index.html")
            with open(index_path) as f:
                return f.read()

        @web_app.get("/health")
        async def health():
            return {
                "status": "ok",
                "mode": "live",
                "model": self.model_name,
            }

        @web_app.websocket("/ws")
        async def websocket_endpoint(ws: fastapi.WebSocket):
            # Auth check
            if self.api_key:
                token = ws.query_params.get("token")
                if token != self.api_key:
                    await ws.close(code=4001, reason="Unauthorized")
                    return

            await ws.accept()
            logger.info("WebSocket client connected")

            loop = asyncio.get_event_loop()
            adapter = WebSocketAdapter(ws, loop)

            # Run the WhisperLive recv_audio loop in a thread
            # (it's synchronous and blocking)
            try:
                await asyncio.to_thread(self._handle_client, adapter)
            except Exception as e:
                logger.error("WebSocket error: %s", e)
            finally:
                logger.info("WebSocket client disconnected")

        return web_app

    def _handle_client(self, websocket: WebSocketAdapter):
        """Run WhisperLive's client handling in a sync context.

        Intercepts the initial options message to extract per-client feature
        configuration and sets up a per-connection post-processing pipeline.
        """
        import json

        from websockets.exceptions import ConnectionClosed
        from whisper_live.server import BackendType

        try:
            self.server.backend = BackendType.FASTER_WHISPER

            # Peek at the options to extract features config before WhisperLive processes it
            # We monkey-patch the recv to capture the first message
            original_recv = websocket.recv
            captured_options = {}

            def intercepting_recv(timeout=None):
                nonlocal captured_options
                data = original_recv(timeout=timeout)
                try:
                    captured_options = (
                        json.loads(data)
                        if isinstance(data, str)
                        else json.loads(data.decode())
                    )
                except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                    pass
                # Restore original recv after first message
                websocket.recv = original_recv
                return data

            websocket.recv = intercepting_recv

            if not self.server.handle_new_connection(
                websocket, self.model_name, None, False, False
            ):
                return

            # Apply per-client feature config if provided
            features = captured_options.get("features")
            if features:
                processor = self._build_post_processor(features=features)
                if processor:
                    self.server.segment_post_processor = processor
                    logger.info(
                        "Applied per-client feature config: %s",
                        [
                            k
                            for k, v in features.items()
                            if isinstance(v, dict) and v.get("enabled")
                        ],
                    )

            while not self.server.client_manager.is_client_timeout(websocket):
                if not self.server.process_audio_frames(websocket):
                    break
        except (ConnectionClosed, ConnectionError):
            logger.info("Client disconnected")
        except Exception as e:
            logger.error("Error in client handler: %s", e)
        finally:
            if self.server.client_manager.get_client(websocket):
                self.server.client_manager.remove_client(websocket)
            # Reset to default post-processor after client disconnects
            self.server.segment_post_processor = self._build_post_processor()
