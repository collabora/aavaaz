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
import json
import logging
import threading
from typing import Optional

import fastapi
import modal
import numpy as np

logger = logging.getLogger("aavaaz.modal.live")
logger.setLevel(logging.INFO)

WHISPER_MODEL = "large-v3"
WHISPERLIVE_DIR = "/opt/whisper_live"
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
        f"python -c \"from faster_whisper import WhisperModel; "
        f"WhisperModel('{WHISPER_MODEL}', device='cpu')\""
    )
    .add_local_dir("../../aavaaz", remote_path="/root/aavaaz_pkg/aavaaz", copy=True)
    .add_local_file(
        "../../pyproject.toml", remote_path="/root/aavaaz_pkg/pyproject.toml", copy=True
    )
    .run_commands("pip install /root/aavaaz_pkg")
    .add_local_dir(
        "/home/aaron/src/WhisperLive/whisper_live",
        remote_path=f"{WHISPERLIVE_DIR}/whisper_live",
    )
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
        future = asyncio.run_coroutine_threadsafe(
            self._ws.send_text(data), self._loop
        )
        future.result(timeout=10)

    def recv(self, timeout=None):
        """Receive data from the client (text or binary)."""
        future = asyncio.run_coroutine_threadsafe(
            self._recv_any(), self._loop
        )
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
            asyncio.run_coroutine_threadsafe(
                self._ws.close(), self._loop
            )


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
        import sys

        sys.path.insert(0, WHISPERLIVE_DIR)

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
        from whisper_live.backend.faster_whisper_backend import ServeClientFasterWhisper
        from faster_whisper import WhisperModel

        ServeClientFasterWhisper.SINGLE_MODEL = WhisperModel(
            model_name, device="cuda", compute_type="float16"
        )
        logger.info("Model loaded and ready for live transcription")

    def _make_client_manager(self, max_clients, max_time):
        import sys
        sys.path.insert(0, WHISPERLIVE_DIR)
        from whisper_live.server import ClientManager
        return ClientManager(max_clients=max_clients, max_connection_time=max_time)

    def _build_post_processor(self):
        import os

        fns = []
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
                await asyncio.to_thread(
                    self._handle_client, adapter
                )
            except Exception as e:
                logger.error("WebSocket error: %s", e)
            finally:
                logger.info("WebSocket client disconnected")

        return web_app

    def _handle_client(self, websocket: WebSocketAdapter):
        """Run WhisperLive's client handling in a sync context."""
        import sys
        sys.path.insert(0, WHISPERLIVE_DIR)

        from whisper_live.server import BackendType
        from websockets.exceptions import ConnectionClosed

        try:
            self.server.backend = BackendType.FASTER_WHISPER
            if not self.server.handle_new_connection(
                websocket, self.model_name, None, False, False
            ):
                return

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
