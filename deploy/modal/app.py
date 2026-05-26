"""Aavaaz — Modal GPU serverless transcription with web demo.

Uses WhisperLive batch inference for GPU-accelerated transcription.

Deploy with:
    modal deploy app.py

Develop with live-reload:
    modal serve app.py

Visit the root URL to access the drag-and-drop transcription demo.
POST to /v1/audio/transcriptions for the OpenAI-compatible API.

Environment variables (set via Modal Secrets):
    AAVAAZ_MODEL          Whisper model name (default: large-v3)
    AAVAAZ_LANGUAGE       Language code, or empty for auto-detect
    AAVAAZ_OUTPUT_FORMAT  json | text | srt | vtt (default: json)
    AAVAAZ_ENABLE_PII     1 to enable PII redaction (default: 0)
    AAVAAZ_ENABLE_FORMAT  1 to enable smart formatting (default: 1)
    AAVAAZ_API_KEY        Optional API key for authentication
    AAVAAZ_STORE_AUDIO    1 to store uploaded audio to a Modal Volume (default: 0)
"""

import logging

import fastapi
import modal

logger = logging.getLogger("aavaaz.modal")
logger.setLevel(logging.INFO)

WHISPER_MODEL = "large-v3"

# Path to the web UI files inside the container.
WEB_DIR = "/web"
# Path where WhisperLive source is mounted.
WHISPERLIVE_DIR = "/opt/whisper_live"

app = modal.App("aavaaz-transcribe")

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
        "tqdm",
    )
    .run_commands(
        f"python -c \"from faster_whisper import WhisperModel; WhisperModel('{WHISPER_MODEL}', device='cpu')\""
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


# Optional: create this secret with `modal secret create aavaaz-config KEY=VALUE ...`
_aavaaz_secret = modal.Secret.from_name("aavaaz-config")

# Optional volume for storing audio uploads (only used when AAVAAZ_STORE_AUDIO=1).
_audio_volume = modal.Volume.from_name("aavaaz-audio-store", create_if_missing=True)
AUDIO_VOLUME_PATH = "/audio_store"


@app.cls(
    image=image,
    gpu="T4",
    timeout=600,
    secrets=[_aavaaz_secret],
    scaledown_window=120,
    volumes={AUDIO_VOLUME_PATH: _audio_volume},
)
@modal.concurrent(max_inputs=4)
class Transcriber:
    @modal.enter()
    def load_model(self):
        import os
        import sys

        # Add WhisperLive to Python path
        sys.path.insert(0, WHISPERLIVE_DIR)

        from whisper_live.batch_inference import BatchInferenceWorker
        from whisper_live.transcriber.transcriber_faster_whisper import WhisperModel

        model_name = os.environ.get("AAVAAZ_MODEL", WHISPER_MODEL)
        logger.info("Loading Whisper model: %s", model_name)
        self.model = WhisperModel(model_name, device="cuda", compute_type="float16")
        self.batch_worker = BatchInferenceWorker(
            self.model, max_batch_size=4, batch_window_ms=100
        )
        self.batch_worker.start()
        self.language = os.environ.get("AAVAAZ_LANGUAGE") or None
        self.api_key = os.environ.get("AAVAAZ_API_KEY")
        self.store_audio = os.environ.get("AAVAAZ_STORE_AUDIO", "0") == "1"
        logger.info("Model loaded. store_audio=%s", self.store_audio)

    @modal.asgi_app()
    def web(self):
        import os

        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles

        web_app = fastapi.FastAPI(title="Aavaaz Transcription Demo")

        # Serve static assets (logo, etc.)
        web_app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @web_app.get("/", response_class=HTMLResponse)
        async def index():
            index_path = os.path.join(WEB_DIR, "index.html")
            with open(index_path) as f:
                return f.read()

        @web_app.post("/v1/audio/transcriptions")
        async def transcribe(request: fastapi.Request):
            return await self._handle_transcription(request)

        @web_app.get("/health")
        async def health():
            return {"status": "ok"}

        return web_app

    async def _handle_transcription(self, request):
        import os
        import shutil
        import tempfile
        import time
        import uuid
        from pathlib import Path


        request_id = uuid.uuid4().hex[:12]
        logger.info("Request received: request_id=%s", request_id)

        # Auth check
        if self.api_key:
            auth = request.headers.get("Authorization")
            if not auth or auth != f"Bearer {self.api_key}":
                logger.warning("Unauthorized request: request_id=%s", request_id)
                raise fastapi.HTTPException(status_code=401, detail="Unauthorized")

        content_type = request.headers.get("content-type", "")
        response_format = None

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                if "multipart/form-data" in content_type:
                    form = await request.form()
                    upload = form.get("file")
                    if upload is None:
                        raise fastapi.HTTPException(
                            status_code=400, detail="No 'file' field"
                        )
                    response_format = form.get("response_format")
                    filename = (
                        getattr(upload, "filename", None) or f"{uuid.uuid4().hex}.wav"
                    )
                    local_path = os.path.join(tmpdir, Path(filename).name)
                    content = await upload.read()
                    Path(local_path).write_bytes(content)
                    logger.info(
                        "Multipart upload: request_id=%s filename=%s size_bytes=%d",
                        request_id,
                        filename,
                        len(content),
                    )
                elif "application/json" in content_type:
                    import base64

                    payload = await request.json()
                    if "audio_base64" in payload:
                        filename = payload.get("filename", f"{uuid.uuid4().hex}.wav")
                        local_path = os.path.join(tmpdir, Path(filename).name)
                        audio_bytes = base64.b64decode(payload["audio_base64"])
                        Path(local_path).write_bytes(audio_bytes)
                        logger.info(
                            "JSON upload: request_id=%s filename=%s size_bytes=%d",
                            request_id,
                            filename,
                            len(audio_bytes),
                        )
                    else:
                        raise fastapi.HTTPException(
                            status_code=400,
                            detail="Provide 'file' (multipart) or 'audio_base64' (JSON)",
                        )
                elif "application/octet-stream" in content_type:
                    local_path = os.path.join(tmpdir, "audio.wav")
                    body = await request.body()
                    Path(local_path).write_bytes(body)
                    logger.info(
                        "Octet-stream upload: request_id=%s size_bytes=%d",
                        request_id,
                        len(body),
                    )
                else:
                    raise fastapi.HTTPException(
                        status_code=400,
                        detail="Unsupported Content-Type. Use multipart/form-data, application/json, or application/octet-stream",
                    )

                # Optional audio storage
                if self.store_audio:
                    store_name = f"{uuid.uuid4().hex}_{os.path.basename(local_path)}"
                    store_path = os.path.join(AUDIO_VOLUME_PATH, store_name)
                    shutil.copy2(local_path, store_path)
                    logger.info(
                        "Stored audio: request_id=%s path=%s", request_id, store_path
                    )

                t0 = time.time()
                result = self._transcribe(local_path)
                elapsed = time.time() - t0
                logger.info(
                    "Transcription complete: request_id=%s duration=%.1fs segments=%d elapsed=%.2fs",
                    request_id,
                    result.get("duration", 0),
                    len(result.get("segments", [])),
                    elapsed,
                )
        except fastapi.HTTPException:
            raise
        except Exception:
            logger.exception("Unhandled error: request_id=%s", request_id)
            raise fastapi.HTTPException(status_code=500, detail="Internal server error")

        fmt = response_format or os.environ.get("AAVAAZ_OUTPUT_FORMAT", "json")
        if fmt == "text":
            text = "\n".join(seg["text"] for seg in result["segments"])
            return fastapi.Response(content=text, media_type="text/plain")
        return result

    def _transcribe(self, audio_path: str) -> dict:
        import os

        from faster_whisper.audio import decode_audio
        from whisper_live.batch_inference import BatchRequest

        file_size = os.path.getsize(audio_path)
        logger.info(
            "Starting transcription: file=%s size_bytes=%d",
            os.path.basename(audio_path),
            file_size,
        )

        # Decode audio to raw float32 samples at 16kHz
        audio = decode_audio(audio_path)

        # Submit to WhisperLive batch inference worker
        req = BatchRequest(audio=audio, language=self.language)
        self.batch_worker.submit(req)
        req.future.wait(timeout=300)

        if req.error:
            logger.error("Transcription failed: %s", req.error)
            raise req.error

        segments = req.result or []
        info = req.info

        pipeline = self._build_pipeline()
        results = []
        for seg in segments:
            entry = {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            }
            if hasattr(seg, "words") and seg.words:
                entry["words"] = [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    }
                    for w in seg.words
                ]
            for fn in pipeline:
                entry["text"] = fn(entry["text"])
            results.append(entry)

        return {
            "language": info.language if info else "unknown",
            "language_probability": info.language_probability if info else 0.0,
            "duration": info.duration if info else 0.0,
            "segments": results,
        }

    def _build_pipeline(self) -> list:
        import os

        fns = []
        if os.environ.get("AAVAAZ_ENABLE_FORMAT", "1") == "1":
            from aavaaz.features.formatting import smart_format

            fns.append(smart_format)
        if os.environ.get("AAVAAZ_ENABLE_PII", "0") == "1":
            from aavaaz.features.pii_redaction import redact_pii

            fns.append(redact_pii)
        return fns
