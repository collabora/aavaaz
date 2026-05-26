"""AWS Lambda handler for batch audio transcription.

Supports three trigger modes:

1. **S3 event** — automatically transcribes files uploaded to a bucket.
2. **API Gateway (JSON)** — POST an audio file URL or base64-encoded payload and
   receive the transcript in the response.
3. **API Gateway (multipart)** — POST a file via multipart/form-data (used by
   the web demo drag-and-drop UI).

The handler also serves the web demo UI on GET /.

Environment variables
---------------------
AAVAAZ_MODEL          Whisper model name (default: ``small.en``)
AAVAAZ_LANGUAGE       Language code, or empty for auto-detect
AAVAAZ_OUTPUT_FORMAT  ``json`` | ``text`` | ``srt`` | ``vtt`` (default: ``json``)
AAVAAZ_OUTPUT_BUCKET  S3 bucket for transcript output (S3 trigger mode)
AAVAAZ_OUTPUT_PREFIX  Key prefix inside output bucket (default: ``transcripts/``)
AAVAAZ_ENABLE_PII     ``1`` to enable PII redaction (default: ``0``)
AAVAAZ_ENABLE_FORMAT  ``1`` to enable smart formatting (default: ``1``)
AAVAAZ_STORE_AUDIO    ``1`` to store uploaded audio in S3 (default: ``0``)
AAVAAZ_AUDIO_BUCKET   S3 bucket for stored audio (defaults to output bucket)
AAVAAZ_AUDIO_PREFIX   Key prefix for stored audio (default: ``audio/``)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# CORS headers for browser access from GitHub Pages.
# ---------------------------------------------------------------------------
_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


def _response(status_code: int, body: str, headers: dict | None = None) -> dict:
    """Build an API Gateway response with CORS headers."""
    h = {**_CORS_HEADERS}
    if headers:
        h.update(headers)
    return {"statusCode": status_code, "headers": h, "body": body}


# ---------------------------------------------------------------------------
# Global model cache — survives across warm Lambda invocations.
# ---------------------------------------------------------------------------
_model: Any = None


def _get_model() -> Any:
    """Return a cached WhisperModel, loading on first call."""
    global _model
    if _model is None:
        try:
            from whisper_live.transcriber.transcriber_faster_whisper import WhisperModel
        except ImportError:
            from faster_whisper import WhisperModel

        name = os.environ.get("AAVAAZ_MODEL", "small.en")
        logger.info("Loading Whisper model %s", name)
        _model = WhisperModel(name, device="cpu", compute_type="int8")
        logger.info("Model loaded")
    return _model


# ---------------------------------------------------------------------------
# Post-processing pipeline (optional)
# ---------------------------------------------------------------------------


def _build_pipeline() -> list[Any]:
    """Return a list of segment transform functions based on env config."""
    fns: list[Any] = []

    if os.environ.get("AAVAAZ_ENABLE_FORMAT", "1") == "1":
        from aavaaz.features.formatting import smart_format

        fns.append(smart_format)

    if os.environ.get("AAVAAZ_ENABLE_PII", "0") == "1":
        from aavaaz.features.pii_redaction import redact_pii

        fns.append(redact_pii)

    return fns


def _apply_pipeline(segment: dict, pipeline: list[Any]) -> dict:
    for fn in pipeline:
        segment["text"] = fn(segment["text"])
    return segment


# ---------------------------------------------------------------------------
# Core transcription
# ---------------------------------------------------------------------------


def _store_audio(audio_path: str, filename: str | None = None) -> str | None:
    """Optionally store audio to S3. Returns the S3 key or None if disabled."""
    if os.environ.get("AAVAAZ_STORE_AUDIO", "0") != "1":
        return None

    bucket = os.environ.get("AAVAAZ_AUDIO_BUCKET") or os.environ.get(
        "AAVAAZ_OUTPUT_BUCKET", ""
    )
    if not bucket:
        logger.warning("AAVAAZ_STORE_AUDIO=1 but no bucket configured")
        return None

    prefix = os.environ.get("AAVAAZ_AUDIO_PREFIX", "audio/")
    name = filename or os.path.basename(audio_path)
    key = f"{prefix}{uuid.uuid4().hex}_{name}"

    try:
        _s3_client().upload_file(audio_path, bucket, key)
        logger.info("Stored audio: s3://%s/%s", bucket, key)
        return key
    except Exception:
        logger.exception("Failed to store audio to s3://%s/%s", bucket, key)
        return None


def _transcribe(audio_path: str) -> dict:
    """Transcribe a local audio file and return a result dict."""
    file_size = os.path.getsize(audio_path)
    logger.info(
        "Starting transcription: file=%s size_bytes=%d model=%s",
        os.path.basename(audio_path),
        file_size,
        os.environ.get("AAVAAZ_MODEL", "small.en"),
    )
    t0 = time.time()

    model = _get_model()
    language = os.environ.get("AAVAAZ_LANGUAGE") or None
    segments, info = model.transcribe(
        audio_path, language=language, word_timestamps=True
    )
    segments = list(segments)

    pipeline = _build_pipeline()
    results = []
    for seg in segments:
        entry = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        }
        if seg.words:
            entry["words"] = [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability,
                }
                for w in seg.words
            ]
        entry = _apply_pipeline(entry, pipeline)
        results.append(entry)

    elapsed = time.time() - t0
    logger.info(
        "Transcription complete: duration=%.1fs segments=%d language=%s elapsed=%.2fs",
        info.duration,
        len(results),
        info.language,
        elapsed,
    )

    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "segments": results,
    }


def _format_output(result: dict) -> str:
    """Format the transcription result according to AAVAAZ_OUTPUT_FORMAT."""
    fmt = os.environ.get("AAVAAZ_OUTPUT_FORMAT", "json")
    if fmt == "text":
        return "\n".join(seg["text"] for seg in result["segments"])
    if fmt == "srt":
        lines = []
        for i, seg in enumerate(result["segments"], 1):
            lines.append(str(i))
            lines.append(f"{_ts(seg['start'])} --> {_ts(seg['end'])}")
            lines.append(seg["text"])
            lines.append("")
        return "\n".join(lines)
    if fmt == "vtt":
        lines = ["WEBVTT", ""]
        for seg in result["segments"]:
            lines.append(f"{_ts(seg['start'])} --> {_ts(seg['end'])}")
            lines.append(seg["text"])
            lines.append("")
        return "\n".join(lines)
    # default: json
    return json.dumps(result, indent=2)


def _ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Lambda handlers
# ---------------------------------------------------------------------------

_s3 = None


def _s3_client():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


def handler(event: dict, context: Any) -> dict:
    """Main Lambda entry point — dispatches to S3, web UI, or API handler."""
    request_id = (
        getattr(context, "aws_request_id", uuid.uuid4().hex)
        if context
        else uuid.uuid4().hex
    )
    logger.info("Request started: request_id=%s", request_id)

    try:
        if "Records" in event:
            return _handle_s3(event, context)

        # API Gateway v2 (HTTP API) uses requestContext.http.method
        http = event.get("requestContext", {}).get("http", {})
        method = http.get("method", event.get("httpMethod", "POST"))
        path = http.get("path", event.get("rawPath", event.get("path", "/")))

        # CORS preflight
        if method == "OPTIONS":
            return _response(204, "")

        if method == "GET":
            return _handle_web_ui(event, path)

        return _handle_api(event, context)
    except Exception:
        logger.exception("Unhandled error: request_id=%s", request_id)
        return _response(500, json.dumps({"error": "Internal server error"}))


def _handle_s3(event: dict, context: Any) -> dict:
    """Process S3 put events — download, transcribe, upload result."""
    output_bucket = os.environ.get("AAVAAZ_OUTPUT_BUCKET", "")
    output_prefix = os.environ.get("AAVAAZ_OUTPUT_PREFIX", "transcripts/")
    s3 = _s3_client()
    results = []

    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])
        obj_size = record["s3"]["object"].get("size", 0)
        logger.info("Processing s3://%s/%s size_bytes=%d", bucket, key, obj_size)

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, os.path.basename(key))
            s3.download_file(bucket, key, local_path)
            _store_audio(local_path, os.path.basename(key))
            result = _transcribe(local_path)

        output = _format_output(result)
        stem = Path(key).stem
        ext = (
            "json"
            if os.environ.get("AAVAAZ_OUTPUT_FORMAT", "json") == "json"
            else "txt"
        )
        out_key = f"{output_prefix}{stem}.{ext}"

        if output_bucket:
            s3.put_object(Bucket=output_bucket, Key=out_key, Body=output.encode())
            logger.info("Wrote s3://%s/%s", output_bucket, out_key)
            results.append(
                {
                    "input": f"s3://{bucket}/{key}",
                    "output": f"s3://{output_bucket}/{out_key}",
                }
            )
        else:
            # Same bucket
            s3.put_object(Bucket=bucket, Key=out_key, Body=output.encode())
            results.append(
                {"input": f"s3://{bucket}/{key}", "output": f"s3://{bucket}/{out_key}"}
            )

    return {"statusCode": 200, "body": json.dumps({"results": results})}


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

_WEB_DIR = os.path.join(os.path.dirname(__file__), "..", "web")


def _handle_web_ui(event: dict, path: str) -> dict:
    """Serve the web demo UI static files."""
    if path.startswith("/static/"):
        filename = path[len("/static/") :]
        # Sanitize: only allow known safe filenames
        safe_names = {"Collabora_Logo.svg": "image/svg+xml"}
        if filename not in safe_names:
            return _response(404, "Not found")
        filepath = os.path.join(_WEB_DIR, filename)
        with open(filepath) as f:
            content = f.read()
        return _response(200, content, {"Content-Type": safe_names[filename]})

    # Default: serve index.html
    filepath = os.path.join(_WEB_DIR, "index.html")
    with open(filepath) as f:
        content = f.read()
    return _response(200, content, {"Content-Type": "text/html"})


# ---------------------------------------------------------------------------
# Multipart form-data parsing
# ---------------------------------------------------------------------------


def _parse_multipart(event: dict) -> tuple[bytes | None, str | None, str | None]:
    """Extract file bytes, filename, and response_format from multipart form-data.

    Returns (file_bytes, filename, response_format) or (None, None, None) on failure.
    """
    content_type = ""
    headers = event.get("headers", {})
    for k, v in headers.items():
        if k.lower() == "content-type":
            content_type = v
            break

    if "boundary=" not in content_type:
        return None, None, None

    boundary = content_type.split("boundary=")[-1].strip()

    body = event.get("body", "")
    if event.get("isBase64Encoded"):
        body_bytes = base64.b64decode(body)
    else:
        body_bytes = body.encode() if isinstance(body, str) else body

    # Parse multipart boundaries
    boundary_bytes = f"--{boundary}".encode()
    parts = body_bytes.split(boundary_bytes)

    file_bytes = None
    filename = None
    response_format = None

    for part in parts:
        if b"Content-Disposition:" not in part and b"content-disposition:" not in part:
            continue

        # Split headers from body at double newline
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            header_end = part.find(b"\n\n")
            if header_end == -1:
                continue
            header_section = part[:header_end].decode(errors="replace")
            part_body = part[header_end + 2 :]
        else:
            header_section = part[:header_end].decode(errors="replace")
            part_body = part[header_end + 4 :]

        # Strip trailing \r\n
        if part_body.endswith(b"\r\n"):
            part_body = part_body[:-2]

        header_lower = header_section.lower()
        if 'name="file"' in header_lower or 'name="file"' in header_section:
            file_bytes = part_body
            # Extract filename
            for token in header_section.split(";"):
                token = token.strip()
                if token.lower().startswith("filename="):
                    filename = token.split("=", 1)[1].strip('" ')
        elif 'name="response_format"' in header_lower:
            response_format = part_body.decode(errors="replace").strip()

    return file_bytes, filename, response_format


def _handle_api(event: dict, context: Any) -> dict:
    """Process API Gateway (REST or HTTP API) requests.

    Accepts:
    - multipart/form-data with a ``file`` field (web demo)
    - ``{"audio_url": "s3://bucket/key"}`` — download from S3
    - ``{"audio_base64": "<base64>", "filename": "audio.wav"}`` — inline audio
    """
    # Check for multipart form-data first
    headers = event.get("headers", {})
    content_type = ""
    for k, v in headers.items():
        if k.lower() == "content-type":
            content_type = v
            break

    if "multipart/form-data" in content_type:
        return _handle_multipart(event)

    try:
        body = event.get("body", "{}")
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode()
        payload = json.loads(body) if isinstance(body, str) else body
    except (json.JSONDecodeError, ValueError):
        logger.warning("Invalid JSON body received")
        return _response(400, json.dumps({"error": "Invalid JSON body"}))

    with tempfile.TemporaryDirectory() as tmpdir:
        if "audio_url" in payload:
            url = payload["audio_url"]
            if not url.startswith("s3://"):
                return _response(
                    400, json.dumps({"error": "Only s3:// URLs supported"})
                )
            parts = url[5:].split("/", 1)
            if len(parts) != 2:
                return _response(400, json.dumps({"error": "Invalid S3 URL"}))
            bucket, key = parts
            local_path = os.path.join(tmpdir, os.path.basename(key))
            _s3_client().download_file(bucket, key, local_path)

        elif "audio_base64" in payload:
            filename = payload.get("filename", f"{uuid.uuid4().hex}.wav")
            safe_name = Path(filename).name
            local_path = os.path.join(tmpdir, safe_name)
            audio_bytes = base64.b64decode(payload["audio_base64"])
            Path(local_path).write_bytes(audio_bytes)

        else:
            return _response(
                400, json.dumps({"error": "Provide 'audio_url' or 'audio_base64'"})
            )

        _store_audio(local_path)
        result = _transcribe(local_path)

    output = _format_output(result)
    fmt = os.environ.get("AAVAAZ_OUTPUT_FORMAT", "json")
    content_type = "application/json" if fmt == "json" else "text/plain"

    return _response(200, output, {"Content-Type": content_type})


def _handle_multipart(event: dict) -> dict:
    """Handle multipart/form-data file upload from the web demo."""
    file_bytes, filename, response_format = _parse_multipart(event)
    if file_bytes is None:
        return _response(400, json.dumps({"error": "No file found in multipart data"}))

    with tempfile.TemporaryDirectory() as tmpdir:
        safe_name = Path(filename or "audio.wav").name
        local_path = os.path.join(tmpdir, safe_name)
        Path(local_path).write_bytes(file_bytes)
        logger.info(
            "Multipart upload: filename=%s size_bytes=%d", safe_name, len(file_bytes)
        )
        _store_audio(local_path, safe_name)
        result = _transcribe(local_path)

    fmt = response_format or os.environ.get("AAVAAZ_OUTPUT_FORMAT", "json")
    if fmt == "text":
        text = "\n".join(seg["text"] for seg in result["segments"])
        return _response(200, text, {"Content-Type": "text/plain"})
    return _response(
        200, json.dumps(result, indent=2), {"Content-Type": "application/json"}
    )
