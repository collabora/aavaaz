# Modal GPU Serverless Deployment

Aavaaz supports serverless GPU transcription on [Modal](https://modal.com).
Containers auto-scale to zero when idle and spin up in seconds with GPU
attached — no infrastructure to manage.

Production note: in this project, Modal is primarily used for live WebSocket
transcription via `deploy/modal/app_live.py`. This page documents the optional
GPU batch HTTP endpoint in `deploy/modal/app.py`.

For live deployment:

```bash
cd deploy/modal
modal deploy app_live.py
```

## Architecture

```
                         ┌──────────────────────────┐
  POST /v1/audio/       │  Modal Container (GPU)    │
  transcriptions ──────▶│  ┌─────────────────────┐  │
  (multipart, JSON,      │  │  faster-whisper      │  │
   or raw audio)         │  │  (CUDA / float16)    │  │
                         │  └──────────┬──────────┘  │
                         │  ┌──────────▼──────────┐  │
                         │  │  Post-processing     │  │
                         │  │  PII / Formatting    │  │
                         │  └──────────┬──────────┘  │
                         └─────────────┼────────────┘
                                       │
                            HTTP JSON response
```

## Quick Start

### 1. Install Modal

```bash
pip install modal
modal setup   # authenticate with modal.com
```

### 2. Deploy

```bash
cd deploy/modal
modal deploy app.py
```

Your endpoint is live at `https://<workspace>--aavaaz-transcribe-transcriber-web.modal.run`.

### 3. Web Demo

Visit the root URL in your browser to access the drag-and-drop transcription demo:

```
https://<workspace>--aavaaz-transcribe-transcriber-web.modal.run
```

Drop or select an audio file (WAV, MP3, OGG, etc.) and the transcript appears in seconds.
You can also choose a language or let the model auto-detect.

### 4. API (programmatic)

### 4. API (programmatic)

```bash
# Multipart file upload (OpenAI-compatible)
curl -X POST https://<workspace>--aavaaz-transcribe-transcriber-web.modal.run/v1/audio/transcriptions \
  -F file=@recording.wav

# Raw binary
curl -X POST https://<workspace>--aavaaz-transcribe-transcriber-web.modal.run/v1/audio/transcriptions \
  -H "Content-Type: application/octet-stream" \
  --data-binary @recording.wav

# Base64 JSON
curl -X POST https://<workspace>--aavaaz-transcribe-transcriber-web.modal.run/v1/audio/transcriptions \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\": \"$(base64 -w0 recording.wav)\"}"
```

### 5. Develop with live-reload

```bash
modal serve app.py
```

Creates a temporary URL that live-updates as you edit `app.py`.

## Configuration

Configuration is via Modal Secrets. Create a secret named `aavaaz-config`:

```bash
modal secret create aavaaz-config \
  AAVAAZ_MODEL=large-v3 \
  AAVAAZ_LANGUAGE= \
  AAVAAZ_OUTPUT_FORMAT=json \
  AAVAAZ_ENABLE_PII=0 \
  AAVAAZ_ENABLE_FORMAT=1 \
  AAVAAZ_API_KEY=my-secret-key
```

| Variable | Default | Description |
|----------|---------|-------------|
| `AAVAAZ_MODEL` | `large-v3` | Whisper model name |
| `AAVAAZ_LANGUAGE` | *(auto)* | Language code (`en`, `fr`, etc.) or empty for auto-detect |
| `AAVAAZ_OUTPUT_FORMAT` | `json` | Output format: `json` or `text` |
| `AAVAAZ_ENABLE_PII` | `0` | Set to `1` to enable PII redaction |
| `AAVAAZ_ENABLE_FORMAT` | `1` | Set to `1` to enable smart formatting |
| `AAVAAZ_API_KEY` | *(none)* | API key for Bearer token auth |

## GPU Selection

Edit `deploy/modal/app.py` to change the GPU type:

```python
@app.cls(gpu="T4", ...)      # cheapest, good for small/base models
@app.cls(gpu="A10G", ...)    # good balance for large-v3
@app.cls(gpu="A100", ...)    # fastest, for high-throughput
```

## Cost Estimates

Modal charges per-second of GPU usage. With container idle timeout of 120s:

| Model | GPU | Latency (30s audio) | Cost per transcription |
|-------|-----|--------------------|-----------------------|
| `small.en` | T4 | ~3s | ~$0.002 |
| `large-v3` | T4 | ~12s | ~$0.008 |
| `large-v3` | A10G | ~6s | ~$0.010 |

Containers stay warm for 2 minutes after the last request, so sequential
requests avoid cold starts entirely.

## Authentication

When `AAVAAZ_API_KEY` is set, all requests must include a Bearer token:

```bash
curl -X POST .../v1/audio/transcriptions \
  -H "Authorization: Bearer my-secret-key" \
  -F file=@recording.wav
```

Note: The web demo does not send an Authorization header.  When `AAVAAZ_API_KEY`
is set, the demo UI will receive 401 errors. Leave the key empty for public demos.

## Limitations

- Maximum request body: 4 GiB
- Default timeout: 600 seconds per request
- WebSocket streaming is not supported (use the main server for real-time)
- Model must fit in GPU memory (large-v3 needs ~6 GB VRAM)
