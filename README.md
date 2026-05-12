# Aavaaz

**Production-grade speech-to-text platform built on [WhisperLive](https://github.com/collabora/WhisperLive).**

Aavaaz (आवाज़, "voice" in Hindi) extends WhisperLive with enterprise features
that compete with Deepgram, ElevenLabs, and AssemblyAI — while keeping the
core transcription engine open-source.

## Features

| Category | Capabilities |
|----------|-------------|
| **Transcription** | Real-time WebSocket streaming, REST API (OpenAI-compatible), batch inference, multichannel audio |
| **Intelligence** | Speaker diarization, sentiment analysis, topic detection, entity extraction, summarization |
| **Post-processing** | Smart formatting, PII redaction, profanity filtering, noise reduction, utterance/paragraph segmentation |
| **Platform** | Webhook delivery, transcript search & tagging, storage backends (local/S3), ACL/auth, GDPR compliance, Prometheus metrics |
| **Deployment** | Docker, Helm charts, GPU auto-detection, model caching, SSE streaming |

## Quick Start

```bash
pip install aavaaz

# Start the server
aavaaz serve --model large-v3

# Transcribe a file
aavaaz transcribe audio.wav

# OpenAI-compatible REST endpoint
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav -F model=large-v3
```

## Architecture

Aavaaz uses WhisperLive as its transcription engine and extends it via the
plugin system:

```
┌─────────────────────────────────────────┐
│              Aavaaz Server               │
│  ┌─────────────────────────────────┐    │
│  │  REST API / WebSocket / Web UI  │    │
│  └──────────────┬──────────────────┘    │
│  ┌──────────────┴──────────────────┐    │
│  │        Plugin Pipeline          │    │
│  │  diarization → formatting →     │    │
│  │  PII redaction → intelligence   │    │
│  └──────────────┬──────────────────┘    │
│  ┌──────────────┴──────────────────┐    │
│  │    WhisperLive Core Engine      │    │
│  │  faster-whisper / TensorRT /    │    │
│  │  OpenVINO                       │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

## Advanced Features

### Word-Level Timestamps
Enable per-word timing and confidence scores in transcription segments:
```python
from aavaaz import AavaazServer

server = AavaazServer()
server.serve(word_timestamps=True)
```
When enabled, each segment includes a `words` array:
```json
{
  "segments": [{
    "start": "0.000", "end": "2.500", "text": "Hello world",
    "words": [
      {"word": "Hello", "start": "0.000", "end": "0.800", "probability": 0.95},
      {"word": " world", "start": "0.900", "end": "2.500", "probability": 0.88}
    ]
  }]
}
```

### Custom Vocabulary / Hotwords
Boost recognition of specific terms (product names, acronyms, domain jargon):
```python
from aavaaz import AavaazServer

server = AavaazServer()
server.serve(hotwords="Aavaaz,TensorRT,OpenVINO")
```
The `hotwords` parameter is a comma-separated string passed directly to faster-whisper's keyword boosting. Also available in the REST API via the `hotwords` form field.

### Speaker Diarization
Real-time speaker identification using pyannote.audio embeddings:
```bash
pip install pyannote.audio
```
```python
from aavaaz import AavaazServer

server = AavaazServer()
server.serve(enable_diarization=True, max_speakers=4)
```
When enabled, completed segments include a `speaker` field:
```json
{"start": "0.000", "end": "2.500", "text": "Hello", "speaker": "SPEAKER_00", "completed": true}
```

### Authentication
Protect both REST API and WebSocket connections with a shared API key:
```bash
aavaaz serve --model large-v3 --api-key "my-secret-key"
```
- **REST API**: Requires `Authorization: Bearer my-secret-key` header
- **WebSocket**: Requires either `Authorization: Bearer my-secret-key` header or `?token=my-secret-key` query parameter

Unauthenticated connections receive HTTP 401 before any GPU resources are allocated.

### Rate Limiting
Limit REST API requests per client IP (sliding 60-second window):
```bash
aavaaz serve --model large-v3 --rate-limit-rpm 60
```
Clients exceeding the limit receive HTTP 429.

### Auto-Reconnect
Automatically reconnect when the WebSocket connection drops unexpectedly:
```python
from whisper_live.client import TranscriptionClient

client = TranscriptionClient(
  "localhost", 9090,
  max_retries=5,
  retry_delay=3,
)
```

### Batch Inference
Batch multiple client sessions into single GPU calls for higher throughput:
```bash
aavaaz serve --model large-v3 --batch-inference --batch-max-size 8 --batch-window-ms 50
```

### Prometheus Metrics
Monitor server health with a Prometheus `/metrics` endpoint:
```bash
aavaaz serve --model large-v3 --metrics-port 9091
```
Tracks active connections, transcription latency, segment counts, and error rates.

### SSE Streaming
Stream transcription results via Server-Sent Events from the REST API:
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav -F stream=true
```
Returns real-time segment events as `text/event-stream`.

### Plugin System
Extend the transcription pipeline with custom post-processors:
```python
from aavaaz.plugins import PluginRegistry

registry = PluginRegistry()
registry.register("my_plugin", my_post_processor_fn, priority=50)

server = AavaazServer(plugin_registry=registry)
server.serve()
```
Plugins receive each transcription segment and can modify, enrich, or filter it before delivery to the client.

## Scaling Guide

### Single GPU
```bash
aavaaz serve --model large-v3 --batch-inference
```

### Multi-GPU (Docker Compose)
```yaml
services:
  aavaaz:
    image: collabora/aavaaz:latest
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    ports:
      - "9090:9090"
      - "8000:8000"
```

### Kubernetes (Helm)
```bash
helm install aavaaz deploy/helm/aavaaz \
  --set model=large-v3 \
  --set replicas=3 \
  --set gpu.enabled=true
```

## Development

```bash
git clone git@github.com:collabora/aavaaz.git
cd aavaaz
pip install -e ".[dev]"
pytest
```

## License

[MPL-2.0](LICENSE)
