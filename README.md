# Aavaaz

**Production-grade speech-to-text platform built on [WhisperLive](https://github.com/collabora/WhisperLive).**

Aavaaz (आवाज़, "voice" in Hindi) is an open source extension of WhisperLive
with enterprise features that compete with Deepgram, ElevenLabs, and AssemblyAI.

## Features

| Category | Capabilities |
|----------|-------------|
| **Transcription** | Real-time WebSocket streaming, REST API (OpenAI-compatible), batch inference, multichannel audio |
| **Intelligence** | Speaker diarization, sentiment analysis, topic detection, entity extraction, summarization |
| **Post-processing** | Smart formatting, PII redaction, profanity filtering, noise reduction, utterance/paragraph segmentation |
| **Platform** | Webhook delivery, transcript search & tagging, storage backends (local/S3), ACL/auth, GDPR compliance, Prometheus metrics |
| **Deployment** | Docker, Helm charts, Terraform (AWS), **serverless (Lambda)**, **Modal (GPU)**, GPU auto-detection, model caching, SSE streaming |

## Quick Start

```bash
# Create a virtualenv (recommended, Python 3.12 for full ML stack)
python -m venv .venv && source .venv/bin/activate

pip install aavaaz whisper-live

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

### AWS (Terraform)
```bash
cd deploy/terraform
terraform init
terraform apply -var="model=large-v3" -var="api_key=my-secret"
```
Provisions VPC, ALB, ECS with GPU instances (g5.xlarge), ECR, and CloudWatch.
See [deploy/terraform/README.md](deploy/terraform/README.md) for full options.

### AWS Lambda (Serverless)

For batch file transcription without managing servers:

```bash
# Build and push the Lambda container image
docker build -f Dockerfile.lambda --build-arg WHISPER_MODEL=small.en -t aavaaz-lambda .

# Deploy infrastructure
cd deploy/terraform-lambda
terraform init
terraform apply

# Upload audio — transcript appears automatically in the output bucket
aws s3 cp recording.wav s3://$(terraform output -raw audio_input_bucket)/

# Or use the REST API
curl -X POST $(terraform output -raw api_endpoint) \
  -H "Content-Type: application/json" \
  -d '{"audio_url": "s3://my-bucket/recording.wav"}'
```

See [docs/SERVERLESS.md](docs/SERVERLESS.md) for full configuration, model
selection, cost estimates, and limitations.

### Modal (GPU Serverless)

Deploy on Modal for on-demand GPU transcription with zero infrastructure:

```bash
cd deploy/modal
pip install modal
modal setup
modal deploy app.py

# Transcribe
curl -X POST https://your-workspace--aavaaz-transcribe.modal.run/v1/audio/transcriptions \
  -F file=@recording.wav -F model=large-v3
```

Auto-scales to zero when idle, GPU containers spin up in seconds.
See [docs/MODAL.md](docs/MODAL.md) for full configuration.

## Development

```bash
git clone git@github.com:collabora/aavaaz.git
cd aavaaz
pip install -e ".[dev]"
pytest
```

## License

[MPL-2.0](LICENSE)
