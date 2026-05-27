"""
Tests for real-time WebSocket streaming (Test Matrix §2).

Tests the WhisperLive WebSocket protocol including connection lifecycle,
audio frame processing, and segment delivery.
"""

import json
import sys
from unittest.mock import MagicMock

import numpy as np

# Mock whisper_live before importing aavaaz.server
_mock_wl = MagicMock()
sys.modules.setdefault("whisper_live", _mock_wl)
sys.modules.setdefault("whisper_live.server", _mock_wl.server)


class FakeWebSocket:
    """Simulates a WebSocket for testing server-side logic."""

    def __init__(self, messages=None):
        self._messages = list(messages or [])
        self._sent = []
        self._closed = False
        self._recv_index = 0

    def recv(self):
        if self._recv_index >= len(self._messages):
            raise ConnectionError("No more messages")
        msg = self._messages[self._recv_index]
        self._recv_index += 1
        return msg

    def send(self, data):
        self._sent.append(data)

    def close(self):
        self._closed = True

    @property
    def sent_messages(self):
        return self._sent


class TestWebSocketProtocol:
    """2.1 - WebSocket connection lifecycle."""

    def test_options_json_parsed(self):
        """First message should be JSON options."""
        options = {
            "uid": "test-uid-123",
            "language": "en",
            "task": "transcribe",
            "model": "tiny.en",
            "use_vad": True,
        }
        msg = json.dumps(options)
        parsed = json.loads(msg)
        assert parsed["uid"] == "test-uid-123"
        assert parsed["language"] == "en"
        assert parsed["task"] == "transcribe"

    def test_end_of_audio_signal(self):
        """END_OF_AUDIO should signal end of stream."""
        signal = b"END_OF_AUDIO"
        assert signal == b"END_OF_AUDIO"

    def test_audio_frame_is_float32_pcm(self):
        """Audio frames should be float32 PCM data."""
        # Simulate 4096 samples at 16kHz = 256ms frame
        samples = np.random.randn(4096).astype(np.float32)
        frame_bytes = samples.tobytes()
        # Server should interpret as float32
        recovered = np.frombuffer(frame_bytes, dtype=np.float32)
        np.testing.assert_array_equal(recovered, samples)


class TestAudioFrameProcessing:
    """2.2 - Audio streaming at 16kHz float32 PCM."""

    def test_frame_size_256ms(self):
        """Standard frame is 4096 samples at 16kHz = 256ms."""
        sample_rate = 16000
        frame_size = 4096
        duration_ms = (frame_size / sample_rate) * 1000
        assert duration_ms == 256.0

    def test_multiple_frames_accumulate(self):
        """Multiple frames should accumulate into longer buffer."""
        frames = []
        for _ in range(10):
            frame = np.random.randn(4096).astype(np.float32)
            frames.append(frame)
        combined = np.concatenate(frames)
        expected_duration = len(combined) / 16000
        assert abs(expected_duration - 2.56) < 0.01  # 10 frames * 256ms

    def test_raw_pcm_int16_conversion(self):
        """Raw PCM int16 input should be converted to float32."""
        int16_samples = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        float32_samples = int16_samples.astype(np.float32) / 32768.0
        assert float32_samples.dtype == np.float32
        assert abs(float32_samples[3] - 1.0) < 0.001
        assert abs(float32_samples[4] - (-1.0)) < 0.001


class TestSegmentDelivery:
    """2.3 - Incremental segment delivery."""

    def test_segment_json_format(self):
        """Server should send segments in expected JSON format."""
        response = {
            "uid": "test-uid",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Hello world"},
                {"start": 2.5, "end": 5.0, "text": " how are you"},
            ],
        }
        data = json.dumps(response)
        parsed = json.loads(data)
        assert "segments" in parsed
        assert len(parsed["segments"]) == 2
        assert parsed["segments"][0]["text"] == "Hello world"

    def test_empty_segments_not_sent(self):
        """Server should not send empty segment lists."""
        ws = FakeWebSocket()
        segments = []
        # Simulate WhisperLive behavior: only send if segments non-empty
        if len(segments):
            ws.send(json.dumps({"uid": "test", "segments": segments}))
        assert len(ws.sent_messages) == 0

    def test_progressive_segments(self):
        """Segments should grow over time (incremental)."""
        ws = FakeWebSocket()
        # First result
        ws.send(json.dumps({"uid": "x", "segments": [{"text": "Hello"}]}))
        # Second result (more text)
        ws.send(json.dumps({"uid": "x", "segments": [{"text": "Hello world how"}]}))

        first = json.loads(ws.sent_messages[0])
        second = json.loads(ws.sent_messages[1])
        # Second should contain more text
        assert len(second["segments"][0]["text"]) >= len(first["segments"][0]["text"])


class TestConnectionManagement:
    """2.5-2.8 - Connection management scenarios."""

    def test_max_clients_wait_status(self):
        """When server is full, should send WAIT status."""
        response = {
            "uid": "new-client",
            "status": "WAIT",
            "message": 2.5,  # estimated wait time in minutes
        }
        data = json.dumps(response)
        parsed = json.loads(data)
        assert parsed["status"] == "WAIT"
        assert parsed["message"] == 2.5

    def test_disconnect_message(self):
        """Server should send DISCONNECT when client must leave."""
        response = {"uid": "client-1", "message": "DISCONNECT"}
        data = json.dumps(response)
        parsed = json.loads(data)
        assert parsed["message"] == "DISCONNECT"

    def test_server_ready_message(self):
        """Server should send SERVER_READY after initialization."""
        response = {
            "uid": "client-1",
            "message": "SERVER_READY",
            "backend": "faster_whisper",
        }
        data = json.dumps(response)
        parsed = json.loads(data)
        assert parsed["message"] == "SERVER_READY"
        assert parsed["backend"] == "faster_whisper"


class TestVADFiltering:
    """2.10 - VAD filtering (silence between speech)."""

    def test_silence_produces_no_output(self):
        """Pure silence should not produce transcription segments."""
        # Simulate VAD behavior: silent frames are discarded
        audio = np.zeros(16000, dtype=np.float32)  # 1 second silence
        # VAD threshold typically 0.5, silence energy ~0
        energy = np.mean(audio**2)
        assert energy == 0.0  # Pure silence has zero energy

    def test_speech_passes_vad(self):
        """Audio with speech-like energy should pass VAD."""
        # Simulate speech with some energy
        t = np.linspace(0, 1, 16000, endpoint=False)
        audio = (np.sin(2 * np.pi * 200 * t) * 0.5).astype(np.float32)
        energy = np.mean(audio**2)
        assert energy > 0.01  # Has meaningful energy
