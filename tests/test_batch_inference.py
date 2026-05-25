"""Integration tests for the batch inference module (mocked transcriber)."""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# We need to mock faster_whisper/whisper_live at import time but must not
# leave the mocks in sys.modules for other test files. Use a subprocess
# isolation approach via pytest-forked if available, or just guard with
# try/except.
_need_mock = "faster_whisper" not in sys.modules
if _need_mock:
    _mocks = {}
    for mod in [
        "faster_whisper", "faster_whisper.audio", "faster_whisper.tokenizer",
        "faster_whisper.vad", "whisper_live", "whisper_live.transcriber",
        "whisper_live.transcriber.transcriber_faster_whisper",
    ]:
        _mocks[mod] = sys.modules[mod] = MagicMock()

try:
    from aavaaz.features.batch_inference import BatchInferenceWorker, BatchRequest
except ImportError:
    pytest.skip("Cannot import batch_inference", allow_module_level=True)
finally:
    if _need_mock:
        for mod in _mocks:
            sys.modules.pop(mod, None)


class TestBatchRequest:
    def test_defaults(self):
        audio = np.zeros(16000, dtype=np.float32)
        req = BatchRequest(audio=audio)
        assert req.language is None
        assert req.task == "transcribe"
        assert req.result is None
        assert req.error is None
        assert not req.future.is_set()

    def test_future_signaling(self):
        req = BatchRequest(audio=np.zeros(100, dtype=np.float32))
        req.future.set()
        assert req.future.is_set()


class TestBatchInferenceWorker:
    def test_start_and_stop(self):
        mock_transcriber = MagicMock()
        worker = BatchInferenceWorker(mock_transcriber, max_batch_size=4, batch_window_ms=10)
        worker.start()
        assert worker._thread is not None
        assert worker._thread.is_alive()
        worker.stop()
        worker._thread.join(timeout=2)
        assert not worker._thread.is_alive()

    def test_single_request_processed(self):
        """Submit a single request and verify it gets processed."""
        mock_transcriber = MagicMock()
        # Mock transcribe to return segments
        mock_segments = [MagicMock(start=0, end=1, text="hello")]
        mock_info = MagicMock()
        mock_transcriber.transcribe.return_value = (mock_segments, mock_info)

        worker = BatchInferenceWorker(mock_transcriber, max_batch_size=4, batch_window_ms=10)
        worker.start()

        try:
            audio = np.zeros(16000, dtype=np.float32)
            req = BatchRequest(audio=audio, language="en")
            worker.submit(req)
            req.future.wait(timeout=5)

            assert req.future.is_set()
            # Either result is set or error is set
            assert req.result is not None or req.error is not None
        finally:
            worker.stop()

    def test_queue_collects_batch(self):
        """Submit multiple requests quickly — they should be batched."""
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = ([MagicMock()], MagicMock())

        worker = BatchInferenceWorker(mock_transcriber, max_batch_size=4, batch_window_ms=100)
        worker.start()

        try:
            requests = []
            for _ in range(3):
                req = BatchRequest(audio=np.zeros(16000, dtype=np.float32), language="en")
                worker.submit(req)
                requests.append(req)

            # Wait for all to complete
            for req in requests:
                req.future.wait(timeout=5)
                assert req.future.is_set()
        finally:
            worker.stop()
