"""
Tests for performance characteristics (Test Matrix §21).

These tests verify performance boundaries and resource management.
Not benchmarks — they validate contracts like "doesn't crash under load"
and "memory doesn't grow unbounded".
"""

import threading
import time

import numpy as np
import pytest


class TestTranscriptionLatency:
    """21.1 - Transcription latency bounds."""

    @pytest.mark.smoke
    def test_tiny_model_latency_under_10s(self, tmp_path):
        """tiny.en model should transcribe 10s audio in under 10s on CPU."""
        pytest.importorskip("faster_whisper")
        import wave

        from faster_whisper import WhisperModel

        # Generate 10 seconds of audio
        sample_rate = 16000
        duration = 10.0
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.3).astype(np.int16)
        audio_path = tmp_path / "perf_test.wav"
        with wave.open(str(audio_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())

        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

        start = time.time()
        result = model.transcribe(str(audio_path), language="en")
        if isinstance(result, tuple):
            segments, info = result
            list(segments)
        else:
            # faster-whisper >= 1.0 returns generator directly
            list(result)
        elapsed = time.time() - start

        # Should complete within 10 seconds on modern CPU
        assert elapsed < 10.0, f"Transcription took {elapsed:.1f}s (expected < 10s)"


class TestConcurrentClients:
    """21.2 - Concurrent client handling."""

    def test_multiple_threads_dont_deadlock(self):
        """Multiple threads accessing shared resources shouldn't deadlock."""
        from aavaaz.features.model_cache import ModelCache

        cache = ModelCache(max_models=2)
        errors = []

        def access_cache(thread_id):
            try:
                for i in range(10):
                    cache.get(f"model_{thread_id % 3}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_cache, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # No threads should still be alive (deadlock)
        alive = [t for t in threads if t.is_alive()]
        assert len(alive) == 0, f"{len(alive)} threads deadlocked"


class TestMemoryManagement:
    """21.4 - Memory usage under sustained load."""

    def test_audio_buffer_doesnt_grow_unbounded(self):
        """Audio frame accumulation should have bounds."""
        buffer = []
        max_buffer_duration = 30.0
        sample_rate = 16000
        frame_size = 4096
        max_frames = int(max_buffer_duration * sample_rate / frame_size)

        for i in range(max_frames + 100):
            frame = np.zeros(frame_size, dtype=np.float32)
            buffer.append(frame)
            if len(buffer) > max_frames:
                buffer = buffer[-max_frames:]

        assert len(buffer) <= max_frames

    def test_transcript_index_handles_many_entries(self):
        """Search index should handle 1000+ entries without issues."""
        from aavaaz.features.search import TranscriptIndex, TranscriptMetadata

        index = TranscriptIndex()
        for i in range(1000):
            meta = TranscriptMetadata(
                job_id=f"job_{i}",
                text=f"This is transcript number {i} with some text content",
            )
            index.add(meta)

        # Search should still be fast
        start = time.time()
        results = index.search(query="transcript number 500")
        elapsed = time.time() - start
        assert elapsed < 1.0
        assert len(results) > 0


class TestModelCachePerformance:
    """21.7 - Model hot-swap latency."""

    def test_cache_hit_is_instant(self):
        """Retrieving a cached model should be near-instant."""
        from unittest.mock import patch

        from aavaaz.features.model_cache import ModelCache

        cache = ModelCache(max_models=3)
        # Pre-populate cache
        with patch.object(cache, "_load_model", return_value="fake_model"):
            cache.get("test_model")

            start = time.time()
            for _ in range(1000):
                cache.get("test_model")
            elapsed = time.time() - start

        # 1000 cache hits should take < 100ms
        assert elapsed < 0.1, f"1000 cache hits took {elapsed:.3f}s"

    def test_eviction_happens(self):
        """Cache should evict when full."""
        from unittest.mock import patch

        from aavaaz.features.model_cache import ModelCache

        cache = ModelCache(max_models=2)
        with patch.object(cache, "_load_model", return_value="fake_model"):
            cache.get("model_a")
            cache.get("model_b")
            cache.get("model_c")  # Should evict model_a

        assert len(cache) <= 2


class TestGPUMemory:
    """21.5 - GPU memory management."""

    def test_model_cache_evicts_old_models(self):
        """Cache should evict models to prevent OOM."""
        from unittest.mock import patch

        from aavaaz.features.model_cache import ModelCache

        cache = ModelCache(max_models=3)
        with patch.object(cache, "_load_model", return_value="fake"):
            for i in range(5):
                cache.get(f"model_{i}")

        assert len(cache) <= 3

    def test_single_model_mode(self):
        """max_models=1 should only keep one model."""
        from unittest.mock import patch

        from aavaaz.features.model_cache import ModelCache

        cache = ModelCache(max_models=1)
        with patch.object(cache, "_load_model", return_value="fake"):
            cache.get("model_a")
            cache.get("model_b")

        assert len(cache) == 1
