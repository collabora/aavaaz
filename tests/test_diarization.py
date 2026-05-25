"""Integration tests for the diarization module (mocked embedding model)."""

from unittest.mock import MagicMock

import numpy as np

from aavaaz.features.diarization import SpeakerDiarizer


class TestSpeakerDiarizer:
    def _make_diarizer(self, **kwargs):
        d = SpeakerDiarizer(**kwargs)
        # Mock the model to return predictable embeddings
        d._model = MagicMock()
        return d

    def _set_embedding(self, diarizer, embedding):
        """Make _compute_embedding return a fixed vector."""
        diarizer._model = MagicMock()  # prevent lazy load
        norm = embedding / np.linalg.norm(embedding)

        def fake_compute(audio_np, sample_rate=16000):
            return norm
        diarizer._compute_embedding = fake_compute

    def test_new_speaker_created(self):
        d = self._make_diarizer()
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._set_embedding(d, emb)
        audio = np.zeros(16000, dtype=np.float32)  # 1 second
        speaker = d.identify_speaker(audio)
        assert speaker == "SPEAKER_00"
        assert len(d.speakers) == 1

    def test_same_speaker_matched(self):
        d = self._make_diarizer(similarity_threshold=0.5)
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._set_embedding(d, emb)
        audio = np.zeros(16000, dtype=np.float32)
        d.identify_speaker(audio)
        # Same embedding → same speaker
        speaker = d.identify_speaker(audio)
        assert speaker == "SPEAKER_00"
        assert len(d.speakers) == 1

    def test_different_speaker(self):
        d = self._make_diarizer(similarity_threshold=0.9)
        audio = np.zeros(16000, dtype=np.float32)

        # First speaker
        self._set_embedding(d, np.array([1.0, 0.0, 0.0]))
        d.identify_speaker(audio)

        # Different embedding → new speaker
        self._set_embedding(d, np.array([0.0, 1.0, 0.0]))
        speaker = d.identify_speaker(audio)
        assert speaker == "SPEAKER_01"
        assert len(d.speakers) == 2

    def test_max_speakers_cap(self):
        d = self._make_diarizer(max_speakers=2, similarity_threshold=0.99)
        audio = np.zeros(16000, dtype=np.float32)

        self._set_embedding(d, np.array([1.0, 0.0, 0.0]))
        d.identify_speaker(audio)
        self._set_embedding(d, np.array([0.0, 1.0, 0.0]))
        d.identify_speaker(audio)

        # Third speaker should be assigned to closest existing
        self._set_embedding(d, np.array([0.0, 0.0, 1.0]))
        speaker = d.identify_speaker(audio)
        assert speaker in ("SPEAKER_00", "SPEAKER_01")
        assert len(d.speakers) == 2

    def test_short_audio_returns_none(self):
        d = self._make_diarizer()
        # _compute_embedding returns None for short audio
        audio = np.zeros(100, dtype=np.float32)  # way too short

        # Need to use real _compute_embedding logic (which checks length)
        # but model is mocked, so we just verify the length check
        d._model = MagicMock()
        result = d._compute_embedding(audio, sample_rate=16000)
        assert result is None

    def test_reset(self):
        d = self._make_diarizer()
        self._set_embedding(d, np.array([1.0, 0.0, 0.0]))
        d.identify_speaker(np.zeros(16000, dtype=np.float32))
        assert len(d.speakers) == 1
        d.reset()
        assert len(d.speakers) == 0
        assert d._speaker_count == 0
