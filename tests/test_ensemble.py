"""Integration tests for the ensemble transcription module."""

import pytest

from aavaaz.features.ensemble import EnsembleTranscriber, Segment


class TestEnsembleTranscriber:
    def _make_transcriber(self, strategy="confidence"):
        et = EnsembleTranscriber(strategy=strategy)
        return et

    def test_add_and_remove_model(self):
        et = self._make_transcriber()
        et.add_model("m1", lambda audio: [])
        assert "m1" in et._models
        et.remove_model("m1")
        assert "m1" not in et._models

    def test_invalid_strategy(self):
        et = self._make_transcriber()
        with pytest.raises(ValueError):
            et.strategy = "invalid"

    def test_longest_strategy(self):
        et = self._make_transcriber(strategy="longest")

        def short_model(audio):
            return [Segment(start=0, end=1, text="hi", confidence=0.9, model_name="short")]

        def long_model(audio):
            return [Segment(start=0, end=2, text="hello world", confidence=0.8, model_name="long")]

        et.add_model("short", short_model)
        et.add_model("long", long_model)

        import numpy as np
        result = et.transcribe(np.zeros(16000, dtype=np.float32))
        assert "hello world" in result.text

    def test_confidence_strategy(self):
        et = self._make_transcriber(strategy="confidence")

        def low_conf(audio):
            return [Segment(start=0, end=1, text="lo", confidence=0.3, model_name="low")]

        def high_conf(audio):
            return [Segment(start=0, end=1, text="hi", confidence=0.95, model_name="high")]

        et.add_model("low", low_conf)
        et.add_model("high", high_conf)

        import numpy as np
        result = et.transcribe(np.zeros(16000, dtype=np.float32))
        assert "hi" in result.text

    def test_empty_models(self):
        et = self._make_transcriber()
        import numpy as np
        result = et.transcribe(np.zeros(16000, dtype=np.float32))
        assert result.text == ""

    def test_segment_dataclass(self):
        s = Segment(start=1.0, end=2.0, text="test", confidence=0.9, model_name="m")
        assert s.start == 1.0
        assert s.model_name == "m"
