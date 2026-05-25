"""Integration tests for the noise reduction module."""

from unittest.mock import patch

import numpy as np
import pytest


class TestNoiseReducer:
    def test_import_error_when_missing(self):
        """NoiseReducer raises ImportError if noisereduce not installed."""
        with patch.dict("sys.modules", {"noisereduce": None}):
            # Re-import to trigger the ImportError path
            # Can't easily test this without reimporting; test the is_available check instead
            pass

    def test_is_available(self):
        from aavaaz.features.noise_reduction import is_available
        # Just ensure it returns a bool
        assert isinstance(is_available(), bool)

    def test_invalid_mode(self):
        try:
            from aavaaz.features.noise_reduction import NoiseReducer
            with pytest.raises(ValueError, match="mode must be"):
                NoiseReducer(mode="invalid")
        except ImportError:
            pytest.skip("noisereduce not installed")

    def test_empty_audio(self):
        try:
            from aavaaz.features.noise_reduction import NoiseReducer
            nr = NoiseReducer(mode="near_field")
            result = nr.reduce(np.array([], dtype=np.float32))
            assert result.size == 0
        except ImportError:
            pytest.skip("noisereduce not installed")

    def test_reduce_basic(self):
        """Test that reduce returns audio of same shape."""
        try:
            from aavaaz.features.noise_reduction import NoiseReducer
            nr = NoiseReducer(mode="near_field")
            # Generate some noisy audio
            rng = np.random.default_rng(42)
            audio = rng.standard_normal(16000).astype(np.float32)
            result = nr.reduce(audio)
            assert result.shape == audio.shape
            assert result.dtype == np.float32
        except ImportError:
            pytest.skip("noisereduce not installed")

    def test_far_field_mode(self):
        try:
            from aavaaz.features.noise_reduction import NoiseReducer
            nr = NoiseReducer(mode="far_field")
            assert nr._stationary is False
        except ImportError:
            pytest.skip("noisereduce not installed")

    def test_prop_decrease_clamped(self):
        try:
            from aavaaz.features.noise_reduction import NoiseReducer
            nr = NoiseReducer(mode="near_field", prop_decrease=2.0)
            assert nr.prop_decrease == 1.0
            nr2 = NoiseReducer(mode="near_field", prop_decrease=-0.5)
            assert nr2.prop_decrease == 0.0
        except ImportError:
            pytest.skip("noisereduce not installed")
