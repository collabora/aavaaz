"""Integration tests for the model cache module."""

from unittest.mock import MagicMock, patch

from aavaaz.features.model_cache import ModelCache


class TestModelCache:
    def test_cache_hit(self):
        cache = ModelCache(max_models=3, default_model="small")
        mock_model = MagicMock()
        with patch.object(cache, "_load_model", return_value=mock_model) as loader:
            m1 = cache.get("small")
            m2 = cache.get("small")
            assert m1 is m2
            loader.assert_called_once()

    def test_different_models(self):
        cache = ModelCache(max_models=3)
        models = {}

        def fake_load(name, device, compute_type):
            m = MagicMock(name=f"model-{name}")
            models[name] = m
            return m

        with patch.object(cache, "_load_model", side_effect=fake_load):
            m1 = cache.get("small")
            m2 = cache.get("large")
            assert m1 is not m2

    def test_eviction(self):
        cache = ModelCache(max_models=2)

        def fake_load(name, device, compute_type):
            return MagicMock(name=f"model-{name}")

        with patch.object(cache, "_load_model", side_effect=fake_load):
            cache.get("a")
            cache.get("b")
            cache.get("c")  # Should evict "a"
            assert len(cache._cache) == 2
            # "a" should be gone
            keys = list(cache._cache.keys())
            assert ("a", "cpu", "int8") not in keys

    def test_default_model(self):
        cache = ModelCache(default_model="tiny")
        with patch.object(cache, "_load_model", return_value=MagicMock()) as loader:
            cache.get(None)
            loader.assert_called_once_with("tiny", "cpu", "int8")

    def test_lru_order(self):
        cache = ModelCache(max_models=2)

        def fake_load(name, device, compute_type):
            return MagicMock()

        with patch.object(cache, "_load_model", side_effect=fake_load):
            cache.get("a")
            cache.get("b")
            # Access "a" again to make it recently used
            cache.get("a")
            # Add "c" — should evict "b" (LRU)
            cache.get("c")
            keys = [k[0] for k in cache._cache.keys()]
            assert "a" in keys
            assert "c" in keys
            assert "b" not in keys
