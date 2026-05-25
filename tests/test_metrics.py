"""Integration tests for the metrics module."""

from unittest.mock import patch


class TestMetrics:
    def test_is_available_returns_bool(self):
        from aavaaz.features.metrics import is_available
        assert isinstance(is_available(), bool)

    def test_start_metrics_server_without_prometheus(self):
        """When prometheus_client is not available, logs a warning."""
        from aavaaz.features import metrics
        with patch.object(metrics, "_AVAILABLE", False):
            # Should not raise
            metrics.start_metrics_server(port=19091)

    def test_metrics_counters_exist_when_available(self):
        from aavaaz.features.metrics import is_available
        if not is_available():
            return
        from aavaaz.features.metrics import (
            CONNECTIONS_ACTIVE,
            CONNECTIONS_TOTAL,
            SEGMENTS_EMITTED,
            TRANSCRIPTION_LATENCY,
        )
        # Verify they're proper prometheus objects
        assert hasattr(CONNECTIONS_TOTAL, "inc")
        assert hasattr(CONNECTIONS_ACTIVE, "set")
        assert hasattr(TRANSCRIPTION_LATENCY, "observe")
        assert hasattr(SEGMENTS_EMITTED, "labels")
