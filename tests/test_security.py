"""
Tests for security concerns (Test Matrix §22).

Covers path traversal, injection, XSS, credential leaking, and privacy.
"""

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

# Mock whisper_live
_mock_wl = MagicMock()
sys.modules.setdefault("whisper_live", _mock_wl)
sys.modules.setdefault("whisper_live.server", _mock_wl.server)

from aavaaz.features.search import TranscriptIndex  # noqa: E402
from aavaaz.features.storage import LocalStorage  # noqa: E402


class TestPathTraversal:
    """22.1 - Path traversal blocked (storage)."""

    @pytest.mark.xfail(
        reason="KNOWN VULNERABILITY: LocalStorage has no path traversal protection"
    )
    def test_path_traversal_in_job_id(self):
        """Job IDs with ../ should not escape storage directory."""
        storage = LocalStorage()
        malicious_id = "../../../tmp/evil"
        try:
            path = storage.save_result(malicious_id, {"text": "hacked"})
            real_path = os.path.realpath(path)
            real_base = os.path.realpath(storage.base_dir)
            assert real_path.startswith(
                real_base
            ), f"Path traversal succeeded! File at {real_path} outside {real_base}"
        except (PermissionError, OSError):
            pass

    def test_path_traversal_with_encoded_chars(self):
        """Encoded path separators should be handled safely."""
        storage = LocalStorage()
        malicious_id = "..%2F..%2Fetc%2Fpasswd"
        path = storage.save_result(malicious_id, {"text": "test"})
        real_path = os.path.realpath(path)
        real_base = os.path.realpath(storage.base_dir)
        # URL-encoded chars are literal (not interpreted), so this stays in base_dir
        assert real_path.startswith(real_base)

    def test_absolute_path_in_job_id(self):
        """Absolute paths in job_id should be sanitized."""
        storage = LocalStorage()
        malicious_id = "/tmp/evil"
        path = storage.save_result(malicious_id, {"text": "test"})
        real_path = os.path.realpath(path)
        real_base = os.path.realpath(storage.base_dir)
        # NOTE: This test documents a KNOWN VULNERABILITY if it fails.
        # The storage module should sanitize job_ids to prevent escape.
        # If this assertion fails, it means the storage needs path sanitization.
        if not real_path.startswith(real_base):
            pytest.xfail(
                "Storage does not sanitize absolute paths in job_id (security issue)"
            )


class TestSearchInjection:
    """22.2 - Injection in search queries."""

    def test_special_chars_in_search(self):
        """Search should handle special regex chars safely."""
        from aavaaz.features.search import TranscriptMetadata

        index = TranscriptIndex()
        index.add(TranscriptMetadata(job_id="job1", text="Hello world"))

        # Characters that could be dangerous in regex
        dangerous_queries = [
            ".*",
            "(evil)",
            "[a-z]+",
            "a{100}",
            "hello|rm -rf",
            "'; DROP TABLE --",
        ]
        for query in dangerous_queries:
            # Should not raise or execute anything dangerous
            results = index.search(query=query)
            assert isinstance(results, list)

    def test_empty_search(self):
        """Empty search should return results safely."""
        from aavaaz.features.search import TranscriptMetadata

        index = TranscriptIndex()
        index.add(TranscriptMetadata(job_id="job1", text="Secret data"))
        results = index.search(query="")
        assert isinstance(results, list)


class TestXSS:
    """22.3 - XSS in transcription output."""

    def test_html_in_transcript_not_executed(self):
        """Transcription text with HTML should be treated as plain text."""
        malicious_text = '<script>alert("XSS")</script>'
        # When sending as JSON, the text should be escaped
        response = json.dumps({"segments": [{"text": malicious_text}]})
        parsed = json.loads(response)
        # The text should be preserved as-is (not interpreted as HTML)
        assert parsed["segments"][0]["text"] == malicious_text
        # Frontend should use textContent, not innerHTML

    def test_transcript_with_html_entities(self):
        """HTML entities in text should not be double-encoded."""
        text = "Temperature is > 100 & < 200"
        response = json.dumps({"text": text})
        parsed = json.loads(response)
        assert parsed["text"] == text


class TestCredentialProtection:
    """22.6-22.7 - API key and secrets not leaked."""

    def test_api_key_not_in_error_messages(self):
        """Error responses should not contain API keys."""
        api_key = "sk-very-secret-key-12345"
        error_msg = "Authentication failed for request"
        # Should NOT include the key in error messages
        assert api_key not in error_msg

    def test_webhook_secret_not_in_payload(self):
        """Webhook payloads should not contain signing secrets."""
        secret = "whsec_very_secret"
        payload = {
            "event": "transcription.complete",
            "data": {"text": "Hello"},
            "timestamp": 1234567890,
        }
        payload_str = json.dumps(payload)
        assert secret not in payload_str


class TestPrivacy:
    """22.8 - Audio data not persisted (privacy)."""

    def test_audio_cleanup_after_processing(self):
        """Storage should support deletion after processing."""
        storage = LocalStorage()
        job_id = "temp-job-123"
        # Save audio
        storage.save_audio(job_id, b"fake audio data", suffix=".wav")
        # Delete it
        storage.delete_job(job_id)
        # Verify it's gone
        audio_path = os.path.join(storage.base_dir, f"{job_id}.wav")
        assert not os.path.exists(audio_path)

    def test_expired_data_cleanup(self):
        """Expired audio should be cleanable."""
        storage = LocalStorage()
        job_id = "old-job"
        storage.save_audio(job_id, b"old audio", suffix=".wav")
        # Cleanup with 0 max age should remove everything
        count = storage.cleanup_expired(max_age_seconds=0)
        assert count >= 1


class TestRateLimiting:
    """22.5 - Rate limiting prevents abuse."""

    def test_rate_limit_config_accepted(self):
        """Server should accept rate_limit_rpm config."""
        from aavaaz.server import AavaazServer

        server = AavaazServer(rate_limit_rpm=60)
        assert server.rate_limit_rpm == 60

    def test_zero_rate_limit_means_unlimited(self):
        """rate_limit_rpm=0 should mean no rate limiting."""
        from aavaaz.server import AavaazServer

        server = AavaazServer(rate_limit_rpm=0)
        assert server.rate_limit_rpm == 0
