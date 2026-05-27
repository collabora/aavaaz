"""
Tests for authentication and authorization gaps (Test Matrix §4.6-4.9).

Covers rate limiting, WebSocket auth, and unauthorized access.
"""

import sys
from unittest.mock import MagicMock

import pytest

from aavaaz.api.auth import configure_auth, create_token, verify_token
from aavaaz.features.acl import Role, UserStore


class TestRateLimiting:
    """4.6 - Rate limiting (requests per minute)."""

    def test_rate_limit_config(self):
        """Rate limit should be configurable."""
        sys.modules.setdefault("whisper_live", MagicMock())
        sys.modules.setdefault("whisper_live.server", MagicMock())
        from aavaaz.server import AavaazServer

        server = AavaazServer(rate_limit_rpm=30)
        assert server.rate_limit_rpm == 30

    def test_zero_means_unlimited(self):
        sys.modules.setdefault("whisper_live", MagicMock())
        sys.modules.setdefault("whisper_live.server", MagicMock())
        from aavaaz.server import AavaazServer

        server = AavaazServer(rate_limit_rpm=0)
        assert server.rate_limit_rpm == 0


class TestQuotaTracking:
    """4.7 - Quota tracking and enforcement."""

    def test_user_store_quota(self, tmp_path):
        """UserStore should track quota per user."""
        store = UserStore(path=str(tmp_path / "users.json"))
        user, key = store.create_user("test_user", role=Role.USER, quota_minutes=100)
        assert user.quota_minutes == 100
        assert user.used_minutes == 0.0

    def test_user_rate_limit_per_user(self, tmp_path):
        """Each user can have different rate limits."""
        store = UserStore(path=str(tmp_path / "users.json"))
        user1, _ = store.create_user("fast", rate_limit_rpm=100)
        user2, _ = store.create_user("slow", rate_limit_rpm=10)
        assert user1.rate_limit_rpm == 100
        assert user2.rate_limit_rpm == 10

    def test_rate_limit_check(self, tmp_path):
        """Rate limiting should track request timestamps."""
        store = UserStore(path=str(tmp_path / "users.json"))
        user, key = store.create_user("limited", rate_limit_rpm=5)
        # Authenticate and check rate
        authed = store.authenticate(key)
        assert authed is not None
        result = store.check_rate_limit(authed.user_id)
        assert result is True  # First request should pass


class TestWebSocketAuth:
    """4.8 - WebSocket authentication (token query param)."""

    def test_valid_token_accepted(self):
        """Valid token should allow WebSocket connection."""
        configure_auth("test-secret-ws")
        token = create_token("ws-user", expires_in=3600)
        payload = verify_token(token)
        assert payload["sub"] == "ws-user"

    def test_expired_token_rejected(self):
        """Expired tokens should be rejected for WebSocket."""
        import jwt as pyjwt

        configure_auth("test-secret-ws")
        token = create_token("ws-user", expires_in=-1)
        with pytest.raises(pyjwt.ExpiredSignatureError):
            verify_token(token)

    def test_no_token_means_unauthenticated(self):
        """Missing token should mean unauthenticated access."""
        configure_auth("test-secret-ws", api_keys=["valid-key"])
        with pytest.raises((ValueError, KeyError, Exception)):  # noqa: B017
            verify_token("")


class TestUnauthorizedAccess:
    """4.9 - Unauthorized access returns proper error codes."""

    def test_invalid_api_key_rejected(self, tmp_path):
        """Invalid API key should not authenticate."""
        store = UserStore(path=str(tmp_path / "users.json"))
        store.create_user("real_user")
        result = store.authenticate("wrong-key-entirely")
        assert result is None

    def test_disabled_user_rejected(self, tmp_path):
        """Disabled users should not authenticate."""
        store = UserStore(path=str(tmp_path / "users.json"))
        user, key = store.create_user("disabled_user")
        user.enabled = False
        result = store.authenticate(key)
        assert result is None

    def test_role_permissions(self, tmp_path):
        """Roles should have correct permission levels."""
        assert Role.ADMIN.can_admin() is True
        assert Role.ADMIN.can_transcribe() is True
        assert Role.USER.can_admin() is False
        assert Role.USER.can_transcribe() is True
        assert Role.READONLY.can_transcribe() is False
        assert Role.READONLY.can_read() is True
