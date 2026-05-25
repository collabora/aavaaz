"""Integration tests for the auth module."""

from unittest.mock import MagicMock

import pytest

from aavaaz.api.auth import configure_auth, create_token, require_auth, verify_token


class TestJWT:
    def setup_method(self):
        configure_auth("test-secret-key", api_keys=["key-123"])

    def test_create_and_verify_token(self):
        token = create_token("user1", expires_in=3600)
        payload = verify_token(token)
        assert payload["sub"] == "user1"

    def test_expired_token(self):
        import jwt as pyjwt
        token = create_token("user1", expires_in=-1)
        with pytest.raises(pyjwt.ExpiredSignatureError):
            verify_token(token)

    def test_invalid_token(self):
        import jwt as pyjwt
        with pytest.raises(pyjwt.InvalidTokenError):
            verify_token("not.a.valid.token")

    def test_custom_claims(self):
        token = create_token("user1", role="admin")
        payload = verify_token(token)
        assert payload["role"] == "admin"

    def test_no_secret_raises(self):
        configure_auth("", api_keys=[])
        with pytest.raises(ValueError):
            create_token("user1")


class TestRequireAuth:
    def setup_method(self):
        configure_auth("test-secret", api_keys=["valid-key"])

    @pytest.mark.asyncio
    async def test_api_key_auth(self):
        request = MagicMock()
        request.headers = {"X-API-Key": "valid-key"}
        result = await require_auth(request, credentials=None)
        assert result["sub"] == "api_key"

    @pytest.mark.asyncio
    async def test_bearer_token_auth(self):
        from fastapi.security import HTTPAuthorizationCredentials
        token = create_token("testuser")
        request = MagicMock()
        request.headers = {}
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        result = await require_auth(request, credentials=creds)
        assert result["sub"] == "testuser"

    @pytest.mark.asyncio
    async def test_no_credentials_raises_401(self):
        from fastapi import HTTPException
        request = MagicMock()
        request.headers = {}
        with pytest.raises(HTTPException) as exc_info:
            await require_auth(request, credentials=None)
        assert exc_info.value.status_code == 401
