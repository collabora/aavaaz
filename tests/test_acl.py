"""Integration tests for the ACL (user management) module."""

import os
import tempfile

from aavaaz.features.acl import (
    Role,
    User,
    UserStore,
    generate_api_key,
)


class TestRole:
    def test_admin_can_all(self):
        assert Role.ADMIN.can_transcribe()
        assert Role.ADMIN.can_admin()
        assert Role.ADMIN.can_read()

    def test_user_can_transcribe(self):
        assert Role.USER.can_transcribe()
        assert not Role.USER.can_admin()
        assert Role.USER.can_read()

    def test_readonly(self):
        assert not Role.READONLY.can_transcribe()
        assert not Role.READONLY.can_admin()
        assert Role.READONLY.can_read()


class TestUser:
    def test_to_dict_roundtrip(self):
        u = User(user_id="u1", name="Alice", role=Role.ADMIN, api_key_hash="abc")
        d = u.to_dict()
        u2 = User.from_dict(d)
        assert u2.user_id == "u1"
        assert u2.role == Role.ADMIN


class TestGenerateApiKey:
    def test_format(self):
        key = generate_api_key()
        assert key.startswith("wl_")
        assert len(key) > 10

    def test_uniqueness(self):
        keys = {generate_api_key() for _ in range(100)}
        assert len(keys) == 100


class TestUserStore:
    def test_create_and_authenticate(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store = UserStore(path=path)
            _user, key = store.create_user("alice", Role.USER)
            user = store.authenticate(key)
            assert user is not None
            assert user.name == "alice"
            assert user.role == Role.USER
        finally:
            os.unlink(path)

    def test_authenticate_invalid(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store = UserStore(path=path)
            assert store.authenticate("invalid-key") is None
        finally:
            os.unlink(path)

    def test_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            store = UserStore(path=path)
            _user, key = store.create_user("bob", Role.ADMIN)

            # Reload from disk
            store2 = UserStore(path=path)
            user = store2.authenticate(key)
            assert user is not None
            assert user.name == "bob"
        finally:
            os.unlink(path)
