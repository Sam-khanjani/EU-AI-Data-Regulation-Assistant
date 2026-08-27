"""``/status`` shows corpus contents and change-detection results -- not for every visitor.

HTTP Basic Auth gates it. The one property worth a dedicated test beyond "wrong password is
rejected" is the empty-password case: ``secrets.compare_digest("", "")`` is ``True``, so an
unconfigured ADMIN_PASSWORD must not silently become "any password works."
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from euaia.api.main import app, require_admin
from euaia.config import settings
from euaia.db.session import engine


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(), reason="Postgres not reachable; run `docker compose up -d db`"
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def _admin_credentials(monkeypatch):
    monkeypatch.setattr(settings, "admin_username", "admin")
    monkeypatch.setattr(settings, "admin_password", "correct-horse-battery-staple")


class TestStatusPageRequiresAuth:
    def test_no_credentials_is_refused(self):
        resp = client.get("/status")
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Basic"

    def test_wrong_password_is_refused(self):
        resp = client.get("/status", auth=("admin", "wrong"))
        assert resp.status_code == 401

    def test_wrong_username_is_refused(self):
        resp = client.get("/status", auth=("not-admin", "correct-horse-battery-staple"))
        assert resp.status_code == 401

    def test_correct_credentials_are_accepted(self):
        resp = client.get("/status", auth=("admin", "correct-horse-battery-staple"))
        assert resp.status_code == 200


class TestCheckEndpointRequiresAuth:
    def test_no_credentials_is_refused(self):
        resp = client.post("/status/check")
        assert resp.status_code == 401

    def test_correct_credentials_are_accepted(self):
        resp = client.post("/status/check", auth=("admin", "correct-horse-battery-staple"))
        assert resp.status_code == 200


class TestPublicRoutesStayOpen:
    def test_the_chat_page_needs_no_credentials(self):
        assert client.get("/").status_code == 200

    def test_healthz_needs_no_credentials(self):
        assert client.get("/healthz").status_code == 200


class TestUnconfiguredPasswordRefusesEverything:
    def test_an_empty_configured_password_refuses_even_an_empty_guess(self, monkeypatch):
        # secrets.compare_digest("", "") is True -- without the explicit `bool(...)` guard
        # in require_admin, leaving ADMIN_PASSWORD unset in .env would let anyone in with a
        # blank password field rather than locking the dashboard out entirely.
        monkeypatch.setattr(settings, "admin_password", "")
        resp = client.get("/status", auth=("admin", ""))
        assert resp.status_code == 401

    def test_the_guard_lives_in_require_admin_not_just_in_this_test(self):
        # Pin the property to the function itself, so a future refactor that moves the
        # check elsewhere (or drops it) fails here rather than only in the route test above.
        import inspect

        source = inspect.getsource(require_admin)
        assert "bool(settings.admin_password)" in source
