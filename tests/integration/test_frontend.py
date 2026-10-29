"""Frontend: API client against the real FastAPI app, and every Streamlit page rendered
with streamlit.testing.AppTest (no browser, no separate server)."""
import httpx
import pytest

from frontend.api_client import APIClient, APIError
from tests.conftest import needs_indexes

pytestmark = needs_indexes()


def test_call_renders_api_errors_from_a_reloaded_module():
    """After a hot reload the cached client raises another module object's APIError; it is still rendered."""
    from frontend.components import ui

    stale = type("APIError", (Exception,), {})       # same name, different class object
    err = stale("gone")
    err.status, err.detail = 404, "Candidate not in graph"

    def fail():
        raise err

    assert not isinstance(err, APIError) and ui.call(fail) is None
    with pytest.raises(ValueError):                  # anything else still propagates
        ui.call(lambda: (_ for _ in ()).throw(ValueError("bug")))


@pytest.fixture(scope="module")
def wired_client(api):
    c = APIClient()
    c._client = api   # Starlette TestClient is an httpx.Client bound to the ASGI app
    return c


def test_client_error_mapping():
    c = APIClient(transport=httpx.MockTransport(lambda r: httpx.Response(404, json={"detail": "nope"})))
    with pytest.raises(APIError) as e:
        c.health()
    assert e.value.status == 404 and e.value.detail == "nope"

    def boom(request):
        raise httpx.ConnectError("down")

    with pytest.raises(APIError) as e:
        APIClient(transport=httpx.MockTransport(boom)).health()
    assert e.value.status == 0


def test_client_roundtrip(wired_client, uploaded):
    assert wired_client.health()["status"] == "ok"
    rec = wired_client.recommend(uploaded, {}, "rrf", 3)
    assert len(rec["results"]) == 3
    assert wired_client.chat(uploaded, "Which jobs best match my resume?")["intent"] == "recommend"


PAGES = ["dashboard", "resume_analyzer", "job_recommendations", "skill_gap", "knowledge_graph", "career_assistant",
         "research_dashboard", "annotation"]


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_without_errors(page, wired_client, uploaded, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import frontend.components.ui as ui

    monkeypatch.setattr(ui, "client", lambda: wired_client)
    wired_client.recommend(uploaded, {}, "final", 5, False, False)
    at = AppTest.from_file(f"../../frontend/pages/{page}.py", default_timeout=120)
    at.session_state["candidate_id"] = uploaded
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.title and at.title[0].value


def test_chat_is_cleared_when_the_profile_changes(wired_client, uploaded, resume_text, monkeypatch):
    """Switching to another profile and back does not bring the earlier conversation back."""
    from streamlit.testing.v1 import AppTest

    import frontend.components.ui as ui

    monkeypatch.setattr(ui, "client", lambda: wired_client)
    wired_client.recommend(uploaded, {}, "final", 5, False, False)
    other = wired_client.upload_resume("second.txt", (resume_text + "\nSecond profile").encode(), "false")["candidate_id"]

    def shell(profile, chat=None):
        at = AppTest.from_file("../../frontend/app.py", default_timeout=120)
        at.session_state["candidate_id"] = profile
        at.session_state["chat_profile"] = uploaded
        at.session_state["chat"] = chat or []
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return at

    said = [{"role": "user", "content": "Which jobs best match my resume?"}]
    assert shell(uploaded, said).session_state["chat"] == said          # same profile: conversation kept
    at = shell(other, said)                                             # switched (or uploaded a new resume)
    assert "chat" not in at.session_state or at.session_state["chat"] == []
    assert at.session_state["chat_profile"] == other

    # the chat page shows this sitting's messages only, never the stored history
    wired_client.chat(uploaded, "Which related occupations should I consider?")
    page = AppTest.from_file("../../frontend/pages/career_assistant.py", default_timeout=120)
    page.session_state["candidate_id"] = uploaded
    page.run()
    assert not page.exception and not page.chat_message
    wired_client.delete_resume(other)
