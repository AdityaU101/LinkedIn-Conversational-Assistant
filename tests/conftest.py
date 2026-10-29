"""Shared fixtures.

Tests read the real datasets/indexes under Data/ (read-only) but write
application state (SQLite, uploads, graph) to a temporary directory.
External services (Adzuna, RemoteOK, LLM providers) are always mocked.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import common  # noqa: E402,F401  (OpenMP bootstrap first)

FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def isolated_state(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("state")
    os.environ["LA_PATH_SQLITE"] = str(tmp / "app.db")
    os.environ["LA_PATH_UPLOADS"] = str(tmp / "uploads")
    os.environ["LA_PATH_ANNOTATIONS"] = str(tmp / "annotations")
    os.environ["LA_PATH_RESULTS"] = str(tmp / "results")
    os.environ.setdefault("DISABLE_CROSS_ENCODER", "1")
    # never reach real providers from tests
    for k in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "PROPRIETARY_API_KEY", "ADZUNA_APP_ID", "ADZUNA_APP_KEY"):
        os.environ.pop(k, None)
    os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:9"  # closed port: Ollama unavailable
    from knowledge_graph import neo4j_client

    store = neo4j_client.MemoryGraphStore(tmp / "graph.pkl")
    neo4j_client.get_graph_store.cache_clear()
    original = neo4j_client.get_graph_store
    neo4j_client.get_graph_store = lambda: store  # type: ignore[assignment]
    import knowledge_graph.neo4j_client as m

    for mod in list(sys.modules.values()):
        if mod and getattr(mod, "get_graph_store", None) is original:
            mod.get_graph_store = neo4j_client.get_graph_store
    from llm import router as router_mod

    router_mod.get_router.cache_clear()
    yield tmp


@pytest.fixture(scope="session")
def resume_text() -> str:
    return (FIXTURES / "resume_data_scientist.txt").read_text()


@pytest.fixture(scope="session")
def api(isolated_state):
    from fastapi.testclient import TestClient

    from backend.main import app

    return TestClient(app)


@pytest.fixture(scope="session")
def uploaded(api, resume_text):
    r = api.post("/api/resumes/upload", files={"file": ("ds_resume.txt", resume_text.encode(), "text/plain")},
                 params={"use_llm": "false"})
    assert r.status_code == 200, r.text
    return r.json()["candidate_id"]


def needs_indexes():
    from matching.store import indexed_path

    return pytest.mark.skipif(not indexed_path().exists(), reason="run scripts/build_indexes.py first")
