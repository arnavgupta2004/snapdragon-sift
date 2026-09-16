from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def _first_user_id() -> int:
    return client.get("/api/users").json()[0]["id"]


class TestUsersEndpoint:
    def test_returns_three_seeded_personas(self):
        resp = client.get("/api/users")
        assert resp.status_code == 200
        users = resp.json()
        assert len(users) == 3
        assert {u["persona_key"] for u in users} == {
            "priya_grad_student", "david_analyst", "maria_freelancer",
        }


class TestQueryEndpoint:
    def test_fast_route_query(self):
        resp = client.post(
            "/api/query", json={"query": "show me my pptx files from last week", "user_id": _first_user_id()}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["routing_trace"]["tier"] == "fast"
        assert "results" in body

    def test_invalid_body_rejected(self):
        resp = client.post("/api/query", json={"query": "no user id"})
        assert resp.status_code == 422


class TestPersonalizationEndpoint:
    def test_returns_insights_shape(self):
        resp = client.get(f"/api/personalization/{_first_user_id()}")
        assert resp.status_code == 200
        body = resp.json()
        assert "preferred_file_types" in body
        assert "recurring_patterns" in body
        assert "active_context_boost_now" in body


class TestBaselineComparisonEndpoint:
    def test_returns_the_checked_in_summary(self):
        resp = client.get("/api/eval/baseline-comparison")
        assert resp.status_code == 200
        body = resp.json()
        assert "full_system" in body
        assert "naive_keyword" in body


class TestIngestEndpoint:
    """Contract tests only (request validation, error mapping, response shape) via
    monkeypatched internals — no real DB/filesystem mutation. The deeper crawl+insert
    logic is already covered in isolation by tests/test_datasources.py and
    tests/test_ingest_datasource.py; a real end-to-end run of this endpoint was also
    manually verified against a real ~300-file Downloads folder during development
    (see REPORT.md)."""

    def test_missing_root_returns_400(self, monkeypatch):
        def fake_init(self, root, max_files=None, exclude_dirs=None):
            raise FileNotFoundError(f"root does not exist: {root}")

        monkeypatch.setattr(
            "app.datasources.filesystem_source.FilesystemDataSource.__init__", fake_init
        )
        resp = client.post("/api/ingest", json={"root": "/definitely/not/a/real/path"})
        assert resp.status_code == 400

    def test_empty_directory_returns_400(self, monkeypatch):
        class EmptySource:
            def __init__(self, root, max_files=None, exclude_dirs=None):
                self.root = root

            def list_files(self):
                return []

        monkeypatch.setattr("app.datasources.filesystem_source.FilesystemDataSource", EmptySource)
        resp = client.post("/api/ingest", json={"root": "/tmp"})
        assert resp.status_code == 400

    def test_happy_path_returns_summary_shape(self, monkeypatch):
        from datetime import datetime

        from app.datasources.base import RawFile

        class FakeSource:
            def __init__(self, root, max_files=None, exclude_dirs=None):
                self.root = root

            def list_files(self):
                return [
                    RawFile(
                        filename="a.md", path="a.md", file_type="md", size_bytes=10,
                        created_at=datetime.now(), modified_at=datetime.now(),
                        extracted_text="hello",
                    )
                ]

        monkeypatch.setattr("app.datasources.filesystem_source.FilesystemDataSource", FakeSource)
        monkeypatch.setattr("data.ingest_datasource.ingest_files", lambda *a, **k: {"md": 1})
        monkeypatch.setattr("app.retrieval.semantic_search.build_index", lambda force=False: 351)

        resp = client.post("/api/ingest", json={"root": "/tmp", "max_files": 10})
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_files_crawled"] == 1
        assert body["by_type"] == {"md": 1}
        assert body["n_indexed_total"] == 351
        assert body["cleared_existing"] is False


class TestFeedbackEndpoint:
    def test_records_valid_feedback(self):
        resp = client.post(
            "/api/feedback",
            json={"user_id": _first_user_id(), "file_id": 1, "query": "test query", "signal": "thumbs_up"},
        )
        assert resp.status_code == 200
        assert resp.json()["recorded"] is True

    def test_rejects_invalid_signal(self):
        resp = client.post(
            "/api/feedback",
            json={"user_id": _first_user_id(), "file_id": 1, "query": "test query", "signal": "shrug"},
        )
        assert resp.status_code == 422


class TestQueryStreamEndpoint:
    def test_stream_ends_with_done_event_containing_trace(self):
        with client.stream(
            "GET", "/api/query/stream",
            params={"query": "gradient_descent_v2.md", "user_id": _first_user_id()},
        ) as resp:
            assert resp.status_code == 200
            events = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    events.append(line[len("data: "):])

        assert events
        import json
        parsed = [json.loads(e) for e in events]
        assert parsed[-1]["type"] == "done"
        assert "routing_trace" in parsed[-1]
        assert any(e["type"] == "route" for e in parsed)
        assert any(e["type"] == "stage" for e in parsed)
