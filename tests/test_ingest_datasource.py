from datetime import datetime

from sqlmodel import Session, SQLModel, create_engine, select

from app.datasources.base import RawFile
from app.models import FileRecord
from data.ingest_datasource import ingest_files


def _isolated_session() -> Session:
    engine = create_engine("sqlite://")  # in-memory, fully isolated from data/db.sqlite
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _raw_file(name: str) -> RawFile:
    return RawFile(
        filename=name, path=name, file_type=name.rsplit(".", 1)[-1],
        size_bytes=100, created_at=datetime.now(), modified_at=datetime.now(),
        extracted_text=f"content of {name}",
    )


class TestIngestFiles:
    def test_inserts_all_files(self):
        session = _isolated_session()
        raw_files = [_raw_file("a.md"), _raw_file("b.py"), _raw_file("c.txt")]

        by_type = ingest_files(raw_files, session)

        assert by_type == {"md": 1, "py": 1, "txt": 1}
        stored = session.exec(select(FileRecord)).all()
        assert len(stored) == 3
        assert {f.filename for f in stored} == {"a.md", "b.py", "c.txt"}

    def test_clear_existing_removes_prior_rows(self):
        session = _isolated_session()
        ingest_files([_raw_file("old.md")], session)
        assert len(session.exec(select(FileRecord)).all()) == 1

        ingest_files([_raw_file("new.md")], session, clear_existing=True)

        stored = session.exec(select(FileRecord)).all()
        assert len(stored) == 1
        assert stored[0].filename == "new.md"

    def test_default_does_not_clear_existing(self):
        session = _isolated_session()
        ingest_files([_raw_file("first.md")], session)
        ingest_files([_raw_file("second.md")], session)

        stored = session.exec(select(FileRecord)).all()
        assert len(stored) == 2

    def test_uncategorized_topic_cluster_preserved(self):
        session = _isolated_session()
        ingest_files([_raw_file("real_file.md")], session)

        stored = session.exec(select(FileRecord)).all()
        assert stored[0].topic_cluster == "uncategorized"
