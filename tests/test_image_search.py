from sqlmodel import select

from app.db import get_session
from app.models import FileRecord
from app.retrieval import image_search


def _some_image_record() -> FileRecord | None:
    with get_session() as session:
        return session.exec(
            select(FileRecord).where(FileRecord.file_type.in_(image_search.IMAGE_EXTENSIONS))
        ).first()


class TestResolveImagePath:
    def test_resolves_relative_synthetic_path(self):
        record = _some_image_record()
        assert record is not None, "expected at least one image in the generated corpus"
        path = image_search.resolve_image_path(record)
        assert path is not None
        assert path.exists()
        assert path.suffix.lstrip(".").lower() in image_search.IMAGE_EXTENSIONS

    def test_absolute_path_used_directly(self, tmp_path):
        from datetime import datetime

        real_file = tmp_path / "photo.png"
        real_file.write_bytes(b"not a real png but exists")
        record = FileRecord(
            id=999999, filename="photo.png", path=str(real_file), file_type="png",
            size_bytes=10, created_at=datetime.now(), modified_at=datetime.now(),
            topic_cluster="uncategorized",
        )
        resolved = image_search.resolve_image_path(record)
        assert resolved == real_file

    def test_missing_file_returns_none(self):
        from datetime import datetime

        record = FileRecord(
            id=999998, filename="gone.png", path="/definitely/not/a/real/path.png",
            file_type="png", size_bytes=10, created_at=datetime.now(),
            modified_at=datetime.now(), topic_cluster="uncategorized",
        )
        assert image_search.resolve_image_path(record) is None


class TestImageSearchIndexAndQuery:
    def test_build_index_covers_all_corpus_images(self):
        with get_session() as session:
            n_images = len(
                session.exec(
                    select(FileRecord).where(FileRecord.file_type.in_(image_search.IMAGE_EXTENSIONS))
                ).all()
            )
        n_indexed = image_search.build_index(force=False)
        assert n_indexed == n_images

    def test_content_query_returns_bounded_scores(self):
        results = image_search.search("a picture with a large circle in it", limit=5)
        assert results
        assert all(0.0 <= r.score <= 1.0 for r in results)
        assert all(r.source == "image_semantic" for r in results)

    def test_different_shape_queries_return_different_top_result_sets(self):
        circles = {r.file_id for r in image_search.search("a large red circle", limit=5)}
        squares = {r.file_id for r in image_search.search("a large blue square", limit=5)}
        # not a strict guarantee for every corpus, but with real CLIP embeddings and
        # distinct drawn subjects, the top-5 sets should not be identical
        assert circles != squares
