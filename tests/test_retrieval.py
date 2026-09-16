from sqlmodel import select

from app.db import get_session
from app.models import FileRecord
from app.retrieval import filename_search, keyword_search, metadata_search, semantic_search
from app.retrieval.metadata_search import MetadataFilters


def _file_by_id(file_id: int) -> FileRecord:
    with get_session() as session:
        return session.get(FileRecord, file_id)


class TestFilenameSearch:
    def test_exact_filename_ranks_first(self):
        with get_session() as session:
            some_file = session.exec(select(FileRecord)).first()
        query = some_file.filename.rsplit(".", 1)[0].replace("_", " ")

        results = filename_search.search(query, limit=5)

        assert results, "expected at least one match"
        assert results[0].file_id == some_file.id

    def test_nonsense_query_returns_no_or_low_confidence_matches(self):
        results = filename_search.search("zzzzqqqqxxxxnonsense", limit=5, score_cutoff=80)
        assert results == []

    def test_scores_are_normalized(self):
        results = filename_search.search("report", limit=10)
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_exact_match_direct_lookup(self):
        with get_session() as session:
            some_file = session.exec(select(FileRecord)).first()

        result = filename_search.exact_match(some_file.filename)

        assert result is not None
        assert result.file_id == some_file.id
        assert result.score == 1.0

    def test_exact_match_missing_filename_returns_none(self):
        assert filename_search.exact_match("definitely_not_a_real_file.docx") is None


class TestMetadataSearch:
    def test_file_type_filter_is_exact(self):
        results = metadata_search.search(MetadataFilters(file_types=["xlsx"]), limit=50)
        assert results
        for r in results:
            assert _file_by_id(r.file_id).file_type == "xlsx"

    def test_topic_filter_is_exact(self):
        results = metadata_search.search(MetadataFilters(topic_cluster="ml_coursework"), limit=50)
        assert results
        for r in results:
            assert _file_by_id(r.file_id).topic_cluster == "ml_coursework"

    def test_results_ranked_by_recency_descending(self):
        results = metadata_search.search(MetadataFilters(file_types=["py"]), limit=50)
        modified_times = [_file_by_id(r.file_id).modified_at for r in results]
        assert modified_times == sorted(modified_times, reverse=True)

    def test_combined_filters_narrow_results(self):
        broad = metadata_search.search(MetadataFilters(file_types=["pdf"]), limit=200)
        narrow = metadata_search.search(
            MetadataFilters(file_types=["pdf"], topic_cluster="finance_reports"), limit=200
        )
        assert len(narrow) <= len(broad)


class TestKeywordSearch:
    def test_topic_keyword_surfaces_relevant_files(self):
        results = keyword_search.search("gradient descent backpropagation", limit=10)
        assert results
        topics = {_file_by_id(r.file_id).topic_cluster for r in results[:5]}
        assert "ml_coursework" in topics

    def test_scores_normalized_and_descending(self):
        results = keyword_search.search("flight dynamics trajectory", limit=10)
        assert results
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_no_matching_tokens_returns_empty(self):
        results = keyword_search.search("zzz nonexistent qqq vocabulary", limit=10)
        assert results == []


class TestSemanticSearch:
    def test_vague_paraphrased_query_surfaces_right_topic(self):
        results = semantic_search.search(
            "find that thing I was working on with my advisor about audio deepfakes", limit=5
        )
        assert results
        topics = {_file_by_id(r.file_id).topic_cluster for r in results[:3]}
        assert "research_drafts" in topics

    def test_scores_bounded(self):
        results = semantic_search.search("quarterly budget numbers", limit=10)
        assert results
        assert all(0.0 <= r.score <= 1.0 for r in results)
