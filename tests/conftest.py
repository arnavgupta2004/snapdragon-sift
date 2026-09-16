import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DB_PATH = REPO_ROOT / "data" / "db.sqlite"


@pytest.fixture(scope="session", autouse=True)
def _require_synthetic_data():
    if not DB_PATH.exists():
        pytest.exit(
            "data/db.sqlite not found. Run `python data/generate_synthetic_data.py` "
            "before running tests — the retrieval/personalization suites are "
            "integration tests against the generated corpus, not isolated unit fixtures."
        )
