# api/deps.py
import sys, os, logging

logger = logging.getLogger(__name__)

# Screening/ root (parent of api/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from db.db_postgres import TalentScoutRepo  # pyre-ignore[21]

_repo = None

def get_repo():
    """Return a TalentScoutRepo, or None if the DB is unreachable.
    Returning None is safe: pipeline.run_pipeline() already handles repo=None
    by skipping session creation and candidate persistence.
    Retries on every call so a temporarily-down DB can recover.
    """
    global _repo
    if _repo is not None:
        return _repo
    try:
        _repo = TalentScoutRepo()
        return _repo
    except Exception as exc:
        logger.warning(
            "Database unavailable — running without persistence: %s", exc
        )
        return None
