"""
entity_extractor.py — NER for candidate names and companies via spaCy.

Ported from talent-scout-screening/core/entity_extractor.py.
No import changes needed (no cross-module deps).
"""

import threading
from typing import Dict, List, Optional
import logging

_NLP_MODEL = None
_nlp_lock = threading.Lock()
logger = logging.getLogger(__name__)


def _get_nlp():
    """Lazy load the spaCy model."""
    global _NLP_MODEL
    if _NLP_MODEL is False:
        return None
    if _NLP_MODEL is not None:
        return _NLP_MODEL
    with _nlp_lock:
        if _NLP_MODEL is False:
            return None
        if _NLP_MODEL is not None:
            return _NLP_MODEL
        import spacy  # type: ignore
        try:
            _NLP_MODEL = spacy.load("en_core_web_lg", disable=["parser", "attribute_ruler", "lemmatizer", "tagger"])
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_lg' not found. "
                "Entity extraction will be disabled. "
                "Install with: python -m spacy download en_core_web_lg"
            )
            _NLP_MODEL = False
            return None
    return _NLP_MODEL


class CandidateEntityExtractor:
    def __init__(self, max_total_chars: int = 3000, max_name_chars: int = 250):
        self.max_total_chars = max_total_chars
        self.max_name_chars = max_name_chars

    def extract(self, text: str) -> Dict[str, Optional[str] | List[str]]:
        if not text:
            return {"candidate_name": None, "recent_companies": []}

        truncated_text = text[:int(self.max_total_chars)]

        nlp = _get_nlp()
        if not nlp:
            return {"candidate_name": None, "recent_companies": []}

        doc = nlp(truncated_text)

        name = None
        for ent in doc.ents:  # type: ignore
            if ent.label_ == "PERSON" and ent.start_char <= self.max_name_chars:
                name = ent.text.strip()
                break

        companies = set()
        for ent in doc.ents:  # type: ignore
            if ent.label_ == "ORG":
                c = ent.text.strip()
                if len(c) > 2 and c.lower() not in {"university", "college", "school", "bachelor", "master", "phd", "inc", "ltd"}:
                    companies.add(c)

        return {
            "candidate_name": name,
            "recent_companies": sorted(list(companies))
        }


_default_extractor = CandidateEntityExtractor()


def extract_entities(text: str) -> Dict[str, Optional[str] | List[str]]:
    return _default_extractor.extract(text)
