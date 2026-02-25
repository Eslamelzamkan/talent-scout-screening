"""
entity_extractor.py

Deterministic Named Entity Recognition (NER) for extracting candidate names
and companies using the fast, offline `spaCy` library.

Design:
- Uses `en_core_web_lg` for high accuracy on Organization names.
- Globally lazy-loads the model to avoid excessive memory spikes during multiprocessing.
- Restricts Person extraction to the first 250 characters to avoid catching references.
- Truncates full parsing to the first 3000 characters to ensure millisecond performance per resume.
"""

from typing import Dict, List, Optional
import logging

# Lazy-loaded globals
_NLP_MODEL = None
logger = logging.getLogger(__name__)


def _get_nlp():
    """Lazy load the spaCy model. Thread-safe for inference."""
    global _NLP_MODEL
    if _NLP_MODEL is False:
        return None
    if _NLP_MODEL is None:
        import spacy  # type: ignore
        try:
            # Try loading the large model
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
    """Extracts base candidate entities (Name, Companies) deterministically."""
    
    def __init__(self, max_total_chars: int = 3000, max_name_chars: int = 250):
        self.max_total_chars = max_total_chars
        self.max_name_chars = max_name_chars
    
    def extract(self, text: str) -> Dict[str, Optional[str] | List[str]]:
        """
        Extract the candidate's name and recent companies.
        """
        if not text:
            return {"candidate_name": None, "recent_companies": []}
            
        # Optimization: Capping total text parsed to save processing time
        truncated_text = text[:int(self.max_total_chars)]  # type: ignore
        
        nlp = _get_nlp()
        if not nlp:
            return {"candidate_name": None, "recent_companies": []}
            
        doc = nlp(truncated_text)
        
        # 1. Extract Candidate Name
        name = None
        for ent in doc.ents:  # type: ignore
            if ent.label_ == "PERSON" and ent.start_char <= self.max_name_chars:
                # Only trust a PERSON entity if it occurs in the very beginning of the document
                # This prevents picking up recruiters, references, or "Developed for John at Apple"
                name = ent.text.strip()
                break # Usually the first name in the header is the candidate's name
        
        # 2. Extract Companies
        companies = set()
        for ent in doc.ents:  # type: ignore
            if ent.label_ == "ORG":
                c = ent.text.strip()
                # Simple heuristic to strip out common noise picked up by NER
                if len(c) > 2 and c.lower() not in {"university", "college", "school", "bachelor", "master", "phd", "inc", "ltd"}:
                    companies.add(c)
                    
        return {
            "candidate_name": name,
            "recent_companies": sorted(list(companies))
        }

# Singleton instance
_default_extractor = CandidateEntityExtractor()

def extract_entities(text: str) -> Dict[str, Optional[str] | List[str]]:
    """Convenience wrapper for the global extractor."""
    return _default_extractor.extract(text)
