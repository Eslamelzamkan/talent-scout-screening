"""
model.py — Stage-1 semantic ranker (SentenceTransformer bi-encoder).

Ported from talent-scout-screening/core/model.py.
"""

import logging
import os
import re
from collections import Counter
from math import exp
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

import numpy as np  # type: ignore
import torch  # type: ignore
from dotenv import load_dotenv  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore

load_dotenv()

# Resolve model path: env var > project-relative default > HuggingFace fallback
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # ai_engine/
DEFAULT_FINETUNED_DIR = os.getenv(
    "FINETUNED_MODEL_DIR",
    str(_PROJECT_ROOT / "models_data" / "fine_tuning" / "resume_matcher_model"),
)
DEFAULT_FALLBACK_MODEL = os.getenv(
    "FALLBACK_MODEL",
    "BAAI/bge-large-en-v1.5",
)
DEFAULT_CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
DEFAULT_ENABLE_CROSS_ENCODER = os.getenv("ENABLE_CROSS_ENCODER", "true").strip().lower() in {
    "1", "true", "yes", "on",
}
DEFAULT_LOCAL_MODELS_ONLY = os.getenv("AI_LOCAL_MODELS_ONLY", "true").strip().lower() in {
    "1", "true", "yes", "on",
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


DEFAULT_SEMANTIC_BI_WEIGHT = _env_float("SEMANTIC_BI_WEIGHT", 0.72)
DEFAULT_SEMANTIC_CROSS_WEIGHT = _env_float("SEMANTIC_CROSS_WEIGHT", 0.28)

_TOKEN_RE = re.compile(r"[a-z][a-z0-9\+#\.\-]{1,}")
_STOPWORDS = {
    "about", "after", "all", "also", "and", "any", "are", "as", "at", "be", "been", "but",
    "by", "can", "could", "do", "for", "from", "had", "has", "have", "if", "in", "into",
    "is", "it", "its", "may", "more", "must", "need", "of", "on", "or", "our", "should",
    "that", "the", "their", "them", "they", "this", "to", "using", "we", "will", "with",
    "you", "your", "years", "year", "experience", "work", "role", "team", "skills",
    "requirements", "responsibilities", "ability", "strong", "good", "knowledge",
}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


def _tokenize(text: str) -> List[str]:
    # Input is already lower-cased and _TOKEN_RE only matches [a-z0-9+#.-],
    # so findall() results need no further normalization.
    return _TOKEN_RE.findall((text or "").lower())


def _extract_focus_terms(text: str, max_terms: int = 48) -> List[str]:
    terms = [
        tok
        for tok in _tokenize(text)
        if len(tok) >= 3 and tok not in _STOPWORDS and not tok.isdigit()
    ]
    if not terms:
        return []
    counts = Counter(terms)
    return [t for t, _ in counts.most_common(max_terms)]


def lexical_alignment_score(job_description: str, resume_text: str, max_terms: int = 48) -> float:
    """
    Lightweight lexical signal to complement embedding similarity.
    Measures how many high-signal JD terms appear in the resume.
    """
    jd_terms = _extract_focus_terms(job_description, max_terms=max_terms)
    if not jd_terms:
        return 0.0

    resume_tokens = set(_tokenize(resume_text))
    if not resume_tokens:
        return 0.0

    hits = sum(1 for t in jd_terms if t in resume_tokens)
    return float(round(_clip01(hits / max(1, len(jd_terms))), 4))


class ResumeRanker:
    """
    Stage-1 ranker (bi-encoder).
    - Loads fine-tuned model folder if it exists.
    - Otherwise falls back to a HF hub model.
    - Uses normalized embeddings so cosine similarity = dot product.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_FINETUNED_DIR,
        fallback_model: str = DEFAULT_FALLBACK_MODEL,
        cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
        device: Optional[str] = None,
        max_seq_length: int = 512,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        use_cross_encoder: bool = DEFAULT_ENABLE_CROSS_ENCODER,
        semantic_bi_weight: float = DEFAULT_SEMANTIC_BI_WEIGHT,
        semantic_cross_weight: float = DEFAULT_SEMANTIC_CROSS_WEIGHT,
        local_models_only: bool = DEFAULT_LOCAL_MODELS_ONLY,
    ):
        chosen = model_path if model_path and os.path.isdir(model_path) else fallback_model
        self.model_id = chosen
        self.model = None
        try:
            self.model = SentenceTransformer(
                chosen,
                device=device,
                local_files_only=local_models_only,
            )
            self.model.max_seq_length = max_seq_length
        except Exception as exc:
            logger.error(
                "SentenceTransformer failed to load (model=%s); falling back to lexical scoring: %s",
                chosen, str(exc)[:200],
            )
            self.model = None
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.cross_encoder_model_id: Optional[str] = None
        self.cross_encoder = None

        bi_w = max(0.0, float(semantic_bi_weight))
        cross_w = max(0.0, float(semantic_cross_weight))
        total = bi_w + cross_w
        if total <= 0:
            bi_w, cross_w, total = 1.0, 0.0, 1.0
        self.semantic_bi_weight = bi_w / total
        self.semantic_cross_weight = cross_w / total

        if use_cross_encoder and CrossEncoder is not None:
            try:
                self.cross_encoder = CrossEncoder(
                    cross_encoder_model,
                    device=device,
                    local_files_only=local_models_only,
                )
                self.cross_encoder_model_id = cross_encoder_model
            except Exception:
                self.cross_encoder = None
                self.cross_encoder_model_id = None

        if self.cross_encoder is None:
            # Keep robust behavior in offline environments: bi-encoder-only scoring.
            self.semantic_bi_weight = 1.0
            self.semantic_cross_weight = 0.0

    @staticmethod
    def _normalize_bi_score(score: float) -> float:
        # Cosine similarity can be -1..1; map into 0..1.
        return _clip01((float(score) + 1.0) / 2.0)

    @staticmethod
    def _normalize_cross_score(score: float) -> float:
        s = float(score)
        if 0.0 <= s <= 1.0:
            return s
        return _clip01(_sigmoid(s))

    def rank(
        self,
        candidates: List[Dict[str, Any]],
        job_description: str,
        top_k: Optional[int] = None,
        resume_text_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        jd = (job_description or "").strip()
        if not jd:
            raise ValueError("job_description is empty")

        texts: List[str] = []
        for c in candidates:
            if resume_text_key and resume_text_key in c:
                t = c[str(resume_text_key)]
            else:
                t = c.get("resume_text") or c.get("resume") or c.get("raw")
            texts.append((t or "").strip())

        if not texts:
            return []

        bi_scores_np: np.ndarray
        bi_scores_are_normalized = False
        if self.model is not None:
            with torch.no_grad():
                jd_emb = self.model.encode(
                    [jd],
                    convert_to_tensor=True,
                    normalize_embeddings=self.normalize_embeddings,
                )
                res_embs = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    normalize_embeddings=self.normalize_embeddings,
                    batch_size=self.batch_size,
                )
                scores = (jd_emb @ res_embs.T).squeeze(0)
            bi_scores_np = scores.detach().cpu().numpy().astype(np.float32)
        else:
            # Fully local fallback that avoids total semantic failure if embedding
            # models are unavailable (e.g., offline/blocked model download).
            bi_scores_np = np.asarray(
                [lexical_alignment_score(jd, txt) for txt in texts],
                dtype=np.float32,
            )
            bi_scores_are_normalized = True
        cross_scores_np: Optional[np.ndarray] = None
        if self.cross_encoder is not None:
            try:
                pairs = [(jd, txt) for txt in texts]
                cross_raw = self.cross_encoder.predict(
                    pairs,
                    batch_size=max(1, min(self.batch_size, 32)),
                    show_progress_bar=False,
                )
                cross_scores_np = np.asarray(cross_raw, dtype=np.float32)
            except Exception:
                cross_scores_np = None

        for idx, c in enumerate(candidates):
            raw_bi = float(bi_scores_np[idx])
            bi_norm = raw_bi if bi_scores_are_normalized else self._normalize_bi_score(raw_bi)
            c["score_bi"] = float(round(bi_norm, 6))
            if cross_scores_np is not None and idx < len(cross_scores_np):
                cross_norm = self._normalize_cross_score(float(cross_scores_np[idx]))
                c["score_cross"] = float(round(cross_norm, 6))
                blended = (self.semantic_bi_weight * bi_norm) + (self.semantic_cross_weight * cross_norm)
                c["score"] = float(round(_clip01(blended), 6))
            else:
                c["score"] = float(round(bi_norm, 6))

        ranked: List[Dict[str, Any]] = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)

        if top_k is not None:
            ranked = ranked[:int(top_k)]

        return ranked


# ------------------------------------------------------------------
# Module-level convenience function
# ------------------------------------------------------------------
_default_ranker: Optional[ResumeRanker] = None


def rank_resumes_stage1(
    job_description: str,
    resume_texts: List[str],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper around ResumeRanker.
    Accepts a list of plain resume strings and returns ranked dicts
    with keys: resume_text, score.
    """
    global _default_ranker
    if _default_ranker is None:
        _default_ranker = ResumeRanker()

    candidates = [{"resume_text": t} for t in resume_texts]
    return _default_ranker.rank(candidates, job_description, top_k=top_k)
