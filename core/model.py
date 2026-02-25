import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np  # type: ignore
import torch  # type: ignore
from dotenv import load_dotenv  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

load_dotenv()

# Resolve model path: env var > project-relative default > HuggingFace fallback
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Talent_scout/
DEFAULT_FINETUNED_DIR = os.getenv(
    "FINETUNED_MODEL_DIR",
    str(_PROJECT_ROOT / "fine_tuning" / "resume_matcher_model"),
)
DEFAULT_FALLBACK_MODEL = os.getenv(
    "FALLBACK_MODEL",
    "sentence-transformers/all-mpnet-base-v2",
)


class ResumeRanker:
    """
    Stage-1 ranker (bi-encoder).
    - Loads your fine-tuned model folder if it exists.
    - Otherwise falls back to a HF hub model.
    - Uses normalized embeddings so cosine similarity = dot product.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_FINETUNED_DIR,
        fallback_model: str = DEFAULT_FALLBACK_MODEL,
        device: Optional[str] = None,
        max_seq_length: int = 256,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
    ):
        chosen = model_path if model_path and os.path.isdir(model_path) else fallback_model
        self.model_id = chosen
        self.model = SentenceTransformer(chosen, device=device)
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

    def rank(
        self,
        candidates: List[Dict[str, Any]],
        job_description: str,
        top_k: Optional[int] = None,
        resume_text_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Adds candidate['score'] and returns candidates sorted by score desc.

        candidates: list of dicts; each dict must contain resume text in one of:
          - resume_text_key (if provided)
          - 'resume_text'
          - 'resume'
          - 'raw'  (fallback)
        """

        jd = (job_description or "").strip()
        if not jd:
            raise ValueError("job_description is empty")

        # Extract resume texts
        texts: List[str] = []
        for c in candidates:
            if resume_text_key and resume_text_key in c:
                t = c[str(resume_text_key)]
            else:
                t = c.get("resume_text") or c.get("resume") or c.get("raw")
            texts.append((t or "").strip())

        if not texts:
            return []

        # Encode -> cosine via dot product (because normalized embeddings)
        with torch.no_grad():
            jd_emb = self.model.encode(
                [jd],
                convert_to_tensor=True,
                normalize_embeddings=self.normalize_embeddings,
            )  # (1, d)

            res_embs = self.model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize_embeddings,
                batch_size=self.batch_size,
            )  # (n, d)

            scores = (jd_emb @ res_embs.T).squeeze(0)  # (n,)

        scores_np = scores.detach().cpu().numpy().astype(np.float32)

        # Attach scores
        for c, s in zip(candidates, scores_np):
            c["score"] = float(s)

        # Sort
        ranked: List[Dict[str, Any]] = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)

        if top_k is not None:
            ranked = ranked[:int(top_k)]  # type: ignore

        return ranked


# ------------------------------------------------------------------
# Module-level convenience function (used by pipeline.py)
# ------------------------------------------------------------------
_default_ranker: Optional[ResumeRanker] = None


def rank_resumes_stage1(
    job_description: str,
    resume_texts: List[str],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper around ResumeRanker for pipeline.py.
    Accepts a list of plain resume strings and returns ranked dicts
    with keys: resume_text, score.
    """
    global _default_ranker
    if _default_ranker is None:
        _default_ranker = ResumeRanker()

    candidates = [{"resume_text": t} for t in resume_texts]
    return _default_ranker.rank(candidates, job_description, top_k=top_k)
