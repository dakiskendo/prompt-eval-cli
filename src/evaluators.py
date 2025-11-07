from __future__ import annotations
from typing import Callable, Sequence, Dict, Any, Optional
import re, math, sacrebleu
import numpy as np

_WS_RE = re.compile(r"\s+")

#normalize text
def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = _WS_RE.sub(" ", s)
    return s

def exact_match(pred: str, ref: str, normalize: bool = True) -> float:
    if normalize:
        pred = normalize_text(pred)
        ref = normalize_text(ref)
    return 1.0 if pred == ref else 0.0

def bleu_score(pred: str, ref: str, max_n: int = 4) -> float:
    try:
        score = sacrebleu.sentence_bleu(pred, [ref]).score
        if math.isnan(score) or math.isinf(score):
            return 0.0
        return float(score) / 100.0
    except Exception:
        return 0.0
    
def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    a = np.asarray(vec1, dtype=np.float32)
    b = np.asarray(vec2, dtype=np.float32)
    if a.ndim != 1 or b.ndim != 1 or a.size == 0 or b.size == 0:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def text_embedding_similarity(
        text1:str,
        text2:str,
        embedder: Callable[[str], Sequence[float]],
) -> float:
    v1 = embedder(text1)
    v2 = embedder(text2)
    return cosine_similarity(v1, v2)

def judge_prompt(pred: str, ref: str) -> str:
    return("You are a strict evaluator. Compare the candidate answer to the reference answer.\n"
           "Scoring rubric:\n"
           "- 10: Fully correct, complete, faithful and concise.\n"
           "- 8: Mostly correct with minor issues: faithful to reference.\n"
           "- 5: Partially correct, noticeable omissions or inaccuracies.\n"
           "- 2: Mostly incorrect, significant issues.\n"
           "- 0: Unrelated or unusable.\n\n"
           "Instructions:\n"
           "- Focus on correctness and faithfulness to the reference.\n"
           "- Ignore stylistic differences unless they harm fidelity.\n"
           "- Output a single numeric score from 0 to 10. Optionally include one short justification line.\n\n"
           f"Reference Answer:\n{ref}\n\n"
           f"Candidate Answer:\n{pred}\n\n"
           "Final Score: (0-10):")

_NUM_RE = re.compile(r"""
    (?P<num>
        (?:\d+(?:\.\d+)?\s*%)                           # 80%
        |
        (?:\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?)           # 7/10, 85 / 100
        |
        (?:\d+(?:\.\d+)?)                               # 0.8, 7.5, 9, 100
    )
""", re.VERBOSE)

def parse_scalar_score(text: str) -> Optional[float]:
    """
    Parse a numeric score and normalize to 0..1. Supports:
    '7', '7.5', '7/10', '3/5', '0.8', 'Score: 85 / 100'
    """
    s = (text or "").strip()
    m = _NUM_RE.search(s)
    if m:
        token = m.group("num")
        if "%" in token:
            try:
                return max(0.0, min(1.0, float(token.replace("%", "")) / 100.0))
            except Exception:
                return None
        if "/" in token:
            try:
                left, right = token.split("/", 1)
                num = float(left.strip())
                den = float(right.strip())
                if den <= 0:
                    return None
                return max(0.0, min(1.0, num / den))
            except Exception:
                return None
        try:
            v = float(token)
            if 0.0 <= v <= 1.0:
                return v
            if 0.0 <= v <= 10.0:
                return v / 10.0
        except Exception:
            return None
        return None
    
def llm_judge_score(pred: str, ref: str, judge_fn: Callable[[str], str]) -> Dict[str, Any]:
    prompt = judge_prompt(pred, ref)
    raw = (judge_fn(prompt) or "").strip()
    score = parse_scalar_score(raw)
    return {"score": score, "raw": raw}






