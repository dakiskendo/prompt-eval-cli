from __future__ import annotations
import math
from src.evaluators import (bleu_score, cosine_similarity, exact_match, parse_scalar_score, text_embedding_similarity)

def test_exact_match_normalized():
    assert exact_match("Hello World", "hello world") == 1.0
    assert exact_match("Hello", "hello", normalize=False) == 0.0

def test_bleu():
    ref = "The quick brown fox jumps over the lazy dog."
    assert bleu_score(ref, ref) > 0.9
    assert bleu_score("Completely different text", ref) < 0.2

def test_cosine_sim():
    assert math.isclose(cosine_similarity([1,0], [1,0]), 1.0, rel_tol=1e-6)
    assert cosine_similarity([1,0], [0,1]) == 0.0

def test_embed_sim():
    def stub_embedder(s: str):
        return [len(s), 1.0]
    
    a = "abc"
    b = "abcd"
    sim = text_embedding_similarity(a, b, stub_embedder)
    assert 0.0 < sim < 1.0

def test_parse_scalar_score():
    assert parse_scalar_score("7/10") == 0.7
    assert parse_scalar_score("3/5") == 0.6
    assert parse_scalar_score("80%") == 0.8
    assert parse_scalar_score("0.8") == 0.8
    assert parse_scalar_score("Score: 9.5/10") == 0.95
    assert parse_scalar_score("Score: 85/100") == 0.85
    assert parse_scalar_score("N/A") is None
