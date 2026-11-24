#!/usr/bin/env python3
"""Cross-encoder reranker (api#95 final slice — operator: build now).

Scores (query, passage) pairs with BAAI/bge-reranker-base, quantized to
int8 (torch.ao.quantization.quantize_dynamic — zero extra dependencies;
the repo venv carries torch+transformers already). Loaded LAZILY on the
first rerank request and sticky for the daemon lifetime — the wake path
(ping/search) stays model-light, and the 600s idle self-exit releases
everything, so the reranker is a transient resident like torch itself.

Budget facts (2026-11-25 measurement): daemon RSS 754MB with bge-small +
index; int8 reranker adds ~300-400MB → peak ~1.2-1.3G, clear of
MemoryHigh=2G. ~0.4-1s per 20-30 pairs on CPU. Upgrade path: swap REPO
to BAAI/bge-reranker-v2-m3 (same load path; +~400MB and 1.5-4s/query —
outside the accepted ~1s budget).

Memory safety: torch/transformers import INSIDE functions so this module
imports cleanly without torch (no-torch tests monkeypatch score_pairs or
_seed with fakes).
"""
import os

REPO = "BAAI/bge-reranker-base"
MAX_PAIRS = 64      # hard cap on pairs per request (api sends ≤30-ish)
MAX_LENGTH = 512    # tokenizer truncation per (q, passage) pair
BATCH = 8
THREADS = min(4, os.cpu_count() or 1)

_state = {}


def _load():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(THREADS)
    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()
    model = torch.ao.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8)
    return tok, model


def _seed(tok, model):
    """Test seam: install a (fake) tokenizer/model pair directly."""
    _state["tok"], _state["model"] = tok, model


def score_pairs(pairs):
    """Cross-encoder relevance logits, aligned with `pairs` (order kept).

    Raw logits (higher = more relevant) — ordering-only by contract; the
    api renders them as opaque scores.
    """
    if not pairs:
        return []
    tok, model = _state.get("tok"), _state.get("model")
    if tok is None:
        tok, model = _load()
        _state["tok"], _state["model"] = tok, model
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH):
            enc = tok(list(pairs[i:i + BATCH]), padding=True,
                      truncation=True, max_length=MAX_LENGTH,
                      return_tensors="pt")
            logits = model(**enc).logits.squeeze(-1)  # (B,) relevance
            out.extend(float(x) for x in logits.tolist())
    return out
