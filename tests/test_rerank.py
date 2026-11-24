"""Rerank op tests — no torch; run from the repo root with the search
venv (python -m pytest tests/). Covers the pure pieces: passage-based
pair building and the rerank op end-to-end with a faked cross-encoder
and an in-memory soil DB."""
import json
import sqlite3

import numpy as np
import pytest

import reranker
from embed_daemon import build_pairs, rerank


@pytest.fixture
def soil(tmp_path):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE item (
        uuid TEXT PRIMARY KEY, _type TEXT, data TEXT,
        metadata TEXT, superseded_by TEXT)""")
    rows = [
        ("itemA", "Email", {"subject": "bill",
                            "description": "october total due <b>html</b>"}),
        ("itemB", "Note", {"title": "note", "description": "run log entry"}),
        ("itemC", "Attachment", {"filename": "receipt.pdf",
                                 "description": "aws charges"}),
        ("gone", "Email", {"subject": "deleted", "description": "body"}),
    ]
    for u, t, d in rows:
        conn.execute("INSERT INTO item VALUES (?,?,?,?,NULL)",
                     (u, t, json.dumps(d), "{}"))
    conn.execute("UPDATE item SET superseded_by='itemA' WHERE uuid='gone'")
    return conn


def test_build_pairs_dedup_and_skip():
    texts = {"a": "t-a", "b": "", "c": "t-c"}
    kept, pairs = build_pairs(["a", "a", "b", "c", "missing"], texts)
    assert kept == ["a", "c"]
    assert pairs == ["t-a", "t-c"]


def test_rerank_op_orders_by_logits(soil, monkeypatch):
    seen_pairs = []

    def fake_score(pairs):
        seen_pairs.extend(pairs)
        return [5.0 if "october total due" in p else -1.0
                for _, p in pairs]

    monkeypatch.setattr(reranker, "score_pairs", fake_score)
    req = {"q": "bill", "uuids": ["itemA", "gone", "itemC", "nope"]}
    out = rerank(req, soil, idx=None, embed_query=None)
    got = [u for u, _ in out["results"]]
    assert got == ["itemA", "itemC"]          # gone superseded, nope absent
    assert out["results"][0] == ["itemA", 5.0]  # best-first
    # passage view: body text present, html stripped, title prepended
    pair_texts = [p for _, p in seen_pairs]
    assert "october total due" in pair_texts[0]
    assert "<b>" not in pair_texts[0]
    assert "bill" in pair_texts[0]
    assert "receipt.pdf" in pair_texts[1]


def test_rerank_op_empty_and_cap(soil, monkeypatch):
    monkeypatch.setattr(reranker, "score_pairs", lambda pairs: [0.0] * len(pairs))
    assert rerank({"q": "x", "uuids": []}, soil, None, None) == {"results": []}
    assert rerank({"q": "", "uuids": ["itemA"]}, soil, None, None) == {"results": []}
    # duplicates are deduped BEFORE the cap (they must not crowd out
    # unique uuids) and the cap truncates head-first
    many = (["itemA"] * 10
            + [f"u{i:03d}" for i in range(reranker.MAX_PAIRS + 5)])
    out = rerank({"q": "q", "uuids": many}, soil, None, None)
    scored = [u for u, _ in out["results"]]
    assert len(scored) <= reranker.MAX_PAIRS
    assert len(set(scored)) == len(scored)          # no duplicates scored


def test_score_pairs_batch_boundary():
    """>BATCH pairs stay aligned across batch splits (fake model returns
    positional logits; misalignment would shuffle scores)."""
    class FakeLogits:
        def __init__(self, seq):
            self._seq = seq
        def squeeze(self, _):
            return self
        def tolist(self):
            return self._seq
    class FakeOut:
        def __init__(self, seq):
            self.logits = FakeLogits(seq)
    class FakeModel:
        def __init__(self):
            self.seen = 0
        def __call__(self, **enc):
            b = len(enc["pairs"])
            seq = list(range(self.seen, self.seen + b))
            self.seen += b
            return FakeOut(seq)
    class FakeTok:
        def __call__(self, pairs, **kw):
            return {"pairs": pairs}
    model = FakeModel()
    reranker._seed(FakeTok(), model)
    try:
        n = reranker.BATCH + 3
        out = reranker.score_pairs([("q", f"p{i}") for i in range(n)])
        assert out == list(range(n))       # identity across both batches
        assert model.seen == n
    finally:
        reranker._state.clear()


def test_score_pairs_lazy_and_seeded():
    # no model loaded, no pairs -> [] without touching torch
    assert reranker.score_pairs([]) == []
    # _seed path: fake tok/model chain (squeeze(-1).tolist())
    class FakeLogits:
        def squeeze(self, _):
            return self
        def tolist(self):
            return [1.5, -2.0]
    class FakeOut:
        logits = FakeLogits()
    class FakeModel:
        def __call__(self, **enc):
            return FakeOut()
    class FakeTok:
        def __call__(self, pairs, **kw):
            return {}
    reranker._seed(FakeTok(), FakeModel())
    try:
        assert reranker.score_pairs([("q", "p1"), ("q", "p2")]) == [1.5, -2.0]
    finally:
        reranker._state.clear()


def test_passage_for_gives_document_view():
    from build_search_index import passage_for
    d = {"title": "Your invoice", "description":
         "<!doctype html><html><body><p>Lazada invoice for order 123</p>"
         + "<p>filler</p>" * 400 + "</body></html>"}
    p = passage_for(d, "Email")
    assert p.startswith("Your invoice — Lazada invoice for order 123")
    assert "<p>" not in p
    assert len(p) <= 1700
    assert passage_for({"title": "only"}, "Note") == "only"
