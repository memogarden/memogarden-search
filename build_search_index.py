#!/usr/bin/env python3
"""Build the corpus-wide CHUNKED embedding index for /items/search (api#95 follow-up).

Search-quality items 1+2 (operator decision 2026-11-24):
  1. CLS pooling — BAAI/bge-small-en-v1.5 is CLS-pool-trained; the old masked
     mean-pooling degraded retrieval. Query side additionally gets the bge v1.5
     passage-retrieval instruction prefix (embed_daemon.py, not here).
  2. Chunked bodies — ~600-char whitespace-aligned windows, 75-char overlap,
     title prepended to every chunk, <=12 chunks/item (replaces the single
     1200-char cap that truncated 61.5% of email bodies; median 1915, p90 16.7k).

Embeds every live text-bearing fact (Email, Attachment, Note, Entry, Event) and
writes (chunk_id = f"{uuid}#{i}", rows grouped per uuid in item order):
    cache/search_index.npy       vecs: (N,384) float32, mmap-able — live items
    cache/search_index_ids.json  [chunk_id, ...] aligned with vecs rows
    cache/search_meta.json       uuid -> {type, scopes, canonical_at, title, n_chunks}
    cache/index_state.json       {"watermark": change_log seq the index reflects}
    cache/search_emb.npz         vecs: chunk-level embedding cache (all items)
    cache/search_emb_ids.json    [chunk_id, ...] aligned with cache rows
    cache/search_emb_meta.json   chunk_id -> sha256(chunk_text)[:16]

Cache is chunk-granular and hash-keyed: unchanged text re-embeds nothing, and
the build is resumable (periodic saves). After a build, the daemon's
catch-up-on-wake only embeds items whose change_log seq postdates the
watermark (see index_store.py). Legacy npz search_index.npz is removed on
write; pre-chunked mean-pooled caches were never reusable anyway.

Run: HF_HOME=$PWD/hf-cache .venv/bin/python build_search_index.py
"""
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
DB = Path(os.environ.get(
    "MEMOGARDEN_SOIL_DB",
    str(Path.home() / ".local/share/memogarden/soil.db")))
CACHE = Path(os.environ.get(
    "MEMOGARDEN_SEARCH_CACHE", str(HERE / "cache")))
VEC_NPY = CACHE / "search_index.npy"     # mmap-able index (0-load at wake)
LEGACY_NPZ = CACHE / "search_index.npz"  # pre-mmap format, removed on write
INDEX_IDS = CACHE / "search_index_ids.json"
INDEX_META = CACHE / "search_meta.json"
EMB_CACHE = CACHE / "search_emb.npz"
EMB_IDS = CACHE / "search_emb_ids.json"
EMB_META = CACHE / "search_emb_meta.json"
REPO = "BAAI/bge-small-en-v1.5"
TYPES = ("Email", "Attachment", "Note", "Entry", "Event")
TAG_RE = re.compile(r"<[^>]+>")
BLOCK_RE = re.compile(r"<(style|script)\b[^>]*>.*?</\1\s*>",
                      re.IGNORECASE | re.DOTALL)
WS_RE = re.compile(r"\s+")

import index_store  # noqa: E402  (after sys.path insert)

CHUNK_CHARS = 600
CHUNK_OVERLAP = 75
MAX_CHUNKS = 12          # ~7.2k chars/item coverage — full for ~p90 email bodies
TITLE_CAP = 200
EMB_BATCH = 96
SAVE_EVERY_BATCHES = 40  # resume point cost: one stacked-npz rewrite (~300MB)


def load_bge():
    """Shared model loader (builder + daemon catch-up)."""
    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModel.from_pretrained(REPO).eval()
    return tok, model


@torch.no_grad()
def embed_texts(texts, tok, model):
    """CLS-pooled + L2-normalized embeddings, batched (bge training pooling)."""
    outs = []
    for i in range(0, len(texts), EMB_BATCH):
        enc = tok(texts[i:i + EMB_BATCH], padding=True, truncation=True,
                  max_length=512, return_tensors="pt")
        o = model(**enc)
        e = o.last_hidden_state[:, 0]
        outs.append(torch.nn.functional.normalize(e, dim=-1))
    return torch.cat(outs).numpy().astype(np.float32)


def chunk_text(text):
    """Whitespace-aligned ~600-char windows with 75-char overlap, capped."""
    if not text:
        return []
    if len(text) <= CHUNK_CHARS:
        return [text]
    step = CHUNK_CHARS - CHUNK_OVERLAP
    out, start = [], 0
    while start < len(text) and len(out) < MAX_CHUNKS:
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + (CHUNK_CHARS // 2), end)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, 0)
        while start < len(text) and text[start] == " ":
            start += 1
    return out


def _title_body(d, _type):
    """(clean title, clean body) shared by the chunker and the rerank
    passage builder — tag-strip + whitespace collapse + title cap, with
    <style>/<script> BLOCK CONTENT dropped before tag-stripping.

    Style-strip is unconditional as of 2026-11-25 (operator approved the
    one-off cache invalidation): CSS pollution made Lazada-invoice chunks
    ~pure stylesheet text (replay finding), degrading every embedding
    built from them. Changing chunk text invalidates the hash-keyed
    embedding cache wholesale → a full multi-hour re-embed, run once with
    the daemon socket + ingest timer paused so no wake races the write.
    """
    if _type == "Attachment":
        title, body = d.get("filename", "") or d.get("title", ""), d.get("description", "")
    else:
        title, body = (d.get("title", "") or d.get("subject", "")), \
                      (d.get("description", "") or d.get("summary", ""))
    if body and "<" in body and ">" in body:
        body = BLOCK_RE.sub(" ", body)
        body = TAG_RE.sub(" ", body)
    body = WS_RE.sub(" ", body or "").strip()
    title = WS_RE.sub(" ", title or "").strip()[:TITLE_CAP]
    return title, body


def chunk_texts_for(d, _type):
    """Title-prepended chunks for one item's data dict."""
    title, body = _title_body(d, _type)
    if not body:
        return [title] if title else []
    pieces = chunk_text(body)
    if title:
        pieces = [f"{title} — {p}" for p in pieces]
    return pieces


def passage_for(d, _type, max_chars=1700):
    """One document-level passage: title + tag-stripped body prefix.

    The reranker's input — unlike the chunked embedder it sees query and
    document JOINTLY, so it wants the widest document view that fits its
    512-token window (~1700 chars) rather than a single chunk. Brand,
    sender and period signals ("Lazada", "Nov 2026") that live near the
    top of the body land inside it; the tokenizer clamps overflow.
    style/script block content is dropped (CSS turned Lazada-invoice
    passages into stylesheet noise — 2026-11-25 replay finding; the
    chunker strips blocks too now, so chunks and passages agree).
    """
    title, body = _title_body(d, _type)
    if not body:
        return title
    if not title:
        return body[:max_chars]
    return f"{title} — {body[:max_chars - len(title) - 3]}"


def main():
    CACHE.mkdir(exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute(
        f"""SELECT uuid, _type, canonical_at, data, metadata FROM item
            WHERE superseded_by IS NULL AND _type IN {TYPES}"""
    ).fetchall()

    # pass 1 — chunk texts, hashes, per-item meta (chunk ids grouped per uuid)
    items = {}
    chunk_texts, chunk_hashes = {}, {}
    for uuid, _type, cat, data, metadata in rows:
        d = json.loads(data)
        m = json.loads(metadata or "{}")
        pieces = chunk_texts_for(d, _type)
        items[uuid] = {
            "type": _type,
            "scopes": m.get("scopes", []),
            "canonical_at": cat or "",
            "title": (d.get("title") or d.get("subject")
                      or d.get("filename") or "")[:160],
            "n_chunks": len(pieces),
        }
        for i, p in enumerate(pieces):
            cid = f"{uuid}#{i}"
            chunk_texts[cid] = p
            chunk_hashes[cid] = hashlib.sha256(p.encode()).hexdigest()[:16]
    n_items = sum(1 for it in items.values() if it["n_chunks"])
    print(f"live text-bearing facts: {len(items)} ({n_items} with text), "
          f"chunks: {len(chunk_texts)}", flush=True)

    tok, model = load_bge()

    # chunk-level cache: chunk_id -> vec, valid while text hash matches
    emb_vecs, emb_hash = {}, {}
    if EMB_CACHE.exists() and EMB_IDS.exists() and EMB_META.exists():
        z = np.load(EMB_CACHE)["vecs"]
        ids = json.loads(EMB_IDS.read_text())
        emb_hash = json.loads(EMB_META.read_text())
        for k, cid in enumerate(ids):
            emb_vecs[cid] = z[k]
        print(f"chunk cache: {len(emb_vecs)} loaded", flush=True)
    todo = [c for c in chunk_texts
            if emb_hash.get(c) != chunk_hashes[c] or c not in emb_vecs]
    print(f"embeddings: {len(emb_vecs)} cached, {len(todo)} to embed", flush=True)

    t0 = time.time()
    batch_no = 0
    for i in range(0, len(todo), EMB_BATCH):
        batch = todo[i:i + EMB_BATCH]
        E = embed_texts([chunk_texts[c] for c in batch], tok, model)
        for cid, e in zip(batch, E):
            emb_vecs[cid] = e
            emb_hash[cid] = chunk_hashes[cid]
        batch_no += 1
        if batch_no % SAVE_EVERY_BATCHES == 0:
            save_emb(emb_vecs, emb_hash)
            print(f"  {i + len(batch)}/{len(todo)} "
                  f"({time.time() - t0:.0f}s) — saved", flush=True)
    save_emb(emb_vecs, emb_hash)
    EMB_META.write_text(json.dumps(emb_hash))
    print(f"embedded {len(todo)} chunks ({time.time() - t0:.0f}s)", flush=True)

    # index: only live items' chunks, grouped per uuid in items order
    idx_ids, idx_rows = [], []
    for uuid, it in items.items():
        for i in range(it["n_chunks"]):
            cid = f"{uuid}#{i}"
            if cid in emb_vecs:
                idx_ids.append(cid)
                idx_rows.append(emb_vecs[cid])
    vecs = np.empty((len(idx_rows), 384), dtype=np.float32)
    for k, r in enumerate(idx_rows):
        vecs[k] = r
    np.save(VEC_NPY, vecs)                    # mmap-able (0-cost daemon load)
    if LEGACY_NPZ.exists():
        LEGACY_NPZ.unlink()
    INDEX_IDS.write_text(json.dumps(idx_ids))
    INDEX_META.write_text(json.dumps(items))
    index_store.write_watermark(
        CACHE, index_store.max_change_seq(con),
        extra={"index_chunks": len(idx_ids)})
    print(f"index written: {len(idx_ids)} chunks over "
          f"{sum(1 for u in items if items[u]['n_chunks'])} items "
          f"({time.time() - t0:.0f}s total)", flush=True)


def save_emb(emb_vecs, emb_hash):
    vecs = np.empty((len(emb_vecs), 384), dtype=np.float32)
    for k, cid in enumerate(emb_vecs):
        vecs[k] = emb_vecs[cid]
    np.savez_compressed(EMB_CACHE, vecs=vecs)
    EMB_IDS.write_text(json.dumps(list(emb_vecs)))


if __name__ == "__main__":
    main()
