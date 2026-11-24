#!/usr/bin/env python3
"""Lazy embed/search daemon for the operator console's semantic search.

Activated ON DEMAND — by systemd socket activation (deploy/
memogarden-embed.socket: systemd owns the listening socket and starts
this daemon on first connect) or, when the units aren't installed, by
memogarden-api's lazy Popen spawn fallback. Either way the architecture
decision stands (2026-11-22): query-time inference never lives in the
serving process. Loads bge-small + the CHUNKED search index once, then
answers search requests over a unix socket owned by the operator (0600),
idle self-exits after 600s (zero resident RSS between searches).

Catch-up-on-wake (api#95 phase 2): on every start the daemon reads the
trigger-maintained change_log (system#12) and re-indexes items whose seq
postdates the stored watermark — new/changed items embed (chunk-level
hash cache makes unchanged text free), superseded/gone items leave the
index — bounded per wake, watermark persisted per pass.

Queries are CLS-pooled (matching the index rebuild) and prefixed with
the bge v1.5 passage-retrieval instruction; scoring is best-passage per
item over ~600-char chunks (the old single 1200-char embedding
truncated 61.5% of email bodies).

Protocol: one JSON request per connection.
  {"op": "ping"}                                  -> {"ok": true}
  {"op": "search", "q": str, "scopes": [uuid]|null,
   "types": [str]|null, "k": int}                 -> {"results": [[uuid, score], …]}
Exit after IDLE_SECONDS without a request (memory returns to zero; the
next search simply cold-starts again).
"""
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
SOCK = Path.home() / ".local/share/memogarden/triage-model/embed.sock"
CACHE_DIR = Path(os.environ.get(
    "MEMOGARDEN_SEARCH_CACHE", str(HERE / "cache")))
IDLE_SECONDS = 600
REPO = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CATCHUP_LIMIT = 300       # items per wake; remainder re-plans next wake
CATCHUP_MAX_PASSES = 50

import sqlite3  # noqa: E402  (after sys.path insert)

import index_store  # noqa: E402
import build_search_index as bsi  # noqa: E402  (chunking + embed helpers)

EMB_CACHE = CACHE_DIR / "search_emb.npz"
EMB_IDS = CACHE_DIR / "search_emb_ids.json"
EMB_META = CACHE_DIR / "search_emb_meta.json"


def _soil_conn():
    conn = sqlite3.connect(f"file:{bsi.DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_emb_cache():
    """Chunk-level embedding cache (chunk_id -> vec, hash-validated)."""
    if EMB_CACHE.exists() and EMB_IDS.exists() and EMB_META.exists():
        z = np.load(EMB_CACHE)["vecs"]
        ids = json.loads(EMB_IDS.read_text())
        emb_hash = json.loads(EMB_META.read_text())
        return {cid: z[k] for k, cid in enumerate(ids)}, emb_hash
    return {}, {}


def save_emb_cache(emb_vecs, emb_hash):
    vecs = np.empty((len(emb_vecs), 384), dtype=np.float32)
    for k, cid in enumerate(emb_vecs):
        vecs[k] = emb_vecs[cid]
    np.savez_compressed(EMB_CACHE, vecs=vecs)
    EMB_IDS.write_text(json.dumps(list(emb_vecs)))
    EMB_META.write_text(json.dumps(emb_hash))


def chunk_texts_for_row(row):
    """Title-prepended chunk texts for a live item row."""
    d = json.loads(row["data"]) if row["data"] else {}
    return bsi.chunk_texts_for(d, row["_type"])


def catch_up(conn, idx, tok, model):
    """Re-index items the change_log says changed since the watermark.

    Bounded per wake (CATCHUP_LIMIT items, CATCHUP_MAX_PASSES); the
    watermark persists per pass, so an interrupted catch-up resumes.
    Returns a summary string.
    """
    wm = index_store.read_watermark(CACHE_DIR)
    top = index_store.max_change_seq(conn)
    if top <= wm:
        return f"catch-up: index current at change_log seq {wm}"
    emb_vecs, emb_hash = load_emb_cache()
    t0 = time.time()
    n_up, n_rm, n_emb = 0, 0, 0
    passes = 0
    while passes < CATCHUP_MAX_PASSES:
        upserts, removes, new_wm = index_store.plan_from_change_log(
            conn, index_store.read_watermark(CACHE_DIR),
            limit=CATCHUP_LIMIT, live_types=bsi.TYPES)
        if not upserts and not removes:
            break
        for u, row in upserts:
            texts = chunk_texts_for_row(row)
            hashes = [hashlib.sha256(t.encode()).hexdigest()[:16]
                      for t in texts]
            todo = [i for i, h in enumerate(hashes)
                    if emb_hash.get(f"{u}#{i}") != h]
            missing = [texts[i] for i in todo]
            if missing:
                E = bsi.embed_texts(missing, tok, model)
                for k, i in enumerate(todo):
                    emb_vecs[f"{u}#{i}"] = E[k]
                    emb_hash[f"{u}#{i}"] = hashes[i]
                    n_emb += 1
            d = json.loads(row["data"]) if row["data"] else {}
            idx.upsert(u, texts,
                       [emb_vecs[f"{u}#{i}"] for i in range(len(texts))],
                       {
                           "type": row["_type"],
                           "scopes": (json.loads(row["metadata"] or "{}")
                                      or {}).get("scopes", []),
                           "canonical_at": row["canonical_at"] or "",
                           "title": (d.get("title") or d.get("subject")
                                     or d.get("filename") or "")[:160],
                       })
            n_up += 1
        if removes:
            idx.remove(removes)
            n_rm += len(removes)
        idx.save(CACHE_DIR, watermark=new_wm)
        save_emb_cache(emb_vecs, emb_hash)
        passes += 1
        if len(upserts) + len(removes) < CATCHUP_LIMIT:
            break
    return (f"catch-up: {passes} pass(es), {n_up} item(s) re-indexed, "
            f"{n_rm} removed, {n_emb} embedded, "
            f"watermark -> {index_store.read_watermark(CACHE_DIR)} "
            f"({time.time() - t0:.1f}s)")


def main():
    import torch

    try:
        idx = index_store.VectorIndex.load(CACHE_DIR)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"search index missing/broken — run build_search_index.py "
              f"first ({exc})", file=sys.stderr)
        sys.exit(2)

    tok, model = bsi.load_bge()
    conn = _soil_conn()
    print(catch_up(conn, idx, tok, model), flush=True)

    # systemd socket activation vs self-bind (see deploy/ units)
    listen_fds = int(os.environ.get("LISTEN_FDS", "0") or 0)
    if listen_fds >= 1:
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, fileno=3)
        srv.settimeout(IDLE_SECONDS)
        print("listening on systemd-passed socket (fd 3)", flush=True)
    else:
        if SOCK.exists():
            SOCK.unlink()
        SOCK.parent.mkdir(parents=True, exist_ok=True)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(SOCK))
        os.chmod(SOCK, 0o600)
        srv.listen(4)
        srv.settimeout(IDLE_SECONDS)
        print(f"listening on {SOCK}", flush=True)

    @torch.no_grad()
    def embed_query(text):
        enc = tok([QUERY_PREFIX + text[:1200]], truncation=True,
                  max_length=512, return_tensors="pt")
        o = model(**enc)
        e = o.last_hidden_state[:, 0]          # CLS pooling, matches the index
        return torch.nn.functional.normalize(e, dim=-1).numpy().astype(
            np.float32)[0]

    def search(req):
        qv = embed_query(req.get("q", ""))
        k = max(1, min(int(req.get("k", 50)), 500))
        scopes = req.get("scopes") or None
        want_types = set(req.get("types") or [])
        if not len(idx.item_ids):
            return {"results": [], "masked_total": 0}
        # type/scope are item-level filters → aggregate first, mask after
        best = np.maximum.reduceat(idx.vecs @ qv, idx.starts)
        mask = np.zeros(len(idx.item_ids), dtype=bool)
        n_scoped = 0
        for j in range(len(idx.item_ids)):
            item = idx.meta[idx.item_ids[j]]
            if want_types and item["type"] not in want_types:
                continue
            if scopes and not set(item.get("scopes", [])) & set(scopes):
                continue
            n_scoped += 1
            mask[j] = True
        hits = []
        idxs = np.flatnonzero(mask)
        if idxs.size:
            sub = np.argsort(-best[idxs])[:k]
            hits = [[idx.item_ids[j], round(float(best[j]), 4)]
                    for j in idxs[sub]]
        return {"results": hits, "masked_total": int(n_scoped)}

    while True:
        try:
            conn_srv, _ = srv.accept()
        except socket.timeout:
            print(f"idle {IDLE_SECONDS}s — exiting", flush=True)
            break
        try:
            req = json.loads(conn_srv.recv(1 << 22).decode())
            op = req.get("op")
            if op == "ping":
                out = {"ok": True}
            elif op == "search":
                out = search(req)
            else:
                out = {"error": f"unknown op {op!r}"}
            conn_srv.sendall(json.dumps(out).encode())
        except Exception as e:  # noqa: BLE001 — daemon must survive bad reqs
            try:
                conn_srv.sendall(json.dumps({"error": str(e)}).encode())
            except OSError:
                pass
        finally:
            conn_srv.close()


if __name__ == "__main__":
    main()
