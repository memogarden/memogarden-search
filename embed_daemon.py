#!/usr/bin/env python3
"""Lazy embed/search daemon for the operator console's semantic search.

Spawned ON DEMAND by memogarden-api when /items/search runs in semantic
mode — never a systemd unit (architecture decision 2026-11-22: inference
stays out of the serving process; the daemon is a model-layer child with
idle self-exit). Loads bge-small + the CHUNKED search index once, then
answers search requests over a unix socket owned by the operator (0600
dir).

Search-quality items 1+2 (2026-11-24): queries are CLS-pooled (matching
the index rebuild) and prefixed with the bge v1.5 passage-retrieval
instruction; scoring is best-passage per item over ~600-char chunks
(the old single 1200-char embedding truncated 61.5% of email bodies).

Protocol: one JSON request per connection.
  {"op": "ping"}                                  -> {"ok": true}
  {"op": "search", "q": str, "scopes": [uuid]|null,
   "types": [str]|null, "k": int}                 -> {"results": [[uuid, score], …]}
Exit after IDLE_SECONDS without a request (memory returns to zero; the
next search simply cold-starts again).
"""
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
IDLE_SECONDS = 600
REPO = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer

    index_npz = HERE / "cache" / "search_index.npz"
    index_ids = HERE / "cache" / "search_index_ids.json"
    index_meta = HERE / "cache" / "search_meta.json"
    if not (index_npz.exists() and index_ids.exists()
            and index_meta.exists()):
        print("chunked search index missing — run build_search_index.py "
              "first", file=sys.stderr)
        sys.exit(2)

    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModel.from_pretrained(REPO).eval()
    vecs = np.load(index_npz)["vecs"]          # (N, 384) float32
    ids = json.loads(index_ids.read_text())
    meta = json.loads(index_meta.read_text())  # uuid -> {type, scopes, n_chunks, …}

    # chunk rows are grouped per uuid (builder guarantees it) — derive the
    # per-item start offsets for np.maximum.reduceat aggregation
    item_ids, starts, prev = [], [], object()
    for i, cid in enumerate(ids):
        u = cid.rsplit("#", 1)[0]
        if u != prev:
            item_ids.append(u)
            starts.append(i)
            prev = u
    if sum(meta[u]["n_chunks"] for u in item_ids) != len(ids):
        print("index ids/meta inconsistent — rebuild the index",
              file=sys.stderr)
        sys.exit(2)
    item_types = [meta[u]["type"] for u in item_ids]
    item_scopes = [set(meta[u]["scopes"]) for u in item_ids]
    starts_arr = np.array(starts, dtype=np.intp)
    print(f"daemon ready: {len(ids)} chunks over {len(item_ids)} items",
          flush=True)

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
        # type/scope are item-level filters → aggregate first, mask after
        best = np.maximum.reduceat(vecs @ qv, starts_arr)
        mask = np.zeros(len(item_ids), dtype=bool)
        n_scoped = 0
        for j in range(len(item_ids)):
            if want_types and item_types[j] not in want_types:
                continue
            if scopes:
                if not item_scopes[j].intersection(scopes):
                    continue
                n_scoped += 1
            else:
                n_scoped += 1
            mask[j] = True
        hits = []
        idxs = np.flatnonzero(mask)
        if idxs.size:
            sub = np.argsort(-best[idxs])[:k]
            hits = [[item_ids[j], round(float(best[j]), 4)]
                    for j in idxs[sub]]
        return {"results": hits, "masked_total": int(n_scoped)}

    while True:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            print(f"idle {IDLE_SECONDS}s — exiting", flush=True)
            break
        try:
            req = json.loads(conn.recv(1 << 22).decode())
            op = req.get("op")
            if op == "ping":
                out = {"ok": True}
            elif op == "search":
                out = search(req)
            else:
                out = {"error": f"unknown op {op!r}"}
            conn.sendall(json.dumps(out).encode())
        except Exception as e:  # noqa: BLE001 — daemon must survive bad reqs
            try:
                conn.sendall(json.dumps({"error": str(e)}).encode())
            except OSError:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
