"""Chunked vector index store + change_log catch-up planning (api#95).

The index is four files in the cache dir:
    search_index.npy      (N, 384) float32, rows grouped per item uuid
    search_index_ids.json ["uuid#i", ...] aligned with rows
    search_meta.json      uuid -> {type, scopes, canonical_at, title, n_chunks}
    index_state.json      {"watermark": change_log seq the index reflects}

The watermark is the catch-up high-water mark: the daemon reads the
trigger-maintained change_log (system#12) on wake and re-indexes only
what changed since it. Pure numpy/json/sqlite3 — no torch — so tests run
without the model venv.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np

STATE_FILE = "index_state.json"
VEC_FILE = "search_index.npy"
IDS_FILE = "search_index_ids.json"
META_FILE = "search_meta.json"


def read_watermark(cache_dir) -> int:
    p = Path(cache_dir) / STATE_FILE
    if not p.exists():
        return 0
    try:
        return int(json.loads(p.read_text()).get("watermark", 0))
    except (json.JSONDecodeError, ValueError, OSError):
        return 0


def write_watermark(cache_dir, watermark, extra=None):
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    doc = {"watermark": int(watermark)}
    if extra:
        doc.update(extra)
    (p / STATE_FILE).write_text(json.dumps(doc))


class VectorIndex:
    """In-memory view of the chunked index; chunk rows are grouped per
    item uuid (the builder and upsert guarantee it), which is what makes
    np.maximum.reduceat best-passage scoring possible."""

    def __init__(self, vecs, ids, meta):
        self.vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        self.ids = list(ids)
        self.meta = dict(meta)
        self._validate_grouping()

    def _validate_grouping(self):
        if len(self.vecs) != len(self.ids):
            raise ValueError(
                f"index rows/ids inconsistent: {len(self.vecs)} vector "
                f"rows vs {len(self.ids)} chunk ids")
        item_ids, starts, prev = [], [], object()
        for i, cid in enumerate(self.ids):
            u = cid.rsplit("#", 1)[0]
            if u != prev:
                item_ids.append(u)
                starts.append(i)
                prev = u
        total = sum(self.meta[u]["n_chunks"] for u in item_ids
                    if u in self.meta)
        if total != len(self.ids):
            raise ValueError(
                f"index ids/meta inconsistent: {total} chunks in ids "
                f"vs {len(self.ids)} rows")
        self.item_ids = item_ids
        self.starts = np.array(starts, dtype=np.intp)

    @classmethod
    def load(cls, cache_dir):
        cache_dir = Path(cache_dir)
        npy = cache_dir / VEC_FILE
        if npy.exists():
            vecs = np.load(npy, mmap_mode="r")
        else:  # pre-mmap legacy format
            vecs = np.load(cache_dir / "search_index.npz")["vecs"]
        ids = json.loads((cache_dir / IDS_FILE).read_text())
        meta = json.loads((cache_dir / META_FILE).read_text())
        return cls(vecs, ids, meta)

    def save(self, cache_dir, watermark=None, extra_state=None):
        """Atomic save: one partial-write failure (suspend, OOM-kill,
        transient I/O fault) must never leave a truncated index live —
        every file lands via tmp + os.replace."""
        import os
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / (VEC_FILE + ".tmp")
        with open(tmp, "wb") as f:  # file handle: np.save won't append .npy
            np.save(f, np.asarray(self.vecs, dtype=np.float32))
        os.replace(tmp, cache_dir / VEC_FILE)
        for name, payload in ((IDS_FILE, json.dumps(self.ids)),
                              (META_FILE, json.dumps(self.meta))):
            tmp = cache_dir / (name + ".tmp")
            tmp.write_text(payload)
            os.replace(tmp, cache_dir / name)
        if watermark is not None:
            write_watermark(cache_dir, watermark, extra_state)

    def _drop_rows(self, uuids):
        drop = set(uuids)
        if not drop:
            return
        keep = np.array([cid.rsplit("#", 1)[0] not in drop
                         for cid in self.ids])
        if keep.all():
            return
        self.vecs = np.ascontiguousarray(self.vecs[keep])
        self.ids = [c for c, k in zip(self.ids, keep) if k]

    def upsert(self, uuid, chunk_texts, vecs_new, item_meta):
        """Replace one item's chunks and metadata (chunk count may change)."""
        self._drop_rows([uuid])
        if chunk_texts:
            self.vecs = np.vstack([
                self.vecs,
                np.ascontiguousarray(vecs_new, dtype=np.float32)])
            self.ids.extend(f"{uuid}#{i}" for i in range(len(chunk_texts)))
        meta = dict(item_meta)
        meta["n_chunks"] = len(chunk_texts)
        self.meta[uuid] = meta
        self._validate_grouping()

    def remove(self, uuids):
        self._drop_rows(uuids)
        for u in uuids:
            self.meta.pop(u, None)
        self._validate_grouping()


def plan_from_change_log(conn, watermark, limit=300, live_types=None):
    """Distinct item uuids touched since `watermark`, bounded to `limit`.

    `conn` must use sqlite3.Row rows. Returns (upserts, removes,
    new_watermark): upserts = [(uuid, row)] for live rows (row carries
    uuid/_type/canonical_at/data/metadata), removes = [uuid] for rows
    now superseded or gone. new_watermark advances only past fully
    planned seqs — the bounded remainder re-plans on the next wake.

    `change_log.uuid` is `item.uuid` verbatim — `soil_`-prefixed, the
    same form the index keys use — so it is used as stored. (Until
    2026-11-26 the `soil_` prefix was stripped here on the false premise
    that `item.uuid` is plain: every lookup missed, every change looked
    like a removal of an id the index did not have, and the watermark
    advanced anyway — the catch-up has been a no-op since it shipped.)
    """
    rows = conn.execute(
        "SELECT seq, uuid FROM change_log WHERE tbl = 'item' AND seq > ? "
        "ORDER BY seq", (watermark,)).fetchall()
    ordered, seen, new_wm = [], set(), watermark
    for seq, item_uuid in rows:
        u = item_uuid
        if u not in seen:
            if len(ordered) >= limit:
                break
            seen.add(u)
            ordered.append(u)
        new_wm = seq
    if not ordered:
        return [], [], watermark
    marks = ",".join("?" * len(ordered))
    live = {}
    for r in conn.execute(
            f"""SELECT uuid, _type, canonical_at, data, metadata FROM item
                WHERE uuid IN ({marks}) AND superseded_by IS NULL""",
            ordered):
        if live_types is None or r["_type"] in live_types:
            live[r["uuid"]] = r
    upserts, removes = [], []
    for u in ordered:
        (upserts.append((u, live[u])) if u in live
         else removes.append(u))
    return upserts, removes, new_wm


def max_change_seq(conn) -> int:
    return conn.execute(
        "SELECT COALESCE(max(seq), 0) FROM change_log").fetchone()[0]
