"""index_store tests — no torch; run with any python that has numpy+pytest
(e.g. the memogarden-api venv). Covers grouping invariants, upsert/remove,
save/load round-trip, watermark state, and change_log planning semantics
(prefixed uuids, bounds, live vs superseded classification)."""
import json
import sqlite3

import numpy as np
import pytest

from index_store import (VectorIndex, plan_from_change_log, read_watermark,
                         write_watermark, max_change_seq)


def make_index():
    # item A: 2 chunks; item B: 3 chunks (grouped per uuid)
    vecs = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
    ids = [f"itemA#{i}" for i in range(2)] + [f"itemB#{i}" for i in range(3)]
    meta = {
        "itemA": {"type": "Email", "scopes": ["s1"], "n_chunks": 2},
        "itemB": {"type": "Note", "scopes": [], "n_chunks": 3},
    }
    return VectorIndex(vecs, ids, meta)


def test_grouping_offsets():
    idx = make_index()
    assert idx.item_ids == ["itemA", "itemB"]
    assert idx.starts.tolist() == [0, 2]


def test_grouping_inconsistency_rejected():
    ids = ["itemA#0", "itemA#1"]
    meta = {"itemA": {"n_chunks": 3}}
    with pytest.raises(ValueError):
        VectorIndex(np.zeros((2, 4), dtype=np.float32), ids, meta)


def test_upsert_changes_chunk_count():
    idx = make_index()
    new_vecs = np.full((3, 4), 99.0, dtype=np.float32)
    idx.upsert("itemA", ["t0", "t1", "t2"], new_vecs,
               {"type": "Email", "scopes": ["s2"]})
    assert idx.meta["itemA"]["n_chunks"] == 3
    # upsert drops the old group and APPENDS the new one — per-uuid
    # contiguity is the invariant, global order is not
    assert idx.ids == ["itemB#0", "itemB#1", "itemB#2",
                       "itemA#0", "itemA#1", "itemA#2"]
    assert idx.starts.tolist() == [0, 3]
    assert (idx.vecs[3:] == 99.0).all()
    assert idx.item_ids == ["itemB", "itemA"]


def test_remove_drops_rows_and_meta():
    idx = make_index()
    idx.remove(["itemA"])
    assert idx.item_ids == ["itemB"]
    assert len(idx.ids) == 3
    assert "itemA" not in idx.meta
    # draining the index entirely is valid (0 rows == 0 ids)
    idx.remove(["itemB"])
    assert idx.item_ids == [] and len(idx.ids) == 0


def test_save_load_roundtrip(tmp_path):
    idx = make_index()
    idx.save(tmp_path, watermark=42, extra_state={"index_chunks": 10})
    loaded = VectorIndex.load(tmp_path)
    assert loaded.item_ids == idx.item_ids
    assert (loaded.vecs == idx.vecs).all()
    assert read_watermark(tmp_path) == 42


def test_watermark_defaults_to_zero(tmp_path):
    assert read_watermark(tmp_path) == 0


@pytest.fixture
def change_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE change_log (seq INTEGER PRIMARY KEY, "
                 "tbl TEXT, uuid TEXT, kind TEXT, changed_at TEXT)")
    conn.execute("""CREATE TABLE item (uuid TEXT PRIMARY KEY, _type TEXT,
                    canonical_at TEXT, data TEXT, metadata TEXT,
                    superseded_by TEXT)""")
    rows = [
        (1, "item", "soil_aaaa-1", "insert", "2026-11-24T10:00:00Z"),
        (2, "item", "soil_bbbb-2", "insert", "2026-11-24T10:00:01Z"),
        (3, "system_relation", "soil_x", "insert", "2026-11-24T10:00:02Z"),
        (4, "item", "soil_cccc-3", "insert", "2026-11-24T10:00:03Z"),
    ]
    conn.executemany("INSERT INTO change_log VALUES (?,?,?,?,?)", rows)
    conn.execute("INSERT INTO item VALUES ('aaaa-1','Email',null,'{}',null,null)")
    # bbbb-2 superseded; cccc-3 not present at all (deleted)
    conn.execute("INSERT INTO item VALUES ('bbbb-2','Note',null,'{}',null,"
                 "'soil_replacement')")
    return conn


def test_plan_live_vs_removed_and_prefix_strip(change_db):
    upserts, removes, wm = plan_from_change_log(change_db, 0, limit=10)
    assert [u for u, _ in upserts] == ["aaaa-1"]      # prefix stripped, live
    assert removes == ["bbbb-2", "cccc-3"]            # superseded / gone
    assert wm == 4                                    # advanced through seq 4
    assert "system_relation" not in {u for u, _ in upserts} | set(removes)


def test_plan_bounds_and_watermark_semantics(change_db):
    # limit 1: only aaaa-1 planned; watermark may only reach seq 1 — the
    # bounded remainder (bbbb-2 at seq 2) re-plans next wake
    upserts, removes, wm = plan_from_change_log(change_db, 0, limit=1)
    assert [u for u, _ in upserts] == ["aaaa-1"]
    assert wm == 1
    upserts, removes, wm = plan_from_change_log(change_db, wm, limit=10)
    assert removes == ["bbbb-2", "cccc-3"]
    assert wm == 4


def test_plan_nothing_new(change_db):
    upserts, removes, wm = plan_from_change_log(change_db, 4, limit=10)
    assert upserts == [] and removes == [] and wm == 4


def test_max_change_seq(change_db):
    assert max_change_seq(change_db) == 4
