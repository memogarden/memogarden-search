# memogarden-search

The search planes for MemoGarden (api#95 "Search build-out: two-plane
architecture"): a lexical plane (SQLite FTS5, lives in
[`memogarden-system`](https://github.com/memogarden/memogarden-system)) and a
semantic plane — **this repo** — the model-layer service that owns embeddings
and vector search. Query-time inference never runs inside the serving process
(architecture decision 2026-11-22): the API talks to this layer over a unix
socket.

## Components

- **`embed_daemon.py`** — lazy search daemon: loads bge-small + the chunked
  vector index once, answers `{"op": "search", ...}` requests over a unix
  socket, idle self-exits after 600s (zero resident RSS between searches).
  Spawned on demand by the API today; systemd socket activation is the
  planned successor (slice B, pending).
- **`build_search_index.py`** — index builder: chunks every live text-bearing
  fact (~600-char whitespace-aligned windows, 75-char overlap, ≤12 chunks per
  item, title prepended to every chunk), embeds with **CLS pooling** +
  L2-norm (bge-small's training pooling; queries additionally get the bge v1.5
  passage-retrieval instruction prefix on the daemon side), and writes a
  chunk-level, hash-keyed cache so rebuilds embed only new/changed text.
- **`tools/smoke_compare.py`** — one-command old-vs-new ranking comparison
  for validating search-quality changes.

## Daemon protocol

One JSON request per connection, socket `~/.local/share/memogarden/triage-model/embed.sock`:

```
{"op": "ping"}                                        -> {"ok": true}
{"op": "search", "q": str, "scopes": [uuid]|null,
 "types": [str]|null, "k": int}                       -> {"results": [[uuid, score], …]}
```

`results` is best-first, aggregated per item (best passage wins). The
daemon's scope mask uses capture-time scope sets — it is **defense-in-depth
only**; enforcement happens at the API boundary under current-time visibility
(RFC-011 §4.9(3)).

## Socket activation (staged — not yet installed)

`deploy/memogarden-embed.socket` + `deploy/memogarden-embed.service` are the
slice-B successor of the API's lazy Popen spawn: systemd owns the listening
socket (`~/.local/share/memogarden/triage-model/embed.sock`, 0600,
`RemoveOnStop`) and starts the daemon on first connect; the daemon adopts the
passed fd (`LISTEN_FDS`) and keeps its in-daemon 600s idle self-exit, so RSS
returns to zero between searches while systemd re-activates instantly on the
next query. Without the units installed, the daemon self-binds exactly as
before — the API's spawn fallback remains as belt-and-suspenders either way.

Install (user units — after the slice-B cutover gives this repo its own
`.venv`; `MEMOGARDEN_SEARCH_CACHE` in the unit covers the migration window):

```
cp deploy/memogarden-embed.{socket,service} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now memogarden-embed.socket
```

Verify: `systemctl --user status memogarden-embed.socket`, then any console
search in semantic mode — first hit shows `memogarden-embed.service`
active, and it self-exits after 10 idle minutes.

## Runtime layout (migration note)

Code graduated here from the untracked `~/memogarden/triage-model/`
scratch dir (2026-11-24). Until the slice-B cutover lands, the **runtime**
venv, caches and logs still live there:

- `~/memogarden/triage-model/.venv/` — the sealed torch venv
- `~/memogarden/triage-model/cache/` — `search_index.npz` (chunked vectors),
  `search_index_ids.json`, `search_meta.json`, chunk cache `search_emb.*`;
  superseded pre-chunked index kept in `cache/pre-chunked-backup/`
- `HF_HOME=~/memogarden/triage-model/hf-cache` for model weights

Rebuild: `HF_HOME=$PWD/../triage-model/hf-cache ../triage-model/.venv/bin/python build_search_index.py`
(from a checkout of this repo; the script's `HERE`-relative cache paths
resolve to wherever the file lives — point `cache/` handling at the runtime
location during the migration window).

## History

- 2026-11-22/23 — search-quality diagnosis (61.5% of email bodies truncated
  out of the old single-embedding index; mean-pooling on a CLS-trained model).
- 2026-11-24 — items 1+2 shipped: CLS pooling + query instruction prefix,
  chunked-body best-passage index (211,858 chunks / 63,782 items).
- api#95 Phase 1 (FTS5 lexical plane + console) lives in memogarden-system +
  memogarden-api; Phase 2 slice A (verb semantic/hybrid strategies) is staged
  in memogarden-api.
