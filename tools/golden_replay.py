#!/usr/bin/env python3
"""Golden-set replay: rerank-off vs rerank-on over the live daemon.

The rerank acceptance gate (operator: build now, replay first). Drives
the REAL daemon socket (op:search then op:rerank per query) and scores
each order against the golden set's expected uuids with MRR@5 + hit@5;
a query wins when rerank's MRR beats baseline's, loses on the inverse,
pars otherwise. Writes a results/ JSON artifact (eval_matrix.py culture:
cached inputs + durable JSON verdicts) and prints the table.

Golden format (tools/golden_set.json):
  {"queries": [{"q": str, "expected": [uuid, ...]}]}   # best uuid first

Daemon control: start it yourself, e.g.
  MEMOGARDEN_EMBED_SOCK=/tmp/golden-embed.sock \
  HF_HOME=$HOME/memogarden/triage-model/hf-cache \
  .venv/bin/python embed_daemon.py &
then:
  MEMOGARDEN_EMBED_SOCK=/tmp/golden-embed.sock .venv/bin/python \
  tools/golden_replay.py
"""
import json
import os
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

GOLDEN = HERE / "tools" / "golden_set.json"
RESULTS = HERE / "results"
K = 10
DEPTH = 10   # rerank candidate pool per query


def _daemon_req(payload, timeout=60):
    sock = os.environ.get("MEMOGARDEN_EMBED_SOCK") or str(
        Path.home() / ".local/share/memogarden/triage-model/embed.sock")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(sock)
    s.sendall(json.dumps(payload).encode())
    buf = b""
    while True:
        chunk = s.recv(1 << 20)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf)


def _is_hit(u, expected):
    """expected entries may be full uuids or unique prefixes."""
    return any(u == e or u.startswith(e) for e in expected)


def mrr_at_5(order, expected):
    for r, u in enumerate(order[:5], start=1):
        if _is_hit(u, expected):
            return 1.0 / r
    return 0.0


def hit_at_5(order, expected):
    return 1.0 if any(_is_hit(u, expected) for u in order[:5]) else 0.0


def main():
    golden = json.loads(GOLDEN.read_text())
    t0 = time.time()
    rows, wins, losses, pars = [], 0, 0, 0
    for entry in golden["queries"]:
        q = entry["q"]
        expected = set(entry["expected"])
        base = _daemon_req({"op": "search", "q": q, "k": K})
        if "results" not in base:
            print(f"SKIP {q!r}: daemon search error: {base.get('error')}")
            rows.append({"q": q, "error": base.get("error"),
                         "verdict": "skip"})
            continue
        base_order = [u for u, _ in base["results"]]
        # 240s: the FIRST rerank call in a fresh daemon downloads + int8-
        # quantizes the cross-encoder (~1.1GB from HF + quantize pass)
        rr = _daemon_req({"op": "rerank", "q": q,
                          "uuids": base_order[:DEPTH]}, timeout=240)
        if "results" in rr:
            scored = dict(rr["results"])
            rr_order = [u for u, _ in
                        sorted(scored.items(), key=lambda kv: -kv[1])]
            # candidates the reranker dropped (unscoreable) keep baseline order
            tail = [u for u in base_order[:DEPTH] if u not in scored]
            rr_order = rr_order + tail + base_order[DEPTH:]
        else:
            rr_order = base_order
        bm, rm = mrr_at_5(base_order, expected), mrr_at_5(rr_order, expected)
        bh, rh = hit_at_5(base_order, expected), hit_at_5(rr_order, expected)
        verdict = "win" if rm > bm else ("loss" if rm < bm else "par")
        wins, losses, pars = (wins + (verdict == "win"),
                              losses + (verdict == "loss"),
                              pars + (verdict == "par"))
        rows.append({"q": q, "base_mrr5": round(bm, 3),
                     "rerank_mrr5": round(rm, 3),
                     "base_hit5": bh, "rerank_hit5": rh,
                     "verdict": verdict,
                     "base_top": base_order[:5], "rerank_top": rr_order[:5]})
        mark = "  <<<" if base_order[:1] != rr_order[:1] else ""
        print(f"{verdict:4}  base {bm:.3f} -> rr {rm:.3f}  {q!r}{mark}")

    scored = [r for r in rows if r.get("verdict") != "skip"]
    n = len(scored)
    print(f"\n{n} queries: {wins} win / {pars} par / {losses} loss  "
          f"({time.time() - t0:.1f}s)")
    print(f"mean MRR@5: base "
          f"{sum(r['base_mrr5'] for r in scored) / max(n, 1):.3f} -> "
          f"rerank {sum(r['rerank_mrr5'] for r in scored) / max(n, 1):.3f}")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"rerank_replay-{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps({
        "golden": str(GOLDEN), "k": K, "depth": DEPTH, "rows": rows,
        "summary": {"queries": n, "win": wins, "par": pars, "loss": losses},
    }, indent=2))
    print(f"artifact: {out}")
    return 0 if losses == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
