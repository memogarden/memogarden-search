#!/usr/bin/env python3
"""One-off smoke: OLD (mean-pooled, 1200-char) vs NEW (CLS, chunked) search.

Old side is reconstructed from cache/pre-chunked-backup/* with the original
pooling/query handling inlined; new side talks to the live embed_daemon
socket exactly like api/v1/operator.py does. Run with the triage-model venv
AFTER build_search_index.py has written the new index.
"""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).parent
REPO = "BAAI/bge-small-en-v1.5"
SOCK = Path.home() / ".local/share/memogarden/triage-model/embed.sock"
BACKUP = HERE / "cache" / "pre-chunked-backup"
QUERIES = [
    "Senoko bill November",
    "Lazada invoice for my order",
    "AWS receipt",
    "order 2847",
    "school fee payment",
]
K = 5


def old_top(q, k=K):
    z = np.load(BACKUP / "search_index.npz")
    uuids = list(z.files)
    E = np.vstack([z[u] for u in uuids])
    meta = json.loads((BACKUP / "search_meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModel.from_pretrained(REPO).eval()
    with torch.no_grad():
        enc = tok([q[:1200]], truncation=True, max_length=512,
                  return_tensors="pt")
        o = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1)
        e = (o.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        qv = torch.nn.functional.normalize(e, dim=-1).numpy().astype(
            np.float32)[0]
    sims = E @ qv
    top = np.argsort(-sims)[:k]
    return [(meta[u].get("title", "")[:60], round(float(sims[i]), 4))
            for i, u in zip(top, [uuids[t] for t in top])]


def daemon_search(q, k=K):
    payload = json.dumps({"op": "search", "q": q, "scopes": None,
                          "types": None, "k": k}).encode()
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(15)
            s.connect(str(SOCK))
            s.sendall(payload)
            buf = b""
            while True:
                chunk = s.recv(1 << 20)
                if not chunk:
                    break
                buf += chunk
            s.close()
            out = json.loads(buf)
            if "error" in out:
                return [("ERROR: " + out["error"], 0.0)]
            hits = out["results"][:k]
            meta = json.loads((HERE / "cache" / "search_meta.json")
                              .read_text())
            return [(meta.get(u, {}).get("title", "")[:60], sc)
                    for u, sc in hits]
        except (ConnectionRefusedError, FileNotFoundError, socket.timeout):
            with open(HERE / "embed-daemon.log", "ab") as log:
                subprocess.Popen([str(HERE / ".venv/bin/python"),
                                  str(HERE / "embed_daemon.py")],
                                 cwd=str(HERE), stdout=log, stderr=log,
                                 start_new_session=True)
            time.sleep(3)
    return [("daemon unreachable", 0.0)]


def main():
    print(f"{'query':<28} | OLD (mean-pool, 1200-char)        | NEW (CLS, chunked)")
    print("-" * 110)
    for q in QUERIES:
        old = old_top(q)
        new = daemon_search(q)
        print(f"{q:<28} |")
        for i in range(K):
            ot, os_ = old[i] if i < len(old) else ("—", 0)
            nt, ns = new[i] if i < len(new) else ("—", 0)
            mark = "  <<<" if i == 0 else ""
            print(f"  {'':<26} | {os_:7.4f} {ot:<32.32} | {ns:7.4f} {nt:<32.32}{mark}")
        print()


if __name__ == "__main__":
    main()
