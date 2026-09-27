"""Rebuild a ViDoRe split's queries.json without downloading its page images.

The benchmark runner only needs query strings and which page each query points
at. When an encode cache exists but the folder that produced it does not (a run
done on Kaggle, say), this rebuilds ``queries.json`` from the public dataset rows
using exactly the rule in :func:`optivision.corpus.load_vidore_subset`: page ``i``
is ``f"{i:05d}::p1"`` and each distinct query string is relevant to the first row
it appears on.

    python scripts/rebuild_vidore_qrels.py vidore/docvqa_test_subsampled \
        --out data/vidore_docvqa_test_subsampled/queries.json \
        --check data/cache/colpali_docvqa_test_subsampled.queries.npz

``--check`` compares the rebuilt strings, in order, with the ones stored in a
query cache and refuses to write on any mismatch. The response is decoded as
UTF-8 explicitly: letting the platform pick the code page (cp1252 on Windows)
mangles curly quotes and silently breaks the match.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset={ds}&config=default"
    "&split=test&offset={off}&length={n}"
)


def fetch_rows(dataset: str, page: int = 100) -> list[tuple[int, str | None]]:
    rows: list[tuple[int, str | None]] = []
    off = 0
    while True:
        url = ROWS_URL.format(ds=dataset, off=off, n=page)
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        batch = data.get("rows", [])
        rows.extend((int(r["row_idx"]), r["row"].get("query")) for r in batch)
        off += len(batch)
        if not batch or off >= int(data.get("num_rows_total", off)):
            break
    rows.sort()
    return rows


def build_queries(rows: list[tuple[int, str | None]]) -> list[dict]:
    queries: list[dict] = []
    seen: set[str] = set()
    for i, query in rows:
        if query and query not in seen:
            seen.add(query)
            queries.append({"qid": f"q{len(queries):04d}", "query": query,
                            "relevant": [f"{i:05d}::p1"]})
    return queries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", help="e.g. vidore/docvqa_test_subsampled")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=Path, help="a *.queries.npz cache to verify against")
    a = ap.parse_args()

    rows = fetch_rows(a.dataset)
    queries = build_queries(rows)
    print(f"{a.dataset}: {len(rows)} rows, {len(queries)} distinct queries")

    if a.check is not None:
        import numpy as np

        cached = json.loads(str(np.load(a.check, allow_pickle=True)["texts"]))
        if cached != [q["query"] for q in queries]:
            print(f"refusing to write: strings differ from {a.check}", file=sys.stderr)
            return 1
        print(f"verified against {a.check}: identical strings, identical order")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(queries, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
