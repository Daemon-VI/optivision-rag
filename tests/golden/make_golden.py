"""Write the golden fixtures that pin the v0.3.0 file formats.

Files written by v0.3.0 must keep reading back, with the same meaning, in every later
version. This script writes a small, seeded example of each format with the library as it
is at v0.3.0 and records in ``expected_v030.json`` what each file must read back as.
``tests/test_golden_formats.py`` loads the committed files with whatever library is
installed and compares: exact for arrays, ids, metadata, pipelines and rankings, 1e-6 for
scores.

    optivision-multivector v1   MultiVectorCorpus.save   corpus_f32_v030.npz, corpus_f16_v030.npz,
                                                         queries_v030.npz
    optivision-compressed v1    compose.save_compressed  compressed_<pipeline>_v030.npz
    numpy index directory       NumpyIndex.save          index_binary_v030/ (codes.npy, offsets.npy,
                                                         meta.json)

Run it ONCE, from the repository root, with the library as it is at v0.3.0:

    PYTHONPATH=src python tests/golden/make_golden.py

and commit what it wrote. It refuses to overwrite existing fixtures. Regenerating them with a
newer library would make them describe that library instead of v0.3.0, which defeats their
purpose: do it only in the same change that deliberately changes a format, and say so there.

Everything is seeded and nothing is pickled. The arrays and the JSON are reproducible; the
.npz containers are not byte-identical between runs (zip timestamps), so no test hashes file
bytes. Digests in the JSON are computed here with hashlib, on purpose independent of
``optivision.core``: dtype string, shape, then the array bytes.

Codecs with fitted state (``overhead_bytes() > 0``: centred binary or int4, per-dimension
int8, lloyd2) cannot be written by ``save_compressed`` and so have no fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

import optivision
from optivision.compose import Pipeline, load_compressed, save_compressed
from optivision.compression import Compressor
from optivision.config import CompressionConfig
from optivision.index.numpy_index import NumpyIndex
from optivision.representation import MultiVectorCorpus
from optivision.scoring import maxsim_matrix, rank
from optivision.stages import (
    AdaptiveMerge,
    BinaryQuantizer,
    Float16Quantizer,
    Int4Quantizer,
    Int8Quantizer,
)
from optivision.types import PageRef, PatchGrid, PrunedPage

HERE = Path(__file__).resolve().parent
EXPECTED_NAME = "expected_v030.json"
SEED = 20261006
DIM = 20  # not a multiple of 8: binary codes carry padding bits
DOC_COUNTS = (6, 9, 4, 7, 0, 5, 8, 6, 3, 10, 7, 5)  # the fifth document is empty
QUERY_COUNTS = (3, 5, 4, 6)
ROWS_SHOWN = 4  # decoded rows recorded per compressed file
MIN_GAP = 1e-4  # scores closer than this could swap order on another BLAS
MAX_TOTAL_BYTES = 100_000

NAMES = {
    "corpus_f32": "corpus_f32_v030.npz",
    "corpus_f16": "corpus_f16_v030.npz",
    "queries": "queries_v030.npz",
    "index": "index_binary_v030",
}

PIPELINES = {
    "binary": lambda: Pipeline([BinaryQuantizer()]),
    "int4": lambda: Pipeline([Int4Quantizer()]),
    "int8_per_vector": lambda: Pipeline([Int8Quantizer("per_vector")]),
    "float16": lambda: Pipeline([Float16Quantizer()]),
    "float32_empty": lambda: Pipeline([]),  # what optimize() returns when nothing is feasible
    "merge_int8_per_vector": lambda: Pipeline(
        [AdaptiveMerge(radius=None, ratio=0.5), Int8Quantizer("per_vector")]
    ),
}


# ------------------------------------------------------------------ helpers


def fixture_names() -> dict[str, str]:
    return {**NAMES, **{f"compressed_{key}": f"compressed_{key}_v030.npz" for key in PIPELINES}}


def fixture_bytes(out: Path) -> int:
    """Bytes of the fixture files only (not of whatever else sits in ``out``)."""
    paths = [out / name for name in [*fixture_names().values(), EXPECTED_NAME]]
    files = [f for p in paths for f in ([p] if p.is_file() else p.rglob("*")) if f.is_file()]
    return sum(f.stat().st_size for f in files)


def array_sha256(a: np.ndarray) -> str:
    """sha256 of dtype string, shape and bytes (independent of optivision.core on purpose)."""
    a = np.ascontiguousarray(a)
    head = f"{a.dtype.str}|{','.join(str(n) for n in a.shape)}|".encode("ascii")
    return hashlib.sha256(head + a.tobytes()).hexdigest()


def numbers(a: np.ndarray) -> Any:
    """Nested lists of floats that read back as the same float32 (9 significant digits)."""
    a = np.asarray(a)
    if a.ndim == 0:
        return None if np.isneginf(a) else float(f"{float(a):.9g}")
    return [numbers(row) for row in a]


def unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def check_gaps(scores: np.ndarray, label: str) -> None:
    for q, row in enumerate(scores):
        finite = np.sort(row[np.isfinite(row)])
        if finite.size > 1 and np.diff(finite).min() < MIN_GAP:
            raise SystemExit(f"{label}: scores of query {q} are within {MIN_GAP} of each other; "
                             "change SEED so the recorded rankings cannot flip on another BLAS")
        if not np.all(np.isfinite(row) | np.isneginf(row)):
            raise SystemExit(f"{label}: a score is NaN or +inf")


def dump(obj: Any, depth: int = 0, max_depth: int = 2) -> str:
    """JSON with the first two levels one entry per line and the rest compact."""
    if isinstance(obj, dict) and depth < max_depth:
        pad = "  " * (depth + 1)
        items = [f"{pad}{json.dumps(key)}: {dump(value, depth + 1, max_depth)}"
                 for key, value in sorted(obj.items())]
        return "{\n" + ",\n".join(items) + "\n" + "  " * depth + "}"
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ------------------------------------------------------------------- the data


def build_corpus(rng: np.random.Generator) -> MultiVectorCorpus:
    """Twelve pages: vectors, protected rows, grid positions (NaN off the grid), weights."""
    n = sum(DOC_COUNTS)
    offsets = np.concatenate([[0], np.cumsum(DOC_COUNTS)]).astype(np.int64)
    protected = np.zeros(n, dtype=bool)
    positions = np.full((n, 2), np.nan, dtype=np.float32)
    for i, count in enumerate(DOC_COUNTS):
        if count == 0:
            continue
        lo = int(offsets[i])
        protected[lo] = True  # one leading token per page stays verbatim
        cells = count - 1
        cols = max(1, int(np.ceil(np.sqrt(cells))))
        rows = -(-cells // cols)
        for j in range(cells):
            r, c = divmod(j, cols)
            positions[lo + 1 + j] = ((r + 0.5) / rows, (c + 0.5) / cols)
    weights = np.where(rng.random(n) < 0.2, 2.0, 1.0).astype(np.float32)
    ids = [f"doc{i // 2}::p{i % 2 + 1}" for i in range(len(DOC_COUNTS))]
    metadata = [{"doc_id": f"doc{i // 2}", "page_no": i % 2 + 1, "n_vectors": c}
                for i, c in enumerate(DOC_COUNTS)]
    metadata[1]["title"] = "caf" + chr(0xE9)  # non-ASCII text in the header
    metadata[2]["nested"] = {"tags": ["a", "b"], "ratio": 0.5}
    return MultiVectorCorpus(
        unit(rng.standard_normal((n, DIM))), offsets, ids=ids, metadata=metadata,
        protected=protected, positions=positions, weights=weights,
        attrs={"model": "synthetic/golden", "encoder": {"name": "synthetic", "dim": DIM},
               "seed": SEED},
    )


def build_queries(rng: np.random.Generator) -> MultiVectorCorpus:
    arrays = [unit(rng.standard_normal((n, DIM))) for n in QUERY_COUNTS]
    return MultiVectorCorpus.from_arrays(arrays, ids=[f"q{i}" for i in range(len(arrays))],
                                         attrs={"role": "queries"})


# ------------------------------------------------------- what each file must read as


def corpus_entry(path: Path, queries: MultiVectorCorpus | None, strict: bool) -> dict[str, Any]:
    c = MultiVectorCorpus.load(path)
    rows = sorted({0, 1, c.num_vectors // 2, c.num_vectors - 1})
    entry: dict[str, Any] = {
        "dtype": str(c.dtype), "shape": list(c.vectors.shape), "ids": c.ids,
        "metadata": c.metadata, "attrs": c.attrs, "counts": c.counts.tolist(),
        "sha256": {"vectors": array_sha256(c.vectors), "offsets": array_sha256(c.offsets)},
        "rows": {str(r): numbers(c.vectors[r].astype(np.float32)) for r in rows},
        "images": None,
    }
    for name in ("protected", "positions", "weights", "importance"):
        array = getattr(c, name)
        entry["sha256"][name] = None if array is None else array_sha256(array)
    if queries is not None:
        scores = maxsim_matrix(queries, c)
        if strict:
            check_gaps(scores, path.name)
        entry["scores"], entry["ranking"] = numbers(scores), rank(scores).tolist()
    return entry


def compressed_entry(path: Path, queries: MultiVectorCorpus, strict: bool) -> dict[str, Any]:
    c = load_compressed(path)
    rows = sorted({0, 1, c.num_vectors // 2, c.num_vectors - 1})[:ROWS_SHOWN]
    scores = maxsim_matrix(queries, c)
    if strict:
        check_gaps(scores, path.name)
    return {
        "n_docs": len(c), "num_vectors": c.num_vectors, "dim": c.dim,
        "code_bytes": int(c.codes.shape[1]), "counts": c.counts.tolist(), "ids": c.ids,
        "stats": c.stats, "pipeline": c.pipeline, "quantizer": c.quantizer.to_dict(),
        "nbytes": c.nbytes,
        "sha256": {"codes": array_sha256(c.codes), "offsets": array_sha256(c.offsets)},
        "decoded_rows": {str(r): numbers(c.decode_rows(r, r + 1)[0]) for r in rows},
        "scores": numbers(scores), "ranking": rank(scores).tolist(),
    }


def write_index(path: Path, corpus: MultiVectorCorpus) -> None:
    """A NumpyIndex over the non-empty pages, each compressed with the legacy binary codec."""
    compressor = Compressor(CompressionConfig(method="binary"))
    pages = []
    for i, page_id in enumerate(corpus.ids):
        n = int(corpus.counts[i])
        if n == 0:
            continue
        ref = PageRef(doc_id=f"doc{i // 2}", page_no=i % 2 + 1, source_path=f"pdfs/doc{i // 2}.pdf")
        assert ref.page_id == page_id
        pruned = PrunedPage(
            ref=ref, embeddings=np.asarray(corpus[i].vectors, dtype=np.float32),
            kept_token_index=np.arange(n, dtype=np.int32), keep_mask=np.ones((1, n), dtype=bool),
            grid=PatchGrid.contiguous(1, n), n_tokens_before=4 * n,
        )
        pages.append(compressor.compress(pruned))
    index = NumpyIndex(path, dim=DIM, method="binary")
    index.add(pages)
    index.save()


def index_entry(path: Path, queries: MultiVectorCorpus, strict: bool) -> dict[str, Any]:
    index = NumpyIndex.load(path)
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    scores = np.stack([index.score_all(np.asarray(q.vectors, dtype=np.float32)) for q in queries])
    if strict:
        check_gaps(scores, path.name)
    search = []
    for q in queries:
        hits = index.search(np.asarray(q.vectors, dtype=np.float32), top_k=5)
        search.append([
            {"page_id": h.ref.page_id, "rank": h.rank, "score": numbers(np.float32(h.score))}
            for h in hits
        ])
    return {
        "n_pages": index.n_pages, "dim": index.dim, "method": index.method,
        "page_ids": [r.page_id for r in index.refs], "meta_keys": sorted(meta),
        "ref_keys": sorted(meta["refs"][0]), "codec": meta["codec"],
        "stats": index.stats(),
        "sha256": {"codes": array_sha256(np.load(path / "codes.npy")),
                   "offsets": array_sha256(np.load(path / "offsets.npy"))},
        "scores": numbers(scores), "search": search,
    }


# ----------------------------------------------------------------------- main


def main(out: Path = HERE, force: bool = False, strict: bool = True) -> dict[str, Any]:
    """Write every fixture and ``expected_v030.json`` into ``out``; return the expectations.

    ``strict=False`` skips the check that recorded scores are well separated; the tests use
    it when they run this script on whatever random stream the installed NumPy provides.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = fixture_names()
    existing = [n for n in [*names.values(), EXPECTED_NAME] if (out / n).exists()]
    if existing and not force:
        raise SystemExit(f"refusing to overwrite {existing}; see the module docstring")

    rng = np.random.default_rng(SEED)
    corpus, queries = build_corpus(rng), build_queries(rng)
    corpus.save(out / names["corpus_f32"])
    corpus.astype(np.float16).save(out / names["corpus_f16"])
    queries.save(out / names["queries"])
    for key, make in PIPELINES.items():
        save_compressed(make().compress(corpus), out / names[f"compressed_{key}"])
    write_index(out / names["index"], corpus)

    q = MultiVectorCorpus.load(out / names["queries"])
    expected = {
        "written_with": {"optivision": optivision.__version__, "numpy": np.__version__,
                         "python": platform.python_version()},
        "seed": SEED, "dim": DIM,
        "queries": {names["queries"]: corpus_entry(out / names["queries"], None, strict)},
        "corpora": {names[k]: corpus_entry(out / names[k], q, strict)
                    for k in ("corpus_f32", "corpus_f16")},
        "compressed": {names[k]: compressed_entry(out / names[k], q, strict)
                       for k in (f"compressed_{key}" for key in PIPELINES)},
        "index": {names["index"]: index_entry(out / names["index"], q, strict)},
    }
    (out / EXPECTED_NAME).write_text(dump(expected) + "\n", encoding="utf-8")

    total = fixture_bytes(out)
    if strict and total > MAX_TOTAL_BYTES:
        raise SystemExit(f"fixtures are {total} bytes; keep them under {MAX_TOTAL_BYTES}")
    return expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=HERE, help="directory to write into")
    parser.add_argument("--force", action="store_true", help="overwrite existing fixtures")
    args = parser.parse_args()
    main(args.out, force=args.force)
    print(f"wrote golden fixtures to {args.out} ({fixture_bytes(args.out)} bytes; optivision "
          f"{optivision.__version__}, numpy {np.__version__}, Python {sys.version.split()[0]})")
