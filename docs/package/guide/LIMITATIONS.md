# Limitations

Status: 2026-10-06, branch `package-universal`. What a careful user should know before relying on
the package. Every statement about research names its origin and its scope. The package does not
promise that a compression keeps retrieval quality on an encoder, corpus, corpus size, modality or
coordinate basis that was not measured. Sources: `../FINAL_RESEARCH_BOUNDARY.md` (the boundary
between package and research), `../METHOD_C_SPEC.md`, `../ARCHITECTURE_AUDIT.md`,
`../RETRIEVAL_BACKEND_AUDIT.md`, `../BACKEND_ARCHITECTURE.md`.

## 1. Scope

- What was studied is late-interaction retrieval scored by MaxSim, over L2-normalised vectors,
  with training-free compression of a fixed encoder's document vectors (queries stay in full
  precision), exact exhaustive search over a fixed corpus, and the operators of v0.3.0
  (`reports/research/NEXT_PHASE_PREREGISTRATION.md`, section 5).
- Out of scope there: any scoring function other than MaxSim, approximate search and engines,
  latency, corpus growth after certification, trained compressors, and corpus sizes beyond 500
  pages and 5,183 abstracts (same section; section 25 lists the non-claims).
- The vector contract does not require unit length (`MultiVectorRepresentation`), and several
  codecs assume it. What happens with unnormalised vectors, or with another scoring function, is
  not known. The certificate records the norms of the vectors it was issued for and carries a
  caveat when they are not unit.
- Only nDCG@k, with binary relevance, was evaluated by the registered experiments. `recall@k`,
  `mrr@k` and `hit@k` are available and are an engineering option.

## 2. What was measured, and on what

| what | scope |
|---|---|
| X1 (agreement against labels), X2 (query budget), X3 (coordinate basis) | seven development stores: ColPali-v1.3, ColQwen2-v1.0 and ColQwen2.5-v0.2, each on ViDoRe V1 DocVQA and InfoVQA, and answerai-colbert-small-v1 on BEIR SciFact; one pool of queries per corpus; the registered list of 40 pipelines (`reports/research/X_RESULTS.md`) |
| XH, the held-out encoder protocol | designed, not registered, not run |
| the package as a whole (`certify`, the certificate, the artifact, the backends) | ENGINEERING: tested on synthetic data and against the frozen research code for numerical equality; no registered experiment evaluated it as a package |

- X1 to X3 are development analyses. They reuse the encoders that shaped the candidate list and
  say nothing about transfer. **Transfer to encoders that took no part in the design is not
  demonstrated.** No real encoder output was encoded or scored in the package work.
- The registered runs recorded the library tree `0b9523bd10f43941a9704cbf6fae1db1804880ee`. The
  package branch has a different tree, because it adds files. X1 to X3 were not re-run.
- The 2,560-dimensional Nemotron results (`../../UNIVERSAL.md`, R12) come from the empirical rule
  of `calibrate()`. The selection rule was not run on that encoder.
- Not validated: ColQwen3, ColNomic, 4,096-d models, ViDoRe V2 and V3, million-page corpora,
  database connectors beyond NumPy and Qdrant (`PACKAGE.md`).

## 3. The default candidate list

It is frozen and the same for every encoder, and it was shaped by results on the development
encoders. It holds centred int4 and plain binary because of results on ColPali and the text
ColBERT (`../METHOD_C_SPEC.md`, section 5.1). It is model-blind in what the procedure inspects,
not neutral in how it was built. 39 of its 40 slots end in `int8(per_vector)`, `int4(mean)` or
`binary`, and the other is `float16`, so every default candidate is basis sensitive
(`BASIS_DEPENDENCE.md`). Without SciPy the Ward families cannot be built, and the default list
raises unless `allow_reduced=True` gives a smaller list with another identity and another level.
The `wide` list was not evaluated as a selection list.

## 4. Assumptions the software cannot check

The rule's statement is claimed under assumptions and conditions that the certificate lists
(`SELECTION_AND_CERTIFICATION.md`). Among them, A1 (independent queries from the deployment
distribution), A5 (an order fixed before outcomes) and A6 (labels taken as given) are properties
of how you obtained the queries and labels. `certify` refuses exact duplicate queries unless
you pass `allow_duplicate_queries=True`, which asserts that they are independent draws and is
recorded in the certificate with a caveat; it cannot see near-duplicates or queries that are clustered
by session or author. `certify` and the command line test the queries in a seeded random order
by default, and record whether the order was seeded or given and whether the seed was chosen
by you; trying several seeds and keeping the best is outside the statement. A2 (a
fixed corpus) is bound by a digest, but the software cannot know that the corpus you serve is the
one you certified unless you pass it to `verify`. A3 (a list fixed before the queries are seen)
is supported by the code path and not proven: the default list depends only on the dimension, the
evidence collector never passes queries to `fit`, and it cannot know whether you edited the list
after a look. The package cannot see several looks at one sample, or several targets tried and one
reported; the one thing it does is a `UserWarning` when `certify` is called again in the same
process on the same queries and corpus with a different target, error level, seed or candidate
space (other processes are invisible to it). `verify` reports the highest level whose checks
actually ran and lists those it skipped; on an ABSTAIN or FALLBACK record `ok` means that the
record is intact, not that an index was certified. `SELECTION_AND_CERTIFICATION.md`, section 6, has the table.

## 5. Abstention

The rule may select nothing, and the statement holds if it never does. On the development stores
at `T = 0.99` the default procedure deployed non-trivial compression in at most 0.010 of the
resampled samples at pool-sized budgets (X2; SIMULATED, pool-supported). At `T = 0.97` the rate
was at least 0.90 on all seven stores, the lowest 0.917. Nothing says how many queries are
enough. Any valid test needs more queries as the target rises, even for a candidate that loses
nothing (`reports/research/E1/DESIGN.md`, derived there and not externally reviewed).

## 6. Agreement is not quality

Retention under `agreement@1` measures fidelity to the uncompressed ranking, mistakes included.
No statistical statement connects it to retention against labels. On the development stores the
two behaved differently (X1: not equivalent, H4b; agreement stricter, N10). With no `qrels`,
`certify` tests agreement by default, and the result and the certificate say so.

## 7. Basis dependence

A certificate is about the vectors as given. X3 found that the measured result of some sign-code
and 4-bit pipelines depends on the coordinate basis under two registered rotations, on these
datasets (`BASIS_DEPENDENCE.md`). Re-encoding, rotating or whitening the vectors means measuring
again.

## 8. Corpus changes

The statement is about the corpus that was certified (A2). Retention moves with corpus size: on
two ColPali splits, adding random distractor pages from 100 to 500 lowered the retention of fixed
configurations, and the lossier ones lowered more (`../../UNIVERSAL.md`, R10; MEASURED, 5 draws).
Nothing measures corpora beyond 5,183 documents. Adding, removing or re-encoding documents gives a
different corpus identity, and the certificate does not cover it.

## 9. Relevance labels

`LabelReference` and `optivision.evaluation.relevant_from_qrels` use binary relevance: every
document listed for a query counts as relevant. When the labels of a query are a mapping
`{document id: grade}`, every grade above zero is treated as relevant and the size of the grade is
not read; an entry with a grade of 0 or below is refused with a `ValueError` (it would be counted
as relevant), and a grade that is not a number with a `TypeError`. A document id that is not in the
corpus is ignored without a warning, so an id mismatch removes queries instead of raising. A query
with no relevant document is removed, for the uncompressed index and every candidate together, and
is listed in `Evidence.dropped_queries`. Check `report()["n_queries_without_relevant_document"]`.

A user with graded labels must choose the grade that counts as relevant and pass only those
documents. The nDCG here has binary gain, so it differs from published graded nDCG.

```python
from optivision.core.evaluate import LabelReference

doc_ids = ["d1", "d3", "d7", "d9"]
graded = {"q1": {"d1": 2, "d7": 0, "d9": 1}, "q2": {"zz": 3}}  # {query id: {document id: grade}}
try:
    LabelReference(graded)
except ValueError as error:
    print("refused:", str(error).split(",")[0])

binary = {q: {d: g for d, g in docs.items() if g > 0} for q, docs in graded.items()}
print(LabelReference(binary).relevant(doc_ids, ["q1", "q2", "q3"]))
```

```text
refused: qrels['q1']['d7'] has grade 0
[array([0, 3]), array([], dtype=int64), array([], dtype=int64)]
```

In the first call `d7`, with grade 0, is refused. In the second, the entries with a grade of 1 and
above are kept: `q2` lists a document the corpus does not hold, and `q3` has no labels.

## 10. Speed and scale

- Scoring is decode-first: codes are decoded to float32 before the inner products. No native
  compressed scoring kernel is shipped. The scan time follows the number of vectors, not the codec
  (`../BACKEND_ARCHITECTURE.md`, section 10, W1). The package makes no claim of speed from
  compression.
- Q5 and Q6 (MEASURED, `reports/research/Q5Q6/REPORT.md`): ColQwen2 on 500 ViDoRe DocVQA pages,
  451 queries as one batch, one 11th-generation Intel laptop CPU, NumPy and torch 2.11.0 on CPU,
  no compiler. Scoring int8 codes directly with torch was 23.8 to 34.1x slower than decoding and
  using BLAS; an exact binary first stage by lookup tables in NumPy was 28 to 30x slower; the
  complete native two-tier path was 9.7 to 11.7x slower end to end. Compiled kernels were not
  tested, and nothing general is claimed either way.
- Search builds a dense float32 `[n_queries, n_docs]` matrix, and the block budget bounds only the
  decoded block and the similarity block. The matrix grows with the corpus. The million-page
  figures of Q8 are projections (`reports/research/Q8/REPORT.md`). Nothing here is a claim about
  million-page corpora.
- Memory of `certify()` is many times the corpus, and the multiple is not a constant. Two separate
  measurements on one machine (tracemalloc, synthetic vectors, d=128, the default 40 candidates):
  the hardening audit measured peaks of 11.4 and 11.1 times the float32 corpus for corpora of
  1.42 MB and 2.78 MB (ABSTAIN path), about 8.4 times of it the cache of transformed corpora
  (`hardening/a10_performance.md`); the engineering benchmark of 2026-10-10 measured 18.8 times
  for its 4.58 MB corpus tested with 1,220 query vectors (`ENGINEERING_BENCHMARKS.md`, section 7).
  The peak holds the cache (default cap 4 GiB, `OPTIVISION_CACHE_BYTES`; past the cap it evicts
  and recomputes, which costs time, not correctness) and one block of query-vector by
  document-vector similarities, which grows with the number of query vectors until
  `max_block_bytes` (256 MiB by default) caps it. On a small corpus plan for ten to twenty times
  its float32 size and measure on your own data; larger corpora were not measured, and this is
  not a scaling law for other data or machines.
- Two tiers keep the small hot store in RAM and read the cold store for a shortlist. That saves
  resident memory. It does not save total storage, and no speed claim is made.
- A Qdrant server searches an approximate graph with its own quantisation and rescoring. It was
  not measured here. Local mode is for convenience.

## 11. A certificate is not a security token without a key

The content digest detects accidental and naive edits. Anyone who can edit the file can recompute
it. An optional HMAC with a key the forger does not hold closes that gap. HMAC is symmetric and
gives no non-repudiation. Schema version 1 has no revocation, expiry or chaining
(`../CERTIFICATE_SPEC.md`, sections 1, 6 and 8).

## 12. float16

float16 is a lossy codec: it rounds each coordinate to half precision. There is no silent float16
fallback. When nothing is certified the answer is the uncompressed index, and float16 is deployed
only if it is itself certified. On one encoder (Nemotron ColEmbed 4B, a 6-page smoke test) the
minimum cosine between float16 and float32 versions of the same vector was 0.923
(`../../UNIVERSAL.md`, R12).

## 13. Size accounting

The size that selection uses is `bytes_per_doc`: codes plus codec overhead. Fitted-state bytes
(for example a PCA matrix that queries need) and offsets are reported beside it
(`ByteAccounting`) and are not in the key. A compression ratio from `compression_vs_float32` does
not include them either.

## 14. Known defects of existing code, left unchanged

Found by the audits and reported, not fixed (the pinned files cannot be edited, and the rest was
left alone by decision). The new wrappers refuse some of these inputs before they reach the
engine.

| where | defect |
|---|---|
| `OptiVisionRAG` | cannot use `method="lloyd2"`; silently falls back to pixel saliency when `pruning.saliency="codebook"`, because no codebook is ever fitted |
| `Config` | `Config.seed` and `CompressionConfig.asymmetric` are never read; `keep_norm` adds a scale to the byte count that `NumpyIndex.add` neither stores nor uses |
| `SyntheticEncoder` | page vectors depend on the order of encoding |
| stale text | `stages/project.py:21-22` says only 128-d models were measured (pinned file); a `CompressedPage` docstring says sign bits only; overstated texts are listed in `../BACKEND_ARCHITECTURE.md`, section 14, and some of them have since been corrected |
| global state | `import optivision` changes PIL's pixel limit process-wide; the first GPU score switches TF32 off for the process; `OPTIVISION_SCORE_DEVICE` is read on every call and honoured only by `maxsim_matrix`; `OPTIVISION_CACHE_BYTES` is read at import |
| `calibration._compress` | rewrites the caller's `pipe.stages` in place with cached stage objects shared across pipelines |
| `Int8Quantizer(scale="fixed")` | warns on every block that clips |
| dtype | float64 inputs are narrowed to float32 in `scoring.py` and `NumpyIndex.search`; `MultiVectorCorpus.from_arrays` promotes mixed dtypes |
| ties | three conventions: `rank()` by position, `NumpyIndex.search` unspecified, Qdrant local unspecified |
| scores | NumPy and Qdrant do not compute the same score; Qdrant receives decoded codes, not float originals, unless `method="none"` |
| empty documents | three conventions: `-inf` (new scorers), 0.0 (legacy helpers), a crash (`QdrantIndex.add`, which also leaves the open index failing until reopened) |
| `NumpyIndex` | `add` is quadratic and not idempotent; `get_index` prefers on-disk metadata over its arguments; its decoded cache holds about 26,000 vectors at 2,560-d, then decodes per query |
| `MultiVectorCorpus.load(mmap=True)` | does not memory-map |
| `OptiVisionRAG` rerank | reloads a pickled cache with `allow_pickle=True` on every query and mixes score scales |
| `relevant_from_qrels` | drops unknown ids silently, and counts a grade of 0 as relevant (`LabelReference` refuses that) |
| `Pipeline` (pinned `compose.py`) | `fit`, `compress` and `transform_queries` given a raw array raise `AttributeError`; `transform_queries` with queries of the wrong width fails inside NumPy's matmul. The retrievers check the width before they reach it |
| `TieredIndex` | `last_timing` is shared state; it fails on an empty corpus and gives NaN below the shortlist if every hot document is empty for a query |
| `to_legacy_pages` | re-splits ids at the first `::p` (`abc` returns as `abc::p1`) and crashes on an id whose suffix is a superscript digit |
| `QdrantIndex` local | rejects an empty query with an opaque `ValueError` and a NaN query with an `AssertionError` |

## 15. Import-time dependencies

Importing any `optivision` module still runs `optivision/__init__.py`, which imports the original
page recipe and so PIL and PyYAML. `import optivision.core` loads NumPy, PIL and yaml in the
development environment. The layering of `optivision.core` is a source-level property, checked by
`tests/test_core_layering.py`. A NumPy-only import closure is deferred (design decision D2).
`pyproject.toml` lists `pillow`, `pypdfium2`, `qdrant-client`, `pyyaml`, `typer` and `rich` as
required dependencies; SciPy is optional and is needed for the registered default list.

## 16. Numerics and platforms

What was tested: the full test suite ran on Python 3.11 on one Windows laptop. The tests of the new
layer ran against the installed wheel on Python 3.10 (NumPy 1.26.4, SciPy 1.13.1), 3.12 (NumPy
2.0.2) and 3.13 (NumPy 2.5.3), all on the same Windows machine; 3.14 had a smoke test only. Not
tested: Linux, macOS, big-endian machines, GPUs, a Qdrant server, and real encoder output through
the new layer.

Replay compares p-values at a relative tolerance of 1e-12 and everything else exactly; bitwise
equality across platforms is not promised. BLAS kernels, thread counts and block sizes change the
last bits of float32 sums, so near-ties can swap. A GPU product agrees with the NumPy one only to
float32 rounding. The `lloyd2` rotation is regenerated from `(dim, seed)` by the local LAPACK, so
codes written on one platform may refuse to load on another. The `code` field of a certificate
identifies source files, not the numerical libraries beneath them. The certificate digest covers
the recorded environment (Python, NumPy and SciPy versions), so two environments give different
digests for the same decision and evidence. A corpus built directly from Fortran-ordered arrays
(not through `ArrayAdapter`, which makes its items C-contiguous) can give last-bit different norms
and therefore a different certificate digest. Known limit, not fixed.
