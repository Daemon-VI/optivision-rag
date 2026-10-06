# Quickstart

Status: 2026-10-06, branch `package-universal`. From your encoder's vectors to a decision, a
verifiable artifact and a search. Every code block on this page runs, in order, in one session
(`tests/test_docs_examples.py` runs them).

The data are synthetic, made by `optivision.testing`. They only exercise the code. Nothing printed
here is a result about any encoder or about retrieval quality, and the outputs shown come from one
machine, so the last digits can differ on yours. The entry point is `optivision.certify()`; its
design is in [API_DESIGN.md](../API_DESIGN.md), and the ideas are in [CONCEPTS.md](CONCEPTS.md).
Install SciPy (`pip install "optivision-rag[merge]"`) for the default candidate list.

## 1. Your vectors in

An encoder gives you one `[n_i, d]` float array per document and per query. `ArrayAdapter` checks
them (floating, finite, one dimension, one dtype, unique ids) and returns the library's corpus
type. It never changes a dtype or moves data between devices for you.

```python
from optivision import certify, verify
from optivision.adapters import ArrayAdapter
from optivision.core.vectors import EncoderIdentity
from optivision.testing import make_corpus, make_queries

corpus = make_corpus(60, 32, seed=0)  # synthetic: 60 documents of 32-d unit vectors
sample, qrels = make_queries(corpus, 200, seed=1, noise=3.0)  # 200 queries with known relevant documents

# What your encoder hands over: one [n_i, d] float array per document and per query.
doc_arrays = [corpus.vectors[slice(*corpus.span(i))] for i in range(len(corpus))]
query_arrays = [sample.vectors[slice(*sample.span(i))] for i in range(len(sample))]

# Check and wrap them: floating, finite, one dimension, one dtype, unique ids.
adapter = ArrayAdapter(EncoderIdentity(model_id="my-encoder", dimension=32, modality="other"))
docs = adapter.encode_documents(doc_arrays, ids=corpus.ids)
queries = adapter.encode_queries(query_arrays, ids=sample.ids)
print(len(docs), "documents,", docs.num_vectors, "vectors;", len(queries), "queries")
```

```text
60 documents, 719 vectors; 200 queries
```

## 2. Decide, with labels

`qrels` maps each query id to the ids of its relevant documents. The ids are those of the two
corpora. `target` is the retention a selected candidate must exceed; `error` is the level of the
rule's statement. Choose both before you look at any result, and use one pair per sample of
queries.

```python
result = certify(docs, queries, qrels, target=0.95, error=0.05, encoder=adapter.identity())
print(result)
print(result.outcome.value, "| certified:", result.certified, "| compression:",
      f"{result.compression_ratio:.2f}x vs float32")
print("certified candidates:", len(result.decision.certified_positions), "of",
      len(result.decision.p_values))
print("selected:", result.pipeline.label(), "| level per test:", round(result.decision.level, 4))
```

```text
CertifiedResult(outcome='select', certified=True, 'adaptive_merge(0.6) > int8(per_vector)', 3.77x vs float32; reference=labels, metric=ndcg@5, target=0.95, error=0.05, n_queries=200)
select | certified: True | compression: 3.77x vs float32
certified candidates: 7 of 40
selected: adaptive_merge(0.6) > int8(per_vector) | level per test: 0.0071
```

The rule tested all 40 candidates of the default list, certified 7, and chose the smallest. The
level per test is `error / 7`, because the list has 7 families. "Certified" means that the rule's
statement covers this choice, under its assumptions. It is not a measured retention. The result
carries the statement and the assumptions, and says whether the index is basis sensitive:

```python
report = result.report()
print(report["statement"])
print(report["assumptions"][0])
print("basis sensitive:", report["basis_sensitive"], "| invariance:", report["basis_invariance"])
```

```text
P(a compressed candidate is selected and its retention under Q is below T) <= delta, where the probability is over the draw of the calibration queries and retention under Q is E_Q[metric of the candidate] / E_Q[metric of the uncompressed index]. It holds for every sample size, with no large-sample approximation, and only under assumptions A1-A6. It makes no statement about power: it also holds if the rule always abstains. It is an application of published tools; the derivation is this project's own and has not been externally reviewed.
A1: the calibration queries are independent draws from one distribution Q, the deployment query distribution.
basis sensitive: True | invariance: signed_permutation
```

Search the chosen index. `result.retriever()` projects the queries with the fitted pipeline and
scores the decoded codes:

```python
ranking = result.retriever().search(queries, k=3)
print(ranking.ids[0], ranking.score_kind)
```

```text
['d000020', 'd000019', 'd000055'] maxsim_ip
```

## 3. What changes without labels

Without `qrels` the default reference is agreement with the uncompressed ranking.

| | with `qrels` | without `qrels` |
|---|---|---|
| reference (`reference="auto"`) | `labels` | `agreement@1` |
| relevant documents of a query | the documents listed for it | the one document the uncompressed index ranks first |
| the uncompressed index scores | its metric against the labels | 1 on every query, by construction |
| what is tested | retention against the labels | fidelity to the uncompressed ranking, mistakes included |
| queries dropped | those with no relevant document in the labels | none |

Agreement is not retrieval quality. The result and the certificate say which construct was
tested. A new call needs a new sample of queries:

```python
fresh, _ = make_queries(corpus, 200, seed=2, noise=3.0)  # a new sample: one (target, error) per sample
no_labels = certify(docs, fresh, target=0.95, error=0.05)
print(no_labels.decision.construct, "|", no_labels.outcome.value, "| certified:", no_labels.certified)
print(no_labels.report()["caveats"][0])
```

```text
agreement@1 | select | certified: True
Retention here means agreement with the uncompressed ranking, which is not retrieval quality. The uncompressed index's own top document is taken as the relevant one, so the measure is fidelity to that ranking, including its mistakes, and no statement in this record is a statement about relevance.
```

On this synthetic sample both references selected a candidate. On the development stores of the
registered experiments agreement was the stricter one (`reports/research/X_RESULTS.md`, X1).
Neither direction carries over to your data.

## 4. What an ABSTAIN looks like

A strict target on another sample. Nothing is certified, no index is compressed, and the answer
is the uncompressed index.

```python
strict, strict_qrels = make_queries(corpus, 200, seed=3, noise=3.0)
stopped = certify(docs, strict, strict_qrels, target=0.99, error=0.05)
print(stopped.outcome.value, "| certified:", stopped.certified, "| compressed:", stopped.compressed,
      "| ratio:", stopped.compression_ratio)
print(stopped.report()["not_certified_note"])
print(type(stopped.retriever()).__name__, "searches the original vectors")
print("smallest p-value:", round(min(stopped.decision.p_values), 4), "| level:", round(stopped.decision.level, 4))
```

```text
abstain | certified: False | compressed: None | ratio: 1.0
Nothing was certified at this target and error level on these queries, so the uncompressed index is the outcome. This is the safe outcome, not an error. It does not say that no candidate would pass: with more queries, or a lower target, a later run may certify one. Decide on that before looking again: every further look at the same queries, with another target, error level or list, is outside the statement of the rule.
ExactRetriever searches the original vectors
smallest p-value: 0.878 | level: 0.0071
```

Abstaining is a correct outcome, not an error. What to do:

1. Use the uncompressed index.
2. Do not change the target, the error level or the list and run again on the same queries.
3. For another attempt, collect a new independent sample, and choose the target and the list
   beforehand. More queries can certify what fewer could not, if a candidate really keeps more than
   the target. Nothing says how many are enough.
4. To accept an uncertified compressed index, pass `fallback=Pipeline([...])`. The outcome is then
   `FALLBACK`, recorded as not certified.

[SELECTION_AND_CERTIFICATION.md](SELECTION_AND_CERTIFICATION.md), section 7, has the detail.

## 5. Save, load and verify

`result.save(path)` writes a directory: the certificate, the codes, the fitted state of every
stage that has one, the evidence, and a manifest of file digests. No pickle is written. `verify`
reports every check it made, and every check it did not make.

```python
import tempfile
from pathlib import Path

from optivision.core.artifact import load_artifact
from optivision.core.evaluate import LabelReference

workdir = tempfile.TemporaryDirectory()
path = result.save(Path(workdir.name) / "artifact")
print(sorted(p.name for p in path.iterdir()))

stored = verify(path)
print("stored artifact ok:", stored.ok, "| not run:", [c.name for c in stored.checks if c.status == "skipped"])
live = verify(path, corpus=docs, queries=queries, reference=LabelReference(qrels))
print("live objects ok:", live.ok, "| not run:", [c.name for c in live.checks if c.status == "skipped"])
```

```text
['MANIFEST.json', 'certificate.json', 'codes.npz', 'evidence.npz', 'fitted.npz']
stored artifact ok: True | not run: ['signature', 'binding.corpus', 'binding.queries', 'binding.evidence_queries', 'binding.reference']
live objects ok: True | not run: ['signature']
```

From the directory alone, `verify` checks integrity, replays the decision from the stored
evidence, and binds the stored codes to the certificate. Passing the live `corpus`, `queries` and
`reference` also checks that the certificate was issued for them. `ok` means that no check failed.
It does not make the certificate true, and without a signature key the digest detects edits but
does not stop a forger.

Load the artifact and search it. The loaded index scores like the saved one:

```python
from optivision.core.retrieve import CompressedRetriever

artifact = load_artifact(path)  # checks the manifest, restores the fitted state, verifies
loaded = CompressedRetriever(artifact.compressed, artifact.pipeline).search(queries, k=3)
print("same ranking after saving and loading:", loaded == ranking)
```

```text
same ranking after saving and loading: True
```

Edit one number in the certificate and the checks fail:

```python
import json

cert_file = path / "certificate.json"
record = json.loads(cert_file.read_text(encoding="utf-8"))
record["target"] = 0.5  # edit one number
cert_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
edited = verify(path)
print("after the edit ok:", edited.ok, "| failed:", [c.name for c in edited.failed()])
workdir.cleanup()
```

```text
after the edit ok: False | failed: ['artifact files', 'digest', 'replay.decision']
```

## 6. Where next

- Runnable scripts: [examples/](examples/), one per task, each printing a short result.
- Your own candidate list: `examples/custom_candidate_space.py`.
- From a shell: `optivision certify --help` and `optivision verify --help`.
- Searching with a backend or two tiers: [BACKENDS_GUIDE.md](BACKENDS_GUIDE.md).
- Your own encoder: [ADAPTERS_GUIDE.md](ADAPTERS_GUIDE.md).
- What the statement covers, and what you must supply:
  [SELECTION_AND_CERTIFICATION.md](SELECTION_AND_CERTIFICATION.md).
- What the package does not do: [LIMITATIONS.md](LIMITATIONS.md).
