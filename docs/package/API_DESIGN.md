# Public API: `certify()` beside `optimize()`

Status: as built, 2026-10-06, branch `package-universal`. `__version__` stays `"0.3.0"` and
nothing here is released (decision D18). Contract: `UNIVERSAL_PACKAGE_DESIGN.md`, section 5.9
and decisions D2, D5, D17, D18. Nothing in this file is a research result; what the research
does and does not support is in `FINAL_RESEARCH_BOUNDARY.md`. Every Python block below is run
as written, in order, in one session. They use synthetic data from `optivision.testing`, which
exercises the code and says nothing about any encoder.

## 1. Two families of entry points

`optimize()` and `calibrate()` are what they were in v0.3.0: same signatures, defaults and
behaviour, and they never call the new layer (a test pins this). `certify()` is a separate
function with a different contract. Pick by the question you are asking.

| | `optimize()` / `calibrate()` (unchanged) | `certify()` (new) |
|---|---|---|
| rule | empirical. The queries are split in two halves (seeded); the smallest candidate whose bootstrap lower bound of retention on the calibration half meets the target is chosen and then measured on the held-out half | a rule built from published tools (testing by betting, fixed-sequence testing). Every candidate is tested on all queries, in a recorded seeded random order (seed 0 unless you choose another) or, if you ask, in the order given; the smallest candidate the rule rejected the hypothesis for is chosen |
| candidates | `search_space=None`: the default list, walked family by family with an early stop | a frozen, ordered, hashed `CandidateSpace`; its identity is recorded |
| nothing qualifies | the corpus comes back as a float32 copy: empty `pipeline`, `target_met is None` | `Outcome.ABSTAIN`: nothing is compressed, `compressed is None`, `store` is your corpus |
| returns | `OptimizationResult(pipeline, compressed, calibration)` | `CertifiedResult` (section 4) |
| states | the retention measured on the held-out half. Nothing about the chosen configuration beyond that check; the bound holds per configuration, not for the one chosen among many | for a selection by one of the library's own rules, the rule's statement (`core.selection.STATEMENT`), claimed only under assumptions A1-A6 and conditions C1-C4 (`report()["assumptions"]`); it is a joint probability and not the probability that this selection is below the target. A rule written by anyone else is never certified. For ABSTAIN and FALLBACK it states nothing about a deployed candidate |
| does not state | that the target is met | a retention figure for the chosen candidate (the certificate keeps descriptive sample statistics under a note that selection biases them); anything about another corpus, query distribution, candidate list or coordinate basis; with the agreement reference, anything about retrieval quality |
| files | `save_compressed` (v1: cannot hold fitted state) | `result.save(dir)`: certificate, codes, fitted state, evidence |

## 2. From vectors to a searchable index and a verifiable artifact

```python
import json

from optivision import Outcome, certify, load_artifact, verify
from optivision.testing import make_corpus, make_queries

# Synthetic stand-ins. Use your encoder's vectors: a list of [n_i, d] float arrays per side.
docs = make_corpus(60, 16, tokens=(8, 12), seed=0)
queries, qrels = make_queries(docs, 150, noise=1.0, seed=1)

result = certify(docs, queries, qrels, target=0.9, error=0.05)  # the default list needs SciPy
print(result)
print(result.outcome is Outcome.SELECT, result.certified, result.compression_ratio)

ranking = result.retriever().search(queries, k=3)  # ranked ids from the chosen index (any outcome)
print(ranking.ids[0])

result.save("artifact")  # certificate, codes, fitted state, evidence
report = verify("artifact", corpus=docs, queries=queries)  # the files, and that they fit these vectors
print(report.ok, report.level)
stored = load_artifact("artifact")  # the same checks; raises ArtifactError if one fails
print(stored.certificate.digest == result.certificate.digest)

docs.save("docs.npz"), queries.save("queries.npz")  # the files the command line below reads
with open("qrels.json", "w") as handle:
    json.dump({q: sorted(d) for q, d in qrels.items()}, handle)
```

The command line does the same on files (`optivision certify --help`, `optivision verify --help`):

```bash
optivision certify docs.npz queries.npz --qrels qrels.json --target 0.9 --out cli_artifact
optivision verify cli_artifact --vectors docs.npz --queries queries.npz --qrels qrels.json
```

`certify` prints the outcome plainly and exits 0 for every outcome, an abstention included
(`--json` prints `report()` on stdout; read `outcome` to branch); invalid input exits 2.
`verify` prints every check and exits 1 when one fails. Further options: `--reference`,
`--space default|wide`, `--budget`, `--order-seed` (default 0), `--given-order`, `--metric`,
`--error`, and for `verify` `--evidence`, `--key-file`, `--strict-code`. The queries are tested in
a seeded random order unless `--given-order` asks for the order of the file.

`verify` accepts an artifact directory, a `certificate.json` path or a `Certificate`. It reports
and never raises on a bad certificate; `report.ok` means no check failed, which does not make
the certificate true. The levels are integrity (the certificate alone), replay (the decision is
recomputed from the stored evidence) and binding (the live `corpus`, `queries`, `reference`,
pipeline and codes are the ones certified). `verify_certificate(cert, evidence=..., ...)` is the
same check on a `Certificate` you already hold.

## 3. `certify` parameters

`certify(vectors, queries, qrels=None, *, target=0.97, error=0.05, metric="ndcg@5",
reference="auto", space=None, rule=None, query_budget=None, order_seed=0, fallback=None,
encoder=None, progress=None, created=None) -> CertifiedResult`

| parameter | meaning |
|---|---|
| `vectors`, `queries` | A `MultiVectorCorpus` or a list of `[n_i, d]` float arrays, as `optimize()` takes them; checked with `validate_corpus` (finite, unique ids, one dimension); never modified; the dtype is kept. One query is enough to run, but certification needs far more: a perfect candidate at target 0.95, error 0.05 and 7 families needs about 116 queries if the uncompressed index scores 1 on every query and about 195 if it scores 0.6 on average, and target 0.99 needs about 600 to 1,000 (`STATISTICAL_REVIEW.md`, section 9). The queries are the sample the statement is about: draw them at random from what the index will serve and remove exact and near-duplicates first. Exact duplicates are counted, recorded in the certificate and warned about once (`UserWarning`), not refused. |
| `qrels` | `{query id: [relevant document ids]}`; ids are those of the two corpora (`"0"`, `"1"`, ... for lists). |
| `target`, `error` | The retention to be exceeded, in (0, 1], and the level of the statement, in (0, 1). One pair per call (section 5). |
| `metric` | `ndcg@k`, `recall@k`, `mrr@k` or `hit@k`. Only nDCG@k was evaluated by the registered experiments. |
| `reference` | `"auto"` (labels if `qrels` is given, else agreement), `"labels"`, `"agreement"`, or a `LabelReference` / `RankingReference`. Agreement is agreement with the uncompressed ranking's own top document, which is not retrieval quality; the result, the report and the certificate say which was tested. `qrels` together with `"agreement"` or with a reference object is refused, not ignored. |
| `space` | `None`: the library's default list (raises `ImportError` with advice without SciPy; never reduced silently). Or a `CandidateSpace`, or a dict `{family: [Pipeline, ...]}` as `calibrate(search_space=...)` takes it, each family ordered from least to most aggressive. |
| `rule` | `FixedSequenceBetting()` by default; `BonferroniBetting()` for a list without a meaningful order. These two are the only rules the library certifies for. Any other object with the protocol's attributes is accepted and recorded with its class and what it declared, but its selection is never `certified`: the certificate says the library has not examined it. |
| `query_budget`, `order_seed` | Test a prefix of the order; and the seed of the random permutation the queries are tested in (default `0`, recorded). `order_seed=None` tests them in the order given, which is valid only if that order is unrelated to the outcomes (the arrival order of independent draws; never sorted by difficulty, date, source or any other query attribute). The default shuffle exists so that a sorted input file cannot break assumption A5. Choose the seed before you look and use it once: trying several seeds and reporting one is outside the statement (in a simulation the best of 20 seeds had an error of 0.076 where 0.05 was stated, `STATISTICAL_REVIEW.md`, section 7.1). A budget keeps a prefix of the order and is safe to choose after looking; a seed is not. |
| `fallback` | A `Pipeline` (or its dict) returned, uncertified, when nothing is certified. No default. |
| `encoder` | An `EncoderIdentity` or its dict, recorded in the certificate and used for nothing else. If omitted, `vectors.attrs["encoder_identity"]` (written by the adapters) is recorded. |
| `progress`, `created` | `progress(position, n_candidates, family)` after each candidate; the creation time to record (ISO string ending in `Z`, or an aware `datetime`). With the same inputs and `created` the certificate is identical. |

Caller obligations, which no code can check and the statement assumes (the assumptions A1-A6 and
conditions C1-C4 in every certificate): the candidate list was fixed before these queries were
looked at (`certify` never lets the queries change it); the queries are independent draws from the
distribution the index will serve; the order they are tested in does not depend on their outcomes;
one `(target, error)`, one rule, one metric and one reference are used once per sample; the
reference is fixed without regard to the candidates; the index you deploy is the one that was
evaluated. The vectors are expected to have unit L2 norm, the only regime the research studied:
the certificate records whether they do and says so in `caveats`, and does not refuse them.

## 4. The result

`CertifiedResult` is a frozen dataclass; `certified` is True only for a selection.

| attribute | content |
|---|---|
| `decision`, `outcome`, `certified`, `selected` | the `Decision` of the rule, its `Outcome`, whether the rule certified the choice, and the chosen `CandidateSpec` (`None` on ABSTAIN; position -1 on FALLBACK) |
| `pipeline`, `compressed` | the fitted pipeline and the corpus compressed with it; both `None` on ABSTAIN |
| `certificate`, `evidence`, `corpus` | the record (`Certificate`), the per-query arrays it was computed from, and the corpus the call ran on |
| `store` | what to search: `compressed`, or your corpus on ABSTAIN |
| `compression_ratio` | float32 bytes of the input over stored bytes, as `OptimizationResult.compression_ratio` defines it; 1.0 when nothing was compressed. A size, not a quality figure |
| `retriever()` | a `CompressedRetriever`, or an `ExactRetriever` over your corpus on ABSTAIN: queries in, a `Ranking` out |
| `save(path, *, include_evidence=True, overwrite=False)` | writes the artifact directory; refuses a non-empty directory unless `overwrite=True` |
| `report()` | a JSON-native summary: outcome, certified, construct, metric, target, error, level, the rule (name, class, built in or not, certifying or not, what it declared), queries used, how many repeat an earlier one exactly (`n_exact_duplicate_queries`), query order, space identity, selected label and pipeline, compression ratio, `basis_sensitive`, the statement and assumptions, `caveats`, `not_certified_note` |

No attribute holds a retention figure. The certificate keeps the descriptive sample statistics of
a selected candidate (`certificate.to_dict()["sample"]`) under a note that selection biases them.

## 5. Outcomes, and what to do on ABSTAIN

| outcome | `certified` | meaning |
|---|---|---|
| `SELECT` | True for the library's own rules | the rule rejected "retention <= target" for at least one candidate; the smallest by the library's size key was chosen |
| `ABSTAIN` | False | nothing was certified; the uncompressed index is the answer. The normal, safe outcome: no exception, no warning |
| `FALLBACK` | False | nothing was certified and you supplied `fallback`; it is returned and recorded as not certified |

```python
few = queries.select(range(8))  # too few queries to certify a target this strict
stopped = certify(docs, few, qrels, target=0.999, error=0.05)
print(stopped.outcome, stopped.certified, stopped.compressed, stopped.pipeline, stopped.compression_ratio)
print(stopped.store is docs, type(stopped.retriever()).__name__)
print(stopped.report()["not_certified_note"])
```

An abstention does not say that no candidate would pass. Your options are to collect new,
independent queries, to choose another target for that new sample, or to keep the index
uncompressed. Each is a decision to take before looking at the new queries: the statement covers one
`(target, error)` on one sample, so trying several targets, error levels, lists, orders or seeds on
the same queries and reporting one spends error level that the statement does not account for (in
simulations such looks multiplied the false-selection probability by 3 to 14,
`STATISTICAL_REVIEW.md`, section 7.7). After an abstention, a new run needs new queries. Several
targets cannot be passed to one call: a list is refused.

## 6. Composing the core pieces

`certify` is the sequence below and adds no selection logic. Use the pieces when you need your
own reference, metric, candidate generator or evidence collection. The obligations stay yours.

```python
from optivision import CandidateSpace, Outcome
from optivision.core.artifact import create_certificate
from optivision.core.evaluate import LabelReference
from optivision.core.evaluate.collect import collect_evidence, compress_selected
from optivision.core.selection import select

CREATED = "2026-10-06T00:00:00Z"
space = CandidateSpace.default(docs.dimension)
reference = LabelReference(qrels)
evidence = collect_evidence(docs, queries, space, reference, metric="ndcg@5")
decision = select(space, evidence, target=0.9, error=0.05, order_seed=0)  # certify's default order
pipeline = compressed = None
if decision.outcome is not Outcome.ABSTAIN:
    pipeline, compressed = compress_selected(
        docs, decision.selected, expected_fitted=evidence.fitted[decision.selected.position])
certificate = create_certificate(decision, evidence=evidence, space=space, corpus=docs,
                                 queries=queries, reference=reference, pipeline=pipeline,
                                 compressed=compressed, created=CREATED)
same = certify(docs, queries, qrels, target=0.9, error=0.05, created=CREATED)
print(certificate.digest == same.certificate.digest)
```

`select` itself tests the queries in the order given unless `order_seed` is passed, which is why the
block passes the seed that `certify` uses by default. For a FALLBACK pass `decision.selected`
without `expected_fitted` (the evidence has no entry for a pipeline outside the list).

## 7. Errors

Arguments are checked before any candidate is scored, so a typo costs nothing.

| condition | raises |
|---|---|
| no queries; an empty list; invalid, non-finite or duplicate-id vectors; dimensions differ | `ValueError` / `ContractError` (a `ValueError`) |
| `target` outside (0, 1], `error` outside (0, 1), or not one real number | `TargetError`, `ErrorLevelError` (both `ValueError`) |
| bad `query_budget` or `order_seed`; a budget above the queries given | `QueryOrderError` (`ValueError`) |
| `reference="labels"` without `qrels`; `qrels` with `"agreement"` or a reference object | `ValueError` |
| wrong type for `space`, `fallback`, `reference` or `encoder`; an invalid encoder dict | `TypeError`; `ContractError` |
| default list without SciPy | `ImportError` (advice in the message) |
| a candidate cannot be evaluated | `CandidateEvaluationError` (a `RuntimeError`); a candidate is never skipped |
| no query has a relevant document; the uncompressed index scores 0 on all of them | `ValueError`; `ZeroBaselineError` |
| the chosen pipeline fits differently than it was evaluated | `FittedStateMismatchError` |
| some tested queries repeat an earlier query exactly | a `UserWarning` (once per call); nothing is refused |
| `save` into a non-empty directory; an unreadable or failing artifact | `FileExistsError`; `ArtifactError` |

```python
for bad in ({"target": 1.5}, {"error": 0.0}, {"reference": "labels"}):
    try:
        certify(docs, queries, **bad)
    except ValueError as exc:
        print(type(exc).__name__, "-", str(exc)[:60])
```

## 8. What is deliberately not in the API

- No automatic fallback to float16 or to anything else: float16 is one candidate and has to be
  certified like any other. Without `fallback=`, nothing certified means the uncompressed index.
- No per-model presets and no `model=` argument: the default list, `target` and `error` are the
  same for every encoder. The list was shaped by results on the development encoders; using it
  on another encoder is an untested transfer, not a finding.
- No "best" configuration: the choice among certified candidates is the smallest by the size
  key (`bytes_per_doc`; fitted-state bytes are listed separately). There is no quality ranking
  and no quality-for-size trade-off.
- No hold-out split and no held-out retention figure: all queries are tested. A check on fresh
  queries is a new run on those queries.
- No retry, no target search, no list edited after a look.
- No refusal of repeated queries or of vectors that are not unit-norm: they are counted or measured,
  recorded in the certificate and said in its `caveats`. Near-duplicates are not detected.
- No certification of a rule the library did not derive: another rule is recorded, never certified.
- No silent conversion: dtypes are kept, and a float16 corpus stays float16 (the ratio is vs
  float32). No device selection and no new environment variable.

## 9. Stability

Everything here is new on this branch and may change before a release: the `certify` parameter
set, the keys of `report()`, the CLI options of `certify` and `verify`, the artifact and
certificate formats (both version 1, versioned by their `schema_version`), and the lazy
top-level names (`certify`, `verify`, `verify_certificate`, `load_artifact`, `save_artifact`,
`CertifiedResult`, `Certificate`, `CandidateSpace`, `Decision`, `Outcome`, `EncoderIdentity`),
which load on demand so that `import optivision` does not pay for the certificate modules.
Unchanged: `optimize`, `calibrate`, the 40 names of `__all__`, the v1 file formats, the
existing CLI commands. Certificates record the library version and a digest of the
decision-relevant source files.

## 10. Against the conceptual sketch

The sketch was `optivision.optimize(reference=..., candidates=..., queries=..., target=0.97,
error=0.05)`. The built function is `certify` for three reasons: (a) `optimize` has callers
and a contract (held-out check, `**calibrate_kwargs`, a float32 result when nothing qualifies);
giving `target`/`error` a second meaning there would change what existing results mean (D17).
(b) the rule needs more than a bag of pipelines: an order inside each family fixed in
advance, and the number of families, which sets the test level. So `candidates=` became
`space=`, a `CandidateSpace` or an ordered dict of families. (c) the result differs in kind:
it may hold no compressed index, and it carries a certificate. `reference` and `queries` map
directly; `qrels` stays for parity with `optimize`. Added: `rule`, `query_budget`, `order_seed`,
`fallback`, `encoder`, `progress`, `created`, `metric`.
