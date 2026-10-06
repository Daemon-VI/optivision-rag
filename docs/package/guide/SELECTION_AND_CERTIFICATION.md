# Selection and certification

Status: 2026-10-06, branch `package-universal`. This page explains how a decision is made, what a
certificate records, and what both do and do not say. Terms are defined in `CONCEPTS.md`. The
exact specifications are `../METHOD_C_SPEC.md` (the rule) and `../CERTIFICATE_SPEC.md` (the
record).

The rule selects a candidate or abstains. Whether it selects anything depends on the encoder, the
coordinate basis, the corpus, the queries, the number of queries, the target and the reference.
Abstaining is a correct outcome.

## 1. What the rule tests

For every candidate `k` of the list, the rule tests the hypothesis `H0_k: retention(k) <= T`.
Retention under the query distribution Q is `E_Q[c_k] / E_Q[b]`, where `c_k` is the per-query
metric of candidate `k` and `b` that of the uncompressed index. Rejecting `H0_k` is the claim that
retention exceeds `T`. A candidate is **certified** when, and only when, its hypothesis is
rejected. Because the mean of `b` is positive (assumption A4), retention exceeds `T` exactly when
the mean of `c_k - T * b` is positive, which is a statement about the mean of a bounded quantity:
the setting of the betting tests below.

## 2. How it tests

1. **A p-value per candidate.** The queries are taken in the order given, or in a recorded seeded
   permutation. A betting test stakes, query by query, on the claim that the candidate scores more
   than `T` times the uncompressed index. Wealth grows when the candidate scores above what the
   target requires. The p-value is 1 divided by the largest wealth reached, at most 1.
2. **An error level per family.** The test level is `error / F`, where `F` is the number of
   distinct family labels in the list passed. Changing the list changes `F`.
3. **Fixed-sequence testing along each family.** Walk a family in list order, from least to most
   aggressive. Certify a candidate when its p-value is at most the level. At the earliest
   candidate that is not certified, stop that family: its later members are never certified,
   however small their p-values.
4. **The choice.** Among all certified candidates, in all families, take the one with the
   smallest `(bytes_per_doc, vectors_per_doc, position)`, compared exactly. `bytes_per_doc` is the
   library's stored size per document: codes plus codec overhead, without fitted-state bytes such
   as a PCA matrix. The choice uses no statistics, and any rule for picking a member of the
   certified set keeps the statement.
5. **Nothing certified.** The outcome is `ABSTAIN`: use the uncompressed index.

Nothing in the rule is random. The same evidence, target, error level and order give the same
decision. `FixedSequenceBetting` is the default rule. `BonferroniBetting` tests every candidate
at `error / K` with no order and no stopping, for lists without a meaningful order.

## 3. A runnable illustration

The numbers are synthetic. They exercise the code and describe no encoder. The two decisions use
two independent samples, because one `(target, error)` belongs to one sample.

```python
import numpy as np

from optivision import AdaptiveMerge, BinaryQuantizer, Int8Quantizer, Pipeline
from optivision.core.evaluate import Evidence
from optivision.core.selection import CandidateSpace, select

space = CandidateSpace.from_families(
    {
        "int8": [Pipeline([Int8Quantizer("per_vector")]),
                 Pipeline([AdaptiveMerge(radius=0.8), Int8Quantizer("per_vector")]),
                 Pipeline([AdaptiveMerge(radius=0.6), Int8Quantizer("per_vector")])],
        "binary": [Pipeline([BinaryQuantizer()]),
                   Pipeline([AdaptiveMerge(radius=0.8), BinaryQuantizer()]),
                   Pipeline([AdaptiveMerge(radius=0.6), BinaryQuantizer()])],
    },
    name="illustration",
)


def sample_evidence(seed, n=450):  # synthetic per-query metric values
    rng = np.random.default_rng(seed)
    baseline = rng.uniform(0.4, 1.0, size=n)
    loss = np.array([0.00, 0.02, 0.10, 0.04, 0.12, 0.40])  # mean fraction of the baseline lost
    candidates = np.clip(baseline * (1.0 - 2 * loss[:, None] * rng.random((6, n))), 0.0, 1.0)
    return Evidence(
        metric="ndcg@5", construct="labels", baseline=baseline, candidates=candidates,
        query_ids=tuple(f"q{i}" for i in range(n)),
        bytes_per_doc=np.array([4000.0, 2000.0, 1300.0, 1000.0, 500.0, 330.0]),
        vectors_per_doc=np.array([100.0, 50.0, 33.0, 100.0, 50.0, 33.0]),
        space_identity=space.identity,
    )


for seed, target in ((0, 0.95), (1, 0.99)):  # one (target, error) per sample of queries
    d = select(space, sample_evidence(seed), target=target, error=0.05)
    chosen = None if d.selected is None else d.selected.position
    print(f"target {target}: {d.outcome.value}, certified={d.certified}, "
          f"certified positions={d.certified_positions}, selected={chosen}, level={d.level:.4f}")
    print("  p-values:", [round(p, 4) for p in d.p_values])
```

```text
target 0.95: select, certified=True, certified positions=(0, 1, 3), selected=3, level=0.0250
  p-values: [0.0, 0.0001, 1.0, 0.024, 0.9986, 1.0]
target 0.99: abstain, certified=False, certified positions=(), selected=None, level=0.0250
  p-values: [0.0895, 1.0, 1.0, 1.0, 1.0, 1.0]
```

In the first decision two families give a level of 0.05 / 2 = 0.025. In `int8`, positions 0 and
1 pass and position 2 does not, so that family stops. In `binary`, position 3 passes (0.024) and
position 4 does not. Three candidates are certified, and position 3 is the smallest by stored
size. In the second decision the candidate that loses nothing on the sample has a p-value of about
0.09, above 0.025, so nothing is certified.

## 4. The statement

This is `STATEMENT` in `src/optivision/core/selection/rules.py`, verbatim:

> P(a compressed candidate is selected and its retention under Q is below T) <= delta, where the
> probability is over the draw of the calibration queries and retention under Q is E_Q[metric of
> the candidate] / E_Q[metric of the uncompressed index]. It holds for every sample size, with no
> large-sample approximation, and only under assumptions A1-A6. It makes no statement about power:
> it also holds if the rule always abstains. It is an application of published tools; the
> derivation is this project's own and has not been externally reviewed.

The assumptions, verbatim from `FixedSequenceBetting.assumptions()`:

> A1: the calibration queries are independent draws from one distribution Q, the deployment query
> distribution.
>
> A2: the corpus is fixed; retention moves with corpus size, so the statement is about this
> corpus.
>
> A3: the candidate list, its family labels and chains, and the order inside each chain are fixed
> before the calibration queries are seen, and every fitted stage is fitted on documents only.
>
> A4: the per-query metric lies in [0, 1] and the expected metric of the uncompressed index under
> Q is positive.
>
> A5: the order in which queries enter the sequential test is fixed before their outcomes are
> seen.
>
> A6: the reference is taken as given (the labels under the label construct, the uncompressed
> index's own top document under the agreement construct); errors in the reference are not
> modelled.

`error` is delta and `target` is `T`. `BonferroniBetting` makes the same statement, and its A3
asks only for a fixed list and fitting rule, because it needs no order inside a family.

## 5. How to read it

- **A probability over repeated samples.** Picture many independent samples of queries from Q and
  the rule run on each. In at most a fraction `error` of those runs does the rule both select a
  compressed candidate and select one whose retention under Q is below `T`. Runs that abstain are
  not counted as wrong. The corpus, list, order and fitted stages are fixed, not random.
- **Not a probability conditional on a selection.** It bounds the joint event "selected and below
  target", not "below target given that the rule selected". A rule that rarely selects can meet
  the statement while many of its selections are below target. It says how often the procedure
  goes wrong, not how reliable one selection is.
- **Not a statement about the retention measured on the sample.** The sample retention of the
  selected candidate is optimistically biased by the selection. The certificate records it under
  `sample` and says it is not certified. The p-values and the statement are the valid quantities.
- **Not about a finite batch of future queries.** Retention under Q is an expectation; the
  retention on the next hundred queries fluctuates around it.
- **Not about power.** The statement also holds if the rule never selects anything.
- **About this corpus and this distribution Q only.** Under `agreement@1`, "retention" means
  agreement with the uncompressed ranking, which is not retrieval quality.

## 6. What you are responsible for

| assumption | what you must do | can the software check it? |
|---|---|---|
| A1 | supply queries that are an honest sample of what the index will serve, drawn independently; near-duplicates weaken independence (`../METHOD_C_SPEC.md`, section 1.4) | no; it records what you supplied |
| A2 | certify the corpus you will serve; adding, removing or re-encoding documents changes it | partly: the certificate binds a digest of the corpus, and `verify` compares it when you pass the live corpus |
| A3 | fix the candidate list before looking at any query result, and do not edit it afterwards | partly: the collector never passes queries to `fit` and refuses a stage that declares `fit_uses_queries`; it cannot know that you edited the list after a look |
| A4 | use a metric in [0, 1]; the uncompressed index must score above zero on average | yes: the rule raises on values outside [0, 1], non-finite values and a zero baseline mean |
| A5 | keep the order of the queries as given, or pass `order_seed` (a seeded permutation independent of the data, recorded in the decision); do not order by outcome or try several orders and keep the best | partly: the order is recorded and reproducible; independence from outcomes cannot be checked |
| A6 | accept the reference as given; label errors are not modelled | no |
| one `(target, error)` per sample | do not try several targets, error levels or lists on one sample and report one | no: the package cannot see your other calls |

## 7. Practical guidance

### 7.1 How many queries

More queries make certification more likely for a candidate whose retention is above the target.
Nothing says how many are enough. For one candidate the p-value can only fall or stay the same as
queries are appended in the same order (`../METHOD_C_SPEC.md`, section 2.3), but the statement is
made for a sample whose size and order were fixed in advance. The project documents do not address
stopping when a candidate happens to be certified, so decide the number of queries before you look.

A floor, not an estimate. The E1 design derives a limit for any valid test
(`reports/research/E1/DESIGN.md`; labelled inference there; not externally reviewed). Even for a
candidate that loses nothing, rejecting its hypothesis with probability at least one half needs
`n >= log(1 / (2 * level)) / -log(1 - eps)`, where `eps = beta (1 - T) / (T + beta (1 - T))` and
`beta` is the mean metric of the uncompressed index.

```python
import math


def queries_needed(target, level, mean_baseline):
    """Lower limit for a candidate that loses nothing (reports/research/E1/DESIGN.md)."""
    eps = mean_baseline * (1 - target) / (target + mean_baseline * (1 - target))
    return math.ceil(math.log(1 / (2 * level)) / -math.log(1 - eps))


for target in (0.99, 0.97, 0.95):  # beta = 0.584 is the mean nDCG@5 of one development store
    print(target, queries_needed(target, 0.05, 0.584), queries_needed(target, 0.05 / 7, 0.584))
```

```text
0.99 392 723
0.97 129 238
0.95 77 141
```

The columns are the level 0.05 and the level 0.05 / 7 of the seven-family default list. A real
candidate loses something, so the number needed is larger.

### 7.2 Strict targets

Strict targets may never certify at realistic budgets. Observed on the seven development stores
(X2; `reports/research/X_RESULTS.md`, sections 3 and 4; SIMULATED, pool-supported: resamples of
one pool of queries per corpus, never more queries than the pool holds):

- At `T = 0.97`, `error = 0.05`, with every query the pool supplied (450, or 300 on SciFact), the
  default procedure deployed non-trivial compression (a candidate other than float16) in at least
  0.90 of the resampled samples on all seven stores. The lowest was 0.917 on SciFact: close to the
  0.90 threshold.
- Three stricter cells failed: `T = 0.99` non-trivial on all seven stores (at most 0.010 of the
  samples on any store, `reports/research/X2/x2_<store>.json`); `T = 0.97` substantial (a deployed
  candidate with at least 10x compression) on the three DocVQA stores and SciFact; `T = 0.99`
  substantial on all seven.
- A result at 900, 1,800 or 3,600 queries redraws the pool's own queries. It does not show what
  that many independent queries would give, and it is not claimed that more queries would remove
  an abstention that the registered budgets did not remove
  (`reports/research/NEXT_PHASE_PREREGISTRATION.md`, section 25, non-claim 7).

These statements are about those stores, one pool each, and the registered list of 40 pipelines.
They are not an estimate for yours. Run the rule once for each `(target, error)` and sample of
queries: changing the target, the error level or the list after a result spends error level that
the statement does not account for (`../METHOD_C_SPEC.md`, section 6.3).

### 7.3 When the outcome is ABSTAIN

1. Use the uncompressed index. It is the outcome, not an error.
2. Read what the decision recorded: `decision.p_values` and `decision.certified_positions` show
   how close each candidate came. Treat that as information. Do not edit the list or the target
   and run again on the same queries.
3. For another attempt, collect a new, independent sample and choose the target and the list
   beforehand. A larger sample can certify a candidate that a small one could not, if the
   candidate really keeps more than the target. Nothing says it will.
4. To accept a compressed index without certification, pass an explicit `fallback` pipeline. The
   outcome is then `FALLBACK`, recorded as not certified, and the statement does not cover it.
5. If you suspect the encoder or its coordinate basis, `BASIS_DEPENDENCE.md` has diagnostics.

## 8. What a certificate records

As specified in `../CERTIFICATE_SPEC.md`; the code is in `src/optivision/core/artifact/`, and you
should read it before relying on a detail. A certificate is issued for every outcome. It records
the candidate list, the identities of the corpus, queries and reference, the rule, the statement
and assumptions verbatim, the target, error level and level, the p-values, and for a selection the
deployed pipeline with digests of its fitted state and codes. A `caveats` field repeats what a
reader must not miss: agreement is not retrieval quality, and a fallback is not certified.

Verification reports instead of raising. Level 1 (integrity) needs the certificate alone. Level 2
(replay) re-runs the decision from the evidence. Level 3 (binding) compares the live corpus,
queries, reference, pipeline and compressed corpus with the recorded identities. Each check is
`pass`, `fail`, `skipped` or `differs`, and `ok` means that no check failed. It does not make the
certificate true. The digest detects accidental and naive edits and can be recomputed by anyone
who can edit the file; an HMAC with a key the forger does not hold closes that gap, is symmetric
and gives no non-repudiation. A certificate says nothing about a re-encoded, rotated, extended or
truncated corpus.

## 9. How this differs from `optimize()` and `calibrate()`

| | `optimize()` / `calibrate()` | `select()` / `optivision.certify()` |
|---|---|---|
| rule | per-candidate bootstrap lower bound on retention, default level 0.999, on a calibration half of the queries | betting p-values on all queries, fixed-sequence testing along each family at `error / F` |
| held-out check | the chosen candidate is evaluated once on the other half; `target_met` reports it | no split: the test uses all queries |
| statement | none for the chosen candidate: the bound holds per configuration, not for one chosen among many | the statement in section 4 |
| nothing found | the float corpus is returned and `target_met` is None | `ABSTAIN` |
| random | seeded split and bootstrap | none, unless `order_seed` is given |
| evidence | out-of-sample figures over 20 splits on the development stores (`../../UNIVERSAL.md`, R7b; R10 on what the bound does not cover) | the registered experiments X1 to X3 |

`optimize()` and `calibrate()` are unchanged. The entry point `optivision.certify()` is documented
in [API_DESIGN.md](../API_DESIGN.md), and [QUICKSTART.md](QUICKSTART.md) runs it.

## 10. Where the rule comes from

The rule applies published tools: testing by betting (Waudby-Smith and Ramdas, J. R. Stat. Soc. B
86(1), 2024), fixed-sequence testing (Westfall and Krishen, J. Statist. Plann. Inference 99,
2001; the same device appears in Learn then Test, Angelopoulos et al., 2021), and a union bound
over families. It is not a new statistical method. The derivation that combines them for this
setting is the project's own and has not been externally reviewed. An external statistical review
should come before any wording says more than the statement above.

What supports the implementation: the package arithmetic equals the frozen research functions bit
for bit on seeded random problems and reproduces the stored E1 selections
(`tests/test_core_statistics_parity.py`; ENGINEERING). The registered validity check of X2 covered
2,800 resampled cells with no confirmed failure. Those are resamples of one pool per corpus: they
check the implementation against the derivation and do not replace it.
