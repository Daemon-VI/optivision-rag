# Coordinate basis and the package

Status: 2026-10-06, branch `package-universal`. This page explains why the result of some
compression stages depends on the coordinate basis the vectors are stored in, what the registered
experiment X3 found, how the package represents the dependence, and what a user should do about
it. Terms are defined in `CONCEPTS.md`.

A model-agnostic interface does not mean model-independent compression quality. The selection
rule can be model-blind in what it is allowed to inspect while the compression it yields depends
on the coordinate basis. Both are true at once.

## 1. What a coordinate basis is

A vector is stored as `d` numbers: its coordinates along `d` axes. Another set of `d` perpendicular
axes describes the same vector with other numbers. Multiplying every document and query vector by
one orthogonal matrix `G` (so that `G^T G = I`) moves to such a set. Every inner product between
a query vector and a document vector stays the same, so float MaxSim scores stay the same, up to
rounding. A signed permutation (axes reordered, some signs flipped) is the simplest orthogonal
matrix.

The package does not know which basis an encoder emits, and it does not assume one.

## 2. Why some stages depend on it and some do not

A stage depends on the basis when its result is computed from the stored coordinates one at a
time.

- **Sign codes (`binary`)** keep the sign of each coordinate, so they store which orthant of the
  stored basis a vector lies in. Reordering or flipping axes maps orthants to orthants, and the
  code is the same up to relabelling, provided no coordinate is exactly zero: a zero is coded
  as -1 whichever sign convention is used, so the statement excludes exact zeros. A rotation
  moves vectors across orthant boundaries.
- **Integer codes (`int8`, `int4`)** round each coordinate on a grid, with a fixed step, a
  per-vector step (the largest absolute coordinate) or a per-dimension range. Reordering or
  flipping axes gives the same codes up to the same reordering. A rotation changes which
  coordinates are large or clip, and where each value falls on the grid.
- **Merging by cosine (`adaptive_merge`, `hierarchical_merge`, `redundancy`)** compares the
  vectors of one document through inner products or distances, which an orthogonal map leaves
  unchanged, and pools members by their mean. In exact arithmetic the same rows merge in any
  orthogonal basis. The statement needs data without exact ties in the similarities or
  distances that decide a merge (rows that are exactly equidistant, such as orthonormal rows):
  ties go to row order, and float32 rounding after a rotation can overturn that.
- **`pca`** is fixed by the data, up to the sign of each component, when the last retained
  eigenvalue is larger than the next one (a tie at the cut leaves the retained subspace
  arbitrary) and, for any coordinate stage after it, the retained eigenvalues are distinct (a
  tie leaves its coordinates an arbitrary rotation within the tied group).
- **`truncate`** keeps the leading coordinates, so what it keeps depends on the axis order.
  **`random_projection`** and **`lloyd2`** apply a seeded random map in the stored coordinates:
  the realisation depends on the basis, the distribution over seeds does not.

These are statements about code. They say nothing about how much retrieval quality a stage keeps.

## 3. What X3 found

X3 is the registered experiment on this question. Scope: seven development stores (three visual
encoders on DocVQA and InfoVQA, one text encoder on SciFact), the registered list of 40 pipelines,
and four fixed maps with registered seeds: a signed coordinate permutation (T1), two rotations
(T2a, T2b) and a reordering of the vectors of each document (T3). Every map leaves every MaxSim
score unchanged and keeps the vectors at unit length. The source is `reports/research/X_RESULTS.md`
(sections 3 and 4) and `reports/research/X3/X3_REPORT.md`.

Per-candidate changes in labelled retention are MEASURED on the transformed vectors. Outcomes of
the default procedure are SIMULATED, pool-supported (resamples of one pool per dataset). A pair
(dataset, map) holds when no candidate's labelled retention changed materially (by at least 0.02,
with an interval excluding 0) and the procedure's outcome did not change materially.

| primary dataset | T1 signed permutation | T2a rotation | T2b rotation | T3 reordering |
|---|---|---|---|---|
| ColPali DocVQA | holds | does not hold (2 candidates material) | holds | holds |
| ColQwen2 InfoVQA | holds | holds | holds | holds |
| ColQwen2.5 DocVQA | holds | holds | holds | holds |
| SciFact | holds | does not hold (1 candidate; procedure outcome) | does not hold (5 candidates; procedure outcome) | holds |

- **H5, invariance under the four maps, fails.** 13 of 16 primary pairs hold; 3 do not.
- The signed permutation and the reordering changed nothing material on any dataset. The three
  failing pairs are rotations. The largest change in labelled retention on a primary pair is 0.043.
- The candidates that changed materially were `binary` pipelines on ColPali DocVQA, and `binary`
  and `int4(mean)` pipelines on SciFact (the list is in `reports/research/X_RESULTS.md`, section 3).
- On SciFact at `T = 0.97`, 300 queries, `error = 0.05`, the share of resampled samples in which
  the default procedure deployed a candidate other than float16 was 0.9175 in the native basis and
  0.0995 and 0.1415 under the two rotations. The median certified compression was 6.6x against
  2.0x (`reports/research/X3/x3_scifact.json`).
- The extension set, reported separately and never substituted for a primary dataset: 8 of 12
  pairs hold. The four that do not are rotations: ColPali InfoVQA (T2a, T2b; the procedure's
  outcome is material, no candidate is) and ColQwen2 DocVQA (T2a, T2b; 2 and 1 candidates
  material). ColQwen2.5 InfoVQA holds under all four maps.
- The direction of a change differs by cell. On SciFact the outcome fell. On ColPali InfoVQA
  (extension set), at `T = 0.97` with 225 queries and `error = 0.05`, the share of resampled
  samples that deployed a candidate other than float16 was 0.4975 in the native basis, 0.698
  under T2a (higher) and 0.2275 under T2b (lower)
  (`reports/research/X3/x3_colpali_infovqa.json`). The record supports this much: the certified
  outcome depends on the basis under rotation, downward on SciFact and in both directions on
  ColPali InfoVQA. Nothing establishes that a rotation helps or harms in general.

What the failure of H5 means, within the registration: on these datasets the measured result of
some sign-code (`binary`) and 4-bit (`int4`) pipelines depends on the coordinate basis under the
two registered rotations, beyond the registered tolerance. It is not a statement about other maps,
other encoders or other candidates. Passing X3 would have shown invariance under four maps on
these datasets, and nothing wider.

## 4. How the package represents it

Every stage declares which orthogonal changes of basis leave its scores unchanged
(`optivision.core.stages.traits`).

| level (`BasisInvariance`) | meaning |
|---|---|
| `orthogonal` | scores unchanged under any orthogonal change of basis, after refitting |
| `signed_permutation` | unchanged only when axes are permuted or sign-flipped |
| `none` | not even that, for example a cut of the leading coordinates |
| `unknown` | nothing declared; counted as `none` |

A level is a statement for data that meet the conditions the stage lists. Each stage declares
two more things (`StageTraits`):

- `conditions`: plain sentences naming what must hold of the data, for example no exact ties at
  a PCA cut, no exact ties among the distances that decide a merge, no coordinate exactly zero for
  a sign code. Nothing in the package checks them against the data. `orthogonal` therefore means
  "unchanged under any orthogonal matrix, for data that meet the listed conditions".
- `declared_by`: `library` for the table below, which the tests exercise on synthetic data with a
  witness for each condition; `third_party` for a `register_traits` call or a `traits` attribute
  made outside that table, which is recorded as declared and is not examined by the library;
  `undeclared` when nothing was declared.

### 4.1 The traits of every registered stage

Generated by `traits_of(name)` for each registered name. For a bare name the values are the
weakest case over the parameters: `pca` and `project` read queries when fitted only with
`basis="joint"`, which the evidence collector refuses; `project` takes the traits of its method.

| stage | kind | basis invariance | canonicalises basis | randomised basis | fit uses queries | conditions | what it does |
|---|---|---|---|---|---|---|---|
| `adaptive_merge` | tokens | orthogonal | no | no | no | 1 | merges by cosine; the same rows merge in any orthogonal basis |
| `binary` | bits | signed_permutation | no | no | no | 1 | keeps the sign of each coordinate: the orthant in the stored basis |
| `float16` | bits | signed_permutation | no | no | no | 0 | rounds each coordinate; a rotation moves scores by about the rounding error |
| `float32` | bits | orthogonal | no | no | no | 0 | stores the values unchanged |
| `hierarchical_merge` | tokens | orthogonal | no | no | no | 1 | Ward clustering on pairwise distances (needs SciPy) |
| `int4` | bits | signed_permutation | no | no | no | 0 | per-vector scale, levels -7..7, on each coordinate |
| `int4norm` | bits | signed_permutation | no | no | no | 0 | int4's levels, with the scale that keeps each vector's norm (experimental) |
| `int8` | bits | signed_permutation | no | no | no | 0 | fixed step, per-vector scale or per-dimension range, on each coordinate |
| `lloyd2` | bits | none | no | yes | no | 0 | an internal seeded rotation, then 2-bit levels |
| `pca` | dimensions | orthogonal | yes | no | yes | 2 | canonical up to sign when the eigenvalues are distinct and not tied at the cut |
| `project` | dimensions | none | no | yes | yes | 0 | facade: the traits of the chosen method (the conditions of `pca` for method `pca`) |
| `random_projection` | dimensions | none | no | yes | no | 0 | seeded Gaussian matrix in the stored coordinates |
| `random_prune` | tokens | orthogonal | no | no | no | 0 | seeded random rows; ignores the values |
| `redundancy` | tokens | orthogonal | no | no | no | 1 | cosine clustering inside one document |
| `rotation` | dimensions | orthogonal | no | yes | no | 0 | one seeded orthogonal matrix; a diagnostic |
| `spatial` | tokens | orthogonal | no | no | no | 0 | page pixels and grid; never reads the values; needs images |
| `truncate` | dimensions | none | no | no | no | 0 | keeps the leading coordinates of the stored basis |

Every row has `declared_by = library`. The `conditions` column is the number of sentences;
`traits_of(name).conditions` has them.

```python
from optivision import AdaptiveMerge, BinaryQuantizer, PCAProjector, Pipeline
from optivision.core.stages import STAGES, RotationProjector, pipeline_traits, traits_of

for name in sorted(STAGES):  # the table above is this loop
    t = traits_of(name)
    print(f"{name:18s} {t.kind:11s} {t.basis_invariance.value:19s} {len(t.conditions)} {t.declared_by}")
```

### 4.2 The pipeline rule

`pipeline_traits(pipeline)` walks the stages in order. A stage that applies a seeded random map
(`randomised_basis`) puts the rest of the pipeline in a mixed state, in which a later
`signed_permutation` stage counts as `none`. The pipeline's invariance is the weakest level met.
`basis_sensitive` is true unless every stage is `orthogonal`. Canonicalisation (PCA) is recorded
and is not credited as invariance. `conditionally_invariant` is true when every stage weaker than
`orthogonal` is a `signed_permutation` stage that receives coordinates fixed by the data from a
PCA stage before it, with no seeded random map in between. That holds only if the retained
eigenvalues are distinct, which is a condition on the data and is not checked.

The pipeline also carries `conditions` (those of every stage, in order, each prefixed with the
stage name), `unconditional` (true only when there are none) and `all_library_declared` (true
when every stage's traits are the library's own). `basis_sensitive=True` is the cautious
statement and needs no condition. `basis_sensitive=False` with `unconditional=False` is a
conditional statement: not sensitive, for data that meet the listed conditions. A third-party
declaration is recorded per stage in `declared_by` and is not examined by the library.

```python
for stages in (
    [AdaptiveMerge(radius=0.7)],
    [BinaryQuantizer()],
    [AdaptiveMerge(radius=0.7), BinaryQuantizer()],
    [PCAProjector(16), BinaryQuantizer()],
    [RotationProjector(seed=0), BinaryQuantizer()],
):
    pipe = Pipeline(stages)
    t = pipeline_traits(pipe)
    print(f"{pipe.label():32s} {t.basis_invariance.value:19s} "
          f"basis_sensitive={t.basis_sensitive!s:5s} conditionally_invariant={t.conditionally_invariant!s:5s} "
          f"unconditional={t.unconditional}")
```

```text
adaptive_merge(0.7)              orthogonal          basis_sensitive=False conditionally_invariant=False unconditional=False
binary                           signed_permutation  basis_sensitive=True  conditionally_invariant=False unconditional=False
adaptive_merge(0.7) > binary     signed_permutation  basis_sensitive=True  conditionally_invariant=False unconditional=False
pca(16) > binary                 signed_permutation  basis_sensitive=True  conditionally_invariant=True  unconditional=False
rotation(0) > binary             none                basis_sensitive=True  conditionally_invariant=False unconditional=False
```

### 4.3 The default candidate list

```python
from collections import Counter

from optivision.core.selection import CandidateSpace

for name, space in (("default", CandidateSpace.default(128)), ("wide", CandidateSpace.wide(128))):
    flags = Counter((pipeline_traits(p).basis_invariance.value,
                     pipeline_traits(p).conditionally_invariant) for p in space.pipelines())
    print(name, len(space), dict(flags))
```

```text
default 40 {('signed_permutation', False): 40}
wide 54 {('signed_permutation', False): 47, ('signed_permutation', True): 7}
```

All 40 candidates of the default list are `basis_sensitive` and none is `conditionally_invariant`
(ENGINEERING; derived from the declarations). Five of them list no condition (`float16` and the
`int8` and `int4` pipelines without a merge stage); the other 35 list at least the conditions of a
merge stage or of the sign code. The `wide` list adds PCA families; 7 of its 54
candidates are `conditionally_invariant`. That is an engineering property that no registered
experiment evaluated, and `wide` itself was not evaluated as a selection list.

## 5. What the traits are, and are not

They are statements about code, taken in exact arithmetic and checked on synthetic data by
`tests/test_core_stage_traits.py`, for data that meet the conditions each stage lists. The tests
include a witness for each condition: data that break it and move the scores. For `orthogonal` the test applies random orthogonal maps and
signed permutations, refits the pipeline and compares the score matrices. For `signed_permutation`
it checks the same under signed permutations and requires a witness that a rotation changes the
scores by clearly more than rounding, so that the label is not empty. For `none` it requires a
witness that a plain permutation of the axes changes the scores. In float32 a value within
rounding of a quantisation boundary can still change level.

They are not statements about quality. A stage declared `orthogonal` can lose retrieval quality,
and a `basis_sensitive` pipeline can keep it. A stage without declared traits is `unknown` and is
treated as basis dependent. Nothing in the package assumes that `codec(x)` and `codec(xG)` behave
alike.

## 6. What follows for a user

- A certificate is about the vectors as given. The corpus identity covers the stored bytes and
  the dtype, so a float16 copy, a re-encoded copy or a rotated copy of an index is a different
  corpus, and the certificate does not carry over.
- If you re-encode (another model revision, precision or pooling), rotate, centre, whiten or
  rescale the vectors, measure again.
- The rule's statement is unaffected by the basis: whatever basis the vectors are in, the rule
  tests the pipeline on those vectors. What changes with the basis is the yield, not the validity.
- The certificate records `basis_sensitive` and the invariance level of the deployed pipeline.
- Do not read `basis_sensitive=False` or `conditionally_invariant=True` as "quality is kept".

## 7. Diagnostics

`RotationProjector` (registry name `rotation`) multiplies documents and queries by one seeded
orthogonal matrix. Put it in front of a codec and compare with the same pipeline without it. It
is an engineering capability not evaluated by the registered experiments, and it is not an
improvement: a rotation in front of a codec has to be measured like any other candidate, and may
abstain.

`optivision.testing.basis_probe` applies signed permutations and rotations to the documents and
queries, refits the pipeline on each copy, and reports how far its compressed scores and its
top-k move, together with a check that the uncompressed scores did not change. It describes what
happened for the seeds and data given. A small change does not show invariance, and a large one
does not show a loss of retrieval quality.

```python
from optivision.testing import basis_probe, make_corpus, make_queries

corpus = make_corpus(60, 32, seed=0, anisotropy=3.0)  # synthetic: exercises the code only
queries, qrels = make_queries(corpus, 20, seed=1, noise=0.5)
report = basis_probe(Pipeline([BinaryQuantizer()]), corpus, queries, seeds=(0, 1, 2))
print("declared:", report["declared_basis_invariance"], "with", len(report["declared_conditions"]),
      "condition(s) | valid symmetry:", report["valid_symmetry"])
for name, s in report["summary"].items():
    print(f"{name:19s} mean top-10 overlap {s['mean_top_k_overlap']:.3f}  "
          f"max relative score change {s['max_relative_score_change']:.3f}")
```

```text
declared: signed_permutation with 1 condition(s) | valid symmetry: True
signed_permutation  mean top-10 overlap 1.000  max relative score change 0.000
rotation            mean top-10 overlap 0.412  max relative score change 0.375
```

The numbers are for this synthetic corpus and say nothing about any encoder.

## 8. Open design possibilities, not results

Nothing in this section was measured. Each item says what would have to be measured before any
claim.

- **Basis-aware candidates.** Candidate families chosen or parameterised so as to be less
  sensitive to the basis. Measure their certified outcome against the default list, under the same
  rule, on encoders and bases that took no part in choosing them.
- **Canonicalising projections before a codec** (PCA, then a codec). Measure whether the outcome
  stays the same under rotations of the vectors, as X3 did, and what happens when the retained
  eigenvalues are not distinct.
- **Learned transforms** that align the coordinates with the codec. Measure on encoders that took
  no part in learning the transform, with every fit using documents only (assumption A3).
- **Basis search**, trying several bases per encoder. Each basis is another candidate and spends
  error level, so measure the yield net of that cost.
