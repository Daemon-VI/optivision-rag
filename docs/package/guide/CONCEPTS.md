# Concepts

Status: 2026-10-06, branch `package-universal`. This page gives the picture the package is built
around and defines each term once. It states no research result. Where it mentions one, it names
the report and the scope.

## 1. The picture

```text
 your encoder's vectors              candidate list (frozen, ordered, hashed)
 documents [n_i, d], queries                  |
        |                                     | one compression pipeline per candidate
        v                                     v
        +--------> compress each candidate, score it on every query
                                              |
 reference: labels, or the                    v
 uncompressed index's own   ------>   evidence: one metric value per query, for the
 top document                         uncompressed index and for each candidate
                                              |
 target T, error level  ------------->  selection rule
                                              |
                  +---------------------------+---------------------------+
                  v                           v                           v
               SELECT                      ABSTAIN                     FALLBACK
        (a certified candidate)     (use the uncompressed        (the caller's pipeline;
                  |                         index)                   not certified)
                  v
   certificate + artifact (codes, fitted state)  ->  retriever or backend  ->  ranked ids
```

The seven steps:

1. Vectors in. Your encoder produces one `[n_vectors, dim]` float array per document and per
   query. The package takes arrays. An adapter can call the encoder for you, but nothing needs it.
2. A compression pipeline: an ordered list of stages of three kinds (section 3.2).
3. A candidate list: a frozen list of pipelines grouped into families (section 3.3).
4. Evidence: every candidate is run on every query and scored against a stated reference.
5. A decision: a rule reads the evidence, a target and an error level, and selects one candidate
   or abstains.
6. A certificate: a record of the decision, of what it was computed from, and of the assumptions
   it is conditional on.
7. An index: the compressed corpus, searched by a retriever, possibly behind a backend.

## 2. What is the same for every encoder, and what is not

A model-agnostic interface does not mean model-independent compression quality. Any
late-interaction encoder that can hand over `[n_vectors, dim]` float arrays can use the same
interface, candidate list, evidence, decision procedure, certificate and backends. Whether a given
pipeline keeps retrieval quality depends on the encoder, on the coordinate basis its vectors are
in, on the corpus and on the queries, and has to be measured each time. The decision procedure may
abstain. Abstaining is a correct outcome.

The package separates five layers. This table is section 1.1 of `../UNIVERSAL_PACKAGE_DESIGN.md`
in other words.

| layer | in plain words | depends on the encoder? | how it is checked | what it says about retrieval quality |
|---|---|---|---|---|
| A. Interface contracts | what a corpus, a pipeline, an evidence array and a certificate must look like: shapes, dtypes, identifiers, saved formats | no | input validation and tests | nothing |
| B. Structural properties of operators | what each stage reads, and which changes to the vectors (a rotation, a permutation of axes) leave its scores unchanged; read from the code | no, per stage | tests that apply the change to synthetic data and compare scores | nothing: statements about code |
| C. Development-informed defaults | the default candidate list, the order inside each family, the bet cap: fixed in advance, but shaped by the encoders used while they were designed | the list is the same for all, but not neutral | declared and frozen; the list has a registered hash | nothing; they decide what can be found |
| D. Validity of the decision procedure | the error statement of the rule: how often it can select a candidate whose retention is below the target | no, given its assumptions: it sees arrays only | a derivation, parity tests against the research code, resampling checks | a conditional statement about the procedure, not about any one encoder |
| E. Empirical outcome | whether anything is selected, how much compression, what retention | yes: encoder, coordinate basis, corpus, queries, number of queries, target, reference | measured on your vectors, each time | this layer is where quality is measured |

Layers A, B and D are the same for every encoder (D given its assumptions). Layer C is the same
list for every encoder and is not neutral. Layer E is not the same. A shared interface does not
make E the same.

"Model-blind" describes what the rule may inspect: arrays only, with no encoder name, dimension or
dataset. It is not evidence that the default list works on an encoder that took no part in its
design. That has not been tested (the held-out encoder experiment XH was not run; see
`LIMITATIONS.md`).

## 3. Terms

### 3.1 Vectors

**Late interaction and MaxSim.** A late-interaction retriever stores several vectors per document
(one per token or image patch) and encodes a query as several vectors. The score of a document for
a query is the sum, over the query's vectors, of the largest inner product with any of the
document's vectors: `score(q, d) = sum_i max_j q_i . d_j`. This is MaxSim. The package computes it
exactly over every document, with no approximate search. The registered experiments covered
vectors of unit length (L2-normalised): `reports/research/NEXT_PHASE_PREREGISTRATION.md`, section
5.

**Corpus vectors and query vectors.** A corpus is a list of documents. Each document is a float
array of shape `[n_i, d]`: `n_i` vectors of dimension `d`, where `n_i` may differ between
documents and `d` is the same for all. The library holds it as a `MultiVectorCorpus` (one
contiguous matrix, an offsets array, one id per document). Queries are a second
`MultiVectorCorpus`. `validate_corpus` checks the contract (finite values, one dimension, unique
ids). It does not require unit length.

**Identity.** An identity is a SHA-256 digest of the canonical form of something. Corpora,
queries, pipelines, candidate lists, evidence and fitted state each have one (`corpus_identity`,
`pipeline_identity`, `CandidateSpace.identity`, `Evidence.digest`). An identity detects change. It
is not a signature: anyone can recompute it.

### 3.2 Compression

A **stage** is one step that changes how the vectors are stored. There are three kinds.

| kind | what it does | registry names | applied to |
|---|---|---|---|
| token reducer | keeps fewer vectors per document, by pruning or merging | `adaptive_merge`, `hierarchical_merge`, `redundancy`, `random_prune` | documents only |
| projection | maps every vector to another coordinate system, usually with fewer dimensions | `pca`, `random_projection`, `truncate`, `project`, `rotation` | documents and queries, with the same map |
| codec (quantizer) | stores each coordinate with fewer bits | `float32`, `float16`, `int8`, `int4`, `binary`, `lloyd2` | documents only |

A fourth registered name, `spatial`, prunes by page pixels and grid position. It needs page images
and lives outside the model-agnostic layer.

Queries stay in float32. They pass through projections, because a projection changes the space
both sides must share. They never pass through a token reducer or a codec: scoring decodes the
document codes and takes inner products with the float query.

A **pipeline** is an ordered list of stages. Token reducers and projections may come in any order.
There is at most one codec and it is last; without one the vectors are stored as float32. A stage
is **fitted** on the documents (it learns what it needs, for example a mean or a PCA basis) and
then applied. `Pipeline.to_dict()` is the specification of a pipeline. The learned values are its
**fitted state**.

```python
from optivision import AdaptiveMerge, BinaryQuantizer, PCAProjector, Pipeline
from optivision.testing import make_corpus, make_queries

docs = make_corpus(40, 32, seed=0)                  # synthetic: exercises the code only
queries, _ = make_queries(docs, 20, seed=1)
pipeline = Pipeline([AdaptiveMerge(radius=0.2), PCAProjector(16), BinaryQuantizer()])
compressed = pipeline.compress(docs)                # fits every stage on the documents, then applies it
projected = pipeline.transform_queries(queries)     # queries pass through the projection only
print("documents:", docs.num_vectors, "->", compressed.num_vectors, "vectors;",
      docs.dimension, "->", compressed.dim, "dimensions")
print("queries:  ", queries.num_vectors, "->", projected.num_vectors, "vectors;",
      queries.dimension, "->", projected.dimension, "dimensions")
```

```text
documents: 475 -> 258 vectors; 32 -> 16 dimensions
queries:   121 -> 121 vectors; 32 -> 16 dimensions
```

### 3.3 Candidates

A **candidate** is one pipeline with a position in a list and a family label. A **family** (the
code also says chain) is the candidates that share a label, listed from least to most aggressive;
for example Ward merging to 0.5, 0.33, 0.25, 0.15 and 0.1 of the vectors, each followed by int8
codes. The rule tests a family in that order.

The **candidate list** (`CandidateSpace`) is the ordered list of all candidates. It is immutable
and has an identity. The default list has 40 candidates in 7 families when SciPy is installed, and
its hash is registered (`REGISTERED_DEFAULT_SPACE_SHA256`). Without SciPy the Ward families cannot
be built: `CandidateSpace.default` raises unless `allow_reduced=True`, which gives a smaller list
that is not the registered one. The default list is development-informed: it was shaped by results
on the development encoders (layer C). Any list can be used in its place, if it is fixed before
the queries are looked at.

### 3.4 Measuring

**Metric.** A per-query value in [0, 1] computed from a ranking and a set of relevant documents,
with binary relevance. The default is `ndcg@5`. Only nDCG@k was evaluated by the registered
experiments; `recall@k`, `mrr@k` and `hit@k` are an engineering option. A query with no relevant
document has no value and is removed, for the uncompressed index and every candidate together.

**Retention.** For one candidate: the mean metric of the compressed index divided by the mean
metric of the uncompressed index, on the same queries. It is a ratio of means, not a mean of
ratios, and it can exceed 1. "Retention under Q" is the same ratio taken over the whole query
distribution Q instead of a sample.

**Reference constructs.** The reference says what "relevant" means when the metric is computed.
There are two, and results under them are never merged.

| construct | relevant documents of a query | a value says | what it is not |
|---|---|---|---|
| `labels` | the documents listed for it in the relevance labels (qrels) | how right the index is, compared with the uncompressed index being right | |
| `agreement@1` | the single document the uncompressed index ranks first | how near the top the index keeps that document | retrieval quality |

Agreement is fidelity to the uncompressed ranking, including its mistakes. A compressed index that
corrects a mistake of the uncompressed index loses agreement. No statistical statement connects
the two constructs. On the seven development stores of the registered experiments they behaved
differently. The constructs were not equivalent (X1, H4b; MEASURED). Agreement was materially
stricter: in 26 of 37 pool-supported cells the median certified compression under agreement was
at most half that under labels (X1, N10; SIMULATED, pool-supported;
`reports/research/X_RESULTS.md`, sections 3 and 5). Those statements are about those stores.

**Evidence.** For one corpus, one query sample, one metric and one reference: the per-query
metric of the uncompressed index and of every candidate, in the order the queries enter the test,
with the stored size of each candidate and the identity of the candidate list. Evidence is what
the rule is allowed to see.

### 3.5 Deciding

**Target and error level.** The target `T` in (0, 1] is the retention a certified candidate must
exceed; 0.97 means more than 97 percent of the uncompressed index's mean metric. The error level
(`error`, delta) in (0, 1) bounds the probability in the rule's statement; 0.05 is typical. A
decision uses one pair per sample of queries.

**Outcomes.** Three, as `Outcome`:

| outcome | meaning | certified? | index to use |
|---|---|---|---|
| `SELECT` | the rule certified at least one candidate and chose the smallest | yes, under a certifying rule | the chosen candidate |
| `ABSTAIN` | nothing was certified. This is the safe outcome, not an error | no | the uncompressed index |
| `FALLBACK` | nothing was certified and the caller had supplied a fallback pipeline | no | the caller's pipeline |

There is no default fallback, and no silent float16 fallback: float16 is a candidate like any
other and must itself be certified.

**Certified.** A candidate is certified when the rule has rejected the hypothesis that its
retention under Q is at most `T`. "Certified" says that the rule's statement, with its
assumptions, covers this selection. It is not a measured retention.

### 3.6 Recording and serving

**Certificate.** A canonical-JSON record of one decision: the candidate list, the corpus and query
identities, the reference, the rule, the statement and assumptions, the target and error level,
the p-values, and the deployed candidate. It is issued for every outcome. It can be verified at
three levels: integrity (the certificate alone), replay (the decision recomputed from the
evidence) and binding (the live objects match the recorded identities). Without a secret key it
gives integrity checking, not security. See `SELECTION_AND_CERTIFICATION.md` and
`../CERTIFICATE_SPEC.md`.

**Artifact.** A directory holding the certificate, the codes, the fitted state of every stage, the
evidence (optional) and a manifest of file digests. Nothing is pickled.

**Retriever and backend.** A retriever answers `search(queries, k)` with a `Ranking`: ranked
document ids with their scores. `ExactRetriever` (float vectors), `CompressedRetriever`
(compressed vectors with the pipeline's query projection), `TieredRetriever` and
`TwoTierRetriever` delegate to the library's scoring and add validation. A backend is a page index
(`NumpyIndex`, `QdrantIndex`) wrapped so that it also states what its scores mean
(`BackendInfo.score_kind`). Scores of different kinds are not comparable. See `BACKENDS_GUIDE.md`.

### 3.7 Coordinate basis

A vector is stored as `d` numbers, its coordinates along `d` axes. Multiplying every document and
query vector by one orthogonal matrix changes every coordinate and no inner product, so float
MaxSim scores do not change (up to rounding). A stage is **basis sensitive** when its result can
change under such a change of basis. Sign codes and per-coordinate integer codes are. Merging by
cosine is not. A pipeline is flagged `basis_sensitive` unless every stage is declared invariant
under any orthogonal change of basis. `BASIS_DEPENDENCE.md` has the detail.

## 4. Pairs that are easy to confuse

| not this | but this | difference |
|---|---|---|
| model-agnostic interface | model-independent quality | the first is true by construction; the second is not claimed |
| compression ratio | retention | one is a size, the other a measured quality ratio |
| certified | measured on the sample | the statement covers the procedure; the sample figure is optimistic by selection |
| agreement | retrieval quality | agreement rewards copying the uncompressed index's mistakes |
| abstain | failure | the rule is allowed not to select, and the statement holds if it never does |
| a certificate | a proof of quality, or a security token | it records a conditional statement; the digest detects edits only |
