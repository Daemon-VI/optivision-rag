# Which retriever or backend to use

Status: 2026-10-06, branch `package-universal`. A short how-to for searching an index. Reference
detail (invariants, score kinds, memory, what was measured) is in
[BACKEND_ARCHITECTURE.md](../BACKEND_ARCHITECTURE.md). Terms are in [CONCEPTS.md](CONCEPTS.md).

A retriever answers `search(queries, k)` with a `Ranking`: ranked document ids and float32 scores
for each query, the store positions for in-process retrievers, and the kind of score and the tie
convention it was produced under. A backend is an existing page index (`NumpyIndex`,
`QdrantIndex`) wrapped so that it also states what its scores mean and answers batch search. The
choice of retriever does not change which compression was certified. It changes how the index is
held and scored.

## 1. Choose

| you want | use | what to know |
|---|---|---|
| the reference ranking: exact MaxSim over your float vectors | `ExactRetriever(corpus)` | builds a dense float32 `[n_queries, n_docs]` matrix |
| to search the index a decision selected | `result.retriever()`, or `CompressedRetriever(compressed, pipeline)` | codes are decoded to float32 before scoring; queries pass through the pipeline's projections only |
| less in RAM | `TieredRetriever(hot, cold, candidates=...)` | the small hot store shortlists, the cold store rescores the shortlist; saves resident memory, not total storage; no speed claim |
| a shortlist from an engine, rescored in process | `TwoTierRetriever(BackendRetriever(backend), cold, candidates)` | the engine's page ids must equal the cold store's ids exactly |
| a NumPy page index on disk | `NumpyBackend(NumpyIndex(path, dim=..., method=...))` | exact over the stored codes; ties by ascending position |
| Qdrant | `QdrantBackend(QdrantIndex(...))` | its score is a sum of cosines, not an inner product; a server searches an approximate graph, which was not measured here |

`qdrant-client` is a declared dependency of the package (`pyproject.toml`, installed with `pip install optivision-rag`); what is optional is using it: nothing imports it until a Qdrant index or backend is constructed, and no server is needed otherwise (Qdrant local mode runs in process).

## 2. What to know first

- **Scoring is decode-first.** Compressed codes become float32 before the inner products. No
  native compressed kernel is shipped. Native compressed scoring was slower in NumPy and torch on
  one laptop CPU (Q5 and Q6; `LIMITATIONS.md`, section 10). The package makes no speed claim.
- **Exact over the stored representation.** A retriever over compressed codes is exact for the
  decoded codes, which are not the float vectors they came from. It is not an exact search of the
  float index.
- **Scores of different kinds are not comparable.** `comparable(a, b)` is true when the score kind
  and tie convention agree. Ranks can be compared across kinds; scores cannot. Even within a kind,
  two codecs differ in scale.
- **A dense matrix grows with the corpus.** `score` and `search` of the in-process retrievers
  allocate `[n_queries, n_docs]`. Nothing here is a claim about large corpora.
- **Ties.** In-process retrievers and `NumpyBackend.search_many` break ties by ascending position.
  The legacy `NumpyIndex.search` and Qdrant local do not say how.
- **Empty documents** score `-inf` and rank last. An empty query scores 0.
- **Checks at the door.** `ExactRetriever(corpus)` scans a float corpus once, when it is built,
  and refuses NaN or infinity with a message that names the first document (a NaN document would
  otherwise be scored NaN and never returned). Queries must be finite and of the dimension the
  store, or the pipeline's projection, expects: a retriever with a pipeline names the queries and
  both dimensions when they differ.
- **Codecs.** `to_legacy_pages`, the bridge to the page backends, accepts float32, plain binary
  and fixed-scale int8 only, and raises for the rest.
- **Ids.** A page backend names a page `<doc_id>::p<page_no>`. The bridge rebuilds each page from
  the corpus id by splitting at the first `::p`, so `abc` comes back as `abc::p1`. Give the corpus
  ids of that form before compressing, and `legacy_id_mismatches(ids)` lists the ones that would
  not survive.

## 3. Two tiers

The hot tier is merged vectors as binary codes, the cold tier is per-vector int8. The synthetic
data only exercise the code. Agreement with the exact index, printed below, is not retrieval
quality.

```python
import numpy as np

from optivision import AdaptiveMerge, BinaryQuantizer, Int8Quantizer, Pipeline
from optivision.core.retrieve import CompressedRetriever, ExactRetriever, TieredRetriever
from optivision.testing import make_corpus, make_queries

corpus = make_corpus(100, 32, seed=0)
queries, qrels = make_queries(corpus, 50, seed=1, noise=1.0)

hot_pipeline = Pipeline([AdaptiveMerge(radius=0.8), BinaryQuantizer()])  # small: stays in RAM
cold_pipeline = Pipeline([Int8Quantizer("per_vector")])  # larger: read for the shortlist only
hot, cold = hot_pipeline.compress(corpus), cold_pipeline.compress(corpus)

tiered = TieredRetriever(
    CompressedRetriever(hot, hot_pipeline), CompressedRetriever(cold, cold_pipeline), candidates=10
)
top_exact = ExactRetriever(corpus).search(queries, k=1).ids
top_tiered = tiered.search(queries, k=1).ids
print("top document equals the exact index's:", np.mean([a == b for a, b in zip(top_exact, top_tiered)]))
print("score kind:", tiered.search(queries, k=3).score_kind)
```

```text
top document equals the exact index's: 1.0
score kind: tiered
```

A tiered ranking mixes cold scores for the shortlist with hot scores shifted below them, so its
scores are on two scales. Quality is bounded by whether the right document reaches the shortlist
and by the cold scores. Both are to be measured on your data. What two tiers buy is resident
memory: the hot store is what has to stay in RAM. `TieredRetriever` raises unless both stores hold
the same ids in the same order.

## 4. The NumPy backend

```python
import tempfile

from optivision.backends import NumpyBackend, comparable
from optivision.index.numpy_index import NumpyIndex
from optivision.storage import to_legacy_pages

corpus = make_corpus(40, 32, seed=0)
corpus.ids = [f"doc{i}::p1" for i in range(len(corpus))]  # the id form the page backends use
queries, qrels = make_queries(corpus, 10, seed=1, noise=1.0)

pipeline = Pipeline([BinaryQuantizer()])
store = pipeline.compress(corpus)

with tempfile.TemporaryDirectory() as tmp:
    backend = NumpyBackend(NumpyIndex(tmp, dim=corpus.dimension, method="binary"))
    backend.add(to_legacy_pages(store))  # only codecs the page backends can decode are accepted
    ranking = backend.search_many(queries, k=5)

    reference = CompressedRetriever(store, pipeline).search(queries, k=5)
    print("backend:", backend.info.name, "| score kind:", backend.info.score_kind)
    print("comparable with the in-process retriever:", comparable(ranking, reference))
    print("queries with the same top-5 ids:", sum(a == b for a, b in zip(ranking.ids, reference.ids)))
    backend.close()
```

```text
backend: numpy | score kind: maxsim_ip
comparable with the in-process retriever: True
queries with the same top-5 ids: 10
```

Equality between two scoring paths is rounding-level, never bitwise, so a near-tie can swap
places between them. Compare scores with a tolerance and rankings outside near-tie groups.

## 5. Reading a `Ranking`

`ranking.ids[q]` is the list of ids for query `q`, best first. `ranking.scores` is float32
`[n_queries, width]`. `ranking.indices` holds store positions (or `None` for an id-only backend).
Rows shorter than `width` are padded with `-inf` and `-1`. `ranking.top(k)` and
`ranking.to_lists()` help. `==` between rankings is exact.

The complete scripts are `examples/two_tier_retrieval.py` and `examples/numpy_backend_search.py`.
