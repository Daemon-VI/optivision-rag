"""Two-tier retrieval: a small hot store shortlists, a larger cold store rescores the shortlist.

The data are synthetic (optivision.testing): they exercise the code and say nothing about any
encoder. Agreement with the exact ranking, printed below, is not retrieval quality.

    PYTHONPATH=src python docs/package/guide/examples/two_tier_retrieval.py
"""

import numpy as np

from optivision import AdaptiveMerge, BinaryQuantizer, Int8Quantizer, Pipeline
from optivision.core.retrieve import CompressedRetriever, ExactRetriever, TieredRetriever
from optivision.testing import make_corpus, make_queries

corpus = make_corpus(100, 32, seed=0)
queries, qrels = make_queries(corpus, 50, seed=1, noise=1.0)

hot_pipeline = Pipeline([AdaptiveMerge(radius=0.8), BinaryQuantizer()])  # small: stays in RAM
cold_pipeline = Pipeline([Int8Quantizer("per_vector")])  # larger: read for the shortlist only
hot, cold = hot_pipeline.compress(corpus), cold_pipeline.compress(corpus)

exact = ExactRetriever(corpus)
tiered = TieredRetriever(
    CompressedRetriever(hot, hot_pipeline), CompressedRetriever(cold, cold_pipeline), candidates=10
)

top_exact = exact.search(queries, k=1).ids
top_tiered = tiered.search(queries, k=1).ids
same = np.mean([a == b for a, b in zip(top_exact, top_tiered)])
print("queries whose top document equals the exact index's:", f"{same:.2f}")

float32_bytes = corpus.num_vectors * corpus.dimension * 4 / len(corpus)
hot_bytes, cold_bytes = hot.report()["bytes_per_doc"], cold.report()["bytes_per_doc"]
print(f"bytes per document: float32 {float32_bytes:.0f}, hot {hot_bytes:.0f}, cold {cold_bytes:.0f}")
print("score kind of the tiered ranking:", tiered.search(queries, k=3).score_kind)
