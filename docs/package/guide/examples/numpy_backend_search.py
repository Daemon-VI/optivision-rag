"""Search a compressed index through the NumPy backend.

The data are synthetic (optivision.testing): they exercise the code and say nothing about any
encoder. Everything is written to a temporary directory.

    PYTHONPATH=src python docs/package/guide/examples/numpy_backend_search.py
"""

import tempfile

from optivision import BinaryQuantizer, Pipeline
from optivision.backends import NumpyBackend, comparable
from optivision.core.retrieve import CompressedRetriever
from optivision.index.numpy_index import NumpyIndex
from optivision.storage import to_legacy_pages
from optivision.testing import make_corpus, make_queries

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
    print("scores comparable with the in-process retriever:", comparable(ranking, reference))
    same = sum(a == b for a, b in zip(ranking.ids, reference.ids))
    print("queries with the same top-5 ids:", same, "of", len(queries))
    backend.close()
