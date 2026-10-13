"""Bring your own vectors; choose a compression candidate or abstain.

The data are synthetic (optivision.testing). They exercise the code and say nothing about any
encoder, so nothing printed here is a result about retrieval quality.

    PYTHONPATH=src python docs/package/guide/examples/certify_synthetic.py
"""

from optivision import certify
from optivision.adapters import ArrayAdapter
from optivision.core.vectors import EncoderIdentity
from optivision.testing import make_corpus, make_queries

corpus = make_corpus(60, 32, seed=0)
sample, qrels = make_queries(corpus, 200, seed=1, noise=3.0)

# What an encoder hands over: one [n_i, d] float array per document and per query.
doc_arrays = [corpus.vectors[slice(*corpus.span(i))] for i in range(len(corpus))]
query_arrays = [sample.vectors[slice(*sample.span(i))] for i in range(len(sample))]

# Check and wrap them: floating, finite, one dimension, one dtype, unique ids.
adapter = ArrayAdapter(EncoderIdentity(model_id="my-encoder", dimension=32, modality="other"))
docs = adapter.encode_documents(doc_arrays, ids=corpus.ids)
queries = adapter.encode_queries(query_arrays, ids=sample.ids)

# With labels, the statement is about retention measured against the relevance labels.
result = certify(docs, queries, qrels, target=0.95, error=0.05, encoder=adapter.identity())
print(result)

# Without labels the reference is agreement with the uncompressed ranking, which is not
# retrieval quality. Each call gets its own queries: one (target, error) per sample.
fresh, _ = make_queries(corpus, 200, seed=2, noise=3.0)
agreement = certify(docs, fresh, target=0.95, error=0.05)
print("reference:", agreement.decision.construct, "| outcome:", agreement.outcome.value)

# A strict target on another sample: nothing is certified, and that is an outcome, not an error.
strict, strict_qrels = make_queries(corpus, 200, seed=3, noise=3.0)
abstained = certify(docs, strict, strict_qrels, target=0.99, error=0.05)
print("strict target:", abstained.outcome.value, "| nothing compressed:", abstained.compressed is None)
