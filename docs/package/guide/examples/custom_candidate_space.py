"""Test a candidate list of your own.

The data are synthetic (optivision.testing): they exercise the code and say nothing about any
encoder.

    PYTHONPATH=src python docs/package/guide/examples/custom_candidate_space.py
"""

from optivision import (
    AdaptiveMerge,
    BinaryQuantizer,
    Int4Quantizer,
    Int8Quantizer,
    Pipeline,
    certify,
)
from optivision.core.selection import CandidateSpace
from optivision.testing import make_corpus, make_queries

# A family is listed from least to most aggressive and is tested in that order. The list, its
# families and their order are fixed here, before any query is looked at.
space = CandidateSpace.from_families(
    {
        "int8": [Pipeline([Int8Quantizer("per_vector")]),
                 Pipeline([AdaptiveMerge(radius=0.8), Int8Quantizer("per_vector")]),
                 Pipeline([AdaptiveMerge(radius=0.6), Int8Quantizer("per_vector")])],
        "int4": [Pipeline([Int4Quantizer(center="mean")]),
                 Pipeline([AdaptiveMerge(radius=0.8, center="mean"), Int4Quantizer(center="mean")])],
        "binary": [Pipeline([BinaryQuantizer()]),
                   Pipeline([AdaptiveMerge(radius=0.8), BinaryQuantizer()])],
    },
    name="my-list",
)
print("identity:", space.identity[:16], "| registered default list:", space.is_registered_default)
print("families:", space.families, "| each test runs at error /", space.n_families)

corpus = make_corpus(40, 32, seed=0)
queries, qrels = make_queries(corpus, 150, seed=1, noise=3.0)
result = certify(corpus, queries, qrels, space=space, target=0.9, error=0.05)
print("outcome:", result.outcome.value, "| level per test:", round(result.decision.level, 4))
print("certified positions:", result.decision.certified_positions)
print("the certificate names this list:", result.certificate.space_identity == space.identity)
