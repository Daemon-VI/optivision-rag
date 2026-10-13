"""Save a decision as an artifact, verify it at three levels, and see a modified one fail.

The data are synthetic (optivision.testing): they exercise the code and say nothing about any
encoder. Everything is written to a temporary directory.

    PYTHONPATH=src python docs/package/guide/examples/verify_artifact.py
"""

import json
import tempfile
from pathlib import Path

from optivision import AdaptiveMerge, BinaryQuantizer, Int8Quantizer, Pipeline, certify, verify
from optivision.core.artifact import ArtifactError, load_artifact
from optivision.core.evaluate import LabelReference
from optivision.testing import make_corpus, make_queries

corpus = make_corpus(40, 32, seed=0)
queries, qrels = make_queries(corpus, 150, seed=1, noise=3.0)
space = {  # the candidate list is fixed before the queries are looked at
    "int8": [Pipeline([Int8Quantizer("per_vector")]),
             Pipeline([AdaptiveMerge(radius=0.8), Int8Quantizer("per_vector")]),
             Pipeline([AdaptiveMerge(radius=0.6), Int8Quantizer("per_vector")])],
    "binary": [Pipeline([BinaryQuantizer()]),
               Pipeline([AdaptiveMerge(radius=0.8), BinaryQuantizer()])],
}
result = certify(corpus, queries, qrels, space=space, target=0.9, error=0.05)
print(result.outcome.value, "| certified:", result.certified)

with tempfile.TemporaryDirectory() as tmp:
    path = result.save(Path(tmp) / "artifact")
    print("files:", sorted(p.name for p in path.iterdir()))

    stored = verify(path)  # integrity, replay from the stored evidence, stored codes
    print("stored artifact ok:", stored.ok)
    print("  not run:", [c.name for c in stored.checks if c.status == "skipped"])

    live = verify(path, corpus=corpus, queries=queries, reference=LabelReference(qrels))
    print("bound to the live objects ok:", live.ok)
    print("  not run:", [c.name for c in live.checks if c.status == "skipped"])

    artifact = load_artifact(path)  # also checks the manifest
    print("loaded pipeline:", None if artifact.pipeline is None else artifact.pipeline.label())

    cert_file = path / "certificate.json"  # edit one number and write the file back
    record = json.loads(cert_file.read_text(encoding="utf-8"))
    record["target"] = 0.5
    cert_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    edited = verify(path)
    print("after editing the target ok:", edited.ok, "| failed:", [c.name for c in edited.failed()])
    try:
        load_artifact(path)
    except ArtifactError as exc:
        print("load_artifact refuses it:", str(exc).splitlines()[0][:72])
