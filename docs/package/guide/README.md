# OptiVision package guide

Status: 2026-10-06, branch `package-universal`. This directory explains the package: the
concepts, how a selection is made and what it covers, why the coordinate basis matters, what the
package does not do, and how the code is organised. It describes what is in the tree. It does not
repeat research results; where it mentions one, it names the report and the scope.

One point runs through every page. A model-agnostic interface does not mean model-independent
compression quality. Any late-interaction encoder that hands over `[n_vectors, dim]` float arrays
can use the same interface, candidate list, evidence, decision procedure, certificate and
backends. Whether a pipeline keeps retrieval quality depends on the encoder, on the coordinate
basis its vectors are in, on the corpus and on the queries, and has to be measured each time. The
decision procedure may abstain, and abstaining is a correct outcome.

## What this package is, and is not

1. It is a library that compresses late-interaction (multi-vector, MaxSim) indexes by composing
   token reducers, projections and codecs.
2. It is a shared interface: one `[n, d]` float array per document and per query is enough to use
   the same candidate list, evidence, decision, certificate and backends.
3. It is a measuring tool: it records per-query evidence for a frozen candidate list against a
   stated reference, and selects one candidate or abstains.
4. It is a record: a certificate states what was decided, on which vectors and queries, under
   which assumptions, and can be verified.
5. It is additive: `optimize()`, `calibrate()`, `OptiVisionRAG` and the v0.3.0 file formats are
   unchanged.
6. It is not a promise that compression keeps retrieval quality. That depends on the encoder, the
   coordinate basis, the corpus and the queries, and is measured each time.
7. It is not model-independent: a model-agnostic interface is not model-independent quality.
8. It is not certain to select anything: the rule may abstain.
9. It is not a new statistical method: the rule applies published tools, and the derivation that
   combines them has not been externally reviewed.
10. It is not validated beyond what was measured: seven development stores, MaxSim over unit
    vectors. Transfer to encoders that took no part in the design has not been tested.

## The guide (this directory)

| page | read it to learn |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | from vectors to a decision, a verifiable artifact and a search, on synthetic data; labels against no labels; what an abstain looks like |
| [CONCEPTS.md](CONCEPTS.md) | the mental model and one definition of every term; the five layers of what is and is not the same for every encoder |
| [SELECTION_AND_CERTIFICATION.md](SELECTION_AND_CERTIFICATION.md) | how a decision is made, the statement with its assumptions, how to read it, what you must supply, what to do on abstain |
| [BASIS_DEPENDENCE.md](BASIS_DEPENDENCE.md) | why some stages depend on the coordinate basis, what X3 found, how stage traits represent it (with the data conditions each holds under and who declared it), diagnostics, open possibilities |
| [LIMITATIONS.md](LIMITATIONS.md) | what to know before relying on the package: scope, what was measured, assumptions, labels, speed, known defects |
| [ARCHITECTURE.md](ARCHITECTURE.md) | the tree as built, the dependency rule and its test, pinned files, the data flow, extension checklists |
| [BACKENDS_GUIDE.md](BACKENDS_GUIDE.md) | which retriever or backend to use when |
| [ADAPTERS_GUIDE.md](ADAPTERS_GUIDE.md) | how to bring your own encoder or arrays |

## Runnable examples

`examples/` holds scripts that run on synthetic data from `optivision.testing`, print a short
result and write only to a temporary directory. From the repository root:
`PYTHONPATH=src python docs/package/guide/examples/<name>.py`.
`tests/test_docs_examples.py` runs every script and every Python block of these pages.

| script | shows |
|---|---|
| `certify_synthetic.py` | arrays in, a decision with labels, without labels, and an abstain |
| `verify_artifact.py` | saving an artifact, verifying it, loading it, and a modified one failing |
| `custom_candidate_space.py` | a candidate list of your own |
| `two_tier_retrieval.py` | a hot store that shortlists and a cold store that rescores |
| `numpy_backend_search.py` | searching a compressed index through the NumPy backend |
| `bring_your_own_encoder.py` | wrapping a function as an adapter and checking the contract |

## Reference documents (`docs/package/`)

| document | what it is for |
|---|---|
| [UNIVERSAL_PACKAGE_DESIGN.md](../UNIVERSAL_PACKAGE_DESIGN.md) | the design contract: names, signatures, the dependency rule, decisions D1 to D20 |
| [FINAL_RESEARCH_BOUNDARY.md](../FINAL_RESEARCH_BOUNDARY.md) | what the package supports, what the research record shows and does not show, what the package does not promise |
| [METHOD_C_SPEC.md](../METHOD_C_SPEC.md) | the specification of the selection procedure, its statement and its assumptions |
| [CERTIFICATE_SPEC.md](../CERTIFICATE_SPEC.md) | the certificate and the artifact bundle: fields, verification levels, signing |
| [BACKEND_ARCHITECTURE.md](../BACKEND_ARCHITECTURE.md) | the retrieval seam and the backends as built: invariants, score kinds, memory, what Q5 and Q6 measured |
| [ADAPTER_ARCHITECTURE.md](../ADAPTER_ARCHITECTURE.md) | adapters, encoder identity, the `ext` and `legacy` namespaces and the testing helpers, as built |
| [API_DESIGN.md](../API_DESIGN.md) | the public entry point `optivision.certify()` and `verify()`: parameters, the result, outcomes, errors |
| [MIGRATION.md](../MIGRATION.md) | moving from v0.3.0: every observable difference, and from `optimize()` to `certify()` |
| [ARCHITECTURE_AUDIT.md](../ARCHITECTURE_AUDIT.md) | the read-only audit of the v0.3.0 code before the restructure |
| [RETRIEVAL_BACKEND_AUDIT.md](../RETRIEVAL_BACKEND_AUDIT.md) | the read-only audit of scoring, indexes and two-tier retrieval |

## Other documents

| document | what it is |
|---|---|
| [X_RESULTS.md](../../../reports/research/X_RESULTS.md) | the record of the registered experiments X1 to X3; read-only |
| [UNIVERSAL.md](../../UNIVERSAL.md) | the results record of v0.3.0 (R1 to R12); not edited |
| [PACKAGE.md](../../../PACKAGE.md) and [README.md](../../../README.md) | the v0.3.0 user documents |
| [ARCHITECTURE.md](../../ARCHITECTURE.md) | the original page recipe only |
