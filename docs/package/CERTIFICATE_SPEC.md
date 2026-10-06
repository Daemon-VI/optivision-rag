# OptiVision certificate and artifact specification

Schema `optivision-certificate`, version 1. Status: specification, 2026-10-06, branch
`package-universal`. Implemented in `optivision.core.artifact` (`certificate.py`:
`Certificate`, `create_certificate`, `sign_certificate`, `verify_certificate`; `verification.py`:
the checks; `report.py`: `Check`, `VerificationReport`; `record.py`: field list, fixed texts,
digests; `bundle.py`: `save_artifact`, `load_artifact`; `codeid.py`: `code_identity`) and, for
the evidence it is computed from, in `optivision.core.evaluate.collect`. Section 9 lists every
point where this document differs from the draft.

## 1. What a certificate is

A certificate is a record of one selection decision: which candidate list was tested, on which
vectors and queries, against which reference, at which target and error level, by which rule
and code, what the rule decided, and under which stated assumptions. It lets someone who holds
the certificate answer: *was this compression decision certified under the stated
configuration, and is this the index it was certified for?*

What it is not:

- It is not a statement that the compressed index keeps retrieval quality. The statement it
  carries is the selection rule's own, with its assumptions (section 5), and nothing wider.
- It is not a security token. The content digest detects accidental or naive modification;
  anyone who can edit the file can recompute it. Tamper resistance against a party who wants to
  forge a certificate needs the optional HMAC with a key that party does not hold (section 6).
- It does not transfer. It binds one decision to one corpus (by digest of the vectors as
  given), one query sample and one candidate list. A re-encoded, rotated, extended or truncated
  corpus has a different identity; the certificate says nothing about it.

## 2. Fields

All values are JSON-native. Digests are lower-case hex SHA-256. "Canonical JSON" is
`optivision.core.identity.canonical_json` (sorted keys, separators `,` and `:`, ASCII, finite
numbers only). A certificate has exactly the 24 fields below; any other top-level field fails
the `schema` check. In a file they are written in this order (nested keys sorted); the digest
does not depend on the order.

| field | content |
|---|---|
| `schema`, `schema_version` | `"optivision-certificate"`, `1` |
| `created_utc` | `YYYY-MM-DDTHH:MM:SSZ`. Given by the caller (a string ending in `Z`, or an aware `datetime`, converted to UTC and cut to seconds) or read from the clock; that default is the only clock read in the package's core |
| `package` | `{name, version}`: the import name (`optivision`) and `__version__` of the library that created it |
| `code` | `{digest, files, python, numpy, scipy}`: section 2.1. `digest` is null (with a further key `note`) when sources cannot be read |
| `encoder` | an `EncoderIdentity` dict declared by the caller, or null. Descriptive only: it is not verified and no decision depends on it. At creation its `dimension` must equal the corpus dimension |
| `corpus` | `CorpusIdentity` fields (`n_docs`, `n_vectors`, `dimension`, `dtype`, `vectors_sha256`, `offsets_sha256`, `ids_sha256`, and `protected_sha256`, `positions_sha256`, `weights_sha256`, `importance_sha256`, each null when the array is absent) plus `digest`, the combined digest of those fields |
| `queries` | `{n_pool, n_used, identity, order}`. `n_pool`: query vectors supplied (kept plus dropped); `n_used`: queries the rule tested; `identity`: as `corpus`, for the query vectors and ids; `order`: the decision's record `{kind, seed, budget, n_available, n_used, indices_sha256}`, where `kind` is `given` or `seeded_permutation` and `indices_sha256` is the digest of the tested positions in test order |
| `reference` | `{construct, metric, identity}`. `construct` is `labels` or `agreement@<depth>` (the studied case is depth 1); `identity` is `{construct, qrels_sha256}` for labels (a digest of the listed relevant documents per query, independent of dict order and of grades) or `{construct, depth}` for agreement |
| `space` | `{name, identity, n_candidates, n_families, families, registered_default, candidates}`: the full ordered candidate list (`{position, family, pipeline}` each), so the certificate is self-contained. `name` is a label and is not part of `identity` |
| `sizes` | `{bytes_per_doc, vectors_per_doc}`: one number per candidate, the values the choice among certified candidates used |
| `rule` | `{name, certifying, config}`: the rule's configuration, including its name and whether it is certifying |
| `statement`, `assumptions` | the rule's statement and its assumptions A1-A6, verbatim (`optivision.core.selection.STATEMENT` and `Decision.assumptions`). For a rule that is not certifying, `statement` is instead a fixed refusal text: "The selection rule that produced this record is not a certifying rule: it makes no error statement about the candidate it selected, and nothing in this record is certified." |
| `caveats` | a list of sentences a reader must not miss, derived from the outcome, the construct and the rule (section 2.2). Often empty |
| `target`, `error`, `level` | the retention target, the error level, and the per-test level the rule derived |
| `evidence` | `{digest, n_queries, dropped_queries, baseline_mean}`: digest of the per-query arrays the decision was computed from; the number of queries in them (those with a relevant document, before any budget); the ids removed for having none; the mean metric of the uncompressed index over all of them |
| `result` | `{outcome, certified, selected, certified_positions, p_values}`. `selected` is the list position of the deployed candidate, `-1` for a fallback, null for an abstention |
| `sample` | for `select` only (null otherwise): `{n, mean, baseline_mean, retention_point, note}`, descriptive statistics of the selected candidate on the tested queries. They are optimistically biased by selection and are **not** certified quantities; `note` says so |
| `candidate` | for `select` and `fallback` (null for `abstain`): `{pipeline, pipeline_sha256, fitted_state, traits, bytes, codes_sha256, offsets_sha256}` |
| `digest` | SHA-256 of the canonical JSON of every field above except `digest` and `signature` |
| `signature` | null, or `{algorithm: "HMAC-SHA256", key_id, value}`: an HMAC over the digest (section 6) |

`candidate` in detail: `pipeline` is `Pipeline.to_dict()` as canonical JSON and `pipeline_sha256`
its digest; `fitted_state` has one fitted-state digest per stage (null for a stage without
state); `traits` is `pipeline_traits(pipeline).to_dict()`, including `basis_invariance`,
`conditionally_invariant` and `basis_sensitive` of the deployed pipeline (structural properties
of the code, not statements about quality; see the design document, section 5.3); `bytes` is
`byte_accounting(compressed, pipeline).to_dict()` (the size key `selection_bytes_per_doc`, and
beside it fitted-state, append-state, offsets and float32 bytes, which are not in the key);
`codes_sha256` and `offsets_sha256` are array digests of the compressed corpus.

### 2.1 The code identity

`code_identity()` hashes, with line endings normalised to LF (so a CRLF checkout and a wheel
built from the same sources agree), every decision-relevant source file. The set, relative to the
package directory, found from the package's own `__file__` (a source tree or an installed
wheel):

- every `.py` file under `core/` (recursively, `__pycache__` excluded);
- `calibration.py`, `compose.py`, `scoring.py`, `evaluation.py`, `representation.py`;
- every `.py` file directly under `stages/` and under `compression/`.

`files` maps each relative POSIX path to its SHA-256, sorted; `digest` is the SHA-256 of the
canonical JSON of that table; `python`, `numpy` and `scipy` are version strings (`scipy` is null
when not installed). When a required module or a whole directory cannot be read the result is
`digest: null`, `files: {}` and a `note`; it never raises. The identity of the running package is
computed once per process and an explicit `root` (used by tests) is never cached. It identifies
source files, not the numerical libraries beneath them, and it is not a signature: anyone who can
edit a certificate can write any identity into it.

### 2.2 Texts recorded beside the statement

`caveats` holds, in this order, whichever apply:

- construct `agreement@...`: "Retention here means agreement with the uncompressed ranking,
  which is not retrieval quality. The uncompressed index's own top document is taken as the
  relevant one, so the measure is fidelity to that ranking, including its mistakes, and no
  statement in this record is a statement about relevance.";
- outcome `fallback`: "The deployed pipeline is a fallback supplied by the caller. It is not
  certified, and the statement above does not cover it.";
- outcome `select` under a rule that is not certifying: "The selected candidate is not certified:
  the rule that chose it makes no error statement."

## 3. Outcomes

| `result.outcome` | `result.certified` | meaning |
|---|---|---|
| `select` | true for a certifying rule, false otherwise | the rule certified at least one candidate and chose among them; `candidate` is the deployed pipeline |
| `abstain` | false | nothing was certified; the uncompressed index is to be used. This is the safe outcome, not an error. `candidate` and `sample` are null |
| `fallback` | false | nothing was certified and the caller supplied an explicit fallback pipeline. It is recorded, it is **not** certified, and the rule's statement does not cover it. `sample` is null (the fallback has no per-query evidence) |

`certified` is true only for `select` under a certifying rule, and that is enforced both when a
certificate is created and by a consistency check when it is verified. A certificate is issued for
every outcome, so an abstention is as checkable as a selection.

## 4. Verification

`verify_certificate(cert, *, evidence=None, corpus=None, queries=None, reference=None,
pipeline=None, compressed=None, key=None, strict_code=False) -> VerificationReport`

The report is a tuple of checks, each `(name, status, detail, level)` with status `pass`, `fail`,
`skipped` or `differs`. `ok` is true when no check failed (`skipped` and `differs` do not fail a
report). `level` is the highest of `integrity`, `replay`, `binding` in which at least one check ran
and `levels` lists all of them; a level can be reached while only some of its checks ran, so read
the checks. `failed()` lists the failures and `summary()` is a readable account. Verification never
raises on a bad certificate: a check that cannot be evaluated is reported as failed, and a record
whose structure is invalid stops after `schema`, `digest` and `signature` with one `remaining`
check marked skipped. It raises `TypeError` only for wrong argument types. `ok` does not make a
certificate true: it is as good as the queries, labels and corpus it was issued for.

**Level 1, integrity (certificate only).**

1. `schema`: schema name and version are known, exactly the 24 fields are present with the right
   types and value domains (target in (0, 1], error in (0, 1), level in (0, 1], the time format).
2. `digest`: the content digest recomputed from the fields equals `digest`.
3. `signature`: with a key, the HMAC is compared with `hmac.compare_digest`. A signed certificate
   checked without a key is `skipped`; a key with an unsigned certificate (or an unknown
   algorithm) is a failure.
4. Internal consistency, recomputed from the certificate alone:
   `consistency.space` (`space.identity` is the digest of `space.candidates`; `n_candidates`,
   `n_families`, `families`, `registered_default` agree with the list);
   `consistency.identities` (the combined digests of `corpus` and `queries.identity` and of
   `code.files` equal their fields; the identity in `reference` names the construct;
   `queries.n_pool` is the query count of the identity);
   `consistency.counts` (lengths of sizes and p-values, finite non-negative sizes, p-values in
   [0, 1], `n_pool = evidence.n_queries + dropped`, `n_used` agrees with the budget, and the
   digest of the tested positions is reproduced from `kind`, `seed` and `budget`);
   `consistency.level` (`level` equals what the rule's configuration derives from `error` and the
   list: `error / n_families` or `error / n_candidates`);
   `consistency.outcome` (`outcome`, `certified`, `selected`, `certified_positions`, `candidate`
   and `sample` agree with each other);
   `consistency.decision` (`certified_positions` and `selected` equal what the rule's
   deterministic decision step gives for the stored p-values, families, sizes and level);
   `consistency.candidate` (`candidate.pipeline_sha256` is the digest of `candidate.pipeline`;
   for `select` that pipeline is the selected candidate's list entry; `fitted_state` has one entry
   per stage; the byte figures agree with each other, with the corpus identity and with
   `sizes.bytes_per_doc` of the selected candidate);
   `consistency.sample` (`retention_point` is `mean / baseline_mean`; `n` is `queries.n_used`).
5. `rule.text`: the rule (name, flag, configuration), `statement`, `assumptions` and `caveats` are
   those the running library writes for the same rule and construct. `candidate.traits`: the
   recorded traits are those the running library derives from `candidate.pipeline`. A difference is
   a `fail` when `code.digest` equals the running code's (the same code cannot write a different
   text), and `differs` otherwise (a different version may word a text differently); it is a `fail`
   under `strict_code`. For a rule that is not built in (or a stage this process does not know) the
   check is `skipped`.
6. `code.files`: the running code's identity compared with `code.digest`. A difference is reported
   as `differs` with the files that changed, were added or were removed, not as a failure, unless
   `strict_code=True`: a different library version is not tampering, but a replay under different
   code is weaker evidence. `code.libraries`: differing Python, NumPy or SciPy versions are
   reported as `differs` and never fail a report, even under `strict_code`.

`consistency.level`, `consistency.decision` and the replay need a built-in rule
(`fixed_sequence_betting`, `bonferroni_betting`); for any other rule they are `skipped`.

**Level 2, replay (needs the evidence).**

7. `evidence.digest`: the digest of the supplied evidence equals `evidence.digest`.
8. `evidence.fields`: the certificate's copies of the evidence's values equal the supplied evidence:
   `n_queries`, `dropped_queries`, `baseline_mean` (relative tolerance 1e-12), `sizes`, the construct
   and metric, the space identity and candidate count, and for `select` the fitted-state digests
   the evidence recorded for the selected candidate. `evidence.sample`: the sample statistics
   recomputed from the evidence on the tested queries (relative tolerance 1e-12).
9. `replay.decision`: `select()` is re-run on the supplied evidence with the recorded target, error,
   rule, fallback (for a fallback outcome), query budget and order seed, and compared with the
   stored result: p-values at relative tolerance 1e-12 (bitwise equality across platforms is not
   promised), and the outcome, `certified`, `selected`, `certified_positions`, `level`,
   `n_queries` and the query-order record exactly. A re-run that raises (for example on evidence
   from another candidate list) is a failure.

**Level 3, binding (needs the live objects; each is optional and checked on its own).**

10. `binding.corpus`, `binding.queries`: the identity of the supplied vectors equals `corpus` and
    `queries.identity`. `binding.evidence_queries` (needs queries and evidence): the evidence's
    query ids are the supplied queries without the dropped ones, in order.
11. `binding.reference`: the supplied reference has the recorded construct and identity.
12. `binding.pipeline` (fitted): its specification digest and per-stage fitted-state digests equal
    `candidate`; an unfitted stateful pipeline fails.
13. `binding.codes`: the digests of the compressed corpus's codes and offsets equal `candidate`, its
    document ids equal the certified corpus's, and its recorded pipeline is the certified one.
    `binding.bytes` (needs pipeline and compressed): the byte accounting recomputed from them
    equals `candidate.bytes`.
    A pipeline or compressed corpus supplied for a certificate that records no deployed candidate
    (`abstain`) fails.

### 4.1 What each modification is caught by

"Alone" is level 1 with the certificate only; "everything" is levels 1-3 with the evidence and
every object supplied. Every row is exercised by `tests/test_core_certificate_tamper.py`, once
with the digest left alone (the `digest` check fails for every row) and once with it recomputed, as
a careless or dishonest editor would.

| modified (digest recomputed) | caught alone by | caught with everything by |
|---|---|---|
| a candidate pipeline parameter, in `candidate` only | `consistency.candidate` | adds `binding.codes`; `binding.pipeline` too if `pipeline_sha256` was rewritten |
| the same parameter rewritten in `candidate`, the list and `space.identity` | not detected | `evidence.fields`, `replay.decision`, `binding.pipeline`, `binding.codes` |
| candidate list: drop, reorder or add one (identity stale) | `consistency.space` (and `consistency.level` if the family count changes) | adds `replay.decision` |
| candidate list rewritten consistently (identity, counts, sizes, p-values, level, decision) | not detected | `evidence.fields`, `replay.decision` |
| target | not detected | `replay.decision` |
| error level (level unchanged) | `consistency.level` | adds `replay.decision` |
| error level and level together | `consistency.decision` when the certified set would change | adds `replay.decision`; a rewritten selection that differs from the deployed pipeline is caught by `consistency.candidate` and `evidence.sample` |
| query identity | `consistency.identities`; if its combined digest is refreshed: not detected | `binding.queries` |
| query count (`n_pool`), dropped queries | `consistency.counts` (and `consistency.identities` for `n_pool`) | the same; dropped queries add `evidence.fields` |
| query order, seed or budget | `consistency.counts` (and `consistency.sample` for a budget); if the position digest is refreshed too: only a budget (`consistency.sample`) | `replay.decision` |
| reference identity | not detected | `binding.reference` |
| construct | `consistency.identities`, `rule.text` (caveats); if rewritten consistently: not detected | adds `evidence.fields`, `binding.reference` |
| fitted-state digest | not detected (a removed entry: `consistency.candidate`) | `binding.pipeline`, `evidence.fields`; and `binding.codes` if the codes digest was changed |
| code identity (files table with its digest refreshed) | not detected; `code.files` is `differs` | `differs`; `fail` under `strict_code`. A digest that does not match its table: `consistency.identities` |
| outcome, `certified`, selected position, certified positions | `consistency.outcome`, `consistency.decision`, `consistency.candidate` | adds `replay.decision` |
| one p-value | `consistency.decision` when it changes the decision; otherwise not detected | `replay.decision` |
| corpus identity | `consistency.identities`; if the combined digest is refreshed: not detected. Vector or document counts: `consistency.candidate` | `binding.corpus` |
| byte figures | `consistency.candidate` for what is tied to something else (parts and totals, the size key, the corpus); the serving-state and append-state figures: not detected | `binding.bytes`; the copies in `sizes`: `evidence.fields` |
| statement, an assumption, caveats, rule configuration | `rule.text` (`fail`, or `differs` if `code.digest` was also changed) | same |
| sample statistics | `consistency.sample` (internal arithmetic) | `evidence.sample` |
| `created_utc`, `package`, `encoder`, `space.name` | not detected | not detected |
| any field, under an HMAC key | `signature` | `signature` |

Supplying objects that are not the certified ones is caught by the binding checks of level 3: a
corpus, queries or labels that differ in one value, a reordered id list, a pipeline fitted on other
data (`binding.pipeline`), codes, offsets or ids that differ (`binding.codes`). Evidence with one
value changed fails `evidence.digest` and the replay.

A party who rewrites the certificate **and** supplies matching forged evidence and objects is
not detected by levels 1-3: a certificate issued honestly from forged evidence verifies. Only a
signature from a key they do not hold closes that gap.

## 5. The statement a certificate carries

For the fixed-sequence betting rule (the port of the research's Method C):

> P( a compressed candidate is selected and its retention under Q is below T ) <= delta,
> over the draw of the calibration queries.

It holds for every sample size and is conditional on all of: (A1) the calibration queries are
independent draws from one distribution Q, the distribution the index will serve; (A2) the
corpus is fixed; (A3) the candidate list, its families and its order were fixed before the
queries were seen, and every fitted stage was fitted on documents only; (A4) the per-query
metric lies in [0, 1] and the baseline's mean is positive; (A5) the order in which queries
enter the test was fixed before their outcomes were seen; (A6) the reference is taken as
given (labels, or the uncompressed index's own top document).

It is not a conditional probability given that something was selected, not a statement about
the retention measured on the sample, not a statement about a finite batch of future queries,
and not a statement about power: the rule may abstain every time. The rule is an application
of published tools (testing by betting; fixed-sequence testing; a union bound). The derivation
that combines them is the project's own and has not been externally reviewed. Under the
`agreement@1` construct, "retention" means agreement with the uncompressed ranking, which is
not retrieval quality.

The certificate cannot check A1, A2, A5 or A6: they are properties of how the caller obtained
the queries and labels. It records what the caller supplied. A3 is supported, not proven, by
the code path: the default list depends only on the dimension, and stages are fitted by
`Pipeline.compress(corpus)` without queries.

The wording above paraphrases the text the certificate records in `statement` and `assumptions`
(the library's own, verbatim, with the same conditions); the sentence about the agreement
construct is recorded in `caveats`.

## 6. Signing

`sign_certificate(cert, key: bytes, key_id: str | None = None) -> Certificate` adds
`signature = {"algorithm": "HMAC-SHA256", "key_id": key_id, "value": hmac(key, digest)}`, where the
HMAC input is the ASCII digest string and `value` is hex. `key` must be a non-empty `bytes` or
`bytearray` (a `str` is refused: its encoding would be a guess; use a random key of at least 32
bytes). A certificate whose digest does not match its content is refused: signing it would
vouch for a modified record. Signing again replaces the signature. `key_id` is a free label and is
not authenticated by the signature. `verify_certificate(..., key=key)` checks the value with a
constant-time comparison (`hmac.compare_digest`).

HMAC is symmetric: whoever can verify can also sign. It protects against modification by parties
without the key; it is not a public-key signature and gives no non-repudiation. With a key,
every content modification fails the signature even when the digest is recomputed, a signature
copied from another certificate fails, and removing the signature fails.

## 7. Artifact bundle

`save_artifact(path, *, certificate, compressed=None, pipeline=None, evidence=None,
overwrite=False) -> Path` writes a directory:

| file | content |
|---|---|
| `certificate.json` | the certificate, indented, UTF-8, LF line endings |
| `codes.npz` | `codes` (uint8), `offsets` (int64) and `header`: canonical JSON bytes in a uint8 array (`format: "optivision-artifact"`, `version: 1`, `dim`, `ids`, `stats`, `pipeline`); absent for `abstain` |
| `fitted.npz` | the fitted state of every stage: arrays `s<i>.<key>` for stage `i`, and a `header` (`format: "optivision-artifact-fitted"`, `version: 1`, `stages: [{stage, arrays, scalars}]`) holding the stage names, the array keys and the scalar entries; absent when no stage has state |
| `evidence.npz` | the per-query evidence as `Evidence.to_npz` writes it (optional; needed for a level-2 check later) |
| `MANIFEST.json` | `{format: "optivision-artifact-manifest", version: 1, files: {name: {sha256, size}}}` for every other file; written last, so a half-written bundle has no manifest and does not load |

`select` and `fallback` certificates need both the fitted `pipeline` and the `compressed` corpus;
an `abstain` certificate takes neither. Before anything is written the certificate is verified
against the objects given (level 1, the pipeline and codes binding, and the replay when `evidence`
is given), and a bundle that would fail its own verification is refused with the failed checks
listed. A non-empty directory is refused unless `overwrite=True`, which replaces only the bundle's
own files (and removes a stale `fitted.npz` or `evidence.npz`); other files are left alone.

No pickle is written or read: arrays are read with `allow_pickle=False`. Generated bundles must not
be committed: the repository's `.gitattributes` routes `*.npz` to Git LFS.

**Manifest rule.** A `.json` file is hashed after CRLF is turned into LF (so a checkout that
converts line endings does not break it); an `.npz` file is hashed as stored. Whether `np.savez`
is byte-stable was checked (NumPy 2.4.6, Python 3.11): it writes the zip epoch as the entry
timestamp, so two runs on one platform give identical bytes, but it leaves the "created by"
byte of each zip entry to the operating system (0 on Windows, 3 elsewhere), so containers written
on different systems differ. Older NumPy versions were not checked. The bundle therefore writes its
`.npz` files itself, uncompressed, with the zip epoch and a fixed creator byte, and hashes the
files; the same arrays give the same bytes on every platform for one NumPy version, and a bundle
written twice from the same inputs with the same `created` has an identical manifest. No
per-array hash is kept. `evidence.npz` is `Evidence.to_npz` output re-packed in the same way.

**What the manifest is.** It detects accidental damage and naive edits of one file (the zip
format's own CRC-32 also refuses a damaged entry). Anyone who can write the directory can write a
matching manifest, so it is not the root of trust: the certificate is (its digest, the codes and
fitted-state digests it records, an optional HMAC), and `load_artifact` checks those too.

`load_artifact(path, *, verify=True, key=None, strict_code=False) -> Artifact` rebuilds the
pipeline from the certified specification, restores the fitted state, and rebuilds the
`CompressedCorpus`. With `verify=True` it checks the manifest (every file listed and
present, hashes and sizes equal, nothing unknown listed), restores the fitted state with the
digests the certificate records, then runs `verify_certificate` with the pipeline, the codes and,
when the bundle holds evidence, the evidence (so the replay runs), passing `key` and `strict_code`
on. Any failed check, or an unreadable file, raises `ArtifactError` listing it (a path that is
not a directory raises `FileNotFoundError`). `Artifact` has
`certificate`, `pipeline` (fitted, or None), `compressed` (or None), `evidence` (or None), and
`verification` (the report, None with `verify=False`) and `path`. With `verify=False` nothing is
checked beyond what is needed to read the files, and a changed value inside a valid container
loads silently.

The loaded `CompressedCorpus` decodes bit for bit like the one that was saved, the loaded
pipeline projects queries bit for bit alike, and `maxsim_matrix` over the loaded store gives the
same scores, provided the fitted state restores exactly (a `lloyd2` codec regenerates its
rotation on the loading machine and refuses to load where that differs from the recorded digest).

This format exists because the library's `optivision-compressed` v1 file cannot hold fitted
state (a PCA basis, a codec's mean). That format stays readable and is not changed.

## 8. Limits

- A certificate is as good as its inputs. It cannot tell whether the queries were an honest
  sample of the deployment distribution, whether several targets were tried on the same sample
  and one reported, or whether the candidate list was edited after a look at the results.
- The `code` field identifies source files, not the numerical libraries beneath them. Replay
  compares p-values at a relative tolerance for that reason.
- The byte figures follow the library's accounting: the size key counts codes and codec
  overhead; fitted-state bytes are listed separately and are not part of the key.
- Schema version 1 has no revocation, expiry or chaining.
- The `code` file set (section 2.1) leaves out modules outside the decision path (page encoders,
  the command line, backends, adapters, storage); a change there does not change the identity.
- The decision step, the level and the replay can be recomputed only for the built-in rules
  named in section 4; for any other rule those checks are `skipped`. A rule that declares itself
  certifying gets the library's `STATEMENT` recorded, which is the statement of the built-in rules:
  a rule that makes a different statement must not be issued a certificate with this function.
- Evidence is checked only when it is supplied (or stored in a bundle). A certificate verified
  without it has had its internal consistency and bindings checked, not its p-values.

## 9. Deviations from the draft of this document

Every point where what is built differs from, or settles something left open in, the draft:

1. New field `caveats` (section 2.2). The draft said "add the agreement sentence" without a place
   for it; `statement` stays the rule's text verbatim, and the sentences that must not be missed
   (agreement is not retrieval quality; a fallback is not certified; a non-certifying rule) sit
   beside it as a checkable list. For a non-certifying rule `statement` is a refusal text, since
   the library has no statement of such a rule.
2. `package.name` is the import name `optivision`, not the distribution name.
3. `code` gains a `note` key only when `digest` is null. The file set, the LF rule, the per-process
   cache and the uncached explicit `root` are fixed in section 2.1.
4. `queries.order` holds the decision's full order record, not only `{kind, seed, budget}`; its
   `indices_sha256` lets level 1 reproduce and check the order. `queries.n_pool`, `queries.n_used`
   and `evidence.n_queries` are defined in section 2 (the draft left their relation open).
5. `reference.identity` is the reference object's own `identity()` (a labels digest or a depth).
6. `sample` gains `n`, `baseline_mean` and `note`, is computed on the tested queries, exists for
   `select` only, and `result.selected` is `-1` for a fallback.
7. The draft's single list of level-1 consistency items is split into named checks, and checks the
   draft did not name were added so that every modification in the brief has a check or a pinned
   limit: `rule.text` (statement, assumptions, caveats, rule configuration), `candidate.traits`,
   `consistency.sample`, `evidence.fields`, `evidence.sample`, `binding.evidence_queries`,
   `binding.bytes`, and the reproduction of the tested positions from the order record.
8. A text or trait that differs from what the running library writes is decisive (`fail`) only when
   `code.digest` equals the running code's, otherwise `differs`; `strict_code` makes it `fail`. A
   difference in Python, NumPy or SciPy versions is `differs` and never fails a report.
9. A key supplied for an unsigned certificate is a failure (the draft was silent), as is an unknown
   algorithm; `key_id` is not authenticated.
10. A report's `level` is the highest level with a check that ran and `levels` lists them (the draft
    did not define "level reached" when objects are supplied without evidence).
11. `verify_certificate` reports unexpected top-level fields as a `schema` failure and stops after
    `schema`, `digest` and `signature` when the structure is invalid.
12. Bundle: `save_artifact` takes `overwrite` and verifies before writing; `load_artifact` takes
    `key` and `strict_code`, runs the replay when the bundle holds evidence, and raises
    `ArtifactError` for every unreadable file; `Artifact` has `verification` and `path`. The
    manifest rule (section 7) writes the `.npz` containers itself; the draft left the choice
    between hashing containers and hashing arrays open, and arrays are not hashed separately.
13. `create_certificate` cross-checks more than the draft listed: the evidence's fitted-state
    digests and sizes against the pipeline and compressed corpus given, the queries against the
    evidence's ids, the reference's construct, and the encoder's dimension against the corpus.
14. `Certificate.from_json` refuses duplicate keys and NaN or infinity; `save` writes ASCII JSON with
    LF line endings.
