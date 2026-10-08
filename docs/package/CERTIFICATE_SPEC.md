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
configuration, and does this record match the index it was issued for?*

What it is not:

- It is not a statement that the compressed index keeps retrieval quality. The statement it
  carries is the selection rule's own, with its assumptions and conditions (section 5), and
  nothing wider; and it carries one only for the library's own rules (section 3).
- It is not a security token. The content digest detects accidental or naive modification;
  anyone who can edit the file can recompute it. Tamper resistance against a party who wants to
  forge a certificate needs the optional HMAC with a key that party does not hold (section 6).
- It does not transfer. It binds one decision to one corpus (by digest of the vectors as
  given), one query sample and one candidate list. A re-encoded, rotated, extended or truncated
  corpus has a different identity; the certificate says nothing about it.

## 2. Fields

All values are JSON-native. Digests are lower-case hex SHA-256. "Canonical JSON" is
`optivision.core.identity.canonical_json` (sorted keys, separators `,` and `:`, ASCII, finite
numbers only). A certificate has exactly the 25 fields below; any other top-level field fails
the `schema` check. That check closes the top-level fields only: an unknown key inside a nested
object (`result`, `queries`, `space`, `code`, `package`, `evidence`, `sizes`, `norms` and so on) is
not rejected by it, and is caught only where another check covers that object (for example
`rule`, `corpus`, `queries.identity`, `space.candidates` and `queries.order`). In a file they are written in this order (nested keys sorted); the digest
does not depend on the order.

| field | content |
|---|---|
| `schema`, `schema_version` | `"optivision-certificate"`, `1` |
| `created_utc` | `YYYY-MM-DDTHH:MM:SSZ`. Given by the caller (a string ending in `Z`, or an aware `datetime`, converted to UTC and cut to seconds) or read from the clock; that default is the only clock read in the package's core |
| `package` | `{name, version}`: the import name (`optivision`) and `__version__` of the library that created it |
| `code` | `{digest, files, python, numpy, scipy}`: section 2.1. `digest` is null (with a further key `note`) when sources cannot be read |
| `encoder` | an `EncoderIdentity` dict declared by the caller, or null. Descriptive only: it is not verified and no decision depends on it. At creation its `dimension` must equal the corpus dimension |
| `corpus` | `CorpusIdentity` fields (`n_docs`, `n_vectors`, `dimension`, `dtype`, `vectors_sha256`, `offsets_sha256`, `ids_sha256`, and `protected_sha256`, `positions_sha256`, `weights_sha256`, `importance_sha256`, each null when the array is absent) plus `digest`, the combined digest of those fields |
| `queries` | `{n_pool, n_used, identity, order, exact_duplicates}`. `n_pool`: query vectors supplied (kept plus dropped); `n_used`: queries the rule tested; `identity`: as `corpus`, for the query vectors and ids; `order`: the decision's record `{kind, seed, budget, n_available, n_used, indices_sha256}`, where `kind` is `given` or `seeded_permutation` and `indices_sha256` is the digest of the tested positions in test order; `exact_duplicates`: how many of the tested queries repeat an earlier tested query exactly (their vector rows have the same digest, or their id repeats), counted by `record.exact_duplicates` over the queries the rule saw, so that verification can recount it from the live queries. Near-duplicates are not detected |
| `norms` | `{tolerance, documents, queries}`, where `documents` and `queries` are `{unit_norm, norm_min, norm_max}`: the extremes of the L2 norms of the vectors (null when there are none) and whether both lie within `tolerance` (0.001) of 1. Computed in blocks of rows, in float64, without a copy of the corpus (`record.norm_record`). Descriptive and never a refusal: the research studied only MaxSim over L2-normalised vectors, and a caveat says so when a flag is false |
| `reference` | `{construct, metric, identity}`. `construct` is `labels` or `agreement@<depth>` (the studied case is depth 1); `identity` is `{construct, qrels_sha256}` for labels (a digest of the listed relevant documents per query, independent of dict order and of grades) or `{construct, depth}` for agreement |
| `space` | `{name, identity, n_candidates, n_families, families, registered_default, candidates}`: the full ordered candidate list (`{position, family, pipeline}` each), so the certificate is self-contained. `name` is a label and is not part of `identity`. `registered_default` is true when `identity` is the hash of the list the preregistration fixed (section 14.1) for the development stores; it names the list and does not say the list was validated for any other encoder |
| `sizes` | `{bytes_per_doc, vectors_per_doc}`: one number per candidate, the values the choice among certified candidates used |
| `rule` | `{name, class, builtin, certifying, declared_certifying, config}`. `class` is the qualified class name. `builtin` is true when the exact type is one of the library's own rules (`FixedSequenceBetting`, `BonferroniBetting`; a subclass is not, and neither is a rule that merely has the same name). `declared_certifying` is what the rule says about itself and `certifying` is what the library accepts: true only for a built-in rule that declares it. `config` is the rule's own configuration |
| `statement`, `assumptions` | for a built-in certifying rule, its statement and its assumptions A1-A6 and conditions C1-C4, verbatim (`optivision.core.selection.STATEMENT` and `Decision.assumptions`). For any other rule, `statement` is a fixed text: "This record was produced by a selection rule that is not part of this library. The library has not examined that rule and makes no statement about it, and nothing in this record is certified. Any statement the rule makes is its author's." and each sentence the rule declared is recorded prefixed with "Declared by the rule's author, not examined by this library: " |
| `caveats` | a list of sentences a reader must not miss, derived from the record's own fields (section 2.2). Often not empty: a given order, repeated queries and removed queries each add one |
| `target`, `error`, `level` | the retention target, the error level, and the per-test level the rule derived |
| `evidence` | `{digest, n_queries, dropped_queries, baseline_mean}`: digest of the per-query arrays the decision was computed from; the number of queries in them (those with a relevant document, before any budget); the ids removed for having none (the statement is about the queries that have one); the mean metric of the uncompressed index over all of them |
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
edit a certificate can write any identity into it. The certificate digest covers this recorded
environment (the Python, NumPy and SciPy versions among it), so two environments give different
digests for the same decision and evidence.

### 2.2 Texts recorded beside the statement

`caveats` holds, in this order, whichever apply. Every one follows from fields the record carries
(`record.caveats_for`), so `rule.text` recomputes the list:

- the rule is not one of the library's own: "The selection rule that produced this record is not
  part of this library. The library has not examined it, so a selected candidate is not certified,
  whatever the rule declares about itself, and the assumptions recorded here are the rule author's,
  not the library's.";
- construct `agreement@...`: "Retention here means agreement with the uncompressed ranking,
  which is not retrieval quality. The uncompressed index's own top document is taken as the
  relevant one, so the measure is fidelity to that ranking, including its mistakes, and no
  statement in this record is a statement about relevance.";
- outcome `fallback`: "The deployed pipeline is a fallback supplied by the caller. It is not
  certified, and the statement above does not cover it.";
- order kind `given`: "The queries were tested in the order supplied. The statement requires that
  order to be unrelated to the outcomes (assumption A5): the arrival order of independent draws
  qualifies; an order sorted by difficulty, date, source or any other query attribute does not.";
- order kind `seeded_permutation` with a seed other than the default 0: "The order seed (N) was
  chosen by the caller, not left at the default. Choosing a seed after looking at results, or trying
  several seeds and keeping one, is outside the statement." The default seed of `certify` and of the
  command line is `optivision.core.selection.DEFAULT_ORDER_SEED`;
- `queries.exact_duplicates` above 0: "N of the M queries tested repeat an earlier query exactly
  (the same vectors). Repeated queries count as independent draws only if they really are; if they
  come from failed de-duplication, logs replayed twice or augmentation, assumption A1 does not hold
  and the stated error level does not apply.";
- a `norms` flag that is false: "Not every vector has unit L2 norm (tolerance 0.001): document
  vectors have norms from a to b [and query vectors have norms from c to d]. The research studied
  only MaxSim over L2-normalised vectors, and nothing is known outside that scope.";
- queries removed for having no relevant document: "K of the N queries given have no relevant
  document under this reference and were removed before testing, for the baseline and for every
  candidate alike. The statement is about the queries that have a relevant document."

## 3. Outcomes

| `result.outcome` | `result.certified` | meaning |
|---|---|---|
| `select` | true for one of the library's own certifying rules, false for any other rule | the rule rejected "retention <= target" for at least one candidate and the smallest of those was chosen; `candidate` is the deployed pipeline |
| `abstain` | false | nothing was certified; the uncompressed index is to be used. This is the safe outcome, not an error. `candidate` and `sample` are null |
| `fallback` | false | nothing was certified and the caller supplied an explicit fallback pipeline. It is recorded, it is **not** certified, and the rule's statement does not cover it. `sample` is null (the fallback has no per-query evidence) |

`certified` is true only for `select` under one of the library's own certifying rules: the exact
types `FixedSequenceBetting` and `BonferroniBetting`, whatever a rule declares about itself (a rule
that sets `certifying = True`, or a subclass that keeps a built-in name, is not one). That is
enforced when the decision is made (`Decision`, `select`), when a certificate is created, and by
consistency checks when it is verified: a record that says `certified: true` for a rule that is not
one of those fails `rule.text`, `consistency.level`, `consistency.decision` and the replay (section
4). A certificate is issued for every outcome, so an abstention is as checkable as a selection.

## 4. Verification

`verify_certificate(cert, *, evidence=None, corpus=None, queries=None, reference=None,
pipeline=None, compressed=None, key=None, strict_code=False) -> VerificationReport`

The report is a tuple of checks, each `(name, status, detail, level)` with status `pass`, `fail`,
`skipped` or `differs`. `ok` is true when no check failed (`skipped` and `differs` do not fail a
report). `level` is the highest of `integrity`, `replay`, `binding` in which at least one check ran
and `levels` lists all of them; a level can be reached while only some of its checks ran, so read
the checks. `failed()` lists the failures and `summary()` is a readable account.
`verify_certificate` on a certificate that has been parsed does not raise on a bad one: a check that
cannot be evaluated is reported as failed, and a record whose structure is invalid stops after
`schema`, `digest` and `signature` with one `remaining` check marked skipped. It raises `TypeError`
for wrong argument types. Reading is a different step: `Certificate.load` of a malformed file raises
`CertificateFormatError`, and `load_artifact` or `verify` on a bundle directory that cannot be read
(a malformed file, a pipeline that cannot be rebuilt) raises `ArtifactError`, or
`FileNotFoundError` for a path that is not a directory. `ok` does not make a
certificate true: it is as good as the queries, labels and corpus it was issued for.

**Level 1, integrity (certificate only).**

1. `schema`: schema name and version are known, exactly the 25 fields are present with the right
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
   [0, 1], `n_pool = evidence.n_queries + dropped`, `n_used` agrees with the budget, the digest of
   the tested positions is reproduced from `kind`, `seed` and `budget`, `exact_duplicates` lies in
   [0, `n_used`), and each `norms` entry follows from its extremes and the tolerance);
   `consistency.level` (`level` equals what the rule's configuration derives from `error` and the
   list: `error / n_families` or `error / n_candidates`; needs a built-in rule, see below);
   `consistency.outcome` (`outcome`, `certified`, `selected`, `certified_positions`, `candidate`
   and `sample` agree with each other, and so do the parts of the `rule` record: `certifying` is
   true exactly when the rule is built in and declares itself certifying, and a rule recorded as
   built in has the name and class of one of the library's own);
   `consistency.decision` (`certified_positions` and `selected` equal what the rule's
   deterministic decision step gives for the stored p-values, families, sizes and level);
   `consistency.candidate` (`candidate.pipeline_sha256` is the digest of `candidate.pipeline`;
   for `select` that pipeline is the selected candidate's list entry; `fitted_state` has one entry
   per stage; the byte figures agree with each other, with the corpus identity and with
   `sizes.bytes_per_doc` of the selected candidate);
   `consistency.sample` (`retention_point` is `mean / baseline_mean`; `n` is `queries.n_used`).
5. `rule.text`: the rule record (name, class, flags, configuration), `statement`, `assumptions` and
   `caveats` are those the running library writes for the same rule, from the record's own
   fields. `candidate.traits`: the recorded traits are those the running library derives from
   `candidate.pipeline`. A difference is a `fail` when `code.digest` equals the running code's (the
   same code cannot write a different text), and `differs` otherwise (a different version may word a
   text differently); it is a `fail` under `strict_code`. For a stage this process does not know,
   `candidate.traits` is `skipped`. A rule counts as built in only when its name, its qualified
   class name and its `builtin` flag all describe one of the library's own rules. For a rule that
   is not, a record that says `certified: true` **fails** (it is never skipped, and editing
   `code.digest` does not turn it into `differs`); a record that says nothing is certified is
   checked for what the library itself writes there (the fixed statement, the label on every
   assumption, the caveats) and passes with a detail that says the rule's configuration was not
   examined.
6. `code.files`: the running code's identity compared with `code.digest`. A difference is reported
   as `differs` with the files that changed, were added or were removed, not as a failure, unless
   `strict_code=True`: a different library version is not tampering, but a replay under different
   code is weaker evidence. `code.libraries`: differing Python, NumPy or SciPy versions are
   reported as `differs` and never fail a report, even under `strict_code`.

`consistency.level`, `consistency.decision` and the replay need one of the library's own rules
(`fixed_sequence_betting`, `bonferroni_betting`; section 2, `rule`). For any other rule a record
that says `certified: true` fails all three, and a record that says nothing is certified skips them.
In that case the report's `notes` (printed by `summary()` and by the command line, and listed by
`to_dict()`) say that the record names a rule that is not part of this library, that nothing in it
is certified and that those checks were not run; in the first case they say that the claim is
refused.

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
    `queries.identity`; once it does, the supplied vectors are the certified ones, so the recorded
    `norms` entry must equal the one they give (extremes at relative tolerance 1e-9) and, for the
    queries, `queries.exact_duplicates` must equal the count recomputed with the recorded dropped
    ids and order. These are the only checks that can confirm a count or a norm: no other check
    has the vectors. `binding.evidence_queries` (needs queries and evidence): the evidence's
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
every object supplied. Every row is exercised by `tests/test_core_certificate_tamper.py` (the
case ids are in the last column), once with the digest left alone (the `digest` check fails for
every row) and once with it recomputed, as a careless or dishonest editor would. The adversary of
the table is that editor: someone who can change the file and recompute the digest, and has no
HMAC key. Where a row says "not detected" the test asserts it, so that the limit is pinned and not
hidden.

| modified (digest recomputed) | caught alone by | caught with everything by | test case |
|---|---|---|---|
| a candidate pipeline parameter, in `candidate` only | `consistency.candidate` | adds `binding.codes`; `binding.pipeline` too if `pipeline_sha256` was rewritten | `pipeline-parameter`, `pipeline-parameter-and-its-digest` |
| a candidate pipeline parameter rewritten in the list and `space.identity` only | `consistency.candidate` (the candidate record no longer matches its list entry) | adds `evidence.fields`, `replay.decision` | `pipeline-parameter-in-the-list` |
| the same parameter rewritten everywhere (`candidate`, its digest, the list and `space.identity`) | not detected | `evidence.fields`, `replay.decision`, `binding.pipeline`, `binding.codes` | `pipeline-parameter-everywhere` |
| candidate list: drop, reorder or add one (identity stale) | `consistency.space` (and `consistency.level` or `consistency.decision` if the count or the families change) | adds `replay.decision` | `list-drop-one`, `list-reorder`, `list-add-one` |
| candidate list rewritten consistently (identity, counts, sizes, p-values, level, decision) | not detected | `evidence.fields`, `replay.decision` | `list-*-consistently` |
| target | not detected | `replay.decision` | `target` |
| error level (level unchanged) | `consistency.level` | adds `replay.decision` | `error-level` |
| error level and level together | `consistency.decision` when the certified set would change | adds `replay.decision`; a rewritten selection that differs from the deployed pipeline is caught by `consistency.candidate` and `evidence.sample` | `error-level-and-level`, `error-level-and-the-decision-for-another-candidate` |
| query identity | `consistency.identities`; if its combined digest is refreshed: not detected | `binding.queries` | `query-identity`, `query-identity-consistently` |
| query count (`n_pool`), dropped queries | `consistency.counts` (and `consistency.identities` for `n_pool`), and `rule.text`, because the sentence about removed queries follows from them | the same; dropped queries add `evidence.fields` and `binding.queries` | `query-pool-size`, `dropped-query-hidden` |
| query order kind or seed | `consistency.counts` (the digest of the tested positions); if the digest is refreshed too: `rule.text` (the order sentence follows from kind and seed); if the sentence is rewritten as well: not detected | `replay.decision` (and `evidence.sample`) | `query-order-seed`, `query-order-seed-and-its-digest`, `query-order-seed-its-digest-and-its-caveat` |
| query budget | `consistency.counts` and `consistency.sample`; if the position digest is refreshed too: `consistency.sample` | `replay.decision`, `evidence.sample` | `query-budget`, `query-budget-and-its-digest` |
| the order sentence, or the sentence about removed queries, deleted | `rule.text` | same | `order-caveat-removed`, `dropped-queries-caveat-removed` |
| `exact_duplicates` lowered or raised | `rule.text` (the sentence follows from the count), `consistency.counts` if outside [0, `n_used`); if the sentence is rewritten as well: not detected | `binding.queries` (the count is recomputed from the live queries) | `duplicate-count-cleared`, `duplicate-count-invented`, `duplicate-count-and-its-caveat-cleared`, `duplicate-count-beyond-the-queries-tested` |
| `norms` (a flag set, extremes or tolerance edited) | `consistency.counts` (the flag must follow from the extremes and the tolerance) and `rule.text`; if rewritten consistently, with the sentence: not detected | `binding.corpus`, `binding.queries` (the norms are recomputed from the live vectors) | `norm-flag-set-to-hide-it`, `norm-extremes-edited-and-the-flag-kept`, `norms-rewritten-to-hide-it`, `norm-tolerance-widened-to-hide-it` |
| reference identity | not detected | `binding.reference` | `reference-identity` |
| construct | `consistency.identities`, `rule.text` (caveats); if rewritten consistently: not detected | adds `evidence.fields`, `binding.reference` | `construct`, `construct-to-hide-the-agreement-caveat` |
| fitted-state digest | not detected (a removed entry: `consistency.candidate`) | `binding.pipeline`, `evidence.fields`; and `binding.codes` if the codes digest was changed | `fitted-state-*` |
| code identity (files table with its digest refreshed) | not detected; `code.files` is `differs` | `differs`; `fail` under `strict_code`. A digest that does not match its table: `consistency.identities` | `code-identity`, `code-identity-digest-only` |
| outcome, `certified`, selected position, certified positions | `consistency.outcome`, `consistency.decision`, `consistency.candidate` | adds `replay.decision` | `outcome-*`, `certified-*`, `selected-position-*`, `certified-position-*` |
| one p-value | `consistency.decision` when it changes the decision; otherwise not detected | `replay.decision` | `p-value-*` |
| **the rule renamed** (`rule.name`), with or without another edit (the target, the error level, every p-value forged): the record says `certified: true` for a rule the library does not know | `rule.text`, `consistency.level`, `consistency.decision`, and `consistency.outcome`: a certified record whose rule is unknown **fails**, it is never skipped, and editing `code.digest` as well does not downgrade `rule.text` to `differs` | adds `replay.decision` | `rule-renamed`, `rule-renamed-and-the-target-edited`, `rule-renamed-and-the-decision-forged`, `rule-renamed-and-the-code-identity-edited` |
| **the rule's class or built-in flag** (`rule.class`, `rule.builtin`) | the same four checks | adds `replay.decision` | `rule-class-renamed`, `rule-builtin-flag-cleared` |
| `rule.declared_certifying` or `rule.certifying`, or another built-in rule's name and class | `consistency.outcome` and `rule.text`; the other built-in rule: `consistency.level` and `rule.text` | the same, plus `replay.decision` for the swap | `rule-declared-certifying-cleared`, `rule-certifying-flag`, `rule-swapped-for-the-other-builtin-rule` |
| a record of a rule of someone else's edited to say `certified: true` (and built in, and with the library's statement) | `consistency.outcome`, `rule.text`, `consistency.level`, `consistency.decision` | adds `replay.decision` | `custom-rule-certified-flag-set`, `custom-rule-claims-to-be-built-in` |
| the same record rewritten completely as one of the library's own (name, class, flags, statement, assumptions, caveats): the texts then pass | `consistency.level` and `consistency.decision`, when the stored level and p-values do not support the decision (in the test every p-value is 1, so the library's rule would certify nothing); p-values and level forged consistently as well: not detected | adds `replay.decision`, unless the evidence is forged too | `custom-rule-given-the-librarys-name-and-class` |
| the statement, the label on a rule's assumptions or its caveats edited in a record of a rule of someone else's | `rule.text` | same | `custom-rule-statement-replaced-by-the-librarys`, `custom-rule-assumptions-unlabelled`, `custom-rule-caveat-removed` |
| a certified record rewritten into an uncertified one (unknown rule, `certified: false`, consistent texts) | not detected, and not a forgery: it claims less. The level, decision and replay checks are `skipped` and the report's notes say that nothing in it is certified | the same | `TestDocumentedLimits` |
| corpus identity | `consistency.identities`; if the combined digest is refreshed: not detected. Vector or document counts: `consistency.candidate` | `binding.corpus` | `corpus-*` |
| byte figures | `consistency.candidate` for what is tied to something else (parts and totals, the size key, the corpus); the serving-state and append-state figures: not detected | `binding.bytes`; the copies in `sizes`: `evidence.fields` | `bytes-*`, `sizes-vectors-per-doc` |
| statement, an assumption, caveats, rule configuration (built-in rule) | `rule.text` (`fail`, or `differs` if `code.digest` was also changed) | same | `statement-text`, `assumption-*`, `rule-configuration`, `agreement-caveat-removed`, `all-caveats-removed`, `statement-and-code-identity` |
| sample statistics | `consistency.sample` (internal arithmetic) | `evidence.sample` | `sample-*` |
| `created_utc`, `package`, `encoder`, `space.name` | not detected | not detected | `created-time`, `package-version`, `encoder-declared`, `space-name` |
| any field, under an HMAC key | `signature` | `signature` | every case |

Supplying objects that are not the certified ones is caught by the binding checks of level 3: a
corpus, queries or labels that differ in one value, a reordered id list, a pipeline fitted on other
data (`binding.pipeline`), codes, offsets or ids that differ (`binding.codes`). Evidence with one
value changed fails `evidence.digest` and the replay.

A party who rewrites the certificate **and** supplies matching forged evidence and objects is
not detected by levels 1-3: a certificate issued honestly from forged evidence verifies. Such a
party cannot make a rule of their own pass as certified (a record that says so fails whatever else
it contains), and can always make a record claim less. Only a signature from a key they do not hold
closes the first gap. Without an HMAC key nothing in the record or the bundle is a root of trust.

## 5. The statement a certificate carries

For the library's own rules (the fixed-sequence betting rule, the port of the research's Method C,
and the Bonferroni rule), `statement` is `optivision.core.selection.STATEMENT` verbatim:

> For the target T and the error level delta of this decision: P(a compressed candidate is selected
> and its retention under Q is below T) <= delta. The probability is over the draw of the
> calibration queries; the corpus, the candidate list and its order, T and delta are fixed, not
> random. Q is the distribution the calibration queries are drawn from, and retention under Q is
> E_Q[metric of the candidate] / E_Q[metric of the uncompressed index]. This is a joint probability.
> It is NOT the probability, given that a candidate was selected, that its retention is below T: a
> rule that rarely selects can satisfy it while most of its selections are below T. It is not a
> statement about the retention measured on the calibration queries, not a statement about a finite
> batch of future queries, and not a statement about power (it also holds if the rule always
> abstains). The project derived it for every sample size, with no large-sample approximation, under
> assumptions A1-A6 and conditions C1-C4. They are sufficient conditions: nothing is claimed if one
> of them fails. It is an application of published tools; the derivation is this project's own and
> has not been externally reviewed.

and `assumptions` are the ten sentences below (the Bonferroni rule's A3 does not ask for an order
inside a family). A1 to A6 are the six assumptions of the research's statement, numbered as in the
preregistration (section 16.2); C1 to C4 are conditions of use that the research protocol satisfied
by construction and that the user must satisfy:

> A1: the calibration queries are independent draws from one distribution Q, the distribution of the
> queries the deployed index will serve. Repeated or near-duplicate queries, queries from one
> session or one author, and a sample sorted or filtered by anything related to difficulty are not
> independent draws.
>
> A2: the corpus is fixed; retention moves with corpus size, so the statement is about this corpus.
>
> A3: the candidate list, its family labels and chains, and the order inside each chain are fixed
> before the calibration queries are seen, and no fitted stage uses the calibration queries (in this
> library stages are fitted on documents only).
>
> A4: the per-query metric lies in [0, 1] and the expected metric of the uncompressed index under Q
> is positive.
>
> A5: the order in which queries enter the sequential test is fixed before their outcomes are seen
> and is unrelated to them: the arrival order of independent draws, or a random permutation drawn
> without reference to the data. An order sorted by any query attribute, or picked among several
> orders after looking, is not.
>
> A6: the reference is taken as given (the labels under the label construct, the uncompressed
> index's own top document under the agreement construct); errors in the reference are not modelled.
>
> C1: one look. One target, one error level, one rule, one metric, one reference and one candidate
> list are fixed before the outcomes are seen, and the procedure is run once on this sample;
> re-running it with other choices on the same queries is outside the statement.
>
> C2: the baseline and every candidate are scored on the same queries with the same metric and the
> same reference.
>
> C3: the reference (the labels, or the uncompressed ranking) is fixed without regard to the
> candidates' results.
>
> C4: the index that is deployed is the one that was evaluated: the same corpus, the same pipeline,
> the same fitted state and the same scoring procedure.

The statement is a joint probability over the draw of the calibration queries. It is not a
conditional probability given that something was selected, not a statement about the retention
measured on the sample, not a statement about a finite batch of future queries, and not a statement
about power: the rule may abstain every time. It says how often the procedure goes wrong, not how
reliable one selection is. The rule is an application of published tools (testing by betting;
fixed-sequence testing; a union bound). The derivation that combines them is the project's own and
has not been externally reviewed (`STATISTICAL_REVIEW.md` is an internal second reading, not an
external review). Under the `agreement@1` construct, "retention" means agreement with the
uncompressed ranking, which is not retrieval quality.

The certificate cannot check A1, A2, A5, A6 or C1 to C4: they are properties of how the caller
obtained the queries and labels and of what the caller deploys. It records what the caller supplied,
counts exact duplicate queries (`queries.exact_duplicates`), records how the queries were ordered
and whether the vectors have unit norm, and says in `caveats` what each of those requires. A3 is
supported, not proven, by the code path: the default list depends only on the dimension, and stages
are fitted by `Pipeline.compress(corpus)` without queries.

For any other rule `statement` is the fixed text of section 2, never this one (section 3).

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
matching manifest, and anyone who can edit the certificate can recompute its digest: without an
HMAC key nothing in the bundle is a root of trust. `load_artifact` checks the certificate too (its
digest, the codes and fitted-state digests it records, an optional HMAC).

`load_artifact(path, *, verify=True, key=None, strict_code=False) -> Artifact` rebuilds the
pipeline from the certified specification, restores the fitted state, and rebuilds the
`CompressedCorpus`. With `verify=True` it checks the manifest (every file listed and
present, hashes and sizes equal, nothing unknown listed), restores the fitted state with the
digests the certificate records, checks that the `dim` and `stats` of the `codes.npz` header
(which no digest covers) agree with the certificate and the pipeline, then runs `verify_certificate` with the pipeline, the codes and,
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
  sample of the deployment distribution (it counts exact duplicates; near-duplicates, sessions,
  authors and curation are invisible to it), whether the order was unrelated to the outcomes,
  whether several targets, seeds or lists were tried on the same sample and one reported, or
  whether the candidate list was edited after a look at the results.
- The `code` field identifies source files, not the numerical libraries beneath them. Replay
  compares p-values at a relative tolerance for that reason.
- The byte figures follow the library's accounting: the size key counts codes and codec
  overhead; fitted-state bytes are listed separately and are not part of the key.
- Schema version 1 has no revocation, expiry or chaining.
- The `code` file set (section 2.1) leaves out modules outside the decision path (page encoders,
  the command line, backends, adapters, storage); a change there does not change the identity.
- The decision step, the level and the replay can be recomputed only for the library's own rules
  named in section 4. A rule written by anyone else is never certified: its record carries the
  fixed text of section 2 instead of the library's statement, its assumptions are labelled as its
  author's, and verification fails any claim of `certified: true` for it (a record that claims
  nothing skips those checks and says so in the report's notes). The library does not vouch for such
  a rule: its author must publish the rule's statement and derivation with it.
- Without an HMAC key the digest and the manifest are integrity checks, not roots of trust.
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
15. After the internal and statistical reviews (`RESEARCH_INTEGRITY_AUDIT.md` RI-01 to RI-06 and
    RI-13, `STATISTICAL_REVIEW.md` SR-1 to SR-5): `rule` records its class, whether it is built in
    and what it declared, and `certified` is true only for the library's own rules; a record that
    claims otherwise fails `rule.text`, `consistency.level`, `consistency.decision` and the replay
    instead of skipping them, and the report has `notes`; `statement` and `assumptions` are the
    reworded ones (ten sentences: A1-A6 and C1-C4); `queries.exact_duplicates`, the new top-level
    field `norms` and the order, duplicate, norm and removed-query caveats were added, all checked by
    `rule.text` and, with the live objects, by `binding.corpus` and `binding.queries`. The field count
    is 25. Nothing was released with the earlier shape of the record, so the schema is still version 1.
16. `certify()` and the command line test the queries in a seeded random order by default (seed
    0); the lower-level `select()` keeps the order given as its default, and the certificate says
    which was used and what it requires.
