# A03: certificate and artifact attack report

Agent 03, branch `harden/a03`, worktree `C:\Users\Lenovo\projects\ov-harden\a03`, base 7a6769a.
No claim of cryptographic security is made anywhere here: the digest and the manifest are
integrity checks, and without an HMAC key a coherent forgery verifies (stated limit, not a finding).

## 1. Scope and what was executed

Attack scripts (synthetic data only, `optivision.testing`) are in
`C:\Users\Lenovo\projects\ov-harden\scratch\a03\`: `matrix.py` (about 130 edits x 2 certificates
(select, abstain) x {digest stale, digest recomputed} x {certificate alone, +evidence, everything}
x {no key, HMAC key with the signature left stale}), `json_attacks.py` (text-level edits, signature and
digest edits), `depth.py`, `bundle_attacks.py` and `bundle2.py` (about 90 edits of bundle files with
the manifest refreshed), `binding.py` (34 substitutions of live objects at level 3), `extra.py`
(stateful candidate, code identity, custom rule). Code read: `certificate.py`, `bundle.py`,
`api.py` (`verify`), `identity.py`, the signature path of `verification.py`, `evidence.py`
(`from_npz`).

Tests run after the fixes: `test_core_artifact_bundle.py`, `test_core_artifact_malformed.py` (new),
`test_core_certificate.py`, `test_core_certificate_tamper.py`, `test_api_certify.py`,
`test_api_compat.py`, `test_integration_serialization.py`, `test_integration_end_to_end.py`:
1080 passed. `ruff check src tests/test_core_artifact_malformed.py` clean.
The other `test_integration_*` files were not run (shared machine).

## 2. Findings

| ID | Sev | Where | Problem | Status |
|---|---|---|---|---|
| A03-01 | MEDIUM | `bundle.py` `_states_from_file`, `_load_deployed`, `_load_evidence`, `_check_manifest`, `_read_header`; `certificate.py` `from_json`/`__post_init__` | Edited files that the spec says raise `ArtifactError` (section 7: "raises `ArtifactError` for every unreadable file") raised raw `AttributeError`, `KeyError`, `TypeError` or `RecursionError` from inside the loader | FIXED in 588e216 |
| A03-02 | MEDIUM | `bundle.py` `_load_deployed` | `dim` and `stats` in the `codes.npz` header are covered by no digest. With the manifest refreshed, `dim` 12 -> 16 or 99 loaded, and `load_artifact` and `verify` reported OK, although `dim` decides how the codes decode (spec section 7: the loaded corpus "decodes bit for bit like the one that was saved") | FIXED in 588e216 |
| A03-03 | LOW | `verification.py` schema check; spec section 2 | Only top-level fields are closed. An unknown key added inside `result`, `queries`, `space`, `code`, `package`, `evidence`, `reference`, `sizes`, `norms`, `candidate`, `sample`, `signature` passes every level when the digest is recomputed (only `rule`, `corpus`, `queries.identity`, `space.candidates[i]`, `candidate.traits` and `queries.order` catch it, by other checks). The spec promises only the top-level check, so this is not a spec violation | PROPOSED |
| A03-04 | LOW | `api.py` `verify` | For a directory whose files are malformed in a way `load_artifact(verify=False)` cannot read (for example `candidate.pipeline` not rebuildable), `verify` raises `ArtifactError` instead of returning a failed report. Matches the docstring ("an unreadable directory raises"), but differs from the spec sentence "verification never raises on a bad certificate" | PROPOSED (wording) |
| A03-05 | NO ACTION | see section 4 | Edits that pass and match the spec | NOT A DEFECT |

### A03-01 detail and evidence

Reproducers (before the fix): `Certificate.from_json("[" * 100000 + "]" * 100000)` raised
`RecursionError` (anything nested deeper than about 490 levels; below that, verification reported
`schema` failed correctly). In a bundle with the manifest refreshed: fitted header entry `7` or `null`
(`AttributeError` in the error message path itself: `entry.get` on a non-dict), `scalars: [1]` or `null`,
`arrays: 5` (`TypeError`); evidence header without `metric`, or `query_ids: 5` (`KeyError`/`TypeError`);
`certificate.json` with `candidate: {}`, `candidate: 5` or `fitted_state: 5` (`KeyError`/`TypeError`);
`MANIFEST.json` deeply nested (`RecursionError`). Fix: validate the shapes, and convert these to
`ArtifactError` / `CertificateFormatError`. Regression: `tests/test_core_artifact_malformed.py`
(19 of 21 cases fail on the old code, the other 2 are the sanity cases). Regression risk: low; only
error paths changed.

### A03-02 detail

Fix: with `verify=True`, `load_artifact` now requires `compressed.dim` to equal the dimension the
certified pipeline gives from the certified corpus dimension (folding `output_dim` over the dimension
reducers), the codes to have `quantizer.code_bytes(dim)` columns, and `stats.vectors_before` /
`stats.dim_before` to equal the certified corpus. `verify=False` is unchanged by contract. Regression
risk: low (an honest bundle has exactly these values; all 6 pipeline scenarios of the existing bundle
tests, including `lloyd2`, rotation and merge, still round trip). Note `dim` 12 -> 16 for a 1-bit
code has the same code width, so the width check alone would not have caught it; the pipeline-derived
dimension does.

### A03-03 proposal

Either (a) spec wording: "unknown keys inside nested objects are not rejected; they are covered by the
digest and the HMAC only", or (b) extend the `schema` check to compare the exact key sets of the nested
objects, which would also make an HMAC-less smuggled note impossible. (b) is more than about 20 lines and
changes what a future schema 1.x may add, so it is a design decision. Not implemented.

### A03-04 proposal

Spec/docstring wording: say that `optivision.verify(directory)` raises `ArtifactError` for a directory
whose files cannot be read at all, and reports everything else. Alternatively catch it and return a report
with one failed `artifact files` check.

## 3. Attack results (condensed)

"Digest recomputed" = the careless or dishonest editor of spec section 4.1. L1 = certificate alone,
+ev = with evidence, all = with evidence, corpus, queries, reference, pipeline, compressed.
Every edit with the digest left alone fails `digest`; every content edit under an HMAC key fails
`signature` even with the digest recomputed (keyed runs: all content edits caught, the not-detected
set below is exactly the set that only the signature catches). Spec promises detection = yes unless
stated.

| attack | digest recomputed: detected by | spec promise | match |
|---|---|---|---|
| target (down or up) | L1 no; +ev `replay.decision` | table: replay | yes |
| error / level | L1 `consistency.level` (+ `replay.decision`) | yes | yes |
| selected position, to null | L1 `consistency.decision`, `consistency.candidate`/`outcome` | yes | yes |
| outcome abstain -> select, certified false -> true | L1 `consistency.outcome` | yes | yes |
| outcome select -> abstain, certified true -> false | L1 `consistency.outcome`, `consistency.decision` | yes | yes |
| p-values all 0, certified_positions all | L1 `consistency.decision` (when the decision changes) | yes | yes |
| one p-value that does not change the decision | L1 no; +ev `replay.decision` | table | yes |
| candidate list drop / identity | L1 `consistency.space` | yes | yes |
| space.name, created_utc, package.version, encoder, code.python | not detected (HMAC only) | table: not detected | yes |
| query identity digest / vectors sha | L1 `consistency.identities`; all `binding.queries` | yes | yes |
| corpus sha / digest / counts | L1 `consistency.identities`; all `binding.corpus` | yes | yes |
| n_pool, n_used, order seed/kind, exact_duplicates, norms flag | L1 `consistency.counts` / `rule.text`; `binding.*` | yes | yes |
| rule name/class/config/certifying/builtin, unknown rule kept certified | L1 `rule.text`, `consistency.*` | yes | yes |
| subclass of a built-in rule flipped to certified, or renamed to the built-in's name/class | `rule.text` (and the rest) | yes | yes |
| statement, assumptions edited | `rule.text` | yes | yes |
| evidence.digest, baseline_mean, reference.metric | L1 no; +ev `evidence.digest` / `evidence.fields` | table (replay level) | yes |
| sizes.bytes_per_doc | L1 `consistency.candidate` (select) / +ev `evidence.fields` (abstain) | table | yes |
| reference identity | L1 no; all `binding.reference` | yes | yes |
| construct | L1 `consistency.identities`, `rule.text` | yes | yes |
| code.digest / one file hash | L1 `consistency.identities`; coherent edit: `code.files` differs, fails under `strict_code` | yes | yes |
| fitted_state digest (changed, nulled, swapped) | L1 no; all `binding.pipeline`, `evidence.fields` | yes | yes |
| codes_sha256 / offsets_sha256 | L1 no; all `binding.codes` | yes | yes |
| candidate.bytes, traits, sample.* | `consistency.candidate`, `candidate.traits`, `consistency.sample` | yes | yes |
| schema_version/name changed; any of 10 fields removed; unknown top-level field | `schema` | yes | yes |
| `digest` removed / null / int / uppercase / truncated / non-ASCII | `schema` and/or `digest` | yes | yes |
| signature stripped (null) with key supplied; wrong key; value truncated, empty, uppercase, non-ASCII, int, null; algorithm none / other / string / empty | `signature` (with key); `schema` for wrong types | yes (spec section 6) | yes |
| signature stripped, no key supplied | not detected: `signature` is `skipped` | yes ("not signed" is skipped) | yes |
| `key_id` changed, extra key inside `signature` | not detected | spec: key_id not authenticated | yes |
| key empty `bytes` / `str` | `ValueError` / `TypeError` | yes | yes |
| key order, whitespace, CRLF, `9e-1`, `5e-2`, `0.050`, `\u0061` escapes | not a modification: verifies | n/a (digest is over canonical JSON) | yes |
| `1.0`/`true` for `schema_version`, `true` for target, `-0.0` target | `schema` | yes | yes |
| NaN, Infinity, `1e999`, duplicate keys (top-level, nested), BOM, trailing data, empty, array, null, 5000-digit int | refused at load with `CertificateFormatError` | spec 9.14 | yes |
| nesting deeper than ~490 | raw `RecursionError` before the fix | spec: clean errors | fixed (A03-01) |
| unicode normalisation (NFD combining mark, fullwidth letters) in the statement | `rule.text` (the bytes differ) | yes | yes |
| HMAC comparison | `hmac.compare_digest` on bytes; non-ASCII value does not raise | yes | yes |
| unknown key in a nested object | not detected | spec silent (A03-03) | gap, LOW |

Level 3 binding substitutions (34, `binding.py`): another corpus of the same shape, one float32 ulp, the
same values as float64, ids reversed or renamed, documents swapped or truncated, weights or a protected
mask added, another query set, queries reordered, truncated, one value changed, ids renamed, labels with
one qrel changed, one query added or removed, `RankingReference` instead of labels, a pipeline and codes
from another run, a flipped code byte, evidence of another run, one flipped baseline entry: all caught.
Not detected, as the spec defines the identity: corpus `metadata` and `attrs`, label dict order, order of
document ids inside one query's label set.

Bundle attacks (manifest refreshed after each edit): fitted-array value +1e-3, truncated, NaN; codes bit
flip; ids permuted; header `pipeline` null or edited; `version` 2; wrong dtype/shape/offsets; codes or
fitted or evidence swapped from another artifact (each alone); evidence header and baseline edited; evidence
forged coherently with its own digest recomputed (caught by `evidence.digest`, because the certificate
holds the digest); pickled object arrays in `evidence.npz` and `codes.npz` (refused, `allow_pickle=False`
everywhere in `bundle.py` and `Evidence.from_npz`; no other unpickling on the load path); manifest keys
`../x`, absolute path, backslash path, `./codes.npz`, entry as a string, wrong `size` type, non-UTF-8:
all refused, and no manifest key is ever used as a path (only the four fixed names are opened), so there
is no traversal. All caught or cleanly refused after the A03-01/02 fixes.

## 4. Edits that pass, and why that is correct

- All files of one coherent other artifact swapped in (certificate, codes, fitted, evidence, manifest
  refreshed) verifies: it is a valid artifact. Spec section 4.1 last paragraph.
- `evidence.npz` deleted and manifest refreshed: loads, replay not run. Spec section 8: evidence is
  checked only when supplied.
- `codes.npz` stored as float64 instead of uint8 with identical values: the constructor normalises to
  uint8, the digest is over the normalised array. Harmless.
- Extra unlisted file in the bundle directory: ignored (spec section 7, other files are left alone).
- `Certificate.load` of a signed certificate and `verify` without key: `signature` skipped (spec).

## 5. What was not checked

Symlinked bundle files or directories (Windows, privileges); zip-bomb style `.npz` sizes; platform
differences (only Python 3.11 on this Windows laptop); the command line `optivision verify` wrapper
(`cli_certify.py`); `Evidence.from_npz` called directly by a user on a malformed file still raises raw
`KeyError`/`TypeError` (the bundle loader now wraps it; the function itself is unchanged); the whole
suite (the lead runs it).
