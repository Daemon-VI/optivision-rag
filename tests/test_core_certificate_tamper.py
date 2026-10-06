"""Tampering with a certificate: what each modification is caught by, and what is not.

Every modification of the table below is tried three ways, as a careless and as a dishonest
editor would:

(a) the field is changed and the digest is left alone: the digest check fails;
(b) the field is changed and the digest is recomputed: the check that
    ``docs/package/CERTIFICATE_SPEC.md`` section 4.1 names must fail, once with the certificate
    alone (level 1) and once with the evidence and every live object supplied; where the
    specification says that a value is tied to nothing (the code identity, descriptive fields, a
    p-value that changes no decision) the test asserts exactly that, so the documented limit is
    pinned and not hidden;
(c) with an HMAC key every modification fails the signature, even after the digest is
    recomputed, and a wrong key fails.

A party who rewrites the certificate and also supplies matching forged evidence and objects is
not detected at any level; that limit is pinned below as well. So is the one thing a key-less
editor can always do, which is to make a record claim less: a certified record rewritten as an
uncertified one verifies, and says in the report's notes that nothing in it is certified.

The rows that rename the rule, change its class or its built-in flag, or claim ``certified`` for
a rule of someone else's are the adversary of ``docs/package/RESEARCH_INTEGRITY_AUDIT.md`` (RI-01
and RI-02): a record that claims ``certified: true`` for a rule the library does not know fails
``rule.text``, ``consistency.level``, ``consistency.decision`` and the replay, and is never
``skipped``, even when the code identity is edited too.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pytest

from optivision.compose import CompressedCorpus
from optivision.core.artifact import Certificate, sign_certificate, verify_certificate
from optivision.core.artifact.record import (
    CAVEAT_DROPPED,
    CAVEAT_ORDER_GIVEN,
    CAVEAT_ORDER_SEED,
    STATEMENT_UNEXAMINED,
    caveats_for,
)
from optivision.core.evaluate import (
    LabelReference,
    RankingReference,
    collect_evidence,
    compress_selected,
)
from optivision.core.identity import array_digest, digest
from optivision.core.selection import (
    STATEMENT,
    FixedSequenceBetting,
    fixed_sequence_by_family,
    rule_record,
    select,
    smallest,
)
from optivision.core.selection.rules import DECLARED_PREFIX
from optivision.representation import MultiVectorCorpus

from .test_core_certificate import KEY, DeclaresCertifying, Issued, issue, with_copies
from .test_core_collect_evidence import make_problem, small_space

WRONG_KEY = b"w" * 32
ZEROS = "0" * 64


# ------------------------------------------------------------- what an editor does


def recompute(d: dict) -> None:
    """What a careless or dishonest editor does after changing a field: refresh the digest."""
    d["digest"] = digest({k: v for k, v in d.items() if k not in ("digest", "signature")})


def fix_identity(ident: dict) -> None:
    """Refresh the combined digest of a corpus or query identity."""
    ident["digest"] = digest({k: v for k, v in ident.items() if k != "digest"})


def set_candidates(d: dict, candidates: list[dict]) -> None:
    """A capable editor's consistent rewrite of the list: positions, identity, counts, families."""
    for i, c in enumerate(candidates):
        c["position"] = i
    s = d["space"]
    s["candidates"] = candidates
    s["identity"] = digest(candidates)
    s["n_candidates"] = len(candidates)
    s["families"] = list(dict.fromkeys(c["family"] for c in candidates))
    s["n_families"] = len(s["families"])


def rederive(d: dict) -> None:
    """Re-run the decision step on the stored p-values, as a capable editor would."""
    families = [c["family"] for c in d["space"]["candidates"]]
    d["level"] = d["error"] / len(dict.fromkeys(families))
    rejected = fixed_sequence_by_family(d["result"]["p_values"], families, d["level"])
    d["result"]["certified_positions"] = list(rejected)
    d["result"]["selected"] = smallest(rejected, d["sizes"]["bytes_per_doc"],
                                       d["sizes"]["vectors_per_doc"])


# --------------------------------------------------------------------- the table


@dataclass(frozen=True)
class Case:
    """One modification and what must catch it.

    ``full``: checks that must fail when the evidence and every object are supplied.
    ``alone``: checks that must fail with the certificate alone; ``None`` means that the
    certificate alone cannot show it (a documented limit, asserted as such).
    ``expect``: statuses that must also hold with everything supplied (for example ``differs``).
    """

    id: str
    fixture: str
    mutate: Callable[[dict, Issued], None]
    full: tuple[str, ...] = ()
    alone: tuple[str, ...] | None = None
    expect: dict[str, str] = field(default_factory=dict)


CASES: list[Case] = []


def case(id, *, fixture="selected", full=(), alone=None, expect=None):
    def register(fn):
        CASES.append(Case(id, fixture, fn, tuple(full), None if alone is None else tuple(alone),
                          expect or {}))
        return fn
    return register


# --- a candidate pipeline parameter (spec 4.1 row 1)


@case("pipeline-parameter", full=("consistency.candidate", "binding.codes"),
      alone=("consistency.candidate",))
def pipeline_parameter(d, out):
    d["candidate"]["pipeline"]["stages"][0]["radius"] = 0.6


@case("pipeline-parameter-and-its-digest",
      full=("consistency.candidate", "binding.pipeline", "binding.codes"),
      alone=("consistency.candidate",))
def pipeline_parameter_and_digest(d, out):
    d["candidate"]["pipeline"]["stages"][0]["radius"] = 0.6
    d["candidate"]["pipeline_sha256"] = digest(d["candidate"]["pipeline"])


@case("pipeline-parameter-in-the-list",
      full=("evidence.fields", "replay.decision"), alone=("consistency.candidate",))
def pipeline_parameter_in_the_list(d, out):
    d["space"]["candidates"][3]["pipeline"]["stages"][0]["radius"] = 0.6
    d["space"]["identity"] = digest(d["space"]["candidates"])


@case("pipeline-parameter-everywhere",
      full=("evidence.fields", "replay.decision", "binding.pipeline", "binding.codes"))
def pipeline_parameter_everywhere(d, out):
    for pipeline in (d["candidate"]["pipeline"], d["space"]["candidates"][3]["pipeline"]):
        pipeline["stages"][0]["radius"] = 0.6
    d["candidate"]["pipeline_sha256"] = digest(d["candidate"]["pipeline"])
    d["space"]["identity"] = digest(d["space"]["candidates"])


# --- the candidate list (row 1): drop, reorder, add


@case("list-drop-one", full=("consistency.space", "replay.decision"),
      alone=("consistency.space", "consistency.level"))
def list_drop_one(d, out):
    d["space"]["candidates"].pop()


@case("list-drop-one-consistently", full=("evidence.fields", "replay.decision"))
def list_drop_one_consistently(d, out):
    set_candidates(d, d["space"]["candidates"][:-1])
    for key in ("bytes_per_doc", "vectors_per_doc"):
        d["sizes"][key] = d["sizes"][key][:-1]
    d["result"]["p_values"] = d["result"]["p_values"][:-1]
    rederive(d)


@case("list-reorder", full=("consistency.space", "replay.decision"),
      alone=("consistency.space",))
def list_reorder(d, out):
    d["space"]["candidates"][1], d["space"]["candidates"][2] = (
        d["space"]["candidates"][2], d["space"]["candidates"][1])
    for i, c in enumerate(d["space"]["candidates"]):
        c["position"] = i


@case("list-reorder-consistently", full=("evidence.fields", "replay.decision"))
def list_reorder_consistently(d, out):
    cands = d["space"]["candidates"]
    cands[1], cands[2] = cands[2], cands[1]
    set_candidates(d, cands)


@case("list-add-one", full=("consistency.space", "replay.decision"),
      alone=("consistency.space", "consistency.level"))
def list_add_one(d, out):
    extra = copy.deepcopy(d["space"]["candidates"][0])
    extra.update(position=7, family="extra")
    d["space"]["candidates"].append(extra)


@case("list-add-one-consistently", full=("evidence.fields", "replay.decision"))
def list_add_one_consistently(d, out):
    extra = copy.deepcopy(d["space"]["candidates"][0])
    extra["family"] = "extra"
    set_candidates(d, [*d["space"]["candidates"], extra])
    for key in ("bytes_per_doc", "vectors_per_doc"):
        d["sizes"][key] = [*d["sizes"][key], d["sizes"][key][0]]
    d["result"]["p_values"] = [*d["result"]["p_values"], 1.0]
    rederive(d)


# --- target and error level (row 2)


@case("target", full=("replay.decision",))
def target(d, out):
    d["target"] = 0.5


@case("error-level", full=("consistency.level", "replay.decision"), alone=("consistency.level",))
def error_level(d, out):
    d["error"] = 0.01


@case("error-level-and-level", full=("consistency.decision", "replay.decision"),
      alone=("consistency.decision",))
def error_level_and_level(d, out):
    d["error"] = 0.01
    d["level"] = 0.01 / 4


@case("error-level-and-the-decision-for-another-candidate",
      full=("consistency.candidate", "evidence.sample"), alone=("consistency.candidate",))
def error_level_and_decision(d, out):
    d["error"] = 0.01
    d["level"] = 0.01 / 4
    d["result"]["certified_positions"] = [0, 1, 2]
    d["result"]["selected"] = 2  # a true statement of the evidence, but not the deployed index


# --- query identity, order and budget (row 3)


@case("query-identity", full=("consistency.identities", "binding.queries"),
      alone=("consistency.identities",))
def query_identity(d, out):
    d["queries"]["identity"]["vectors_sha256"] = ZEROS


@case("query-identity-consistently", full=("binding.queries",))
def query_identity_consistently(d, out):
    d["queries"]["identity"]["ids_sha256"] = ZEROS
    fix_identity(d["queries"]["identity"])


@case("query-pool-size", full=("consistency.counts", "consistency.identities"),
      alone=("consistency.counts", "consistency.identities"))
def query_pool_size(d, out):
    d["queries"]["n_pool"] += 1


@case("dropped-query-hidden", full=("consistency.counts", "evidence.fields"),
      alone=("consistency.counts",))
def dropped_query_hidden(d, out):
    d["evidence"]["dropped_queries"].remove("q0")


@case("query-order-seed", full=("consistency.counts", "replay.decision"),
      alone=("consistency.counts",))
def query_order_seed(d, out):
    d["queries"]["order"].update(kind="seeded_permutation", seed=3)


@case("query-order-seed-and-its-digest", full=("rule.text", "replay.decision"),
      alone=("rule.text",))
def query_order_seed_and_digest(d, out):
    order = d["queries"]["order"]
    order.update(kind="seeded_permutation", seed=3)
    order["indices_sha256"] = array_digest(
        np.random.default_rng(3).permutation(order["n_available"]).astype(np.int64))


@case("query-order-seed-its-digest-and-its-caveat", full=("replay.decision",))
def query_order_seed_digest_and_caveat(d, out):
    query_order_seed_and_digest(d, out)
    d["caveats"] = [CAVEAT_ORDER_SEED.format(seed=3) if c == CAVEAT_ORDER_GIVEN else c
                    for c in d["caveats"]]


@case("order-caveat-removed", full=("rule.text",), alone=("rule.text",))
def order_caveat_removed(d, out):
    d["caveats"] = [c for c in d["caveats"] if c != CAVEAT_ORDER_GIVEN]


@case("dropped-queries-caveat-removed", full=("rule.text",), alone=("rule.text",))
def dropped_queries_caveat_removed(d, out):
    d["caveats"] = [c for c in d["caveats"] if c != CAVEAT_DROPPED.format(k=3, n=80)]


# --- exact duplicates and norms: counts that only the live queries and vectors can confirm


@case("duplicate-count-cleared", fixture="duplicated", full=("rule.text", "binding.queries"),
      alone=("rule.text",))
def duplicate_count_cleared(d, out):
    d["queries"]["exact_duplicates"] = 0


@case("duplicate-count-and-its-caveat-cleared", fixture="duplicated", full=("binding.queries",))
def duplicate_count_and_caveat_cleared(d, out):
    d["queries"]["exact_duplicates"] = 0
    d["caveats"] = [c for c in d["caveats"] if "repeat an earlier query" not in c]


@case("duplicate-count-invented", full=("rule.text", "binding.queries"), alone=("rule.text",))
def duplicate_count_invented(d, out):
    d["queries"]["exact_duplicates"] = 5


@case("duplicate-count-beyond-the-queries-tested", fixture="duplicated",
      full=("consistency.counts",), alone=("consistency.counts",))
def duplicate_count_beyond(d, out):
    d["queries"]["exact_duplicates"] = d["queries"]["n_used"]


@case("norm-flag-set-to-hide-it", fixture="scaled",
      full=("consistency.counts", "rule.text", "binding.corpus"),
      alone=("consistency.counts", "rule.text"))
def norm_flag_set(d, out):
    d["norms"]["documents"]["unit_norm"] = True


@case("norms-rewritten-to-hide-it", fixture="scaled", full=("binding.corpus",))
def norms_rewritten(d, out):
    d["norms"]["documents"].update(unit_norm=True, norm_min=1.0, norm_max=1.0)
    d["caveats"] = [c for c in d["caveats"] if "unit L2 norm" not in c]


@case("norm-tolerance-widened-to-hide-it", fixture="scaled", full=("binding.corpus",))
def norm_tolerance_widened(d, out):
    d["norms"]["tolerance"] = 0.6
    d["norms"]["documents"]["unit_norm"] = True
    d["caveats"] = [c for c in d["caveats"] if "unit L2 norm" not in c]


@case("norm-extremes-edited-and-the-flag-kept", fixture="scaled",
      full=("consistency.counts", "binding.corpus"), alone=("consistency.counts",))
def norm_extremes_edited(d, out):
    d["norms"]["documents"]["norm_min"] = 1.0
    d["norms"]["documents"]["norm_max"] = 1.0


# --- the rule: its name, its class, its built-in flag and what it declares (RI-01, RI-02)

RULE_CHECKS = ("rule.text", "consistency.level", "consistency.decision")
RULE_CHECKS_AND_REPLAY = (*RULE_CHECKS, "replay.decision")


@case("rule-renamed", full=RULE_CHECKS_AND_REPLAY, alone=RULE_CHECKS)
def rule_renamed(d, out):
    d["rule"]["name"] = "fixed_sequence_betting_v2"


@case("rule-renamed-and-the-target-edited", full=RULE_CHECKS_AND_REPLAY, alone=RULE_CHECKS)
def rule_renamed_and_retargeted(d, out):
    d["target"] = 0.999
    d["rule"]["name"] = "fixed_sequence_betting_v2"


@case("rule-renamed-and-the-decision-forged", full=RULE_CHECKS_AND_REPLAY, alone=RULE_CHECKS)
def rule_renamed_and_decision_forged(d, out):
    n = len(d["result"]["p_values"])
    d["rule"]["name"] = "my_rule"
    d["error"] = d["level"] = 0.5
    d["result"]["p_values"] = [1e-9] * n
    d["result"]["certified_positions"] = list(range(n))


@case("rule-renamed-and-the-code-identity-edited", full=RULE_CHECKS_AND_REPLAY,
      alone=RULE_CHECKS)
def rule_renamed_and_code_edited(d, out):
    """The code identity is edited too: a text difference would then only be ``differs``, but a
    record that claims ``certified: true`` for a rule the library does not know fails anyway."""
    d["rule"]["name"] = "fixed_sequence_betting_v2"
    d["code"]["files"]["core/selection/rules.py"] = ZEROS
    d["code"]["digest"] = digest(d["code"]["files"])


@case("rule-class-renamed", full=RULE_CHECKS_AND_REPLAY, alone=(*RULE_CHECKS, "consistency.outcome"))
def rule_class_renamed(d, out):
    d["rule"]["class"] = "my.module.MyRule"


@case("rule-builtin-flag-cleared", full=RULE_CHECKS_AND_REPLAY,
      alone=(*RULE_CHECKS, "consistency.outcome"))
def rule_builtin_flag_cleared(d, out):
    d["rule"]["builtin"] = False


@case("rule-declared-certifying-cleared", full=("consistency.outcome", "rule.text"),
      alone=("consistency.outcome", "rule.text"))
def rule_declared_certifying_cleared(d, out):
    d["rule"]["declared_certifying"] = False


@case("rule-swapped-for-the-other-builtin-rule", full=("rule.text", "consistency.level"),
      alone=("rule.text", "consistency.level"))
def rule_swapped(d, out):
    d["rule"].update(name="bonferroni_betting",
                     **{"class": "optivision.core.selection.rules.BonferroniBetting"})


@case("custom-rule-certified-flag-set", fixture="custom",
      full=("consistency.outcome", *RULE_CHECKS_AND_REPLAY),
      alone=("consistency.outcome", *RULE_CHECKS))
def custom_rule_certified_flag_set(d, out):
    d["result"]["certified"] = True


@case("custom-rule-claims-to-be-built-in", fixture="custom",
      full=("consistency.outcome", *RULE_CHECKS_AND_REPLAY),
      alone=("consistency.outcome", *RULE_CHECKS))
def custom_rule_claims_builtin(d, out):
    d["rule"].update(builtin=True, certifying=True)
    d["result"]["certified"] = True
    d["statement"] = STATEMENT


@case("custom-rule-given-the-librarys-name-and-class", fixture="custom",
      full=("consistency.level", "consistency.decision", "replay.decision"),
      alone=("consistency.level", "consistency.decision"))
def custom_rule_given_the_librarys_identity(d, out):
    """Every text and flag now says "built in" and is consistent: the texts pass, and what fails
    is that the library's own rule, run on the stored p-values (all 1), would certify nothing."""
    fs = FixedSequenceBetting()
    d["rule"] = json.loads(json.dumps(rule_record(fs)))
    d["result"]["certified"] = True
    d["statement"] = STATEMENT
    d["assumptions"] = list(fs.assumptions())
    d["caveats"] = caveats_for(d)


@case("custom-rule-statement-replaced-by-the-librarys", fixture="custom",
      full=("rule.text",), alone=("rule.text",))
def custom_rule_statement_replaced(d, out):
    d["statement"] = STATEMENT


@case("custom-rule-assumptions-unlabelled", fixture="custom", full=("rule.text",),
      alone=("rule.text",))
def custom_rule_assumptions_unlabelled(d, out):
    d["assumptions"] = [a.removeprefix(DECLARED_PREFIX) for a in d["assumptions"]]


@case("custom-rule-caveat-removed", fixture="custom", full=("rule.text",), alone=("rule.text",))
def custom_rule_caveat_removed(d, out):
    d["caveats"] = [c for c in d["caveats"] if "not part of this library" not in c]


@case("query-budget", full=("consistency.counts", "replay.decision"),
      alone=("consistency.counts", "consistency.sample"))
def query_budget(d, out):
    d["queries"]["order"].update(budget=50, n_used=50)
    d["queries"]["n_used"] = 50


@case("query-budget-and-its-digest", full=("replay.decision", "evidence.sample"),
      alone=("consistency.sample",))
def query_budget_and_digest(d, out):
    d["queries"]["order"].update(budget=50, n_used=50,
                                 indices_sha256=array_digest(np.arange(50, dtype=np.int64)))
    d["queries"]["n_used"] = 50


# --- reference identity and construct (row 4)


@case("reference-identity", full=("binding.reference",))
def reference_identity(d, out):
    d["reference"]["identity"]["qrels_sha256"] = ZEROS


@case("construct", full=("evidence.fields", "binding.reference"),
      alone=("consistency.identities", "rule.text"))
def construct(d, out):
    d["reference"]["construct"] = "agreement@1"


@case("construct-to-hide-the-agreement-caveat", fixture="agreement",
      full=("evidence.fields", "binding.reference"))
def construct_hides_the_caveat(d, out):
    d["reference"]["construct"] = "labels"
    d["reference"]["identity"] = {"construct": "labels", "qrels_sha256": ZEROS}
    d["caveats"] = [c for c in d["caveats"] if "agreement with the uncompressed ranking" not in c]


@case("agreement-caveat-removed", fixture="agreement", full=("rule.text",), alone=("rule.text",))
def agreement_caveat_removed(d, out):
    d["caveats"] = [c for c in d["caveats"] if "agreement with the uncompressed ranking" not in c]


@case("all-caveats-removed", fixture="agreement", full=("rule.text",), alone=("rule.text",))
def all_caveats_removed(d, out):
    d["caveats"] = []


@case("metric", full=("evidence.fields",))
def metric(d, out):
    d["reference"]["metric"] = "recall@5"


# --- fitted state (row 5)


@case("fitted-state-digest", full=("binding.pipeline", "evidence.fields"))
def fitted_state_digest(d, out):
    d["candidate"]["fitted_state"][0] = ZEROS


@case("fitted-state-and-codes-digest", full=("binding.pipeline", "binding.codes"))
def fitted_state_and_codes(d, out):
    d["candidate"]["fitted_state"][0] = ZEROS
    d["candidate"]["codes_sha256"] = ZEROS


@case("fitted-state-stage-dropped", full=("consistency.candidate", "binding.pipeline"),
      alone=("consistency.candidate",))
def fitted_state_stage_dropped(d, out):
    d["candidate"]["fitted_state"].pop()


# --- the code identity (row 6): not detectable beyond `differs`


@case("code-identity", expect={"code.files": "differs"})
def code_identity_files(d, out):
    d["code"]["files"]["core/identity.py"] = ZEROS
    d["code"]["digest"] = digest(d["code"]["files"])


@case("code-identity-digest-only", full=("consistency.identities",),
      alone=("consistency.identities",))
def code_identity_digest_only(d, out):
    d["code"]["digest"] = ZEROS


# --- outcome, certification, selection, p-values (row 7)


@case("outcome-select-to-abstain", full=("consistency.outcome", "replay.decision"),
      alone=("consistency.outcome", "consistency.decision"))
def outcome_to_abstain(d, out):
    d["result"]["outcome"] = "abstain"


@case("outcome-select-to-fallback", full=("consistency.outcome", "replay.decision"),
      alone=("consistency.outcome",))
def outcome_to_fallback(d, out):
    d["result"]["outcome"] = "fallback"


@case("outcome-abstain-to-select", fixture="abstained",
      full=("consistency.outcome", "consistency.decision", "replay.decision"),
      alone=("consistency.outcome", "consistency.decision"))
def outcome_abstain_to_select(d, out):
    d["result"].update(outcome="select", selected=0, certified=True, certified_positions=[0])


@case("certified-flag-cleared", full=("consistency.outcome", "replay.decision"),
      alone=("consistency.outcome",))
def certified_flag_cleared(d, out):
    d["result"]["certified"] = False


@case("abstention-claimed-certified", fixture="abstained",
      full=("consistency.outcome", "replay.decision"), alone=("consistency.outcome",))
def abstention_claimed_certified(d, out):
    d["result"]["certified"] = True


@case("rule-certifying-flag", full=("consistency.outcome", "rule.text"),
      alone=("consistency.outcome", "rule.text"))
def rule_certifying_flag(d, out):
    d["rule"]["certifying"] = False


@case("selected-position-another-certified", full=("consistency.decision", "consistency.candidate"),
      alone=("consistency.decision", "consistency.candidate"))
def selected_other(d, out):
    d["result"]["selected"] = 4


@case("selected-position-not-certified", full=("consistency.outcome",),
      alone=("consistency.outcome",))
def selected_uncertified(d, out):
    d["result"]["selected"] = 6


@case("p-value-that-changes-the-decision", full=("consistency.decision", "replay.decision"),
      alone=("consistency.decision",))
def p_value_changes_decision(d, out):
    d["result"]["p_values"][3] = 0.5


@case("p-value-that-changes-no-decision", full=("replay.decision",))
def p_value_changes_nothing(d, out):
    d["result"]["p_values"][6] = 0.7


@case("p-value-in-its-ninth-digit", full=("replay.decision",))
def p_value_ninth_digit(d, out):
    d["result"]["p_values"][5] *= 1.0 + 1e-9


@case("certified-position-dropped", full=("consistency.decision",),
      alone=("consistency.decision",))
def certified_dropped(d, out):
    d["result"]["certified_positions"].remove(4)


@case("certified-position-added", full=("consistency.decision",),
      alone=("consistency.decision",))
def certified_added(d, out):
    d["result"]["certified_positions"].append(5)


# --- corpus identity (row 8)


@case("corpus-identity", full=("consistency.identities", "binding.corpus"),
      alone=("consistency.identities",))
def corpus_identity_vectors(d, out):
    d["corpus"]["vectors_sha256"] = ZEROS


@case("corpus-identity-consistently", full=("binding.corpus",))
def corpus_identity_consistently(d, out):
    d["corpus"]["vectors_sha256"] = ZEROS
    fix_identity(d["corpus"])


@case("corpus-vector-count", full=("consistency.candidate", "binding.corpus"),
      alone=("consistency.candidate",))
def corpus_vector_count(d, out):
    d["corpus"]["n_vectors"] += 1
    fix_identity(d["corpus"])


@case("corpus-document-count", full=("consistency.candidate", "binding.corpus"),
      alone=("consistency.candidate",))
def corpus_document_count(d, out):
    d["corpus"]["n_docs"] += 1
    fix_identity(d["corpus"])


# --- byte figures


@case("bytes-code-bytes", full=("consistency.candidate", "binding.bytes"),
      alone=("consistency.candidate",))
def bytes_code_bytes(d, out):
    d["candidate"]["bytes"]["code_bytes"] += 100


@case("bytes-selection-size-coherent", full=("consistency.candidate", "binding.bytes"),
      alone=("consistency.candidate",))
def bytes_selection_coherent(d, out):
    acc = d["candidate"]["bytes"]
    acc["code_bytes"] += 100
    acc["selection_bytes_per_doc"] = (acc["code_bytes"] + acc["codec_overhead_bytes"]) / acc["n_docs"]
    acc["total_bytes_per_doc"] = (acc["code_bytes"] + acc["codec_overhead_bytes"]
                                  + acc["fitted_state_bytes"] + acc["offsets_bytes"]) / acc["n_docs"]


@case("bytes-and-sizes-coherent", full=("evidence.fields", "binding.bytes"))
def bytes_and_sizes_coherent(d, out):
    acc = d["candidate"]["bytes"]
    acc["code_bytes"] += acc["n_docs"]  # one more byte per document: still the smallest certified
    acc["selection_bytes_per_doc"] = (acc["code_bytes"] + acc["codec_overhead_bytes"]) / acc["n_docs"]
    acc["total_bytes_per_doc"] = (acc["code_bytes"] + acc["codec_overhead_bytes"]
                                  + acc["fitted_state_bytes"] + acc["offsets_bytes"]) / acc["n_docs"]
    d["sizes"]["bytes_per_doc"][3] = acc["selection_bytes_per_doc"]


@case("bytes-fitted-state", full=("binding.bytes",))
def bytes_fitted_state(d, out):
    acc = d["candidate"]["bytes"]
    acc["fitted_state_bytes"] += 1000
    acc["total_bytes_per_doc"] = (acc["code_bytes"] + acc["codec_overhead_bytes"]
                                  + acc["fitted_state_bytes"] + acc["offsets_bytes"]) / acc["n_docs"]


@case("bytes-append-state", full=("binding.bytes",))
def bytes_append_state(d, out):
    d["candidate"]["bytes"]["append_state_bytes"] += 5


@case("bytes-offsets", full=("consistency.candidate", "binding.bytes"),
      alone=("consistency.candidate",))
def bytes_offsets(d, out):
    d["candidate"]["bytes"]["offsets_bytes"] += 8


@case("sizes-vectors-per-doc", full=("evidence.fields",))
def sizes_vectors(d, out):
    d["sizes"]["vectors_per_doc"][2] += 1.0


# --- statement, assumptions, rule configuration (the lead's extra rows)


@case("statement-text", full=("rule.text",), alone=("rule.text",))
def statement_text(d, out):
    d["statement"] = "The compressed index keeps 99% of retrieval quality on every query."


@case("assumption-dropped", full=("rule.text",), alone=("rule.text",))
def assumption_dropped(d, out):
    d["assumptions"] = [a for a in d["assumptions"] if not a.startswith("A5")]


@case("assumption-reworded", full=("rule.text",), alone=("rule.text",))
def assumption_reworded(d, out):
    d["assumptions"][0] = "A1: the calibration queries are whatever was convenient."


@case("rule-configuration", full=("rule.text",), alone=("rule.text",))
def rule_configuration(d, out):
    d["rule"]["config"]["bet_cap"] = 0.99


@case("statement-and-code-identity", expect={"rule.text": "differs", "code.files": "differs"})
def statement_and_code_identity(d, out):
    d["statement"] = "The compressed index keeps 99% of retrieval quality on every query."
    d["code"]["files"]["core/selection/rules.py"] = ZEROS
    d["code"]["digest"] = digest(d["code"]["files"])


# --- sample statistics and the evidence summary


@case("sample-retention-point", full=("consistency.sample",), alone=("consistency.sample",))
def sample_retention_point(d, out):
    d["sample"]["retention_point"] = 0.999


@case("sample-coherently", full=("evidence.sample",))
def sample_coherently(d, out):
    d["sample"]["mean"] = d["sample"]["baseline_mean"]
    d["sample"]["retention_point"] = 1.0


@case("evidence-digest", full=("evidence.digest",))
def evidence_digest(d, out):
    d["evidence"]["digest"] = ZEROS


@case("evidence-baseline-mean", full=("evidence.fields",))
def evidence_baseline_mean(d, out):
    d["evidence"]["baseline_mean"] = 0.9


# --- descriptive fields that nothing ties to anything (documented limits)


@case("created-time")
def created_time(d, out):
    d["created_utc"] = "2020-01-01T00:00:00Z"


@case("package-version")
def package_version(d, out):
    d["package"]["version"] = "9.9.9"


@case("encoder-declared")
def encoder_declared(d, out):
    d["encoder"] = {"model_id": "some/other-model", "revision": None, "dimension": 16,
                    "modality": "text", "normalized": None, "inference_dtype": None, "extra": {}}


@case("space-name")
def space_name(d, out):
    d["space"]["name"] = "a different name"


@case("schema-version", full=("schema",), alone=("schema",))
def schema_version(d, out):
    d["schema_version"] = 2


IDS = [c.id for c in CASES]


# ---------------------------------------------------- shared fixtures of this module


@pytest.fixture(scope="module")
def problem():
    return make_problem()


@pytest.fixture(scope="module")
def space():
    return small_space()


@pytest.fixture(scope="module")
def selected(problem, space):
    return issue(problem, space)


@pytest.fixture(scope="module")
def agreement(problem, space):
    return issue(problem, space, kind="agreement", target=0.9)


@pytest.fixture(scope="module")
def abstained(problem, space):
    return issue(problem, space, target=0.95, error=0.3)


@pytest.fixture(scope="module")
def custom(problem, space):
    """A certificate issued under a rule of someone else's that declares itself certifying."""
    out = issue(problem, space, rule=DeclaresCertifying(), target=0.8)
    assert out.decision.outcome.value == "select" and out.certificate.certified is False
    return out


@pytest.fixture(scope="module")
def duplicated(space):
    """Every one of the 80 queries counted twice (the copies under new ids), tested at random."""
    corpus, doubled, labels = with_copies(make_problem(n_unlabelled=0))
    out = issue((corpus, doubled, labels), space, order_seed=0, target=0.7)
    assert out.certificate.to_dict()["queries"]["exact_duplicates"] == 80
    assert out.decision.outcome.value == "select"
    return out


@pytest.fixture(scope="module")
def scaled(problem, space):
    """Documents of norm 1.5 instead of 1: the vectors are not unit-norm."""
    corpus, queries, qrels = problem
    docs = MultiVectorCorpus(corpus.vectors * np.float32(1.5), corpus.offsets, ids=corpus.ids)
    out = issue((docs, queries, qrels), space)
    assert out.certificate.to_dict()["norms"]["documents"]["unit_norm"] is False
    return out


def edited(out: Issued, c: Case, *, digest_too: bool, signed: bool = False) -> Certificate:
    base = sign_certificate(out.certificate, KEY, key_id="k") if signed else out.certificate
    d = base.to_dict()
    c.mutate(d, out)
    if digest_too:
        recompute(d)
    return Certificate.from_dict(d)


def failed_names(report) -> set[str]:
    return {c.name for c in report.failed()}


def test_the_table_covers_every_row_of_the_specification_and_more():
    assert len(CASES) == len(set(IDS)) >= 85
    needed = {"pipeline-parameter", "list-drop-one", "list-reorder", "list-add-one", "target",
              "error-level", "query-identity", "query-budget", "query-order-seed",
              "reference-identity", "construct", "fitted-state-digest", "code-identity",
              "outcome-select-to-abstain", "selected-position-another-certified",
              "p-value-that-changes-no-decision", "certified-position-dropped",
              "corpus-identity", "bytes-code-bytes", "statement-text", "assumption-dropped",
              "rule-renamed", "rule-renamed-and-the-target-edited",
              "rule-renamed-and-the-decision-forged", "rule-class-renamed",
              "rule-builtin-flag-cleared", "custom-rule-certified-flag-set",
              "duplicate-count-cleared", "norm-flag-set-to-hide-it"}
    assert needed <= set(IDS)


# ------------------------------------------------------------ (a) the careless editor


@pytest.mark.parametrize("c", CASES, ids=IDS)
def test_without_recomputing_the_digest_the_digest_check_fails(request, c):
    out = request.getfixturevalue(c.fixture)
    cert = edited(out, c, digest_too=False)
    assert cert != out.certificate and cert.digest == out.certificate.digest
    for report in (verify_certificate(cert), out.verify(cert)):
        assert report.status("digest") == "fail", c.id
        assert not report.ok


# ----------------------------------------------------- (b) the dishonest editor


@pytest.mark.parametrize("c", CASES, ids=IDS)
def test_with_the_digest_recomputed_everything_supplied_catches_what_the_table_says(request, c):
    out = request.getfixturevalue(c.fixture)
    report = out.verify(edited(out, c, digest_too=True))
    assert report.status("digest") == "pass", "the editor fixed the digest"
    failed = failed_names(report)
    if c.full:
        assert set(c.full) <= failed, f"{c.id}: expected {c.full}, failed {sorted(failed)}"
        assert not report.ok
    else:
        assert report.ok and not failed, f"{c.id}: a documented limit, yet {sorted(failed)} failed"
    for name, status in c.expect.items():
        assert report.status(name) == status, (c.id, name)


@pytest.mark.parametrize("c", CASES, ids=IDS)
def test_with_the_digest_recomputed_the_certificate_alone_catches_what_the_table_says(request, c):
    out = request.getfixturevalue(c.fixture)
    report = verify_certificate(edited(out, c, digest_too=True))
    assert report.status("digest") == "pass" and report.level == "integrity"
    failed = failed_names(report)
    if c.alone is None:
        assert report.ok and not failed, (
            f"{c.id} is documented as not detectable from the certificate alone, yet "
            f"{sorted(failed)} failed")
    else:
        assert set(c.alone) <= failed, f"{c.id}: expected {c.alone}, failed {sorted(failed)}"
        assert not report.ok


@pytest.mark.parametrize("c", [c for c in CASES if c.expect], ids=lambda c: c.id)
def test_a_different_code_identity_fails_only_under_strict_code(request, c):
    out = request.getfixturevalue(c.fixture)
    cert = edited(out, c, digest_too=True)
    assert out.verify(cert).ok
    strict = out.verify(cert, strict_code=True)
    assert not strict.ok and strict.status("code.files") == "fail"


# ------------------------------------------------------------------- (c) the HMAC


@pytest.mark.parametrize("c", CASES, ids=IDS)
def test_with_a_key_every_modification_fails_the_signature_even_after_the_digest_is_fixed(
        request, c):
    out = request.getfixturevalue(c.fixture)
    signed = sign_certificate(out.certificate, KEY, key_id="k")
    assert out.verify(signed, key=KEY).ok
    cert = edited(out, c, digest_too=True, signed=True)  # the editor cannot refresh the signature
    assert cert.signed and cert.digest != signed.digest
    report = out.verify(cert, key=KEY)
    assert report.status("digest") == "pass" and report.status("signature") == "fail", c.id
    assert not report.ok
    # removing the signature does not help when the verifier holds a key
    stripped = cert.to_dict()
    stripped["signature"] = None
    assert out.verify(Certificate.from_dict(stripped), key=KEY).status("signature") == "fail"
    # signing the edited record with the editor's own key does not satisfy the verifier's key
    resigned = sign_certificate(edited(out, c, digest_too=True), WRONG_KEY)
    assert out.verify(resigned, key=KEY).status("signature") == "fail"


@pytest.mark.parametrize("which", ["selected", "agreement", "abstained"])
def test_a_wrong_key_fails_an_unmodified_signed_certificate(request, which):
    out = request.getfixturevalue(which)
    signed = sign_certificate(out.certificate, KEY)
    assert out.verify(signed, key=KEY).ok
    report = out.verify(signed, key=WRONG_KEY)
    assert not report.ok and failed_names(report) == {"signature"}


def test_hmac_is_symmetric_so_whoever_holds_the_key_can_sign_a_modified_record(selected):
    """Pinned: the HMAC gives no non-repudiation. The key holder's edit verifies."""
    c = next(c for c in CASES if c.id == "statement-text")
    cert = edited(selected, c, digest_too=True)
    holder = sign_certificate(cert, KEY)
    report = selected.verify(holder, key=KEY)
    assert report.status("signature") == "pass"
    assert report.status("rule.text") == "fail" and not report.ok  # but the text still fails


def test_a_signature_copied_from_another_certificate_does_not_verify(selected, abstained):
    stolen = sign_certificate(abstained.certificate, KEY).to_dict()["signature"]
    d = selected.certificate.to_dict()
    d["signature"] = stolen
    assert selected.verify(Certificate.from_dict(d), key=KEY).status("signature") == "fail"


# ------------------------------------------------- every leaf is covered by the digest


def leaves(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, (*path, key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from leaves(value, (*path, i))
    else:
        yield path


def get(d, path):
    for key in path:
        d = d[key]
    return d


def put(d, path, value):
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = value


def perturbed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if value is None:
        return "x"
    return value + "x"


class TestEveryLeafOfTheRecord:
    @pytest.fixture(scope="class")
    def paths(self, selected):
        d = selected.certificate.to_dict()
        return [p for p in leaves(d) if p[0] != "signature"]

    def test_there_are_many_leaves(self, paths):
        assert len(paths) > 200

    def test_changing_any_one_leaf_fails_the_digest_check(self, selected, paths):
        wrong = []
        for path in paths:
            d = selected.certificate.to_dict()
            put(d, path, perturbed(get(d, path)))
            report = verify_certificate(Certificate.from_dict(d))
            if report.status("digest") != "fail":
                wrong.append(path)
        assert wrong == []

    def test_changing_any_one_leaf_and_the_digest_still_fails_a_signature(self, selected, paths):
        signed = sign_certificate(selected.certificate, KEY)
        wrong = []
        for path in paths:
            d = signed.to_dict()
            put(d, path, perturbed(get(d, path)))
            if path != ("digest",):
                recompute(d)
            report = verify_certificate(Certificate.from_dict(d), key=KEY)
            if report.status("signature") != "fail":
                wrong.append(path)
        assert wrong == []

    def test_the_signature_leaves_are_covered_by_the_key_not_by_the_digest(self, selected):
        signed = sign_certificate(selected.certificate, KEY, key_id="k")
        for leaf, value in (("value", "0" * 64), ("algorithm", "HMAC-SHA512")):
            d = signed.to_dict()
            d["signature"][leaf] = value
            report = verify_certificate(Certificate.from_dict(d), key=KEY)
            assert report.status("digest") == "pass" and report.status("signature") == "fail"


# ---------------------------------------------------- binding to the live objects


class TestLiveObjects:
    def test_a_different_corpus_fails_the_corpus_binding(self, selected, problem):
        corpus, _, _ = problem
        vectors = corpus.vectors.copy()
        vectors[17, 3] += 1e-3  # one coordinate of one vector
        other = MultiVectorCorpus(vectors, corpus.offsets, ids=corpus.ids)
        report = selected.verify(corpus=other)
        assert failed_names(report) == {"binding.corpus"}
        assert "vectors_sha256" in report.get("binding.corpus").detail

    def test_a_corpus_with_other_ids_or_order_fails_the_corpus_binding(self, selected, problem):
        corpus, _, _ = problem
        ids = list(corpus.ids)
        ids[0], ids[1] = ids[1], ids[0]
        report = selected.verify(corpus=MultiVectorCorpus(corpus.vectors, corpus.offsets, ids=ids))
        assert "binding.corpus" in failed_names(report)
        assert "ids_sha256" in report.get("binding.corpus").detail

    def test_a_corpus_in_another_dtype_fails_the_corpus_binding(self, selected, problem):
        corpus, _, _ = problem
        report = selected.verify(corpus=corpus.astype(np.float16))
        assert "binding.corpus" in failed_names(report)

    def test_different_queries_fail_the_query_binding(self, selected, problem):
        _, queries, _ = problem
        vectors = queries.vectors.copy()
        vectors[5, 0] = -vectors[5, 0] + 0.01
        other = MultiVectorCorpus(vectors, queries.offsets, ids=queries.ids)
        report = selected.verify(queries=other)
        assert failed_names(report) == {"binding.queries"}
        assert report.status("binding.evidence_queries") == "pass"  # the ids are unchanged

    def test_queries_with_other_ids_fail_the_query_binding_and_the_evidence_link(self, selected,
                                                                                 problem):
        _, queries, _ = problem
        ids = list(queries.ids)
        ids[10], ids[11] = ids[11], ids[10]
        other = MultiVectorCorpus(queries.vectors, queries.offsets, ids=ids)
        report = selected.verify(queries=other)
        assert failed_names(report) == {"binding.queries", "binding.evidence_queries"}

    def test_different_labels_fail_the_reference_binding(self, selected, problem):
        _, _, qrels = problem
        changed = {**qrels, "q10": ["d39"] if qrels["q10"] != ["d39"] else ["d0"]}
        report = selected.verify(reference=LabelReference(changed))
        assert failed_names(report) == {"binding.reference"}
        assert "labels digest or depth" in report.get("binding.reference").detail

    def test_the_label_digest_ignores_dict_order_and_grades(self, selected, problem):
        _, _, qrels = problem
        reordered = dict(reversed(list(qrels.items())))
        assert selected.verify(reference=LabelReference(reordered)).ok

    def test_the_other_construct_fails_the_reference_binding(self, selected):
        report = selected.verify(reference=RankingReference())
        assert failed_names(report) == {"binding.reference"}
        assert "construct" in report.get("binding.reference").detail

    def test_agreement_depth_is_part_of_the_reference_identity(self, agreement):
        report = agreement.verify(reference=RankingReference(depth=2))
        assert "binding.reference" in failed_names(report)

    def test_a_pipeline_fitted_on_other_data_fails_the_pipeline_binding(self, selected):
        other_corpus, _, _ = make_problem(seed=5)
        other, _ = compress_selected(other_corpus, selected.decision.selected)
        report = selected.verify(pipeline=other)
        assert failed_names(report) == {"binding.pipeline"}
        assert "fitted-state digests differ" in report.get("binding.pipeline").detail

    def test_an_unfitted_pipeline_fails_the_pipeline_binding(self, selected, space):
        unfitted = space.pipelines()[selected.decision.selected.position]
        report = selected.verify(pipeline=unfitted)
        assert failed_names(report) == {"binding.pipeline", "binding.bytes"}
        assert "not fitted" in report.get("binding.pipeline").detail

    def test_a_pipeline_with_another_specification_fails_the_pipeline_binding(self, selected,
                                                                              problem, space):
        other, _ = compress_selected(problem[0], space[4])
        report = selected.verify(pipeline=other)
        assert "binding.pipeline" in failed_names(report)
        assert "specification differs" in report.get("binding.pipeline").detail

    def test_tampered_codes_fail_the_codes_binding(self, selected):
        comp = selected.compressed
        codes = comp.codes.copy()
        codes[7, 1] ^= 0x01  # one bit of one code
        tampered = CompressedCorpus(codes, comp.offsets, comp.dim, comp.quantizer, comp.ids,
                                    stats=comp.stats, pipeline=comp.pipeline)
        report = selected.verify(compressed=tampered)
        assert failed_names(report) == {"binding.codes"}
        assert "codes differ" in report.get("binding.codes").detail

    def test_tampered_offsets_fail_the_codes_binding(self, selected):
        comp = selected.compressed
        offsets = comp.offsets.copy()
        offsets[5] -= 1  # a document boundary moves by one vector
        tampered = CompressedCorpus(comp.codes, offsets, comp.dim, comp.quantizer, comp.ids,
                                    stats=comp.stats, pipeline=comp.pipeline)
        assert "binding.codes" in failed_names(selected.verify(compressed=tampered))

    def test_compressed_corpus_with_other_ids_fails_the_codes_binding(self, selected):
        comp = selected.compressed
        ids = list(comp.ids)
        ids[0] = "renamed"
        tampered = CompressedCorpus(comp.codes, comp.offsets, comp.dim, comp.quantizer, ids,
                                    stats=comp.stats, pipeline=comp.pipeline)
        report = selected.verify(compressed=tampered)
        assert "binding.codes" in failed_names(report)
        assert "document ids" in report.get("binding.codes").detail

    def test_codes_of_another_pipeline_fail_the_codes_binding(self, selected, problem, space):
        _, other = compress_selected(problem[0], space[6])
        assert "binding.codes" in failed_names(selected.verify(compressed=other))

    def test_the_certified_objects_pass_every_binding_check(self, selected):
        report = selected.verify()
        assert report.ok
        assert [c.status for c in report.checks if c.name.startswith("binding.")] == ["pass"] * 7


# ----------------------------------------------------------------- the evidence


class TestSuppliedEvidence:
    def replaced(self, out: Issued, **change):
        return dataclasses.replace(out.evidence, **change)

    def test_one_value_changed_fails_the_digest_and_the_replay(self, selected):
        ev = selected.evidence
        candidates = ev.candidates.copy()
        candidates[3, 10] = 1.0 - candidates[3, 10] if candidates[3, 10] != 0.5 else 0.0
        report = selected.verify(evidence=self.replaced(selected, candidates=candidates))
        assert not report.ok
        assert {"evidence.digest", "replay.decision"} <= failed_names(report)
        assert "p_values differ" in report.get("replay.decision").detail

    def test_a_changed_baseline_value_fails_the_digest_and_the_replay(self, selected):
        baseline = selected.evidence.baseline.copy()
        baseline[0] = 0.25 if baseline[0] != 0.25 else 0.5
        report = selected.verify(evidence=self.replaced(selected, baseline=baseline))
        assert {"evidence.digest", "replay.decision", "evidence.fields"} <= failed_names(report)

    def test_a_changed_size_fails_the_digest_and_the_fields(self, selected):
        sizes = selected.evidence.bytes_per_doc.copy()
        sizes[6] += 1.0
        report = selected.verify(evidence=self.replaced(selected, bytes_per_doc=sizes))
        assert {"evidence.digest", "evidence.fields"} <= failed_names(report)

    def test_reordered_queries_fail_the_digest_and_change_the_p_values(self, selected):
        ev = selected.evidence
        order = np.arange(ev.n_queries)[::-1].copy()
        swapped = ev.subset(order)
        report = selected.verify(evidence=swapped)
        assert {"evidence.digest", "replay.decision"} <= failed_names(report)

    def test_a_dropped_query_fails_the_digest_the_fields_and_the_replay(self, selected):
        ev = selected.evidence
        fewer = ev.subset(np.arange(ev.n_queries - 1))
        report = selected.verify(evidence=fewer)
        assert {"evidence.digest", "evidence.fields", "replay.decision"} <= failed_names(report)

    def test_a_changed_construct_fails_the_digest_and_the_fields(self, selected):
        report = selected.verify(evidence=self.replaced(selected, construct="agreement@1"))
        assert {"evidence.digest", "evidence.fields"} <= failed_names(report)

    def test_evidence_from_another_candidate_space_fails(self, selected, problem, space):
        corpus, queries, qrels = problem
        other_space = type(space)("other", space.candidates[:-1])
        other = collect_evidence(corpus, queries, other_space, LabelReference(qrels))
        report = selected.verify(evidence=other)
        assert {"evidence.digest", "evidence.fields", "replay.decision"} <= failed_names(report)
        assert "select() could not be re-run" in report.get("replay.decision").detail

    def test_evidence_of_the_other_construct_fails(self, selected, agreement):
        report = selected.verify(evidence=agreement.evidence)
        assert not report.ok and "evidence.digest" in failed_names(report)

    def test_p_values_are_compared_at_a_relative_tolerance_of_1e_12(self, selected):
        d = selected.certificate.to_dict()
        d["result"]["p_values"][5] *= 1.0 + 1e-13  # rounding noise: accepted
        recompute(d)
        assert selected.verify(Certificate.from_dict(d)).status("replay.decision") == "pass"
        d["result"]["p_values"][5] *= 1.0 + 1e-10  # not noise: refused
        recompute(d)
        report = selected.verify(Certificate.from_dict(d))
        assert report.status("replay.decision") == "fail"
        assert "relative difference up to" in report.get("replay.decision").detail

    def test_everything_else_in_the_replay_is_compared_exactly(self, selected):
        d = selected.certificate.to_dict()
        d["level"] = np.nextafter(d["level"], 1.0)  # one unit in the last place
        recompute(d)
        assert "replay.decision" in failed_names(selected.verify(Certificate.from_dict(d)))


# --------------------------------------------------------- limits that are pinned


class TestDocumentedLimits:
    def test_forged_evidence_with_a_matching_certificate_verifies_at_every_level(
            self, problem, space, selected):
        """A party who supplies matching forged evidence and objects is not detected."""
        corpus, queries, _ = problem
        ev = selected.evidence
        forged_values = ev.candidates.copy()
        forged_values[5:] = ev.baseline  # pretend the two most aggressive candidates lose nothing
        forged = dataclasses.replace(ev, candidates=forged_values)
        decision = select(space, forged, target=0.8, error=0.2)
        assert decision.selected.position not in selected.decision.certified_positions
        pipeline, compressed = compress_selected(
            corpus, decision.selected, expected_fitted=forged.fitted[decision.selected.position])
        from optivision.core.artifact import create_certificate

        cert = create_certificate(decision, evidence=forged, space=space, corpus=corpus,
                                  queries=queries, reference=selected.reference,
                                  pipeline=pipeline, compressed=compressed,
                                  created="2026-10-06T12:00:00Z")
        full = {"evidence": forged, "corpus": corpus, "queries": queries,
                "reference": selected.reference, "pipeline": pipeline, "compressed": compressed}
        report = verify_certificate(cert, **full)
        assert report.ok and report.level == "binding"  # not detected: the documented limit
        # the honest evidence shows it, and so does the honest party's key
        assert "evidence.digest" in failed_names(verify_certificate(
            cert, **{**full, "evidence": ev}))
        assert verify_certificate(cert, key=KEY).status("signature") == "fail"
        assert verify_certificate(sign_certificate(selected.certificate, KEY), key=KEY).ok

    def test_a_certified_record_can_be_rewritten_into_an_uncertified_one_and_says_so(
            self, selected):
        """Pinned: claiming less is always possible without a key. The record then names a rule
        the library does not know, claims nothing, and the report says that nothing in it is
        certified: the three checks that need the rule are skipped, not passed."""
        d = selected.certificate.to_dict()
        d["rule"].update(name="my_rule", builtin=False, certifying=False)
        d["rule"]["class"] = "my.module.MyRule"
        d["result"]["certified"] = False
        d["statement"] = STATEMENT_UNEXAMINED
        d["assumptions"] = [DECLARED_PREFIX + a for a in d["assumptions"]]
        d["caveats"] = caveats_for(d)
        recompute(d)
        assert verify_certificate(Certificate.from_dict(d)).ok  # alone, and with everything:
        report = selected.verify(Certificate.from_dict(d))
        assert report.ok, report.summary()
        for name in ("consistency.level", "consistency.decision", "replay.decision"):
            assert report.status(name) == "skipped", name
        assert report.status("rule.text") == "pass"
        assert any("nothing in this record is certified" in n for n in report.notes)
        assert "note: this record names a rule" in report.summary()
        assert report.to_dict()["notes"] == list(report.notes)
        # ... and it is not certified: nothing in it can be read as the library's statement
        assert Certificate.from_dict(d).certified is False
        assert Certificate.from_dict(d).statement == STATEMENT_UNEXAMINED != STATEMENT

    def test_a_record_that_claims_certified_for_an_unknown_rule_is_failed_never_skipped(
            self, selected):
        for edit in ({"name": "someone_elses_rule"}, {"class": "my.module.MyRule"},
                     {"builtin": False}):
            d = selected.certificate.to_dict()
            d["rule"].update(edit)
            recompute(d)
            for report in (verify_certificate(Certificate.from_dict(d)),
                           selected.verify(Certificate.from_dict(d))):
                assert not report.ok
                for name in RULE_CHECKS:
                    check = report.get(name)
                    assert check.status == "fail", (edit, name, check.status)
                    assert "certified: true" in check.detail and "not one of this library's own" \
                        in check.detail
                assert any("claims certified: true" in n or "says certified: true" in n
                           for n in report.notes)
        # a skip is correct only where nothing is claimed: see the test above
        assert selected.certificate.certified is True

    def test_descriptive_fields_are_not_verified(self, selected):
        for case_id in ("created-time", "package-version", "encoder-declared", "space-name"):
            c = next(c for c in CASES if c.id == case_id)
            assert selected.verify(edited(selected, c, digest_too=True)).ok, case_id

    def test_a_forged_code_identity_is_only_reported_as_differs(self, selected):
        c = next(c for c in CASES if c.id == "code-identity")
        report = selected.verify(edited(selected, c, digest_too=True))
        assert report.ok and report.status("code.files") == "differs"
        assert not selected.verify(edited(selected, c, digest_too=True), strict_code=True).ok

    def test_a_statement_edit_is_decisive_only_when_the_code_identity_is_the_running_one(
            self, selected):
        text = next(c for c in CASES if c.id == "statement-text")
        both = next(c for c in CASES if c.id == "statement-and-code-identity")
        assert selected.verify(edited(selected, text, digest_too=True)).status("rule.text") == "fail"
        report = selected.verify(edited(selected, both, digest_too=True))
        assert report.status("rule.text") == "differs" and report.ok  # the limit
        assert selected.verify(edited(selected, both, digest_too=True),
                               strict_code=True).status("rule.text") == "fail"

    def test_a_p_value_that_changes_no_decision_needs_the_evidence_to_be_seen(self, selected):
        c = next(c for c in CASES if c.id == "p-value-that-changes-no-decision")
        cert = edited(selected, c, digest_too=True)
        assert verify_certificate(cert).ok
        assert verify_certificate(cert, evidence=selected.evidence).status(
            "replay.decision") == "fail"

    def test_the_target_needs_the_evidence_to_be_seen(self, selected):
        c = next(c for c in CASES if c.id == "target")
        cert = edited(selected, c, digest_too=True)
        assert verify_certificate(cert).ok
        assert verify_certificate(cert, evidence=selected.evidence).status(
            "replay.decision") == "fail"
