"""The DT campaign (document-type presets, FINAL_DESIGN 5.8) and the Tier B batch (FD 5.3.2).

Preregistered in ``reports/easy/PREREG_EASY_PART3.md`` (part 3); frozen with it.

A type preset ships only if the type has 2 or more datasets, measured with 4 or more models, and
the low end of every unit is at or above its target. ViDoRe V3 hr / computer_science /
pharmaceuticals / industrial and TechSlides are the B1-B3 corpora: their vectors are reused here,
not encoded again. The DT notebooks add the V1-format sets below for the four design models.
Datasets with fewer than 150 queries are encoded and reported, but they do not count toward the
2-datasets-per-type rule (``REPORTED_ONLY``).
"""

from __future__ import annotations

from easy_validation import units
from easy_validation.units import PROBE_FILES, Batch, Corpus, Job

_V1 = "vidore_v1"

#: revisions read from the Hugging Face API on 2026-10-22; the card licences are quoted
DT_CORPORA: dict[str, Corpus] = {c.key: c for c in (
    Corpus("v1_arxivqa", "vidore/arxivqa_test_subsampled", "b8a106812c8682bab08935cf5d1b4566c82562de", _V1, 500, 500,
           "cc-by-sa-4.0 (card)", ("DT",), "figures", note="rows are pages; one query per row"),
    Corpus("v1_tabfquad", "vidore/tabfquad_test_subsampled", "16c8e633612fbda7400bfcbbc31d61a7534f580f", _V1, 280, 280,
           "mit (card)", ("DT",), "tables", note="French tables"),
    Corpus("v1_tatdqa", "vidore/tatdqa_test", "b46fea43695e14697510104a3331d9e88683a416", _V1, 1663, 1663,
           "cc-by-4.0 (card)", ("DT",), "tables",
           note="financial-report tables; the query count is the row count, distinct queries are read at ingest"),
    Corpus("v1_syn_government", "vidore/syntheticDocQA_government_reports_test",
           "91cf66572d89c9cccac0661de227acaf04b44f64", _V1, 1000, 100, "mit (card)", ("DT",), "reports",
           note="100 queries: reported, not scored"),
    Corpus("v1_syn_ai", "vidore/syntheticDocQA_artificial_intelligence_test",
           "5694fec64c57f7380f918435f9234b96d817b82a", _V1, 1000, 100, "mit (card)", ("DT",), "academic",
           note="100 queries: reported, not scored"),
)}

#: below ``units.MIN_QUERIES`` queries: encoded and reported, never counted toward a type
REPORTED_ONLY = {k for k, c in DT_CORPORA.items() if c.queries < units.MIN_QUERIES}

#: type -> datasets (registry keys of ``units.CORPORA`` or ``DT_CORPORA``). Types with fewer than two
#: datasets here (scans: DocVQA; infographics: InfoVQA) stay under the worst-dataset rule.
DT_TYPES: dict[str, tuple[str, ...]] = {
    "slides": ("v3_pharmaceuticals", "rmm_techslides"),
    "reports": ("v3_hr", "v1_syn_government"),
    "academic": ("v3_computer_science", "v1_arxivqa", "v1_syn_ai"),
    "tables": ("v1_tabfquad", "v1_tatdqa"),
}
DT_MODELS = tuple(units.DESIGN)
#: the share of the 12-20 h that V9, the DT campaign and the OCR/VLM runs share (FD 14): DT's part
GPU_BUDGET_HOURS = 14.0

_NEEDS = "prereg parts 1, 2 and 3 frozen and pushed (reports/easy/PREREG_EASY.frozen.json)"


def _jobs(corpora: tuple[str, ...]) -> tuple[Job, ...]:
    return tuple(Job(m, corpora, dict(spec)) for m, spec in units.DESIGN.items())


DT_BATCHES: tuple[Batch, ...] = (
    Batch("e4a_easy_dt_tables_figures", "DT: ArxivQA, TabFQuAD, TAT-DQA, the four design models", "DT",
          _jobs(("v1_arxivqa", "v1_tabfquad", "v1_tatdqa")), parts=(1, 2, 3), needs_text=_NEEDS),
    Batch("e4b_easy_dt_synthetic", "DT: synthetic government and AI reports (reported, 100 queries each)", "DT",
          _jobs(("v1_syn_government", "v1_syn_ai")), parts=(1, 2, 3), needs_text=_NEEDS),
)

VLM_BATCH = Batch(
    "e5_easy_tierb_vlm", "Tier B practice questions: local VLM questions on hr, computer_science, pharmaceuticals",
    "VLM", (Job(units.VLM_MODEL["model"], units.VLM_CORPORA, {"dtype": units.VLM_MODEL["dtype"]}, role="vlm"),),
    parts=(1, 2, 3), packages=(units.ST, "qwen-vl-utils>=0.0.14"), needs_frozen=PROBE_FILES + (
        "src/optivision/easy/_probes_vlm.py",),
    needs_text=_NEEDS + "; B1 and B2 ingested and pushed (the VLM questions are encoded by their models)",
    extra_hours=(0.5, 1.5))


def budget() -> tuple[float, float]:
    """GPU hours (low, high) of the DT notebooks."""
    lo = sum(units.batch_hours(b)[0] for b in DT_BATCHES)
    hi = sum(units.batch_hours(b)[1] for b in DT_BATCHES)
    return round(lo, 2), round(hi, 2)
