"""Corpora, models, Kaggle batches and thresholds of the easy-mode validation programme.

One place for every number the notebooks, the CPU analyses and the preregistration share
(FINAL_DESIGN section 8; reports/easy/PREREG_EASY.md part 1, PREREG_EASY_PART2.md,
PREREG_EASY_PART3.md). Hugging Face revisions, licences and counts were read on 2026-10-21;
``reports/easy/UNITS.md`` records where each one came from.

Pure data and arithmetic: no NumPy, no network, so the notebook generator and the tests
import it anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- constants

#: the targets every page and text unit is scored at (FD 8, "Units")
TARGETS = (0.90, 0.95, 0.97, 0.98)
#: units with fewer English-original queries are reported but not scored (FD 8)
MIN_QUERIES = 150
PAGE_METRIC = "ndcg@5"
TEXT_METRIC = "ndcg@10"
#: retention interval: paired bootstrap over queries, as the catalogue (s13: n_boot=1000, seed 0)
N_BOOT = 1000
BOOT_SEED = 0
#: the size study (arm S): corpus sizes, the full corpus last
S_SIZES = (250, 500, 1000, "full")
S_SEED = 0
#: the fit sample of a build (FD 5.7: ``_fit_sample`` caps rows at 200,000, seed 0)
FIT_SAMPLE_ROWS = 200_000
FIT_SEED = 0
#: growth check of V3: fit on a seeded half of the pages, then reuse on all of them
GROWTH_FRACTION = 0.5
GROWTH_SEED = 0
#: practice check (FD 5.5.1): bootstrap resamples and seed, minimum hard probes
PRACTICE_B = 2000
PRACTICE_SEED = 0
PROBE_SEED = 0
PROBE_MAX_PAGES = 600
PROBE_PER_PAGE = 3
HARD_MIN = 60
#: probe sources, scored and reported separately (FD 8: text layer first, the datasets' OCR text apart)
PROBE_SOURCES = ("text_layer", "ocr", "vlm")
#: the planted faults of V5a, as pipeline dicts (``random`` is a ranking, not a pipeline)
FAULTS: dict[str, dict | None] = {
    "random scores": None,
    "project(16) > binary": {"stages": [{"dim": 16, "method": "pca", "stage": "project"}, {"stage": "binary"}]},
    "adaptive_merge(0.6) > int4": {"stages": [
        {"center": "mean", "max_vectors": None, "min_vectors": 1, "n_importance_seeds": 0, "radius": 0.6,
         "ratio": None, "refine": 0, "representative": "mean", "seeding": "farthest", "stage": "adaptive_merge"},
        {"stage": "int4"}]},
}
FAULT_SEED = 0
#: text rule (FD 5.1.2 step 2): one flat margin up to the largest measured text corpus
TEXT_MARGIN = 0.01
TEXT_MAX_DOCS = 15_183
#: size margin of the frozen rule (FD 5.1.2), per rung class; s(N) = max(0, log(N/500)/log 5)
SIZE_COEFFICIENTS = {"codec": 0.005, "merge>codec": 0.0175, "binary": 0.0175, "merge>binary": 0.04}
ONE_BIT_MARGIN = 0.025
CLOSE_CALL_SLACK = 0.005
CODECS_ONLY_ABOVE = 20_000
ROUND_STEP = 0.0025


def s_of_n(n_pages: int) -> float:
    """The size term of the frozen rule: 0 at 500 pages, 1 at 2,500 (FD 5.1.2)."""
    return max(0.0, math.log(n_pages / 500) / math.log(5))


def s_signed(n_pages: int) -> float:
    """The same without the floor at 0 (arm S reports the side below 500 pages)."""
    return math.log(n_pages / 500) / math.log(5)


def round_up(x: float, step: float = ROUND_STEP) -> float:
    """``x`` rounded up to a multiple of ``step`` (the one rounding rule of FD 5.1.2 and arm S)."""
    return round(math.ceil(round(x / step, 9)) * step, 6)


def round_down(x: float, step: float = ROUND_STEP) -> float:
    """``x`` rounded down to a multiple of ``step`` (arm S's side below 500 pages: never looser than measured)."""
    return round(math.floor(round(x / step, 9)) * step, 6)


# --------------------------------------------------------------------------- corpora


@dataclass(frozen=True)
class Corpus:
    """One evaluation corpus, as encoded on Kaggle.

    ``pages``/``queries`` are the counts read on Hugging Face (English-original queries for the
    multilingual ViDoRe V3 sets); ``format`` tells ``encode_unit.py`` how to read it.
    ``text`` names the page text available for practice questions: ``"pdf"`` (the dataset ships
    the PDFs: text layer, plus its OCR markdown as a second source), ``"ocr"`` or ``None``.
    """

    key: str
    hf_id: str
    revision: str
    format: str
    pages: int
    queries: int
    licence: str
    roles: tuple[str, ...]
    doc_type: str
    text: str | None = None
    modality: str = "page"
    qrels_id: str | None = None
    qrels_revision: str | None = None
    subsample: int | None = None
    note: str = ""
    queries_all_languages: int | None = None
    split: str = "test"
    config: str | None = None

    @property
    def scored(self) -> bool:
        return self.queries >= MIN_QUERIES


CORPORA: dict[str, Corpus] = {c.key: c for c in (
    Corpus("v3_hr", "vidore/vidore_v3_hr", "0cdf0979f2c5a0fd3e335e6373b9da48a9fe3bc3", "vidore_v3",
           1110, 318, "cc-by-4.0 (card; all 14 documents cc-by-4.0, European Union reports)",
           ("V1", "V5", "V1u", "V4"), "reports", text="pdf", queries_all_languages=1908),
    Corpus("v3_computer_science", "vidore/vidore_v3_computer_science", "d5cc75883d92e294f0c0fc2662551c9708a06ebc",
           "vidore_v3", 1360, 215, "cc-by-4.0 (card; the 2 OpenStax textbooks are CC BY 4.0)",
           ("V1", "V5", "V1u", "V4"), "textbooks", text="pdf", queries_all_languages=1290),
    Corpus("v3_pharmaceuticals", "vidore/vidore_v3_pharmaceuticals", "3abd4aa8a9445fb5538a78a19ba50bd57bd22b5c",
           "vidore_v3", 2313, 364, "cc-by-4.0 (card; the 52 FDA documents are U.S. public domain)",
           ("V1", "V5", "V4"), "slides", text="pdf", queries_all_languages=2184,
           note="50 of the 52 documents are slide decks"),
    Corpus("v3_industrial", "vidore/vidore_v3_industrial", "e26c864724f5dd71a3d7d739272d95637764cee9",
           "vidore_v3", 5244, 283, "cc-by-4.0 (card; the 27 USAF technical orders are approved for public release, "
           "distribution unlimited)", ("V2", "S"), "technical manuals", text="pdf", queries_all_languages=1698),
    Corpus("rmm_techslides", "ibm-research/REAL-MM-RAG_TechSlides_BEIR", "614ad5cac2edd86756045f04075d335a3825a692",
           "beir_image", 1963, 1354, "cdla-permissive-2.0 (card)", ("V1-unchecked",), "slides", text=None,
           note="BEIR layout of ibm-research/REAL-MM-RAG_TechSlides; no page text, so every build on it is unchecked"),
    Corpus("beir_fiqa10k", "BeIR/fiqa", "979c07a7cb5ccc6ca009792241fa1250b98055dd", "beir_text", 10_000, 648,
           "cc-by-sa-4.0 (BEIR card; the card states no separate licence for the FiQA-2018 source)", ("V1t",),
           "financial question answering", modality="text", qrels_id="BeIR/fiqa-qrels",
           qrels_revision="252958f2d646e22cab6d0c72dd3f0d5de6d0655a", subsample=10_000,
           note="57,638 documents; the 1,706 judged-relevant ones of the 648 test queries kept, the rest drawn "
                "with seed 0 to 10,000"),
    Corpus("beir_arguana", "BeIR/arguana", "9bcf8fc0320c4e3860e25b8b85fa205fcdd7078f", "beir_text", 8674, 1406,
           "cc-by-sa-4.0 (BEIR card)", ("V1t",), "counter-arguments", modality="text", qrels_id="BeIR/arguana-qrels",
           qrels_revision="ae5468c6f1c198109a8af5f0d4dc58bd18b6fea7",
           note="1,298 of the 1,406 queries are themselves corpus documents with the same id; that document is "
                "excluded from its own query's ranking (BEIR ignore_identical_ids)"),
)}

#: corpora whose ArguAna-style self matches are excluded at scoring time
SELF_MATCH_EXCLUDED = {"beir_arguana"}
V1_CORPORA = ("v3_hr", "v3_computer_science", "v3_pharmaceuticals")
V1U_CORPORA = ("v3_hr", "v3_computer_science")
V2_CORPORA = ("v3_industrial",)
UNCHECKED_CORPORA = ("rmm_techslides",)
TEXT_CORPORA = ("beir_fiqa10k", "beir_arguana")

# --------------------------------------------------------------------------- models

F32 = {"dtype": "float32", "multi_gpu": True}
ST16 = {"dtype": "float16", "batch_size": 2}

#: the four design models (FD 8), with the setup the campaign rates were measured with
DESIGN: dict[str, dict] = {
    "vidore/colpali-v1.3": {**F32},
    "vidore/colqwen2-v1.0-hf": {**ST16},
    "vidore/colqwen2.5-v0.2": {**F32},
    "vidore/colSmol-500M": {**F32},
}
V2_MODELS = ("vidore/colSmol-500M", "vidore/colqwen2-v1.0-hf")
TEXT_MODELS: dict[str, dict] = {
    "colbert-ir/colbertv2.0": {"kind": "text"},
    "answerdotai/answerai-colbert-small-v1": {"kind": "text"},
    "lightonai/LateOn": {"kind": "text"},
    "lightonai/GTE-ModernColBERT-v1": {"kind": "text"},
}


@dataclass(frozen=True)
class HeldOut:
    """A V1u candidate: a late-interaction page retriever outside the registry (FD 8).

    ``downloads`` is the Hugging Face 30-day count read on ``read_on``; the notebook tries
    the candidates in this order and keeps the first two that load, fit on two T4s and pass
    ``check_retriever`` (12 pairs, at most 2 misses).
    """

    model: str
    revision: str
    downloads: int
    loader: str
    licence: str
    spec: dict = field(default_factory=dict)
    note: str = ""
    read_on: str = "2026-10-21"


HELDOUT_QUERY = ("https://huggingface.co/api/models?pipeline_tag=visual-document-retrieval&sort=downloads"
                 "&direction=-1&limit=100 and ?filter=colpali&sort=downloads&direction=-1&limit=100, "
                 "read 2026-10-21; registry = optivision.adapters.MODELS (65 ids) at this commit")
HELDOUT_CANDIDATES: tuple[HeldOut, ...] = (
    HeldOut("OpenSearch-AI/Ops-Colqwen3-4B", "e9046ac9487626e3963f20e522dab882ac6d9ba6", 2960,
            "remote-code:ops-colqwen3", "apache-2.0",
            {"dtype": "float16", "dims": 128, "batch_size": 2, "trust_remote_code": True,
             "pip": ["qwen-vl-utils>=0.0.14"]},
            "Qwen3-VL-4B multi-vector model with a prefix projection head; the card's own loader "
            "(AutoModel, trust_remote_code) at 128 dimensions per vector, the width most registry page "
            "models use (the card default is 2,560, about 6.5 MB per page in float16)"),
    HeldOut("tencent/EVIE-Preview-4.5B", "15a12c13a321a891d238d0ad441e680ed015546a", 791, "sentence-transformers",
            "apache-2.0", {"dtype": "float16", "batch_size": 2},
            "the preview checkpoint before the registry's tencent/EVIE-4.5B (different weights, same family); "
            "128 dimensions"),
    HeldOut("webAI-Official/webAI-ColVec1-9b", "", 349, "remote-code:colvec", "other (webAI licence)",
            {"dtype": "float16", "batch_size": 1, "multi_gpu": True, "trust_remote_code": True, "revision": "latest"},
            "fallback only; the 9B sibling of the registry's ColVec1 4B; 640 dimensions"),
)
#: what was excluded from the ranked list above, and why (UNITS.md has the full list)
HELDOUT_EXCLUDED = {
    "jinaai/jina-embeddings-v4-vllm-retrieval": "a vLLM repackaging of the registry's jina-embeddings-v4",
    "nomic-ai/nomic-embed-multimodal-3b": "single-vector (dense) model",
    "Sahil-Kabir/colqwen2.5-v0.2-hf": "a format conversion of the registry's vidore/colqwen2.5-v0.2",
    "MrLight/dse-qwen2-2b-mrl-v1": "single-vector (dense) model",
    "Tevatron/OmniEmbed-v0.1-multivent": "single-vector video/omni embedding",
    "nvidia/Nemotron-3.5-Embed-1B-BF16": "single-vector (mean-pooled) model",
    "nvidia/Nemotron-3.5-Embed-8B-BF16": "single-vector (mean-pooled) model",
    "tsystems/colqwen2.5-3b-base": "an untrained base checkpoint",
    "thkim93/colpali-hf-1layer": "a one-layer test checkpoint",
    "michaelfeil/colpali-v12-random-testing": "random test weights",
    "Qdrant/colmodernvbert": "an ONNX conversion of the registry's ModernVBERT/colmodernvbert",
    "Hcompany/NeoMME-260M-Retriever": "the same model as the registry's NeoMME-260M-Retriever-ST-late, other packaging",
    "Hcompany/NeoMME-800M-Retriever": "the same model as the registry's NeoMME-800M-Retriever-ST-late, other packaging",
    "DataScience-UIBK/Argus-Colqwen3.5-*": "page vectors depend on the query (query-conditioned routing), so "
                                           "pages cannot be stored as fixed vectors",
}

# --------------------------------------------------------------------------- rates and batches

#: encode-step seconds per page including model load, from campaign run.json (FD 8 table)
RATES: dict[str, tuple[float, float]] = {
    "vidore/colpali-v1.3": (2.28, 2.33),
    "vidore/colSmol-500M": (1.44, 2.22),
    "vidore/colqwen2-v1.0-hf": (0.69, 0.82),
    "vidore/colqwen2.5-v0.2": (3.08, 3.28),  # colqwen2.5-v0.1 as proxy (FD 8)
}
HELDOUT_RATE = (2.3, 3.6)  # assumed (FD 8)
#: text encode seconds per document incl. load: SciFact/NFCorpus/SciDocs steps of the four text models
TEXT_RATE = (0.008, 0.033)
#: Tier B practice questions: minutes per 1,000 pages on a T4 (FD 5.3.2, a research estimate)
VLM_MIN_PER_1000 = (35.0, 70.0)
#: fixed costs of a session: clone, installs, dataset downloads, zip (hours)
SESSION_OVERHEAD = (0.25, 0.5)
#: per (model, corpus) job: probe and query encoding, shard writes (hours)
JOB_OVERHEAD = (0.02, 0.05)
#: V6: one representative model per registry loader that loads on Kaggle, plus the two faults
V6_LOADERS: dict[str, dict] = {
    "colpali-engine:colpali": {"model": "vidore/colpali-v1.3", **F32},
    "colpali-engine:colqwen2": {"model": "vidore/colqwen2-v0.1", **F32},
    "colpali-engine:colqwen2.5": {"model": "vidore/colqwen2.5-v0.2", **F32},
    "colpali-engine:colsmol": {"model": "vidore/colSmol-256M", **F32},
    "colpali-engine:colsmol-500m": {"model": "vidore/colSmol-500M", **F32},
    "colpali-engine:colgemma3": {"model": "Cognitive-Lab/ColNetraEmbed", **F32},
    "colpali-engine:colqwen2.5-omni": {"model": "vidore/colqwen-omni-v0.1", **F32},
    "colpali-engine:colqwen3.5": {"model": "vultr/VultronRetrieverFlash-Qwen3.5-0.8B", **F32},
    "sentence-transformers (pages)": {"model": "vidore/colqwen2-v1.0-hf", **ST16},
    "sentence-transformers (text)": {"model": "answerdotai/answerai-colbert-small-v1", "kind": "text"},
    "remote-code:tomoro": {"model": "TomoroAI/tomoro-colqwen3-embed-4b", "dtype": "float16", "multi_gpu": True,
                           "trust_remote_code": True},
    "remote-code:colvec": {"model": "webAI-Official/webAI-ColVec1-4b", **F32, "trust_remote_code": True,
                           "revision": "latest"},
    "nemotron": {"model": "nvidia/nemotron-colembed-vl-4b-v2", **F32},
}
#: the models the two V6 faults (re-initialised norm, query/document dtype mismatch) are planted in
V6_FAULT_MODELS = ("vidore/colqwen2.5-v0.2", "vidore/colqwen2-v1.0-hf")
V6_HOURS = (0.6, 1.0)
#: Tier B generator (FD 5.3.2: Qwen2-VL-2B or SmolVLM class, after a licence check)
VLM_MODEL = {"model": "Qwen/Qwen2-VL-2B-Instruct", "licence": "apache-2.0", "dtype": "float16"}
VLM_CORPORA = ("v3_hr", "v3_computer_science", "v3_pharmaceuticals")


@dataclass(frozen=True)
class Job:
    """One model over one or more corpora inside a notebook (``encode_unit.py`` runs it)."""

    model: str
    corpora: tuple[str, ...]
    spec: dict = field(default_factory=dict)
    kind: str = "image"
    keep_float32: tuple[str, ...] = ()
    role: str = "encode"  # encode | heldout | wiring | vlm

    def as_dict(self) -> dict:
        return {"model": self.model, "corpora": list(self.corpora), "kind": self.kind, "role": self.role,
                "keep_float32": list(self.keep_float32), **self.spec}


@dataclass(frozen=True)
class Batch:
    """One Kaggle notebook: what it encodes, what it needs before it may run, and its time."""

    name: str
    title: str
    prereg_batch: str
    jobs: tuple[Job, ...]
    parts: tuple[int, ...]
    packages: tuple[str, ...] = ()
    needs_files: tuple[str, ...] = ()
    needs_frozen: tuple[str, ...] = ()
    needs_text: str = ""
    remote_code: bool = False
    extra_hours: tuple[float, float] = (0.0, 0.0)

    @property
    def notebook(self) -> str:
        return f"batch_{self.name}.ipynb"


ST = "sentence-transformers[image]>=6.0.0"
#: files every validation notebook needs in the cloned branch (besides the frozen ones)
COMMON_FILES = ("scripts/easy_validation/encode_unit.py", "scripts/easy_validation/units.py",
                "scripts/encode_vectors.py", "scripts/encode_text_vectors.py")
PROBE_FILES = ("src/optivision/easy/_probes.py", "src/optivision/easy/_docs.py")
HELDOUT_SELECTED = "reports/easy/e1b_easy_b1_heldout_v6/heldout_selected.json"


def _design_jobs(corpora: tuple[str, ...], keep32: tuple[str, ...] = ()) -> tuple[Job, ...]:
    return tuple(Job(m, corpora, dict(spec), keep_float32=keep32) for m, spec in DESIGN.items())


BATCHES: tuple[Batch, ...] = (
    Batch("e1a_easy_b1_design", "B1: ViDoRe V3 hr + computer_science, the four design models", "B1",
          _design_jobs(("v3_hr", "v3_computer_science"), keep32=("v3_hr",)), parts=(1, 2),
          needs_frozen=PROBE_FILES,
          needs_text="prereg parts 1 and 2 frozen and pushed (reports/easy/PREREG_EASY.frozen.json)"),
    Batch("e1b_easy_b1_heldout_v6", "B1: held-out encoders (V1u) on hr, and the wiring test (V6)", "B1",
          (Job("heldout", ("v3_hr",), {}, role="heldout"),
           Job("wiring", ("v3_hr",), {}, role="wiring")), parts=(1, 2), packages=(ST, "qwen-vl-utils>=0.0.14"),
          needs_frozen=PROBE_FILES + ("src/optivision/easy/_encode.py",), remote_code=True,
          needs_text="prereg parts 1 and 2 frozen and pushed; optivision.easy._encode (wiring_check) on the branch",
          extra_hours=V6_HOURS),
    Batch("e2a_easy_b2_pharma", "B2: ViDoRe V3 pharmaceuticals, the four design models", "B2",
          _design_jobs(("v3_pharmaceuticals",)), parts=(1, 2), needs_frozen=PROBE_FILES,
          needs_text="prereg parts 1 and 2 frozen and pushed"),
    Batch("e2b_easy_b2_techslides", "B2: REAL-MM-RAG TechSlides (unchecked), the four design models", "B2",
          _design_jobs(("rmm_techslides",)), parts=(1, 2),
          needs_text="prereg parts 1 and 2 frozen and pushed"),
    Batch("e3a_easy_b3_industrial", "B3: ViDoRe V3 industrial (5,244 pages), colSmol-500M and colqwen2-v1.0-hf (V2)",
          "B3", tuple(Job(m, ("v3_industrial",), dict(DESIGN[m])) for m in V2_MODELS), parts=(1, 2),
          needs_frozen=PROBE_FILES, needs_text="prereg parts 1 and 2 frozen and pushed"),
    Batch("e3b_easy_b3_heldout_text", "B3: held-out encoders on computer_science (V1u), text models on FiQA + ArguAna "
          "(V1t)", "B3", (Job("heldout", ("v3_computer_science",), {"selected_from": HELDOUT_SELECTED}, role="heldout"),
                          *(Job(m, TEXT_CORPORA, dict(s), kind="text") for m, s in TEXT_MODELS.items())),
          parts=(1, 2), packages=(ST, "einops", "qwen-vl-utils>=0.0.14"), needs_frozen=PROBE_FILES,
          needs_files=(HELDOUT_SELECTED,), remote_code=True,
          needs_text="prereg parts 1 and 2 frozen; e1b ingested and pushed (its heldout_selected.json names the two "
                     "held-out encoders)"),
)


def batch(name: str) -> Batch:
    for b in all_batches():
        if b.name == name:
            return b
    raise KeyError(name)


def all_batches() -> tuple[Batch, ...]:
    from . import dt  # the DT campaign and the Tier B batch are defined beside their analyses

    return BATCHES + dt.DT_BATCHES + (dt.VLM_BATCH,)


def corpus(key: str) -> Corpus:
    if key in CORPORA:
        return CORPORA[key]
    from . import dt

    return dt.DT_CORPORA[key]


def job_hours(job: Job) -> tuple[float, float]:
    """GPU hours of one job: pages times the model's rate (FD 8), plus per-corpus overhead."""
    if job.role == "wiring":
        return V6_HOURS
    if job.role == "vlm":
        pages = sum(min(PROBE_MAX_PAGES, corpus(c).pages) for c in job.corpora)
        lo, hi = VLM_MIN_PER_1000
        return pages * lo / 60 / 1000, pages * hi / 60 / 1000
    pages = sum(corpus(c).pages for c in job.corpora)
    if job.kind == "text":
        lo, hi = TEXT_RATE
    elif job.role == "heldout":
        lo, hi = (2 * HELDOUT_RATE[0], 2 * HELDOUT_RATE[1])  # two held-out encoders
    else:
        lo, hi = RATES.get(job.model, HELDOUT_RATE)
    n = len(job.corpora) * (2 if job.role == "heldout" else 1)
    return pages * lo / 3600 + n * JOB_OVERHEAD[0], pages * hi / 3600 + n * JOB_OVERHEAD[1]


def batch_hours(b: Batch) -> tuple[float, float]:
    """Estimated session hours (low, high) of a batch: jobs + session overhead + extras."""
    lo = sum(job_hours(j)[0] for j in b.jobs if j.role != "wiring") + SESSION_OVERHEAD[0]
    hi = sum(job_hours(j)[1] for j in b.jobs if j.role != "wiring") + SESSION_OVERHEAD[1]
    return round(lo + b.extra_hours[0], 2), round(hi + b.extra_hours[1], 2)


#: a notebook stops encoding at this many hours and zips what it has (Kaggle allows 12)
DEADLINE_HOURS = 11.0

# --------------------------------------------------------------------------- predictions (FD 8)

#: every threshold of FD 8, as the analyses use them (also printed into PREREG_EASY.md part 1)
THRESHOLDS: dict[str, dict] = {
    "V1a": {"rung": "int8(per_vector)", "point_min": 0.97, "share": 1.0},
    "V1b": {"target": 0.97, "point_min": 0.97, "share": 0.90, "all_point_min": 0.94},
    "V1c": {"target": 0.95, "point_min": 0.95, "share": 0.90, "low_min": 0.93, "low_share": 0.90},
    "V1d": {"targets": (0.98, 0.90), "share": 0.90},
    "V1-unchecked": {"as": ("V1b", "V1c")},
    "V1-arms": {"severe_below": 0.02},
    "V1t": {"targets": (0.97, 0.95), "share": 0.90},
    "V1u": {"share": 1.0},
    "V2": {"targets": (0.95, 0.97), "n_pages": 5244, "int8_point_min": 0.97},
    "V3": {"abs_delta_max": 0.005, "share": 0.95},
    "V4": {"share": 0.70},
    "V5a": {"real_below": 0.80, "fire_share": 0.95},
    "V5b": {"fire_share_max": 0.10},
    "V5c": {"miss_below": 0.03, "fire_share": 0.50, "min_pairs": 6},
    "V5d": {"median_ratio_min": 0.80},
}
