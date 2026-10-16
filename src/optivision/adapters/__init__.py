"""Encoders in, :class:`~optivision.representation.MultiVectorCorpus` out.

The compression layer never talks to a model. It takes vectors. This module is
the thin, replaceable bridge from a model to those vectors, and it is explicit
about how much of each bridge has actually been exercised.

Every known model carries one of three statuses:

``measured``
    Retrieval quality with and without compression has been measured by this
    repository on real encoder output (the numbers live in ``reports/``).
``untested``
    The loading path is known to work: it is the one a measured sibling model uses, or
    this checkpoint passed a :func:`check_retriever` run here (the evidence of the entry
    says which). Compression of this checkpoint's vectors has not been measured here.
``unverified``
    The adapter is written against a library's documented API, but that library
    has never been installed and run here. Treat it as a starting point.

The most general path needs no adapter at all: hand
:meth:`MultiVectorCorpus.from_arrays` the ``[n_i, d]`` arrays any model produced.

Three ways in
-------------
``ArrayAdapter`` takes arrays some model already produced and checks them;
``CallableAdapter`` wraps two functions that produce such arrays; ``load_adapter`` builds
the adapter of a model family this repository has a loader for. A custom adapter can reuse
the checked conversion, :func:`optivision.adapters.arrays.corpus_from_arrays`.

What an adapter is, and is not
------------------------------
An adapter encodes, returns a :class:`~optivision.representation.MultiVectorCorpus`, and
can describe the model behind the vectors (:meth:`EncoderAdapter.identity`,
:attr:`EncoderAdapter.dimension`). Adapters produce vectors and descriptive identity
only. An adapter never chooses or alters compression settings by model, never moves data
between devices or changes its dtype unless the caller said so, and nothing in the core
branches on the identity it reports. That an adapter exists for a model says nothing about
how well any pipeline compresses that model's vectors: that has to be measured on those
vectors, in the basis they are in.

Where layout lives
------------------
Model-specific layout (patch grids, protected instruction tokens, page images) is not part
of the adapter interface. ``PageEncoderAdapter`` carries it into the corpus it returns: the
aligned arrays ``positions`` and ``protected``, the per-document ``metadata`` and ``images``.
The core reads those arrays and never sees a ``PatchGrid`` or a page type.
"""

from __future__ import annotations

import abc
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.vectors import MODALITIES, EncoderIdentity
from ..representation import MultiVectorCorpus

MEASURED = "measured"
UNTESTED = "untested"
UNVERIFIED = "unverified"


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    family: str
    modality: str  # "image" (page screenshots) or "text"
    loader: str  # how the adapter obtains vectors
    dimension: int | None  # width of one output vector; None when no file of the model states it
    status: str
    evidence: str
    license: str | None = None  # as the model repository declares it; check it before use
    revision: str | None = None  # the commit the facts of this entry were read from or measured at
    notes: str = ""  # what a caller has to know to load it (extra packages, arguments)
    #: keyword arguments load_adapter passes to the loader unless the caller gives them, for a
    #: setting the model card states but the repository files do not (as (name, value) pairs)
    load_options: tuple[tuple[str, Any], ...] = ()


_MODELS: list[ModelInfo] = [
    ModelInfo("vidore/colpali-v1.3-merged", "colpali", "image", "colpali-engine:colpali", 128,
              MEASURED, "E2/E3: reports/colpali_*; replayed 2026-09-27 (docs/AUDIT-2026-09-27.md)"),
    ModelInfo("vidore/colSmol-256M", "colsmol", "image", "colpali-engine:colsmol", 128,
              MEASURED, "E1/E4: reports/colsmol*; replayed 2026-09-27"),
    ModelInfo("vidore/colSmol-500M", "colsmol", "image", "colpali-engine:colsmol-500m", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~890 vectors/page), float32 over two Kaggle T4s,"
              " at the revision below, 2026-10-14; Kaggle batch 02_colqwen_small; "
              "reports/universal/campaign/vidore--colSmol-500M",
              revision="0aaa9726104ce485884c7b8faa8a58a72d5fdbe7"),
    ModelInfo("vidore/colqwen2-v1.0-merged", "colqwen", "image", "colpali-engine:colqwen2", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each), encoded 2026-10-01 on a Kaggle T4 in "
              "float32; reports/universal/*/colqwen2-v1.0-merged_* (docs/UNIVERSAL.md, R11)"),
    ModelInfo("vidore/colqwen2-v1.0", "colqwen", "image", "colpali-engine:colqwen2", 128,
              MEASURED, "adapter-only repo, merged explicitly on load; gives the same vectors as the "
              "benchmarked vidore/colqwen2-v1.0-merged (median cosine 0.99999 on 6 pages, R11)"),
    ModelInfo("vidore/colqwen2.5-v0.2", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each), adapter merged explicitly over "
              "vidore/colqwen2.5-base, float32 split over two T4s, 2026-10-01; "
              "reports/universal/*/colqwen2.5-v0.2_* (docs/UNIVERSAL.md, R11)"),
    ModelInfo("nvidia/nemotron-colembed-vl-4b-v2", "nemotron-colembed", "image", "nemotron", 2560,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~750 vectors/page), float32 over two "
              "Kaggle T4s, pinned revision 0ed152d9, 2026-10-02; reports/universal/wide (docs/UNIVERSAL.md, R12); "
              "CC-BY-NC-4.0"),
    ModelInfo("nvidia/nemotron-colembed-vl-8b-v2", "nemotron-colembed", "image", "nemotron", 4096,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~750 vectors/page), float16 inference over "
              "two Kaggle T4s (no float32 encoding exists, so the baseline is the float16-inference one), pinned "
              "revision 34b64061, 2026-10-05/06; reports/universal/wide/NEMOTRON_8B.md; CC-BY-NC-4.0",
              license="cc-by-nc-4.0", revision="34b640612f311ed05a6c7c62c6564847ed555f5f"),
    ModelInfo("athrael-soju/colqwen3.5-4.5B-v3", "colqwen3.5", "image", "colpali-engine:colqwen3.5", 320,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~750 vectors/page), float32 over two Kaggle "
              "T4s, colpali-engine 0.3.18, 2026-10-06; reports/universal/wide/colqwen3.5-4.5B-v3_*. Encoded by "
              "the script stored beside the logs (reports/universal/wide/logs/colqwen3.5-4.5B-v3/enc_generic.py), "
              "whose calls the in-package loader repeats; the in-package loader itself has not been run on this "
              "checkpoint",
              license="apache-2.0", revision="c4ea4b92ff3322d68bfd6c65e492c69d0176f910"),
    ModelInfo("TomoroAI/tomoro-colqwen3-embed-4b", "colqwen3", "image", "remote-code:tomoro", 320,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1,100 to 1,200 vectors/page), float32 over two Kaggle "
              "T4s, 2026-10-06; reports/universal/wide/tomoro-colqwen3-embed-4b_*. Encoded by the script stored "
              "beside the logs (reports/universal/wide/logs/tomoro-colqwen3-embed-4b/enc_generic.py), whose calls "
              "the in-package recipe repeats; the in-package loader itself has not been run on this checkpoint",
              license="apache-2.0", revision="13517a29e8c5e408f7f2684337ed407df3acb212",
              notes="runs the repository's own modelling code: pass trust_remote_code=True"),
    ModelInfo("nomic-ai/colnomic-embed-multimodal-7b", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~740 vectors/page), float16 inference (no "
              "float32 encoding, so the baseline is the float16-inference one) over two Kaggle T4s, at the "
              "revision below, 2026-10-14; Kaggle batch 04_colqwen_7b; "
              "reports/universal/campaign/nomic-ai--colnomic-embed-multimodal-7b",
              license="apache-2.0", revision="09dbc9502b66605d5be56d2226019b49c9fd3293",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("colbert-ir/colbertv2.0", "colbert", "text", "sentence-transformers", 128,
              MEASURED, "BEIR SciFact (5,183 documents, 300 queries, ~157 vectors/document) through "
              "sentence-transformers 6.1.0 on one Kaggle T4, at the revision below (the Hub's main branch "
              "when it ran), 2026-10-15; Kaggle batch 11_text; "
              "reports/universal/campaign/colbert-ir--colbertv2.0",
              revision="c1e84128e85ef755c096a95bdb06b47793b13acf"),
    ModelInfo("answerdotai/answerai-colbert-small-v1", "colbert", "text", "sentence-transformers", 96,
              MEASURED, "BEIR SciFact (5,183 docs / 300 queries) via sentence-transformers 6.1.0 on "
              "2026-09-27; float nDCG@10 74.56 vs 74.77 on the model card; reports/universal/text"),
    ModelInfo("lightonai/GTE-ModernColBERT-v1", "colbert", "text", "sentence-transformers", 128,
              MEASURED, "BEIR SciFact (5,183 documents, 300 queries, ~230 vectors/document) through "
              "sentence-transformers 6.1.0 on one Kaggle T4, at the revision below (the Hub's main branch "
              "when it ran), 2026-10-15; Kaggle batch 11_text; "
              "reports/universal/campaign/lightonai--GTE-ModernColBERT-v1",
              revision="25f6f7bb8237b7ae25ae1d9b805ce17c0d1cc639",
              notes="also loads with loader='pylate' (check_retriever passed with PyLate here, 2026-10-12)"),
    # ---- added 2026-10-12: facts read from each model's card and repository files; those measured since
    # ---- (Kaggle campaign, reports/universal/campaign) say so in their evidence
    ModelInfo("vidore/colpali", "colpali", "image", "colpali-engine:colpali", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1030 vectors/page), float32 over two Kaggle "
              "T4s, at the revision below, 2026-10-14; Kaggle batch 01_colpali; "
              "reports/universal/campaign/vidore--colpali",
              license="mit", revision="7754422bfa37f8e3cd771e13d4d7f07de3103c95",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("vidore/colpali-v1.1", "colpali", "image", "colpali-engine:colpali", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1030 vectors/page), float32 over two Kaggle "
              "T4s, at the revision below, 2026-10-14; Kaggle batch 01_colpali; "
              "reports/universal/campaign/vidore--colpali-v1.1",
              license="mit", revision="855a60dd9e466987f27c385609710152f8868f67",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("vidore/colpali-v1.2", "colpali", "image", "colpali-engine:colpali", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1030 vectors/page), float32 over two Kaggle "
              "T4s, at the revision below, 2026-10-14; Kaggle batch 01_colpali; "
              "reports/universal/campaign/vidore--colpali-v1.2",
              license="mit", revision="6e6cf48566f6fbb8eb429a4bf90f39006d703317",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("vidore/colpali-v1.2-merged", "colpali", "image", "colpali-engine:colpali", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1030 vectors/page), float32 over two Kaggle "
              "T4s, at the revision below, 2026-10-14; Kaggle batch 01_colpali; "
              "reports/universal/campaign/vidore--colpali-v1.2-merged",
              license="gemma", revision="cd3977f763d2495ec7cfb605fb9f5aec65330509"),
    ModelInfo("vidore/colpali-v1.3", "colpali", "image", "colpali-engine:colpali", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1030 vectors/page), float32 over two Kaggle "
              "T4s, at the revision below, 2026-10-14; Kaggle batch 01_colpali; "
              "reports/universal/campaign/vidore--colpali-v1.3",
              license="mit", revision="b5c6dd62125326e6f0b540c1f7b36e901cdc0a11",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("vidore/colpali-v1.2-hf", "colpali", "image", "sentence-transformers", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1030 vectors/page), float16 inference (no "
              "float32 encoding, so the baseline is the float16-inference one) on one Kaggle T4, at the "
              "revision below, 2026-10-15; Kaggle batch 05_st_small; "
              "reports/universal/campaign/vidore--colpali-v1.2-hf",
              license="gemma", revision="3c712e95d5557c4d72a4107659cee01a40c1ece6",
              notes="transformers-native weights (ColPaliForRetrieval), not a colpali-engine "
                    "checkpoint"),
    ModelInfo("vidore/colpali-v1.3-hf", "colpali", "image", "sentence-transformers", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1030 vectors/page), float16 inference (no "
              "float32 encoding, so the baseline is the float16-inference one) on one Kaggle T4, at the "
              "revision below, 2026-10-15; Kaggle batch 05_st_small; "
              "reports/universal/campaign/vidore--colpali-v1.3-hf",
              license="gemma", revision="133a9eb02947310513f52f8f0d39d622e0eab8dc",
              notes="transformers-native weights (ColPaliForRetrieval), not a colpali-engine "
                    "checkpoint"),
    ModelInfo("vidore/colqwen2-v0.1", "colqwen", "image", "colpali-engine:colqwen2", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~740 vectors/page), float32 over two Kaggle T4s,"
              " at the revision below, 2026-10-14; Kaggle batch 02_colqwen_small; "
              "reports/universal/campaign/vidore--colqwen2-v0.1",
              license="apache-2.0", revision="38b8ebb6198e2704bc24fe0027b3743f25f03bfc",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("vidore/colqwen2-v1.0-hf", "colqwen", "image", "sentence-transformers", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~740 vectors/page), float16 inference (no "
              "float32 encoding, so the baseline is the float16-inference one) on one Kaggle T4, at the "
              "revision below, 2026-10-15; Kaggle batch 05_st_small; "
              "reports/universal/campaign/vidore--colqwen2-v1.0-hf",
              license="apache-2.0", revision="ddc07d2317c80f75fc742b7362ee9ad1912908f9",
              notes="transformers-native weights (ColQwen2ForRetrieval), not a colpali-engine "
                    "checkpoint"),
    ModelInfo("vidore/colqwen2.5-v0.1", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~740 vectors/page), float32 over two Kaggle T4s,"
              " at the revision below, 2026-10-15; Kaggle batch 12_retry; pages capped at 768 visual "
              "tokens, as its model card states; reports/universal/campaign/vidore--colqwen2.5-v0.1",
              license="mit", revision="4379ecfd4c9446836ad0ef4dd7390b1c748f68d8",
              notes="adapter-only repository: merged explicitly over its base model on load; pages capped at 768 "
              "visual tokens as the model card states (its preprocessor config allows 16384, which runs a T4 out of "
              "memory)",
              load_options=(("max_num_visual_tokens", 768),)),
    ModelInfo("vidore/colqwen-omni-v0.1", "colqwen", "image", "colpali-engine:colqwen2.5-omni", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1020 vectors/page), float32 over two Kaggle "
              "T4s, at the revision below, 2026-10-14; Kaggle batch 03_colqwen_multilingual; "
              "reports/universal/campaign/vidore--colqwen-omni-v0.1",
              license="mit", revision="c0f50696f483b12247c3a4bc23a524b29aba2c13",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("nomic-ai/colnomic-embed-multimodal-3b", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1200 vectors/page), float32 over two Kaggle "
              "T4s, at the revision below, 2026-10-14; Kaggle batch 02_colqwen_small; "
              "reports/universal/campaign/nomic-ai--colnomic-embed-multimodal-3b",
              license=None, revision="86627b4a9b0cade577851a70afa469084f9863a4",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("tsystems/colqwen2.5-3b-multilingual-v1.0", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~740 vectors/page), float32 over two Kaggle T4s,"
              " at the revision below, 2026-10-14; Kaggle batch 03_colqwen_multilingual; "
              "reports/universal/campaign/tsystems--colqwen2.5-3b-multilingual-v1.0",
              license="mit", revision="d5bd3db7ca57a1de3dc862828f343904ca07efd9",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("tsystems/colqwen2.5-3b-multilingual-v1.0-merged", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~740 vectors/page), float32 over two Kaggle T4s,"
              " at the revision below, 2026-10-14; Kaggle batch 03_colqwen_multilingual; "
              "reports/universal/campaign/tsystems--colqwen2.5-3b-multilingual-v1.0-merged",
              license="mit", revision="db0d27eb032de3fa5bcd626a6558e7a560a2beb8"),
    ModelInfo("Metric-AI/ColQwen2.5-3b-multilingual-v1.0", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~740 vectors/page), float32 over two Kaggle T4s,"
              " at the revision below, 2026-10-15; Kaggle batch 12_retry; pages capped at 768 visual "
              "tokens, as its model card states; "
              "reports/universal/campaign/Metric-AI--ColQwen2.5-3b-multilingual-v1.0",
              license="mit", revision="e2a1c05d053dcf4ad6e39b6c48ced9d6a81071f0",
              notes="adapter-only repository: merged explicitly over its base model on load; pages capped at 768 "
              "visual tokens as the model card states (its preprocessor config allows 16384, which runs a T4 out of "
              "memory)",
              load_options=(("max_num_visual_tokens", 768),)),
    ModelInfo("Metric-AI/ColQwen2.5-7b-multilingual-v1.0", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              UNVERIFIED, "cannot be loaded: its adapter names the base model Metric-AI/colqwen2.5-7b-base, which is "
              "not on the Hugging Face Hub (Kaggle 2026-10-14; checked again 2026-10-15); "
              "reports/universal/campaign/Metric-AI--ColQwen2.5-7b-multilingual-v1.0",
              license="mit", revision="8874f274c038e7ddd093b5ea4b33218db710fa5f",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~980 vectors/page), float16 inference (no "
              "float32 encoding, so the baseline is the float16-inference one) over two Kaggle T4s, at the "
              "revision below, 2026-10-14; Kaggle batch 04_colqwen_7b; "
              "reports/universal/campaign/ApsaraStackMaaS--EvoQwen2.5-VL-Retriever-7B-v1",
              license="apache-2.0", revision="b4c6d894958789e861db16d444cd811b8eb1be75"),
    ModelInfo("vidore/colsmolvlm-v0.1", "colsmol", "image", "colpali-engine:colsmol", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1210 vectors/page), float32 over two Kaggle "
              "T4s, at the revision below, 2026-10-14; Kaggle batch 02_colqwen_small; "
              "reports/universal/campaign/vidore--colsmolvlm-v0.1",
              license=None, revision="c14f4090e6ac69f0420d4ddb4425d28f8cd9cc80",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("ModernVBERT/colmodernvbert", "colmodernvbert", "image", "sentence-transformers", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~900 vectors/page), float32 on one Kaggle T4, at"
              " the revision below, 2026-10-15; Kaggle batch 05_st_small; "
              "reports/universal/campaign/ModernVBERT--colmodernvbert",
              license="mit", revision="4c604428a6e1a7e33d0fb34837f0cdecbc22026e",
              notes="the colpali-engine route of the card (ColModernVBert) needs "
                    "trust_remote_code; loader='colpali-engine:colmodernvbert' is available and "
                    "equally unrun"),
    ModelInfo("TomoroAI/tomoro-colqwen3-embed-8b", "colqwen3", "image", "remote-code:tomoro", 320,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1170 vectors/page), float16 inference (no "
              "float32 encoding, so the baseline is the float16-inference one) over two Kaggle T4s, at the "
              "revision below, 2026-10-15; Kaggle batch 08_remote_code; "
              "reports/universal/campaign/TomoroAI--tomoro-colqwen3-embed-8b",
              license="apache-2.0", revision="7901e21544bf360868ead6b384b37b3abdf119cf",
              notes="runs the repository's own modelling code: pass trust_remote_code=True"),
    ModelInfo("vultr/VultronRetrieverPrime-Qwen3.5-8B", "colqwen3.5", "image", "colpali-engine:colqwen3.5", 320,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~730 vectors/page), float16 inference (no "
              "float32 encoding, so the baseline is the float16-inference one) over two Kaggle T4s, at the "
              "revision below, 2026-10-15; Kaggle batch 07_qwen35_vultr; "
              "reports/universal/campaign/vultr--VultronRetrieverPrime-Qwen3.5-8B",
              license="apache-2.0", revision="1201db5ae23bc4d5977c188748f35cf3be54c568",
              notes="the card pins a colpali-engine git commit and needs causal-conv1d and "
                    "flash-linear-attention"),
    ModelInfo("vultr/VultronRetrieverCore-Qwen3.5-4.5B", "colqwen3.5", "image", "colpali-engine:colqwen3.5", 320,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1170 vectors/page), float32 over two Kaggle "
              "T4s, at the revision below, 2026-10-15; Kaggle batch 07_qwen35_vultr; "
              "reports/universal/campaign/vultr--VultronRetrieverCore-Qwen3.5-4.5B",
              license="apache-2.0", revision="b0857609ad6db741009a7026e02cf30bd384f54d",
              notes="the card pins a colpali-engine git commit and needs causal-conv1d and "
                    "flash-linear-attention"),
    ModelInfo("vultr/VultronRetrieverFlash-Qwen3.5-0.8B", "colqwen3.5", "image", "colpali-engine:colqwen3.5", 320,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1170 vectors/page), float32 over two Kaggle "
              "T4s, at the revision below, 2026-10-15; Kaggle batch 07_qwen35_vultr; "
              "reports/universal/campaign/vultr--VultronRetrieverFlash-Qwen3.5-0.8B",
              license="apache-2.0", revision="6f6b6757e9ea3ad93f4f2cbd839bc6f2f0b93739",
              notes="the card pins a colpali-engine git commit and needs causal-conv1d and "
                    "flash-linear-attention"),
    ModelInfo("tencent/EVIE-4.5B", "colqwen3.5", "image", "sentence-transformers", 2048,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~960 vectors/page), float16 inference (no "
              "float32 encoding, so the baseline is the float16-inference one) on one Kaggle T4, at the "
              "revision below, 2026-10-15; Kaggle batch 06_st_qwen3; "
              "reports/universal/campaign/tencent--EVIE-4.5B",
              license="apache-2.0", revision="498388f8f4c92af8d99fffeececd56c0a4c428d0",
              notes="2,048-d vectors (the card describes shorter prefixes as usable); its colpali "
                    "route needs the fork bundled in the repository"),
    ModelInfo("tencent/EVIE-8B", "colqwen3.5", "image", "sentence-transformers", 4096,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; split over two "
              "T4s on Kaggle 2026-10-16, it passed check_retriever (11 of 12) but ran GPU 0 out of memory in "
              "attention on a page of ~8500 visual tokens (its processor allows 16,384 a page); the retry "
              "uses its card's 1,024-token evaluation budget, not yet re-run; "
              "reports/universal/campaign/tencent--EVIE-8B",
              license="apache-2.0", revision="cdc8f60e56d168591354cf919bb5a50a5d1f617b"),
    ModelInfo("VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1", "colqwen3", "image", "sentence-transformers", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1170 vectors/page), float16 inference (no "
              "float32 encoding, so the baseline is the float16-inference one) on one Kaggle T4, at the "
              "revision below, 2026-10-15; Kaggle batch 06_st_qwen3; "
              "reports/universal/campaign/VAGOsolutions--SauerkrautLM-ColQwen3-2b-v0.1",
              license="apache-2.0", revision="60a0200a0d9c76df7acee02d7fd074db8c81cf00"),
    ModelInfo("VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1", "colqwen3", "image", "sentence-transformers", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1170 vectors/page), float16 inference (no "
              "float32 encoding, so the baseline is the float16-inference one) on one Kaggle T4, at the "
              "revision below, 2026-10-15; Kaggle batch 06_st_qwen3; "
              "reports/universal/campaign/VAGOsolutions--SauerkrautLM-ColQwen3-4b-v0.1",
              license="apache-2.0", revision="de6b8ac536cae74fab7595de5d53e24f5e6c4295"),
    ModelInfo("webAI-Official/webAI-ColVec1-4b", "colvec", "image", "remote-code:colvec", 640,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~1170 vectors/page), float32 over two Kaggle "
              "T4s, at the revision below, 2026-10-15; Kaggle batch 13_retry; "
              "reports/universal/campaign/webAI-Official--webAI-ColVec1-4b",
              license="other", revision="cb5af9aab4894bfa1c7f2427d6970f19055c7226",
              notes="runs the repository's own modelling code: pass trust_remote_code=True"),
    ModelInfo("webAI-Official/webAI-ColVec1.1-4b", "colvec", "image", "sentence-transformers", 640,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; loaded on Kaggle "
              "2026-10-15 (float16 inference, one T4) but failed check_retriever: 9 of 12 queries ranked "
              "their own page first, where 10 are required; its card states bfloat16 weights and "
              "evaluation, which a T4 cannot run natively; in float32 over two T4s (2026-10-16) its own "
              "forward (modeling_colqwen35_bidirection.py) multiplied the projected vectors, on the last "
              "GPU, by the attention mask on the first; the adapter now moves the projection head to the "
              "input GPU, not yet re-run; reports/universal/campaign/webAI-Official--webAI-ColVec1.1-4b",
              license="other", revision="d88d0155d700b06382124ff985ddb07099155bb4",
              notes="runs the repository's own modelling code: pass trust_remote_code=True"),
    ModelInfo("webAI-Official/webAI-ColVec1.1-8b", "colvec", "image", "sentence-transformers", 640,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; never encoded "
              "here: in float16 it does not fit one 16 GB T4, and split over two (Kaggle 2026-10-16) its "
              "own forward (modeling_colqwen35_bidirection.py) multiplied the projected vectors, on the "
              "last GPU, by the attention mask on the first; the adapter now moves the projection head "
              "to the input GPU, not yet re-run; "
              "reports/universal/campaign/webAI-Official--webAI-ColVec1.1-8b",
              license="other", revision="0f712a2777fbba29b2a21bb68ffcfbb5272505ad",
              notes="runs the repository's own modelling code: pass trust_remote_code=True"),
    ModelInfo("Hcompany/NeoMME-260M-Retriever-ST-late", "neomme", "image", "sentence-transformers", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~2150 vectors/page), float32 on one Kaggle T4, "
              "at the revision below, 2026-10-15; Kaggle batch 13_retry; "
              "reports/universal/campaign/Hcompany--NeoMME-260M-Retriever-ST-late",
              license="apache-2.0", revision="023be2a8ab9d797f5aa76f5bf8b5dde78d819659"),
    ModelInfo("Hcompany/NeoMME-800M-Retriever-ST-late", "neomme", "image", "sentence-transformers", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~2150 vectors/page), float32 on one Kaggle T4, "
              "one page per batch (4 ran out of memory in attention), at the revision below, 2026-10-16; "
              "Kaggle batch 14b_neomme_retry; "
              "reports/universal/campaign/Hcompany--NeoMME-800M-Retriever-ST-late",
              license="apache-2.0", revision="ad4c17419c28745c3b3045981b0d8abb72cd142b"),
    ModelInfo("Cognitive-Lab/ColNetraEmbed", "colgemma", "image", "colpali-engine:colgemma3", 128,
              MEASURED, "ViDoRe V1 DocVQA + InfoVQA (500 pages each, ~270 vectors/page), float32 over two Kaggle T4s,"
              " at the revision below, 2026-10-15; Kaggle batch 12_retry; loaded with the released "
              "colpali-engine 0.3.18, its language_model.model.* keys mapped to language_model.* "
              "(loading_report key_remap); reports/universal/campaign/Cognitive-Lab--ColNetraEmbed",
              license="gemma", revision="a27fe0bdcf65a54116393853621f7f80e3cd9ca3",
              notes="the card installs a fork of colpali-engine (github.com/adithya-s-k/colpali); the "
                    "released colpali-engine loads it here once the checkpoint's nested keys are mapped"),
    ModelInfo("ibm-granite/granite-vision-3.3-2b-embedding", "granite-vision", "image", "remote-code:processor", 128,
              UNVERIFIED, "remote-code recipe written against the model card (process_images / process_queries); cannot"
              " be loaded under transformers 5: its modelling code reads a language_model attribute that "
              "LlavaNext no longer has (Kaggle 2026-10-15, transformers 5.16.1); "
              "reports/universal/campaign/ibm-granite--granite-vision-3.3-2b-embedding",
              license="apache-2.0", revision="baa53d1e1ac95baefb15e4aa9fdcc471ac728c83",
              notes="runs the repository's own modelling code: pass trust_remote_code=True; the "
                    "card pins transformers==4.50"),
    ModelInfo("5CD-AI/Vintern-Embedding-1B", "colinternvl", "image", "remote-code:processor", 128,
              UNVERIFIED, "remote-code recipe written against the model card (process_images / process_queries); cannot"
              " be loaded under transformers 5: its configuration_internvl_chat.py reads "
              "llm_config['architectures'], which is not there (Kaggle 2026-10-15, transformers 5.16.1); "
              "reports/universal/campaign/5CD-AI--Vintern-Embedding-1B",
              license=None, revision="ee3f8a72164d194c0885df3f9be3d4a3f0626cc9",
              notes="runs the repository's own modelling code: pass trust_remote_code=True; the "
                    "card pins transformers==4.48.2 and casts its inputs to bfloat16 by hand"),
    ModelInfo("jinaai/jina-embeddings-v4", "jina-v4", "image", "remote-code:jina-v4", 128,
              UNVERIFIED, "remote-code recipe written against the model card (encode_image / encode_text with "
              "return_multivector=True); cannot be loaded under transformers 5: its qwen2_5_vl.py asks for "
              "the 'default' RoPE initialiser, which transformers 5 removed (Kaggle 2026-10-15, "
              "transformers 5.16.1); reports/universal/campaign/jinaai--jina-embeddings-v4",
              license=None, revision="853c867b65b749f3c3c72a06868140d842e04f06",
              notes="runs the repository's own modelling code: pass trust_remote_code=True; "
                    "single-vector by default, the recipe asks for the 128-d multi-vector output; "
                    "the card names the Qwen Research License"),
    ModelInfo("nvidia/llama-nemoretriever-colembed-1b-v1", "nemoretriever-colembed", "image", "remote-code:nvidia-colembed", None,
              UNVERIFIED, "remote-code recipe written against the model card (forward_images / forward_queries); cannot"
              " be loaded on a T4: its modelling code imports flash_attn (Kaggle 2026-10-15), and "
              "FlashAttention 2 needs an Ampere or newer GPU; "
              "reports/universal/campaign/nvidia--llama-nemoretriever-colembed-1b-v1",
              license="other", revision="78361b5fa933f700305cfab3accae44b759de590",
              notes="runs the repository's own modelling code: pass trust_remote_code=True; the "
                    "card requires transformers==4.49.0 and flash-attn; no file of the repository "
                    "states the output width"),
    ModelInfo("nvidia/llama-nemoretriever-colembed-3b-v1", "nemoretriever-colembed", "image", "remote-code:nvidia-colembed", None,
              UNVERIFIED, "remote-code recipe written against the model card (forward_images / forward_queries); cannot"
              " be loaded on a T4: its modelling code imports flash_attn (Kaggle 2026-10-15), and "
              "FlashAttention 2 needs an Ampere or newer GPU; "
              "reports/universal/campaign/nvidia--llama-nemoretriever-colembed-3b-v1",
              license="other", revision="75f7e7311c7a5965c535587fd5e563c5e4a3c583",
              notes="runs the repository's own modelling code: pass trust_remote_code=True; the "
                    "card requires transformers==4.49.0 and flash-attn; no file of the repository "
                    "states the output width"),
    ModelInfo("nvidia/llama-nemotron-colembed-vl-3b-v2", "nemoretriever-colembed", "image", "remote-code:nvidia-colembed", 3072,
              UNVERIFIED, "remote-code recipe written against the model card (forward_images / forward_queries); loaded"
              " over two T4s and passed check_retriever on Kaggle 2026-10-15, then the process was killed "
              "for host memory after encoding the 500 DocVQA pages (~1800 vectors of 3072 dimensions per "
              "page, about 11 GB as float32); not yet re-run; "
              "reports/universal/campaign/nvidia--llama-nemotron-colembed-vl-3b-v2",
              license="other", revision="9907f8fdf1eae6802c5b9ee4287ef9009acfdae9",
              notes="runs the repository's own modelling code: pass trust_remote_code=True; the "
                    "card requires flash-attn"),
    ModelInfo("lightonai/colbertv2.0", "colbert", "text", "sentence-transformers", 128,
              UNTESTED, "check_retriever passed here on 2026-10-12 (8 text pairs, sentence-transformers 6.1.0, CPU: 8"
              " of 8 queries ranked their own document first, 128-d unit-norm vectors; "
              "reports/universal/loader_checks/2026-10-12.json); Kaggle batch 11_text measured it, but "
              "colbert-ir/colbertv2.0 in the same batch has the same short name and overwrote its results "
              "(only its selection study survived, in "
              "reports/universal/campaign/batches/11_text/orphans/colbertv2.0); not yet re-run",
              license="mit", revision="0b9b1e380096a3b72b6b4877f68d59effb62e0f2",
              notes="the sentence-transformers conversion of colbert-ir/colbertv2.0"),
    ModelInfo("jinaai/jina-colbert-v2", "colbert", "text", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; never run here "
              "(Kaggle batch 11_text ran out of time before it); model card and repository files read "
              "2026-10-12",
              license="cc-by-nc-4.0", revision="a9dc5cd7293d4c71dbbba04829923ba4d0e4f6ea",
              notes="runs the repository's own modelling code: pass trust_remote_code=True; needs "
                    "einops"),
    ModelInfo("lightonai/Reason-ModernColBERT", "colbert", "text", "sentence-transformers", 128,
              MEASURED, "BEIR SciFact (5,183 documents, 300 queries, ~274 vectors/document) through "
              "sentence-transformers 6.1.0 on one Kaggle T4, at the revision below (the Hub's main branch "
              "when it ran), 2026-10-15; Kaggle batch 11_text; "
              "reports/universal/campaign/lightonai--Reason-ModernColBERT",
              license="cc-by-nc-4.0", revision="e0bd6968566909365aa553cb54f2219c2d851ed0",
              notes="also loads with loader='pylate'"),
    ModelInfo("lightonai/ColBERT-Zero", "colbert", "text", "sentence-transformers", 128,
              MEASURED, "BEIR SciFact (5,183 documents, 300 queries, ~270 vectors/document) through "
              "sentence-transformers 6.1.0 on one Kaggle T4, at the revision below (the Hub's main branch "
              "when it ran), 2026-10-15; Kaggle batch 11_text; "
              "reports/universal/campaign/lightonai--ColBERT-Zero",
              license="apache-2.0", revision="43aefda07cf4fea7a13dbdab0eef8ecf78580d3d",
              notes="also loads with loader='pylate'"),
    ModelInfo("lightonai/LateOn", "colbert", "text", "sentence-transformers", 128,
              MEASURED, "BEIR SciFact (5,183 documents, 300 queries, ~230 vectors/document) through "
              "sentence-transformers 6.1.0 on one Kaggle T4, at the revision below (the Hub's main branch "
              "when it ran), 2026-10-15; Kaggle batch 11_text; reports/universal/campaign/lightonai--LateOn",
              license="apache-2.0", revision="62911e105059585d244384c7d17826e35f669c17",
              notes="also loads with loader='pylate'"),
    ModelInfo("lightonai/mLateOn", "colbert", "text", "sentence-transformers", 128,
              MEASURED, "BEIR SciFact (5,183 documents, 300 queries, ~313 vectors/document) through "
              "sentence-transformers 6.1.0 on one Kaggle T4, at the revision below (the Hub's main branch "
              "when it ran), 2026-10-15; Kaggle batch 11_text; "
              "reports/universal/campaign/lightonai--mLateOn",
              license="apache-2.0", revision="bbd883ae1fadb563056ff73b1061ae8172a3fcda",
              notes="also loads with loader='pylate'"),
    ModelInfo("mixedbread-ai/mxbai-edge-colbert-v0-17m", "colbert", "text", "sentence-transformers", 48,
              MEASURED, "BEIR SciFact (5,183 documents, 300 queries, ~267 vectors/document) through "
              "sentence-transformers 6.1.0 on one Kaggle T4, at the revision below (the Hub's main branch "
              "when it ran), 2026-10-15; Kaggle batch 11_text; "
              "reports/universal/campaign/mixedbread-ai--mxbai-edge-colbert-v0-17m",
              license="apache-2.0", revision="592c6417c1c6687572043408ed1ae5196bce16b1",
              notes="also loads with loader='pylate'"),
    ModelInfo("mixedbread-ai/mxbai-edge-colbert-v0-32m", "colbert", "text", "sentence-transformers", 64,
              MEASURED, "BEIR SciFact (5,183 documents, 300 queries, ~267 vectors/document) through "
              "sentence-transformers 6.1.0 on one Kaggle T4, at the revision below (the Hub's main branch "
              "when it ran), 2026-10-15; Kaggle batch 11_text; "
              "reports/universal/campaign/mixedbread-ai--mxbai-edge-colbert-v0-32m",
              license="apache-2.0", revision="bb13a29ec9b1e7edd4ba8f7a0776c48b55cbad66",
              notes="also loads with loader='pylate'"),
    ModelInfo("mixedbread-ai/mxbai-colbert-large-v1", "colbert", "text", "sentence-transformers", 128,
              MEASURED, "BEIR SciFact (5,183 documents, 300 queries, ~212 vectors/document) through "
              "sentence-transformers 6.1.0 on one Kaggle T4, at the revision below (the Hub's main branch "
              "when it ran), 2026-10-15; Kaggle batch 11_text; "
              "reports/universal/campaign/mixedbread-ai--mxbai-colbert-large-v1",
              license="apache-2.0", revision="d092c678f449647d1945569dc17b4d8917a687e0"),
    ModelInfo("LiquidAI/LFM2-ColBERT-350M", "colbert", "text", "sentence-transformers", 128,
              MEASURED, "BEIR SciFact (5,183 documents, 300 queries, ~286 vectors/document) through "
              "sentence-transformers 6.1.0 on one Kaggle T4, at the revision below (the Hub's main branch "
              "when it ran), 2026-10-15; Kaggle batch 11_text; "
              "reports/universal/campaign/LiquidAI--LFM2-ColBERT-350M",
              license="other", revision="9d727d7964d3e4fd093a0f01c7287ae814264e31",
              notes="also loads with loader='pylate'"),
    ModelInfo("LiquidAI/LFM2.5-ColBERT-350M", "colbert", "text", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; encoded BEIR "
              "SciFact and passed check_retriever on Kaggle 2026-10-15 (batch 11_text), then the session "
              "ran out of time in the fixed-pipeline study; not yet re-run; "
              "reports/universal/campaign/LiquidAI--LFM2.5-ColBERT-350M",
              license="other", revision="9772bdf797255d8693b83e84aa98e9b2d36dd0be",
              notes="runs the repository's own modelling code: pass trust_remote_code=True; also "
                    "loads with loader='pylate'"),
    ModelInfo("NeuML/colbert-bert-tiny", "colbert", "text", "sentence-transformers", 128,
              MEASURED, "BEIR SciFact (5,183 documents, 300 queries, ~157 vectors/document) through "
              "sentence-transformers 6.1.0 on one Kaggle T4, at the revision below (the Hub's main branch "
              "when it ran), 2026-10-15; Kaggle batch 11_text; "
              "reports/universal/campaign/NeuML--colbert-bert-tiny",
              license="apache-2.0", revision="5967f70792ff2315c72c53a74e7cd5de2af4e209"),
]

MODELS: dict[str, ModelInfo] = {m.model_id: m for m in _MODELS}


def supported_models(status: str | None = None) -> list[ModelInfo]:
    """Known models, optionally filtered by status. Unknown ids can still be used
    through :class:`SentenceTransformersAdapter` or precomputed vectors."""
    return [m for m in _MODELS if status is None or m.status == status]


# ------------------------------------------------------------------ adapters


def _positive_int(value: Any) -> int | None:
    """``value`` as a Python int if it is a positive integer (bool excluded), else None."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        return None
    return int(value) if value > 0 else None


def _text(value: Any) -> str | None:
    """``value`` if it is a non-empty string, else None."""
    return value if isinstance(value, str) and value else None


def _dtype_name(dtype: Any) -> str | None:
    """A dtype object as a plain name (``torch.float32`` -> ``"float32"``), or None."""
    return None if dtype is None else str(dtype).removeprefix("torch.")


def _identity_from_info(info: ModelInfo | None) -> EncoderIdentity | None:
    """What a registry entry states about a model, or None without a known dimension.

    The revision, the normalisation flag and the inference dtype are not in the registry
    (its evidence text is prose and is never parsed), so they stay None. The registry
    status is not part of an identity: it says how well a model has been tested here, not
    which model it is.
    """
    if info is None or _positive_int(info.dimension) is None:
        return None
    return EncoderIdentity(
        model_id=info.model_id,
        dimension=int(info.dimension),
        modality=info.modality if info.modality in MODALITIES else "other",
        extra={"family": info.family, "loader": info.loader},
    )


class EncoderAdapter(abc.ABC):
    """Anything that turns documents and queries into multi-vector corpora.

    ``encode_documents`` and ``encode_queries`` are the contract: each returns a
    :class:`~optivision.representation.MultiVectorCorpus` with one document per input,
    with the padding rows of a batch already dropped. ``identity()`` and ``dimension``
    describe the model behind the vectors; they are concrete, with defaults built from
    ``info``, so a subclass written without them keeps working.
    """

    info: ModelInfo | None = None

    @abc.abstractmethod
    def encode_documents(self, documents: Sequence[Any], ids: Sequence[str] | None = None) -> MultiVectorCorpus: ...

    @abc.abstractmethod
    def encode_queries(self, queries: Sequence[str], ids: Sequence[str] | None = None) -> MultiVectorCorpus: ...

    def identity(self) -> EncoderIdentity | None:
        """Descriptive identity of the model behind the vectors, or None when not known.

        The default is built from ``info`` when it names a model with a known dimension.
        Subclasses override it with whatever they can state; a field they cannot state
        stays None, and a revision or a normalisation flag is never guessed. Nothing in
        the core branches on the result.
        """
        return _identity_from_info(self.info)

    @property
    def dimension(self) -> int | None:
        """Vector dimension the adapter produces, or None when it is not known up front.

        Taken from :meth:`identity` unless the subclass assigned ``self.dimension``.
        """
        declared = _positive_int(getattr(self, "_declared_dimension", None))
        if declared is not None:
            return declared
        identity = self.identity()
        return None if identity is None else identity.dimension

    @dimension.setter
    def dimension(self, value: int | None) -> None:
        self._declared_dimension = value


class PageEncoderAdapter(EncoderAdapter):
    """Wraps any :class:`~optivision.encoders.BaseEncoder` (ColPali family, synthetic).

    Documents are ``(PageRef, PIL.Image)`` pairs, as produced by
    :func:`optivision.ingest.iter_pages`. Grid positions, protected instruction
    tokens and the page images are carried into the corpus, so both the
    vector-only stages and the pixel-space pruner can run on the result.
    """

    def __init__(self, encoder: Any, info: ModelInfo | None = None) -> None:
        self.encoder = encoder
        self.info = info

    def encode_documents(self, documents: Sequence[Any], ids: Sequence[str] | None = None) -> MultiVectorCorpus:
        refs = [ref for ref, _ in documents]
        images = [img for _, img in documents]
        encodings = []
        for ref, img in zip(refs, images, strict=True):
            encodings.extend(self.encoder.encode_pages([img], [ref]))
        corpus = MultiVectorCorpus.from_page_encodings(encodings, images=images, attrs=self._attrs())
        if ids is not None:
            if len(ids) != len(corpus):
                raise ValueError(f"got {len(ids)} ids for {len(corpus)} encoded pages")
            corpus.ids = [str(i) for i in ids]
        return corpus

    def encode_queries(self, queries: Sequence[str], ids: Sequence[str] | None = None) -> MultiVectorCorpus:
        vectors = self.encoder.encode_queries(list(queries))
        return MultiVectorCorpus.from_arrays(vectors, ids=ids, attrs=self._attrs())

    def _attrs(self) -> dict[str, Any]:
        attrs = {"encoder": getattr(self.encoder, "name", type(self.encoder).__name__)}
        checkpoint = getattr(self.encoder, "checkpoint", None)
        if checkpoint:
            attrs["model"] = checkpoint
        if self.info is not None:
            attrs["model_status"] = self.info.status
        return attrs

    def identity(self) -> EncoderIdentity | None:
        """From the wrapped encoder where it states a fact, else from ``info``.

        Model id: the encoder's ``checkpoint``, else ``info.model_id``, else its ``name``.
        Dimension: the encoder's ``dim``, else ``info.dimension``. Modality: ``info`` if
        given, else ``"image"`` (this adapter encodes page images). Revision and inference
        dtype are read only if the encoder carries them (``revision``, ``torch_dtype``).
        The normalisation flag stays None: the ``BaseEncoder`` contract asks for unit-norm
        rows but an arbitrary wrapped encoder is not checked here.
        """
        enc, info = self.encoder, self.info
        model_id = _text(getattr(enc, "checkpoint", None)) or (info.model_id if info else None)
        name = _text(getattr(enc, "name", None))
        model_id = model_id or name
        try:
            dim = _positive_int(enc.dim)
        except Exception:  # noqa: BLE001 -- a property of an arbitrary encoder may fail to answer
            dim = None
        dim = dim or (_positive_int(info.dimension) if info else None)
        if model_id is None or dim is None:
            return None
        modality = ("image" if info is None
                    else info.modality if info.modality in MODALITIES else "other")
        extra = {"encoder": name} if name else {}
        if info is not None:
            extra["family"] = info.family
        cap = _positive_int(getattr(enc, "max_num_visual_tokens", None))
        if cap is not None:  # a resolution cap changes the vectors, so it is part of the identity
            extra["max_num_visual_tokens"] = str(cap)
        return EncoderIdentity(
            model_id=model_id,
            revision=_text(getattr(enc, "revision", None)),
            dimension=dim,
            modality=modality,
            inference_dtype=_dtype_name(getattr(enc, "torch_dtype", None)),
            extra=extra,
        )


def _keep_placement(model: Any) -> None:
    """Stop ``model.to(...)`` from moving a model that ``device_map`` spread over several GPUs.

    sentence-transformers' ``encode`` calls ``self.to(self.device)`` on every call, which
    gathers every layer back onto the first GPU (that ran tencent/EVIE-8B and
    webAI-Official/webAI-ColVec1.1-8b out of memory on a 16 GB T4, Kaggle 2026-10-15).
    accelerate's hooks already move the activations between the devices.
    """
    model.to = lambda *args, **kwargs: model


def _split_module(model: Any) -> Any:
    """The module ``device_map`` dispatched (it carries ``hf_device_map``), or ``None``."""
    modules = getattr(model, "modules", None)
    for module in modules() if callable(modules) else ():
        device_map = getattr(module, "hf_device_map", None)
        if isinstance(device_map, dict) and device_map:
            return module
    return None


#: a top-level module holding at most this share of the parameters counts as a head
_HEAD_SHARE = 0.02


def _heads_on_input_device(model: Any) -> list[str]:
    """Move the small top-level modules of a split model onto the device its inputs are on.

    ``device_map="auto"`` puts the last modules (the projection head) on the last GPU, but
    some models' own forward multiplies the head's output by an input tensor:
    webAI-Official/webAI-ColVec1.1-4b and -8b multiply their projected vectors by the
    attention mask (modeling_colqwen35_bidirection.py), which failed with tensors on
    cuda:1 and cuda:0 (Kaggle 2026-10-16). On the input device the head's output meets the
    inputs there; accelerate's hook copies the hidden states across. Returns the names moved.
    """
    split = _split_module(model)
    if split is None:
        return []
    import torch

    device = getattr(model, "device", None)
    if not isinstance(device, torch.device):
        device = next(split.parameters()).device
    total = sum(p.numel() for p in split.parameters())
    target = device.index if device.type == "cuda" else str(device)
    moved = []
    for name, placed in list(split.hf_device_map.items()):
        if "." in name or not name or placed == target:
            continue
        head = split.get_submodule(name)
        if sum(p.numel() for p in head.parameters()) > _HEAD_SHARE * total:
            continue
        from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_module

        remove_hook_from_module(head, recurse=True)
        head.to(device)
        add_hook_to_module(head, AlignDevicesHook(execution_device=device, io_same_device=False))
        split.hf_device_map[name] = target
        moved.append(name)
    return moved


def _device_map_placement(model: Any, heads: Sequence[str] = ()) -> dict[str, Any] | None:
    split = _split_module(model)
    if split is None:
        return None
    placement = {"devices": sorted({str(d) for d in split.hf_device_map.values()}), "device_map": "auto"}
    if heads:
        placement["heads_on_input_device"] = list(heads)
    return placement


class SentenceTransformersAdapter(EncoderAdapter):
    """sentence-transformers ``MultiVectorEncoder`` (>= 6.0).

    Written against the API the model cards document -- ``encode_document(inputs)``
    and ``encode_query(texts)`` returning one ``[n_i, d]`` tensor or array per
    input. Run for real with sentence-transformers 6.1.0 on text: answerai-colbert-small-v1
    on SciFact (its float baseline matched the model card), and ``check_retriever`` runs on
    the text models whose registry evidence says so. The image path ran on Kaggle T4s
    (sentence-transformers 6.1.0, 2026-10-15) for the image models it measured; each
    other model's status is its own registry entry's. A ``device_map`` placement is kept:
    ``encode`` would otherwise move the model back onto one device, and the model's small
    top-level heads go to the device its inputs are on. Pass ``model``
    to inject an already-constructed encoder (that is how the tests exercise the
    glue without the library).

    Inputs are encoded ``batch_size`` at a time and each chunk is copied to host memory as
    float32 before the next one starts, so the device holds one chunk of outputs at a time
    (a page of a wide model is megabytes of vectors). ``dtype`` (``"float16"``, ...) is the
    inference dtype, passed to the model as ``model_kwargs={"torch_dtype": ...}``;
    ``encode_kwargs`` go to every ``encode_document`` / ``encode_query`` call; any other
    keyword goes to ``MultiVectorEncoder`` (``device``, ``trust_remote_code``, ``revision``,
    ``model_kwargs`` such as ``{"device_map": "auto"}``).
    """

    def __init__(self, model_id: str, model: Any = None, *, batch_size: int = 32, dtype: str | None = None,
                 encode_kwargs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        self.model_id = model_id
        self.batch_size = batch_size
        self.encode_kwargs = dict(encode_kwargs or {})
        self.info = MODELS.get(model_id)
        heads: list[str] = []
        if self.info is None or self.info.status == UNVERIFIED:
            warnings.warn(
                f"{model_id}: the sentence-transformers adapter is unverified in this "
                "repository; check retrieval quality before relying on it",
                stacklevel=2,
            )
        if model is None:
            try:
                from sentence_transformers import MultiVectorEncoder  # type: ignore[attr-defined]
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise ImportError(
                    "SentenceTransformersAdapter needs sentence-transformers>=6.0 "
                    "(pip install 'sentence-transformers[image]>=6')"
                ) from exc
            if dtype is not None:
                import torch

                kwargs["model_kwargs"] = {**(kwargs.get("model_kwargs") or {}), "torch_dtype": getattr(torch, dtype)}
            model = MultiVectorEncoder(model_id, **kwargs)
            if (kwargs.get("model_kwargs") or {}).get("device_map") is not None:
                _keep_placement(model)
                heads = _heads_on_input_device(model)
        self.model = model
        self.dtype = dtype
        self.revision = kwargs.get("revision")
        self.placement = _device_map_placement(model, heads)

    @staticmethod
    def _host(output: Any) -> np.ndarray:
        if hasattr(output, "detach"):
            output = output.detach().cpu().float().numpy()
        return np.asarray(output, dtype=np.float32)

    def _encode(self, method: Any, items: Sequence[Any]) -> list[np.ndarray]:
        items = list(items)
        arrays: list[np.ndarray] = []
        for start in range(0, len(items), self.batch_size):
            arrays.extend(self._host(o) for o in method(items[start : start + self.batch_size], **self.encode_kwargs))
        return arrays

    def _to_corpus(self, arrays: Sequence[np.ndarray], ids: Sequence[str] | None) -> MultiVectorCorpus:
        attrs = {"model": self.model_id, "loader": "sentence-transformers",
                 "model_status": self.info.status if self.info else UNVERIFIED}
        if self.dtype is not None:
            attrs["inference_dtype"] = self.dtype
        return MultiVectorCorpus.from_arrays(list(arrays), ids=ids, attrs=attrs)

    def encode_documents(self, documents: Sequence[Any], ids: Sequence[str] | None = None) -> MultiVectorCorpus:
        return self._to_corpus(self._encode(self.model.encode_document, documents), ids)

    def encode_queries(self, queries: Sequence[str], ids: Sequence[str] | None = None) -> MultiVectorCorpus:
        return self._to_corpus(self._encode(self.model.encode_query, queries), ids)

    def identity(self) -> EncoderIdentity | None:
        """From the registry entry only. The wrapped model's API is unverified in this
        repository, so it is not asked for a dimension or a revision."""
        return _identity_from_info(self.info)


def supported_loaders() -> list[str]:
    """Loader names ``load_adapter(model_id, loader=...)`` accepts for a model not in the registry."""
    from ..encoders.colvlm import BACKENDS
    from ..encoders.late import COLENGINE_FAMILIES, RECIPES

    engine = sorted({*BACKENDS, *COLENGINE_FAMILIES})
    return [*(f"colpali-engine:{name}" for name in engine), "nemotron",
            *(f"remote-code:{name}" for name in sorted(RECIPES)), "pylate", "sentence-transformers"]


def load_adapter(model_id: str, loader: str | None = None, **kwargs: Any) -> EncoderAdapter:
    """Build the adapter a model id calls for.

    The registry entry of ``model_id`` names its loader; ``loader`` overrides it and is how a
    checkpoint that is not in the registry (a fine-tune of a known family, say) is loaded:
    ``load_adapter("org/my-colqwen", loader="colpali-engine:colqwen2.5")``. An id that is
    neither registered nor given a loader goes through sentence-transformers, as before.

    ``colpali-engine:<family>``
        the repository's own page encoder for the families whose patch grid it knows
        (ColPali, ColQwen2, ColQwen2.5, ColSmol), which keeps the layout the pixel-space
        stage needs; :class:`~optivision.encoders.late.ColEngineTokenEncoder` (token vectors
        only) for the other ``Col*`` classes of colpali-engine.
    ``nemotron``
        :class:`~optivision.adapters.nemotron.NemotronAdapter`, with the registry's width as
        the width the checkpoint must have.
    ``remote-code:<recipe>``
        :class:`~optivision.encoders.late.RemoteCodeEncoder`. The checkpoint's own code runs
        on load, so ``trust_remote_code=True`` has to be passed here, and the revision is the
        registry's pin unless ``revision=`` says otherwise.
    ``pylate``, ``sentence-transformers``
        text ColBERT models through those libraries.

    Which of these paths this repository has run is the registry status of each model; see
    :func:`optivision.adapters.check_retriever` for the check to run on the others.
    """
    info = MODELS.get(model_id)
    name = loader if loader is not None else (info.loader if info is not None else "sentence-transformers")
    width = info.dimension if info is not None else None
    for key, value in info.load_options if info is not None else ():
        kwargs.setdefault(key, value)
    if name.startswith("colpali-engine:"):
        from ..encoders.colvlm import BACKENDS

        family = name.split(":", 1)[1]
        if family in BACKENDS:
            from ..encoders.colvlm import ColVLMEncoder  # needs torch + colpali-engine

            return PageEncoderAdapter(ColVLMEncoder(backend=family, model_name=model_id, **kwargs), info=info)
        from ..encoders.late import ColEngineTokenEncoder
        from .tokens import TokenEncoderAdapter

        kwargs.setdefault("expected_dim", width)
        return TokenEncoderAdapter(ColEngineTokenEncoder(model_id, family, **kwargs), info=info)
    if name == "nemotron":
        from ..encoders.nemotron import NemotronColEmbedEncoder  # needs torch + transformers

        kwargs.setdefault("expected_dim", width)
        return NemotronAdapter(NemotronColEmbedEncoder(model_id, **kwargs), info=info)
    if name.startswith("remote-code:"):
        from ..encoders.late import RemoteCodeEncoder
        from .tokens import TokenEncoderAdapter

        kwargs.setdefault("expected_dim", width)
        kwargs.setdefault("known_revision", info.revision if info is not None else None)
        return TokenEncoderAdapter(RemoteCodeEncoder(model_id, name.split(":", 1)[1], **kwargs), info=info)
    if name == "pylate":
        from ..encoders.late import PyLateEncoder
        from .tokens import TokenEncoderAdapter

        kwargs.setdefault("expected_dim", width)
        return TokenEncoderAdapter(PyLateEncoder(model_id, **kwargs), info=info, modality="text")
    if name == "sentence-transformers":
        return SentenceTransformersAdapter(model_id, **kwargs)
    raise ValueError(f"unknown loader {name!r}; choose from {supported_loaders()}")


# The adapters below subclass ``EncoderAdapter``, so they are imported once it exists.
from .arrays import ArrayAdapter, CallableAdapter, corpus_from_arrays  # noqa: F401
from .measured import measured_models, measured_results, measured_shortlist  # noqa: F401
from .nemotron import NemotronAdapter
from .tokens import TokenEncoderAdapter, check_retriever  # noqa: F401
