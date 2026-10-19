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


_MODELS: list[ModelInfo] = [
    ModelInfo("vidore/colpali-v1.3-merged", "colpali", "image", "colpali-engine:colpali", 128,
              MEASURED, "E2/E3: reports/colpali_*; replayed 2026-09-27 (docs/AUDIT-2026-09-27.md)"),
    ModelInfo("vidore/colSmol-256M", "colsmol", "image", "colpali-engine:colsmol", 128,
              MEASURED, "E1/E4: reports/colsmol*; replayed 2026-09-27"),
    ModelInfo("vidore/colSmol-500M", "colsmol", "image", "colpali-engine:colsmol-500m", 128,
              UNTESTED, "same colpali-engine class as ColSmol-256M; never encoded here"),
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
              UNTESTED, "same colpali-engine class (ColQwen2_5) as the measured vidore/colqwen2.5-v0.2 (its model "
              "card loads it that way); never encoded here; model card and repository files read 2026-10-12",
              license="apache-2.0", revision="09dbc9502b66605d5be56d2226019b49c9fd3293",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("colbert-ir/colbertv2.0", "colbert", "text", "sentence-transformers", 128,
              UNVERIFIED, "text late interaction; loading path not exercised here"),
    ModelInfo("answerdotai/answerai-colbert-small-v1", "colbert", "text", "sentence-transformers", 96,
              MEASURED, "BEIR SciFact (5,183 docs / 300 queries) via sentence-transformers 6.1.0 on "
              "2026-09-27; float nDCG@10 74.56 vs 74.77 on the model card; reports/universal/text"),
    ModelInfo("lightonai/GTE-ModernColBERT-v1", "colbert", "text", "sentence-transformers", 128,
              UNTESTED, "check_retriever passed here on 2026-10-12 (8 text pairs, sentence-transformers "
              "6.1.0, CPU: 8 of 8 queries ranked their own document first, 128-d unit-norm "
              "vectors; reports/universal/loader_checks/2026-10-12.json); compression never "
              "measured here; also loads with loader='pylate' (the same check passed with PyLate)"),
    # ---- added 2026-10-12: facts read from each model's card and repository files; none was benchmarked here
    ModelInfo("vidore/colpali", "colpali", "image", "colpali-engine:colpali", 128,
              UNTESTED, "same colpali-engine class (ColPali) as the measured vidore/colpali-v1.3-merged; "
              "never encoded here; model card and repository files read 2026-10-12",
              license="mit", revision="7754422bfa37f8e3cd771e13d4d7f07de3103c95",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("vidore/colpali-v1.1", "colpali", "image", "colpali-engine:colpali", 128,
              UNTESTED, "same colpali-engine class (ColPali) as the measured vidore/colpali-v1.3-merged; "
              "never encoded here; model card and repository files read 2026-10-12",
              license="mit", revision="855a60dd9e466987f27c385609710152f8868f67",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("vidore/colpali-v1.2", "colpali", "image", "colpali-engine:colpali", 128,
              UNTESTED, "same colpali-engine class (ColPali) as the measured vidore/colpali-v1.3-merged; "
              "never encoded here; model card and repository files read 2026-10-12",
              license="mit", revision="6e6cf48566f6fbb8eb429a4bf90f39006d703317",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("vidore/colpali-v1.2-merged", "colpali", "image", "colpali-engine:colpali", 128,
              UNTESTED, "same colpali-engine class (ColPali) as the measured vidore/colpali-v1.3-merged; "
              "never encoded here; model card and repository files read 2026-10-12",
              license="gemma", revision="cd3977f763d2495ec7cfb605fb9f5aec65330509"),
    ModelInfo("vidore/colpali-v1.3", "colpali", "image", "colpali-engine:colpali", 128,
              UNTESTED, "same colpali-engine class (ColPali) as the measured vidore/colpali-v1.3-merged; "
              "never encoded here (the measured checkpoint is the pre-merged copy of this "
              "adapter); model card and repository files read 2026-10-12",
              license="mit", revision="b5c6dd62125326e6f0b540c1f7b36e901cdc0a11",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("vidore/colpali-v1.2-hf", "colpali", "image", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; the "
              "image path of that adapter has never been run here; model card and repository files "
              "read 2026-10-12",
              license="gemma", revision="3c712e95d5557c4d72a4107659cee01a40c1ece6",
              notes="transformers-native weights (ColPaliForRetrieval), not a colpali-engine "
                    "checkpoint"),
    ModelInfo("vidore/colpali-v1.3-hf", "colpali", "image", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; the "
              "image path of that adapter has never been run here; model card and repository files "
              "read 2026-10-12",
              license="gemma", revision="133a9eb02947310513f52f8f0d39d622e0eab8dc",
              notes="transformers-native weights (ColPaliForRetrieval), not a colpali-engine "
                    "checkpoint"),
    ModelInfo("vidore/colqwen2-v0.1", "colqwen", "image", "colpali-engine:colqwen2", 128,
              UNTESTED, "same colpali-engine class (ColQwen2) as the measured vidore/colqwen2-v1.0; never "
              "encoded here; model card and repository files read 2026-10-12",
              license="apache-2.0", revision="38b8ebb6198e2704bc24fe0027b3743f25f03bfc",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("vidore/colqwen2-v1.0-hf", "colqwen", "image", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; the "
              "image path of that adapter has never been run here; model card and repository files "
              "read 2026-10-12",
              license="apache-2.0", revision="ddc07d2317c80f75fc742b7362ee9ad1912908f9",
              notes="transformers-native weights (ColQwen2ForRetrieval), not a colpali-engine "
                    "checkpoint"),
    ModelInfo("vidore/colqwen2.5-v0.1", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              UNTESTED, "same colpali-engine class (ColQwen2_5) as the measured vidore/colqwen2.5-v0.2; "
              "never encoded here; model card and repository files read 2026-10-12",
              license="mit", revision="4379ecfd4c9446836ad0ef4dd7390b1c748f68d8",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("vidore/colqwen-omni-v0.1", "colqwen", "image", "colpali-engine:colqwen2.5-omni", 128,
              UNVERIFIED, "colpali-engine ColQwen2_5Omni per its model card; token-vector loader written "
              "against the card, never run here; model card and repository files read 2026-10-12",
              license="mit", revision="c0f50696f483b12247c3a4bc23a524b29aba2c13",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("nomic-ai/colnomic-embed-multimodal-3b", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              UNTESTED, "same colpali-engine class (ColQwen2_5) as the measured vidore/colqwen2.5-v0.2; "
              "never encoded here; model card and repository files read 2026-10-12",
              license=None, revision="86627b4a9b0cade577851a70afa469084f9863a4",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("tsystems/colqwen2.5-3b-multilingual-v1.0", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              UNTESTED, "same colpali-engine class (ColQwen2_5) as the measured vidore/colqwen2.5-v0.2; "
              "never encoded here; model card and repository files read 2026-10-12",
              license="mit", revision="d5bd3db7ca57a1de3dc862828f343904ca07efd9",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("tsystems/colqwen2.5-3b-multilingual-v1.0-merged", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              UNTESTED, "same colpali-engine class (ColQwen2_5) as the measured vidore/colqwen2.5-v0.2; "
              "never encoded here; model card and repository files read 2026-10-12",
              license="mit", revision="db0d27eb032de3fa5bcd626a6558e7a560a2beb8"),
    ModelInfo("Metric-AI/ColQwen2.5-3b-multilingual-v1.0", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              UNTESTED, "same colpali-engine class (ColQwen2_5) as the measured vidore/colqwen2.5-v0.2; "
              "never encoded here; model card and repository files read 2026-10-12",
              license="mit", revision="e2a1c05d053dcf4ad6e39b6c48ced9d6a81071f0",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("Metric-AI/ColQwen2.5-7b-multilingual-v1.0", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              UNTESTED, "same colpali-engine class (ColQwen2_5) as the measured vidore/colqwen2.5-v0.2; "
              "never encoded here; model card and repository files read 2026-10-12",
              license="mit", revision="8874f274c038e7ddd093b5ea4b33218db710fa5f",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1", "colqwen", "image", "colpali-engine:colqwen2.5", 128,
              UNTESTED, "same colpali-engine class (ColQwen2_5) as the measured vidore/colqwen2.5-v0.2; "
              "never encoded here; model card and repository files read 2026-10-12",
              license="apache-2.0", revision="b4c6d894958789e861db16d444cd811b8eb1be75"),
    ModelInfo("vidore/colsmolvlm-v0.1", "colsmol", "image", "colpali-engine:colsmol", 128,
              UNTESTED, "same colpali-engine class (ColIdefics3) as the measured vidore/colSmol-256M; never "
              "encoded here; model card and repository files read 2026-10-12",
              license=None, revision="c14f4090e6ac69f0420d4ddb4425d28f8cd9cc80",
              notes="adapter-only repository: merged explicitly over its base model on load"),
    ModelInfo("ModernVBERT/colmodernvbert", "colmodernvbert", "image", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; the "
              "image path of that adapter has never been run here; model card and repository files "
              "read 2026-10-12",
              license="mit", revision="4c604428a6e1a7e33d0fb34837f0cdecbc22026e",
              notes="the colpali-engine route of the card (ColModernVBert) needs "
                    "trust_remote_code; loader='colpali-engine:colmodernvbert' is available and "
                    "equally unrun"),
    ModelInfo("TomoroAI/tomoro-colqwen3-embed-8b", "colqwen3", "image", "remote-code:tomoro", 320,
              UNTESTED, "same remote code and calls as the measured TomoroAI/tomoro-colqwen3-embed-4b; never "
              "encoded here; model card and repository files read 2026-10-12",
              license="apache-2.0", revision="7901e21544bf360868ead6b384b37b3abdf119cf",
              notes="runs the repository's own modelling code: pass trust_remote_code=True"),
    ModelInfo("vultr/VultronRetrieverPrime-Qwen3.5-8B", "colqwen3.5", "image", "colpali-engine:colqwen3.5", 320,
              UNVERIFIED, "colpali-engine ColQwen3_5 per its model card; never run here; model card and "
              "repository files read 2026-10-12",
              license="apache-2.0", revision="1201db5ae23bc4d5977c188748f35cf3be54c568",
              notes="the card pins a colpali-engine git commit and needs causal-conv1d and "
                    "flash-linear-attention"),
    ModelInfo("vultr/VultronRetrieverCore-Qwen3.5-4.5B", "colqwen3.5", "image", "colpali-engine:colqwen3.5", 320,
              UNVERIFIED, "colpali-engine ColQwen3_5 per its model card; never run here; model card and "
              "repository files read 2026-10-12",
              license="apache-2.0", revision="b0857609ad6db741009a7026e02cf30bd384f54d",
              notes="the card pins a colpali-engine git commit and needs causal-conv1d and "
                    "flash-linear-attention"),
    ModelInfo("vultr/VultronRetrieverFlash-Qwen3.5-0.8B", "colqwen3.5", "image", "colpali-engine:colqwen3.5", 320,
              UNVERIFIED, "colpali-engine ColQwen3_5 per its model card; never run here; model card and "
              "repository files read 2026-10-12",
              license="apache-2.0", revision="6f6b6757e9ea3ad93f4f2cbd839bc6f2f0b93739",
              notes="the card pins a colpali-engine git commit and needs causal-conv1d and "
                    "flash-linear-attention"),
    ModelInfo("tencent/EVIE-4.5B", "colqwen3.5", "image", "sentence-transformers", 2048,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; the "
              "image path of that adapter has never been run here; model card and repository files "
              "read 2026-10-12",
              license="apache-2.0", revision="498388f8f4c92af8d99fffeececd56c0a4c428d0",
              notes="2,048-d vectors (the card describes shorter prefixes as usable); its colpali "
                    "route needs the fork bundled in the repository"),
    ModelInfo("tencent/EVIE-8B", "colqwen3.5", "image", "sentence-transformers", 4096,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; the "
              "image path of that adapter has never been run here; model card and repository files "
              "read 2026-10-12",
              license="apache-2.0", revision="cdc8f60e56d168591354cf919bb5a50a5d1f617b"),
    ModelInfo("VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1", "colqwen3", "image", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; the "
              "image path of that adapter has never been run here; model card and repository files "
              "read 2026-10-12",
              license="apache-2.0", revision="60a0200a0d9c76df7acee02d7fd074db8c81cf00"),
    ModelInfo("VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1", "colqwen3", "image", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; the "
              "image path of that adapter has never been run here; model card and repository files "
              "read 2026-10-12",
              license="apache-2.0", revision="de6b8ac536cae74fab7595de5d53e24f5e6c4295"),
    ModelInfo("webAI-Official/webAI-ColVec1-4b", "colvec", "image", "remote-code:colvec", 640,
              UNVERIFIED, "remote-code recipe written against the model card (process_images / "
              "process_queries); never run here; model card and repository files read 2026-10-12",
              license="other", revision="cb5af9aab4894bfa1c7f2427d6970f19055c7226",
              notes="runs the repository's own modelling code: pass trust_remote_code=True"),
    ModelInfo("webAI-Official/webAI-ColVec1.1-4b", "colvec", "image", "sentence-transformers", 640,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; the "
              "image path of that adapter has never been run here; model card and repository files "
              "read 2026-10-12",
              license="other", revision="d88d0155d700b06382124ff985ddb07099155bb4",
              notes="runs the repository's own modelling code: pass trust_remote_code=True"),
    ModelInfo("webAI-Official/webAI-ColVec1.1-8b", "colvec", "image", "sentence-transformers", 640,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; the "
              "image path of that adapter has never been run here; model card and repository files "
              "read 2026-10-12",
              license="other", revision="0f712a2777fbba29b2a21bb68ffcfbb5272505ad",
              notes="runs the repository's own modelling code: pass trust_remote_code=True"),
    ModelInfo("Hcompany/NeoMME-260M-Retriever-ST-late", "neomme", "image", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; the "
              "image path of that adapter has never been run here; model card and repository files "
              "read 2026-10-12",
              license="apache-2.0", revision="023be2a8ab9d797f5aa76f5bf8b5dde78d819659"),
    ModelInfo("Hcompany/NeoMME-800M-Retriever-ST-late", "neomme", "image", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; the "
              "image path of that adapter has never been run here; model card and repository files "
              "read 2026-10-12",
              license="apache-2.0", revision="ad4c17419c28745c3b3045981b0d8abb72cd142b"),
    ModelInfo("Cognitive-Lab/ColNetraEmbed", "colgemma", "image", "colpali-engine:colgemma3", 128,
              UNVERIFIED, "colpali-engine ColGemma3 per its model card; never run here; model card and "
              "repository files read 2026-10-12",
              license="gemma", revision="a27fe0bdcf65a54116393853621f7f80e3cd9ca3",
              notes="the card installs a fork of colpali-engine (github.com/adithya-s-k/colpali)"),
    ModelInfo("ibm-granite/granite-vision-3.3-2b-embedding", "granite-vision", "image", "remote-code:processor", 128,
              UNVERIFIED, "remote-code recipe written against the model card (process_images / "
              "process_queries); never run here; model card and repository files read 2026-10-12",
              license="apache-2.0", revision="baa53d1e1ac95baefb15e4aa9fdcc471ac728c83",
              notes="runs the repository's own modelling code: pass trust_remote_code=True; the "
                    "card pins transformers==4.50"),
    ModelInfo("5CD-AI/Vintern-Embedding-1B", "colinternvl", "image", "remote-code:processor", 128,
              UNVERIFIED, "remote-code recipe written against the model card (process_images / "
              "process_queries); never run here; model card and repository files read 2026-10-12",
              license=None, revision="ee3f8a72164d194c0885df3f9be3d4a3f0626cc9",
              notes="runs the repository's own modelling code: pass trust_remote_code=True; the "
                    "card pins transformers==4.48.2 and casts its inputs to bfloat16 by hand"),
    ModelInfo("jinaai/jina-embeddings-v4", "jina-v4", "image", "remote-code:jina-v4", 128,
              UNVERIFIED, "remote-code recipe written against the model card (encode_image / encode_text with "
              "return_multivector=True); never run here; model card and repository files read "
              "2026-10-12",
              license=None, revision="853c867b65b749f3c3c72a06868140d842e04f06",
              notes="runs the repository's own modelling code: pass trust_remote_code=True; "
                    "single-vector by default, the recipe asks for the 128-d multi-vector output; "
                    "the card names the Qwen Research License"),
    ModelInfo("nvidia/llama-nemoretriever-colembed-1b-v1", "nemoretriever-colembed", "image", "remote-code:nvidia-colembed", None,
              UNVERIFIED, "remote-code recipe written against the model card (forward_images / "
              "forward_queries); never run here; model card and repository files read 2026-10-12",
              license="other", revision="78361b5fa933f700305cfab3accae44b759de590",
              notes="runs the repository's own modelling code: pass trust_remote_code=True; the "
                    "card requires transformers==4.49.0 and flash-attn; no file of the repository "
                    "states the output width"),
    ModelInfo("nvidia/llama-nemoretriever-colembed-3b-v1", "nemoretriever-colembed", "image", "remote-code:nvidia-colembed", None,
              UNVERIFIED, "remote-code recipe written against the model card (forward_images / "
              "forward_queries); never run here; model card and repository files read 2026-10-12",
              license="other", revision="75f7e7311c7a5965c535587fd5e563c5e4a3c583",
              notes="runs the repository's own modelling code: pass trust_remote_code=True; the "
                    "card requires transformers==4.49.0 and flash-attn; no file of the repository "
                    "states the output width"),
    ModelInfo("nvidia/llama-nemotron-colembed-vl-3b-v2", "nemoretriever-colembed", "image", "remote-code:nvidia-colembed", 3072,
              UNVERIFIED, "remote-code recipe written against the model card (forward_images / "
              "forward_queries); never run here; model card and repository files read 2026-10-12",
              license="other", revision="9907f8fdf1eae6802c5b9ee4287ef9009acfdae9",
              notes="runs the repository's own modelling code: pass trust_remote_code=True; the "
                    "card requires flash-attn"),
    ModelInfo("lightonai/colbertv2.0", "colbert", "text", "sentence-transformers", 128,
              UNTESTED, "check_retriever passed here on 2026-10-12 (8 text pairs, sentence-transformers "
              "6.1.0, CPU: 8 of 8 queries ranked their own document first, 128-d unit-norm "
              "vectors; reports/universal/loader_checks/2026-10-12.json); compression never "
              "measured here; loads through sentence-transformers MultiVectorEncoder per its "
              "model card; model card and repository files read 2026-10-12",
              license="mit", revision="0b9b1e380096a3b72b6b4877f68d59effb62e0f2",
              notes="the sentence-transformers conversion of colbert-ir/colbertv2.0"),
    ModelInfo("jinaai/jina-colbert-v2", "colbert", "text", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; this "
              "checkpoint has never been run here; model card and repository files read 2026-10-12",
              license="cc-by-nc-4.0", revision="a9dc5cd7293d4c71dbbba04829923ba4d0e4f6ea",
              notes="runs the repository's own modelling code: pass trust_remote_code=True; needs "
                    "einops"),
    ModelInfo("lightonai/Reason-ModernColBERT", "colbert", "text", "sentence-transformers", 128,
              UNTESTED, "check_retriever passed here on 2026-10-12 (8 text pairs, sentence-transformers "
              "6.1.0, CPU: 8 of 8 queries ranked their own document first, 128-d unit-norm "
              "vectors; reports/universal/loader_checks/2026-10-12.json); compression never "
              "measured here; loads through sentence-transformers MultiVectorEncoder per its "
              "model card; model card and repository files read 2026-10-12",
              license="cc-by-nc-4.0", revision="e0bd6968566909365aa553cb54f2219c2d851ed0",
              notes="also loads with loader='pylate'"),
    ModelInfo("lightonai/ColBERT-Zero", "colbert", "text", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; this "
              "checkpoint has never been run here; model card and repository files read 2026-10-12",
              license="apache-2.0", revision="43aefda07cf4fea7a13dbdab0eef8ecf78580d3d",
              notes="also loads with loader='pylate'"),
    ModelInfo("lightonai/LateOn", "colbert", "text", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; this "
              "checkpoint has never been run here; model card and repository files read 2026-10-12",
              license="apache-2.0", revision="62911e105059585d244384c7d17826e35f669c17",
              notes="also loads with loader='pylate'"),
    ModelInfo("lightonai/mLateOn", "colbert", "text", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; this "
              "checkpoint has never been run here; model card and repository files read 2026-10-12",
              license="apache-2.0", revision="bbd883ae1fadb563056ff73b1061ae8172a3fcda",
              notes="also loads with loader='pylate'"),
    ModelInfo("mixedbread-ai/mxbai-edge-colbert-v0-17m", "colbert", "text", "sentence-transformers", 48,
              UNTESTED, "check_retriever passed here on 2026-10-12 (8 text pairs, sentence-transformers "
              "6.1.0, CPU: 8 of 8 queries ranked their own document first, 48-d unit-norm "
              "vectors; reports/universal/loader_checks/2026-10-12.json); compression never "
              "measured here; loads through sentence-transformers MultiVectorEncoder per its "
              "model card; model card and repository files read 2026-10-12",
              license="apache-2.0", revision="592c6417c1c6687572043408ed1ae5196bce16b1",
              notes="also loads with loader='pylate'"),
    ModelInfo("mixedbread-ai/mxbai-edge-colbert-v0-32m", "colbert", "text", "sentence-transformers", 64,
              UNTESTED, "check_retriever passed here on 2026-10-12 (8 text pairs, sentence-transformers "
              "6.1.0, CPU: 8 of 8 queries ranked their own document first, 64-d unit-norm "
              "vectors; reports/universal/loader_checks/2026-10-12.json); compression never "
              "measured here; loads through sentence-transformers MultiVectorEncoder per its "
              "model card; model card and repository files read 2026-10-12",
              license="apache-2.0", revision="bb13a29ec9b1e7edd4ba8f7a0776c48b55cbad66",
              notes="also loads with loader='pylate'"),
    ModelInfo("mixedbread-ai/mxbai-colbert-large-v1", "colbert", "text", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; this "
              "checkpoint has never been run here; model card and repository files read 2026-10-12",
              license="apache-2.0", revision="d092c678f449647d1945569dc17b4d8917a687e0"),
    ModelInfo("LiquidAI/LFM2-ColBERT-350M", "colbert", "text", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; this "
              "checkpoint has never been run here; model card and repository files read 2026-10-12",
              license="other", revision="9d727d7964d3e4fd093a0f01c7287ae814264e31",
              notes="also loads with loader='pylate'"),
    ModelInfo("LiquidAI/LFM2.5-ColBERT-350M", "colbert", "text", "sentence-transformers", 128,
              UNVERIFIED, "loads through sentence-transformers MultiVectorEncoder per its model card; this "
              "checkpoint has never been run here; model card and repository files read 2026-10-12",
              license="other", revision="9772bdf797255d8693b83e84aa98e9b2d36dd0be",
              notes="runs the repository's own modelling code: pass trust_remote_code=True; also "
                    "loads with loader='pylate'"),
    ModelInfo("NeuML/colbert-bert-tiny", "colbert", "text", "sentence-transformers", 128,
              UNTESTED, "check_retriever passed here on 2026-10-12 (8 text pairs, sentence-transformers "
              "6.1.0, CPU: 8 of 8 queries ranked their own document first, 128-d unit-norm "
              "vectors; reports/universal/loader_checks/2026-10-12.json); compression never "
              "measured here; loads through sentence-transformers MultiVectorEncoder per its "
              "model card; model card and repository files read 2026-10-12",
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
        identity = self.identity()
        if identity is not None:
            attrs["encoder_identity"] = identity.to_dict()  # certify() records it from here
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
        return EncoderIdentity(
            model_id=model_id,
            revision=_text(getattr(enc, "revision", None)),
            dimension=dim,
            modality=modality,
            inference_dtype=_dtype_name(getattr(enc, "torch_dtype", None)),
            extra=extra,
        )


class SentenceTransformersAdapter(EncoderAdapter):
    """sentence-transformers ``MultiVectorEncoder`` (>= 6.0).

    Written against the API the model cards document -- ``encode_document(inputs)``
    and ``encode_query(texts)`` returning one ``[n_i, d]`` tensor or array per
    input. Run for real with sentence-transformers 6.1.0 on text: answerai-colbert-small-v1
    on SciFact (its float baseline matched the model card), and ``check_retriever`` runs on
    the text models whose registry evidence says so. The image path has never been run
    here, and every other model remains unverified. Pass ``model``
    to inject an already-constructed encoder (that is how the tests exercise the
    glue without the library).
    """

    def __init__(self, model_id: str, model: Any = None, **kwargs: Any) -> None:
        self.model_id = model_id
        self.info = MODELS.get(model_id)
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
            model = MultiVectorEncoder(model_id, **kwargs)
        self.model = model

    def _to_corpus(self, outputs: Sequence[Any], ids: Sequence[str] | None) -> MultiVectorCorpus:
        arrays = [np.asarray(o.detach().cpu().float().numpy() if hasattr(o, "detach") else o) for o in outputs]
        attrs: dict[str, Any] = {"model": self.model_id, "loader": "sentence-transformers",
                                 "model_status": self.info.status if self.info else UNVERIFIED}
        identity = self.identity()
        if identity is not None:
            attrs["encoder_identity"] = identity.to_dict()  # certify() records it from here
        return MultiVectorCorpus.from_arrays(arrays, ids=ids, attrs=attrs)

    def encode_documents(self, documents: Sequence[Any], ids: Sequence[str] | None = None) -> MultiVectorCorpus:
        return self._to_corpus(self.model.encode_document(list(documents)), ids)

    def encode_queries(self, queries: Sequence[str], ids: Sequence[str] | None = None) -> MultiVectorCorpus:
        return self._to_corpus(self.model.encode_query(list(queries)), ids)

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
    if name.startswith("colpali-engine:"):
        from ..encoders.colvlm import BACKENDS

        family = name.split(":", 1)[1]
        if family in BACKENDS:
            from ..encoders.colvlm import ColVLMEncoder  # needs torch + colpali-engine

            return PageEncoderAdapter(ColVLMEncoder(backend=family, model_name=model_id, **kwargs), info=info)
        from ..encoders.late import ColEngineTokenEncoder
        from .tokens import TokenEncoderAdapter

        kwargs.setdefault("expected_dim", width)
        if info is not None and info.revision is not None:
            kwargs.setdefault("revision", info.revision)  # the commit the registry entry was measured at
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
