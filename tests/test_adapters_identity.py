"""What an adapter says about the model behind its vectors, and what it refuses to guess."""

from __future__ import annotations

import numpy as np
import pytest

from optivision.adapters import (
    MODELS,
    EncoderAdapter,
    ModelInfo,
    PageEncoderAdapter,
    SentenceTransformersAdapter,
    supported_models,
)
from optivision.core.vectors import MODALITIES, EncoderIdentity, MultiVectorCorpus
from optivision.encoders import SyntheticEncoder
from optivision.testing import check_adapter
from optivision.types import PageRef

from .conftest import make_page

#: the registry as this branch found it; a status changes only with a committed measurement
STATUSES = {
    "vidore/colpali-v1.3-merged": "measured",
    "vidore/colSmol-256M": "measured",
    "vidore/colSmol-500M": "untested",
    "vidore/colqwen2-v1.0-merged": "measured",
    "vidore/colqwen2-v1.0": "measured",
    "vidore/colqwen2.5-v0.2": "measured",
    "nvidia/nemotron-colembed-vl-4b-v2": "measured",
    "nvidia/nemotron-colembed-vl-8b-v2": "measured",  # run committed 2026-10-12 (reports/universal/wide)
    "nomic-ai/colnomic-embed-multimodal-7b": "untested",  # its card loads it with the measured ColQwen2_5 class
    "colbert-ir/colbertv2.0": "unverified",
    "answerdotai/answerai-colbert-small-v1": "measured",
    "lightonai/GTE-ModernColBERT-v1": "untested",  # passed check_retriever here 2026-10-12
}


def _info(**changes) -> ModelInfo:
    args = {"model_id": "org/model", "family": "fam", "modality": "text", "loader": "x",
            "dimension": 32, "status": "unverified", "evidence": "none"}
    args.update(changes)
    return ModelInfo(**args)


class _Minimal(EncoderAdapter):
    """A subclass written before identity() and dimension existed."""

    def encode_documents(self, documents, ids=None):
        arrays = [np.ones((1, 4), np.float32) for _ in documents]
        return MultiVectorCorpus.from_arrays(arrays, ids=ids)

    def encode_queries(self, queries, ids=None):
        arrays = [np.ones((1, 4), np.float32) for _ in queries]
        return MultiVectorCorpus.from_arrays(arrays, ids=ids)


# ------------------------------------------------------- the base class defaults


def test_a_subclass_that_knows_nothing_has_no_identity_and_no_dimension():
    adapter = _Minimal()
    assert adapter.info is None and adapter.identity() is None and adapter.dimension is None


def test_the_default_identity_is_built_from_info_and_states_nothing_else():
    class WithInfo(_Minimal):
        info = _info(model_id="vidore/colqwen2-v1.0", family="colqwen", modality="image",
                     loader="colpali-engine:colqwen2", dimension=128, status="measured")

    identity = WithInfo().identity()
    assert isinstance(identity, EncoderIdentity)
    assert (identity.model_id, identity.dimension, identity.modality) == (
        "vidore/colqwen2-v1.0", 128, "image")
    assert identity.revision is None and identity.normalized is None
    assert identity.inference_dtype is None
    assert identity.extra == {"family": "colqwen", "loader": "colpali-engine:colqwen2"}
    assert WithInfo().dimension == 128


def test_the_registry_status_is_not_part_of_an_identity():
    class WithInfo(_Minimal):
        info = _info(status="measured", evidence="a long piece of prose with revision 0ed152d9")

    identity = WithInfo().identity()
    assert "status" not in identity.to_dict() and "measured" not in str(identity.to_dict())
    assert identity.revision is None  # the evidence text is prose; it is never parsed


@pytest.mark.parametrize("dimension", [None, 0, -4])
def test_without_a_known_dimension_there_is_no_identity(dimension):
    class WithInfo(_Minimal):
        info = _info(dimension=dimension)

    assert WithInfo().identity() is None and WithInfo().dimension is None


def test_an_unexpected_modality_is_described_as_other():
    class WithInfo(_Minimal):
        info = _info(modality="audio")

    assert WithInfo().identity().modality == "other"
    assert "other" in MODALITIES


def test_a_subclass_may_assign_or_override_dimension_as_it_could_before():
    class Assigns(_Minimal):
        def __init__(self):
            self.dimension = 64

    class Declares(_Minimal):
        dimension = 32

    class Property(_Minimal):
        @property
        def dimension(self):
            return 16

    assert Assigns().dimension == 64 and Declares().dimension == 32 and Property().dimension == 16
    assert Assigns().identity() is None  # the assignment does not invent an identity


def test_every_registry_entry_gives_an_identity_without_invented_facts():
    for info in supported_models():
        class Adapter(_Minimal):
            pass

        Adapter.info = info
        identity = Adapter().identity()
        if info.dimension is None:  # no file of the model states its width: it is not guessed
            assert identity is None
            continue
        assert identity is not None and identity.dimension == info.dimension
        assert identity.model_id == info.model_id and identity.modality == info.modality
        assert identity.revision is None and identity.normalized is None
        assert identity.inference_dtype is None


def test_the_registry_statuses_have_not_moved():
    """The twelve entries of v0.3.0, and every status of the entries added since."""
    statuses = {m.model_id: m.status for m in supported_models()}
    assert {k: statuses[k] for k in STATUSES} == STATUSES
    added = {k: v for k, v in statuses.items() if k not in STATUSES}
    measured = {k for k, v in added.items() if v == "measured"}
    assert measured == {"athrael-soju/colqwen3.5-4.5B-v3", "TomoroAI/tomoro-colqwen3-embed-4b"}
    assert len(added) == 53 and set(added.values()) == {"measured", "untested", "unverified"}
    assert all(m.evidence for m in supported_models())


def test_an_entry_that_was_not_measured_here_says_so_and_names_its_revision():
    for info in supported_models():
        if info.model_id in STATUSES:
            continue
        assert info.revision and len(info.revision) == 40, info.model_id
        if info.status != "measured":
            assert "never" in info.evidence, info.model_id  # "never encoded here" / "never run here"
        assert info.dimension is None or info.dimension > 0
    without_width = [m.model_id for m in supported_models() if m.dimension is None]
    assert without_width == ["nvidia/llama-nemoretriever-colembed-1b-v1",
                             "nvidia/llama-nemoretriever-colembed-3b-v1"]


def test_single_vector_and_untrained_checkpoints_are_not_registered():
    """Checked on their model cards 2026-10-12: dense (one vector per item) or an untrained base."""
    for model_id in ("nomic-ai/nomic-embed-multimodal-3b", "nomic-ai/nomic-embed-multimodal-7b",
                     "vidore/ColSmolVLM-Instruct-256M-base"):
        assert model_id not in MODELS


# ------------------------------------------------------------- PageEncoderAdapter


def test_a_page_adapter_reads_what_the_wrapped_encoder_states():
    identity = PageEncoderAdapter(SyntheticEncoder(dim=16, grid=8)).identity()
    assert (identity.model_id, identity.dimension, identity.modality) == ("synthetic", 16, "image")
    assert identity.revision is None and identity.normalized is None
    assert identity.inference_dtype is None and identity.extra == {"encoder": "synthetic"}


class _Encoder:
    """The attributes a page encoder may carry, each one optional."""

    def __init__(self, **attributes):
        self.__dict__.update(attributes)


def test_the_encoders_own_facts_come_before_the_registry_and_nothing_is_added():
    info = MODELS["vidore/colqwen2-v1.0-merged"]
    encoder = _Encoder(checkpoint="vidore/colqwen2-v1.0-merged", name="colqwen2", dim=128,
                       torch_dtype="torch.bfloat16", revision="abc123")
    identity = PageEncoderAdapter(encoder, info=info).identity()
    assert identity.model_id == "vidore/colqwen2-v1.0-merged" and identity.dimension == 128
    assert identity.revision == "abc123" and identity.inference_dtype == "bfloat16"
    assert identity.extra == {"family": "colqwen", "encoder": "colqwen2"}
    assert identity.normalized is None  # a BaseEncoder is asked for unit norm, not verified
    bare = PageEncoderAdapter(_Encoder(checkpoint="x/y", dim=8)).identity()
    assert bare.revision is None and bare.inference_dtype is None  # never guessed


def test_a_dimension_the_encoder_cannot_give_falls_back_to_the_registry_or_to_nothing():
    class Broken:
        checkpoint = "x/y"

        @property
        def dim(self):
            raise RuntimeError("model not loaded")

    assert PageEncoderAdapter(Broken()).identity() is None
    info = _info(model_id="x/y", dimension=48, modality="image")
    assert PageEncoderAdapter(Broken(), info=info).identity().dimension == 48
    assert PageEncoderAdapter(_Encoder(dim="128"), info=_info(dimension=None)).identity() is None


def test_a_page_adapter_with_a_deterministic_encoder_conforms():
    adapter = PageEncoderAdapter(SyntheticEncoder(dim=16, grid=8, noise=0.0))
    pages = [(PageRef("a", 1), make_page()), (PageRef("b", 1), make_page(ink_rows=1))]
    assert check_adapter(adapter, pages, ["tax invoice", "memo"]) == []
    assert adapter.dimension == 16


# ------------------------------------------------------- SentenceTransformersAdapter


class _StubModel:
    def encode_document(self, inputs):
        return [np.ones((2, 96), np.float32) for _ in inputs]

    def encode_query(self, texts):
        return [np.ones((3, 96), np.float32) for _ in texts]


def test_a_registered_text_model_gets_its_identity_from_the_registry_only():
    model_id = "answerdotai/answerai-colbert-small-v1"
    adapter = SentenceTransformersAdapter(model_id, model=_StubModel())
    identity = adapter.identity()
    assert (identity.model_id, identity.dimension, identity.modality) == (
        "answerdotai/answerai-colbert-small-v1", 96, "text")
    assert identity.revision is None and identity.normalized is None
    assert adapter.dimension == 96
    assert check_adapter(adapter, ["a document", "another"], ["a query"]) == []


def test_an_unregistered_model_has_no_identity_because_its_dimension_is_not_asked_for():
    with pytest.warns(UserWarning, match="unverified"):
        adapter = SentenceTransformersAdapter("some/unknown-model", model=_StubModel())
    assert adapter.identity() is None and adapter.dimension is None


# ------------------------------------------------------------------ 0.4.1 regressions


def test_page_and_text_adapter_corpora_carry_the_encoder_identity():
    adapter = PageEncoderAdapter(SyntheticEncoder(dim=16, grid=8, noise=0.0))
    pages = [(PageRef("a", 1), make_page()), (PageRef("b", 1), make_page(ink_rows=1))]
    docs = adapter.encode_documents(pages)
    assert docs.attrs["encoder_identity"] == adapter.identity().to_dict()
    assert adapter.encode_queries(["memo"]).attrs["encoder_identity"]["dimension"] == 16
    text = SentenceTransformersAdapter("answerdotai/answerai-colbert-small-v1", model=_StubModel())
    attrs = text.encode_documents(["a document"]).attrs
    assert attrs["encoder_identity"]["model_id"] == "answerdotai/answerai-colbert-small-v1"
    assert text.encode_queries(["q"]).attrs["encoder_identity"] == text.identity().to_dict()
