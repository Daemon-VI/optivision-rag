"""The NVIDIA ColEmbed adapter and the ``load_adapter`` branch for it, without any weights.

``NemotronColEmbedEncoder`` is built the way ``tests/test_nemotron_encoder.py`` builds it,
with ``transformers`` and the strict loader replaced, but here the replacement returns a
tiny working model and a tiny processor, so ``encode_images`` and ``encode_queries`` run for
real (padding, masking, normalisation) on vectors that no model produced. Nothing is
downloaded and no number here says anything about the real encoder.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from optivision.adapters import (
    MEASURED,
    MODELS,
    UNTESTED,
    UNVERIFIED,
    EncoderAdapter,
    NemotronAdapter,
    PageEncoderAdapter,
    SentenceTransformersAdapter,
    load_adapter,
)
from optivision.core.vectors import (
    ContractError,
    EncoderIdentity,
    MultiVectorCorpus,
    validate_corpus,
)
from optivision.testing import check_adapter
from optivision.types import PageRef

DIM = 8
FOUR_B = "nvidia/nemotron-colembed-vl-4b-v2"
EIGHT_B = "nvidia/nemotron-colembed-vl-8b-v2"


# --------------------------------------------------------------------- the fakes


def _config(width: int):
    return SimpleNamespace(model_type="qwen3_vl_nemotron_embed", pooling="colbert",
                           text_config=SimpleNamespace(hidden_size=width))


@pytest.fixture
def tiny(monkeypatch):
    """Replace the three places the encoder reaches the outside; returns a control dict."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from optivision.encoders import nemotron

    control = {"width": DIM, "loads": []}

    class TinyModel(torch.nn.Module):
        """Hidden state of a token = a fixed random vector per token id."""

        def __init__(self, width):
            super().__init__()
            self.embed = torch.nn.Embedding(64, width)
            with torch.no_grad():
                self.embed.weight.copy_(
                    torch.randn(64, width, generator=torch.Generator().manual_seed(0)))

        def forward(self, input_ids, attention_mask=None):
            return SimpleNamespace(last_hidden_state=self.embed(input_ids))

    class TinyProcessor:
        """A page has one token per 8 pixels of width plus one, a query one per word; shorter
        items in a batch are padded, and padding tokens have id 0 and mask 0."""

        @staticmethod
        def _batch(token_lists):
            width = max(len(t) for t in token_lists)
            ids = torch.zeros(len(token_lists), width, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, tokens in enumerate(token_lists):
                ids[i, : len(tokens)] = torch.tensor(tokens)
                mask[i, : len(tokens)] = 1
            return {"input_ids": ids, "attention_mask": mask}

        def process_documents(self, chunk):
            lists = []
            for item in chunk:
                image = np.asarray(item["image"])
                start = int(image[..., 0].mean())
                lists.append([(start + i) % 64 for i in range(image.shape[1] // 8 + 1)])
            return self._batch(lists)

        def process_queries(self, queries):
            return self._batch([[sum(map(ord, w)) % 64 for w in q.split()] for q in queries])

    def fake_load(model_cls, checkpoint, dtype, **kwargs):
        control["loads"].append({"checkpoint": checkpoint, "dtype": dtype, **kwargs})
        return TinyModel(control["width"])

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        lambda *a, **k: _config(control["width"]))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained",
                        lambda *a, **k: TinyProcessor())
    monkeypatch.setattr(nemotron, "_from_pretrained", fake_load)
    return control


def _page(width: int, shade: int = 200) -> Image.Image:
    return Image.new("RGB", (width, 16), (shade, shade, shade))


def _encoder(**kwargs):
    from optivision.encoders.nemotron import NemotronColEmbedEncoder

    return NemotronColEmbedEncoder(FOUR_B, device="cpu", expected_dim=DIM, **kwargs)


# ------------------------------------------------------- load_adapter's new branch


@pytest.mark.parametrize(("model_id", "width"), [(FOUR_B, 2560), (EIGHT_B, 4096)])
def test_load_adapter_builds_the_nemotron_adapter_for_the_registry_loader(tiny, model_id, width):
    tiny["width"] = width
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter = load_adapter(model_id, device="cpu")
    # the sentence-transformers fall-through would warn that the adapter is "unverified"
    assert not [w for w in caught if "unverified" in str(w.message)]
    assert isinstance(adapter, NemotronAdapter) and isinstance(adapter, EncoderAdapter)
    assert not isinstance(adapter, (SentenceTransformersAdapter, PageEncoderAdapter))
    assert adapter.info is MODELS[model_id]
    assert adapter.encoder.dim == width == MODELS[model_id].dimension
    (load,) = tiny["loads"]
    assert load["checkpoint"] == model_id
    assert load["trust_remote_code"] is True
    from optivision.encoders.nemotron import KNOWN

    assert load["revision"] == KNOWN[model_id][1]  # the pinned revision, never a floating one


def test_load_adapter_takes_the_expected_width_from_the_registry_unless_told(tiny):
    tiny["width"] = DIM  # a checkpoint with another width than the registry says
    with pytest.raises(RuntimeError, match="does not match the expected 2560"):
        load_adapter(FOUR_B, device="cpu")
    adapter = load_adapter(FOUR_B, device="cpu", expected_dim=DIM)
    assert adapter.dimension == DIM


def test_other_models_still_take_their_old_paths():
    with pytest.warns(UserWarning, match="unverified"):
        adapter = load_adapter("some/unregistered-model", model=object())
    assert isinstance(adapter, SentenceTransformersAdapter)


def test_the_registry_entries_follow_the_committed_measurements():
    """Registry honesty: an adapter does not change a status; only a committed measurement does.

    The 8B became ``measured`` with the reports of its run (``reports/universal/wide``,
    2026-10-05/06), and its evidence says what that run was: float16 inference.
    """
    assert MODELS[FOUR_B].status == MEASURED and MODELS[FOUR_B].loader == "nemotron"
    assert MODELS[EIGHT_B].status == MEASURED and MODELS[EIGHT_B].loader == "nemotron"
    assert MODELS[FOUR_B].dimension == 2560 and MODELS[EIGHT_B].dimension == 4096
    assert "never encoded here" not in MODELS[FOUR_B].evidence
    assert "float16 inference" in MODELS[EIGHT_B].evidence
    assert "NEMOTRON_8B.md" in MODELS[EIGHT_B].evidence
    # the last untested entry, lightonai/colbertv2.0, was measured in batch 14a_text_retry
    assert {m.status for m in MODELS.values()} <= {MEASURED, UNTESTED, UNVERIFIED}


# ------------------------------------------------------- the adapter, end to end


def test_documents_come_out_as_a_corpus_without_padding(tiny):
    adapter = NemotronAdapter(_encoder(page_batch_size=3))
    pages = [_page(16), _page(48), _page(32)]  # 3, 7 and 5 tokens: the batch is padded to 7
    corpus = adapter.encode_documents(pages)
    assert corpus.counts.tolist() == [3, 7, 5]
    assert corpus.dimension == DIM and corpus.dtype == np.float32
    validate_corpus(corpus)
    norms = np.linalg.norm(corpus.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)  # padded rows would be zero vectors
    assert corpus.positions is None and corpus.protected is None and corpus.images is None


def test_queries_come_out_as_a_corpus_without_padding(tiny):
    adapter = NemotronAdapter(_encoder(query_batch_size=4))
    corpus = adapter.encode_queries(["a", "two words", "three little words"], ids=["x", "y", "z"])
    assert corpus.ids == ["x", "y", "z"]
    assert corpus.counts.tolist() == [1, 2, 3]
    assert np.allclose(np.linalg.norm(corpus.vectors, axis=1), 1.0, atol=1e-6)


def test_the_same_page_in_a_larger_batch_gets_the_same_vectors(tiny):
    """Padding rows are dropped, not left in: batching must not change a page's vectors."""
    alone = NemotronAdapter(_encoder(page_batch_size=1)).encode_documents([_page(16)])
    batched = NemotronAdapter(_encoder(page_batch_size=2)).encode_documents([_page(16), _page(64)])
    assert np.allclose(alone.vectors, batched[0].vectors, atol=1e-6)


def test_a_page_ref_pair_gives_the_page_id_and_keeps_the_reference(tiny):
    adapter = NemotronAdapter(_encoder())
    refs = [PageRef("report", 1), PageRef("report", 2, source_path="report.pdf")]
    corpus = adapter.encode_documents(list(zip(refs, [_page(16), _page(24)], strict=True)))
    assert corpus.ids == ["report::p1", "report::p2"]
    assert corpus.metadata[1]["page_ref"]["source_path"] == "report.pdf"
    again = adapter.encode_documents(list(zip(refs, [_page(16), _page(24)], strict=True)),
                                     ids=["a", "b"])
    assert again.ids == ["a", "b"]  # explicit ids win


def test_pairs_and_bare_images_cannot_be_mixed_and_a_pair_needs_a_page_ref(tiny):
    adapter = NemotronAdapter(_encoder())
    with pytest.raises(ContractError, match="all page images or all"):
        adapter.encode_documents([_page(16), (PageRef("a", 1), _page(16))])
    with pytest.raises(ContractError, match="must be a PageRef"):
        adapter.encode_documents([("not a ref", _page(16))])


def test_the_corpus_says_where_it_came_from(tiny):
    from optivision.encoders.nemotron import KNOWN

    adapter = NemotronAdapter(_encoder(), info=MODELS[FOUR_B])
    attrs = adapter.encode_queries(["hello"]).attrs
    assert attrs["model"] == FOUR_B and attrs["loader"] == "nemotron"
    assert attrs["revision"] == KNOWN[FOUR_B][1]
    assert attrs["model_status"] == MEASURED  # the registry's word, since info was passed
    assert attrs["encoder_identity"] == adapter.identity().to_dict()
    assert attrs["encoder_info"]["dim"] == DIM and attrs["encoder_info"]["device"] == "cpu"
    without_info = NemotronAdapter(_encoder()).encode_queries(["hello"]).attrs
    assert "model_status" not in without_info  # a status is attached on purpose, never looked up


def test_the_corpus_and_its_provenance_survive_a_save_and_a_load(tiny, tmp_path):
    adapter = NemotronAdapter(_encoder(), info=MODELS[FOUR_B])
    corpus = adapter.encode_documents([_page(16), _page(24)])
    loaded = MultiVectorCorpus.load(corpus.save(tmp_path / "pages.npz"))
    assert np.array_equal(loaded.vectors, corpus.vectors) and loaded.ids == corpus.ids
    assert loaded.attrs["encoder_identity"] == corpus.attrs["encoder_identity"]
    assert loaded.attrs["encoder_info"]["dim"] == DIM and loaded.attrs["model_status"] == MEASURED


def test_the_identity_states_only_what_the_encoder_knows(tiny):
    from optivision.encoders.nemotron import KNOWN

    identity = NemotronAdapter(_encoder()).identity()
    assert isinstance(identity, EncoderIdentity)
    assert (identity.model_id, identity.dimension, identity.modality) == (FOUR_B, DIM, "image")
    assert identity.revision == KNOWN[FOUR_B][1]
    assert identity.inference_dtype == "float32"
    assert identity.normalized is True  # this class normalises every row (checked above)
    assert identity.extra == {"family": "nemotron-colembed", "attn_implementation": "sdpa"}
    assert NemotronAdapter(_encoder()).dimension == DIM


def test_a_real_nemotron_adapter_conforms(tiny):
    adapter = NemotronAdapter(_encoder(page_batch_size=2, query_batch_size=2))
    pages = [_page(16), _page(40), _page(24)]
    assert check_adapter(adapter, pages, ["one", "two words here", "x"]) == []


# ------------------------------------- the adapter with a stub, which needs no torch


class _Stub:
    """The two methods the adapter calls, returning whatever the test says."""

    checkpoint = "stub/encoder"
    dim = 4

    def __init__(self, arrays):
        self.arrays = arrays

    def encode_images(self, images):
        return self.arrays[: len(images)]

    def encode_queries(self, queries):
        return self.arrays[: len(queries)]


def _arrays(dim=4):
    rng = np.random.default_rng(0)
    return [rng.standard_normal((n, dim)).astype(np.float32) for n in (3, 2)]


def test_an_object_without_the_two_methods_is_refused():
    with pytest.raises(TypeError, match="encode_images"):
        NemotronAdapter(object())


def test_a_stub_is_checked_like_any_other_adapter_output():
    adapter = NemotronAdapter(_Stub(_arrays()))
    corpus = adapter.encode_documents([_page(8), _page(8)])
    assert corpus.counts.tolist() == [3, 2]
    identity = adapter.identity()
    assert identity.model_id == "stub/encoder" and identity.dimension == 4
    assert identity.normalized is None and identity.revision is None  # not the real class


def test_the_output_is_refused_when_it_does_not_match_the_declared_width():
    adapter = NemotronAdapter(_Stub(_arrays(dim=6)))  # the stub says 4 and returns 6
    with pytest.raises(ContractError, match="has dimension 6"):
        adapter.encode_documents([_page(8), _page(8)])


def test_non_finite_output_is_refused():
    arrays = _arrays()
    arrays[1][0, 0] = np.nan
    with pytest.raises(ContractError, match="non-finite"):
        NemotronAdapter(_Stub(arrays)).encode_queries(["a", "b"])
