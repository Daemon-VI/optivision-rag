"""The layout-free encoders and their adapter, without any weights.

Tiny stand-in models and processors are injected, so the padding, masking and conversion
code runs for real on vectors no model produced. Nothing is downloaded and no number here
says anything about a real encoder.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from optivision.adapters import (
    MODELS,
    TokenEncoderAdapter,
    check_retriever,
    load_adapter,
    supported_loaders,
)
from optivision.core.vectors import validate_corpus
from optivision.encoders.late import (
    COLENGINE_FAMILIES,
    RECIPES,
    ColEngineTokenEncoder,
    PyLateEncoder,
    Recipe,
    RemoteCodeEncoder,
)
from optivision.types import PageRef

torch = pytest.importorskip("torch")

DIM = 8


class TinyModel(torch.nn.Module):
    """Token vector = a fixed unit vector per token id, zeroed where the mask is 0."""

    def __init__(self, wrap: bool = False, width: int = DIM):
        super().__init__()
        table = torch.randn(64, width, generator=torch.Generator().manual_seed(0))
        self.table = torch.nn.functional.normalize(table, dim=-1)
        self.wrap = wrap
        self.rope_deltas = "stale"

    def forward(self, input_ids, attention_mask=None):
        emb = self.table[input_ids] * attention_mask.unsqueeze(-1)
        return SimpleNamespace(embeddings=emb) if self.wrap else emb


class TinyProcessor:
    """A page has one token per 8 pixels of width; a query one token per word. Padded with id 0."""

    @staticmethod
    def _batch(token_lists):
        width = max(len(t) for t in token_lists)
        ids = torch.zeros(len(token_lists), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, tokens in enumerate(token_lists):
            ids[i, : len(tokens)] = torch.tensor(tokens)
            mask[i, : len(tokens)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    @staticmethod
    def _page_tokens(image):
        array = np.asarray(image)
        return [(int(array[0, 0, 0]) + i) % 63 + 1 for i in range(array.shape[1] // 8)]

    @staticmethod
    def _query_tokens(text):
        return [int(w) % 63 + 1 for w in text.split()]

    def process_images(self, images=None):
        return self._batch([self._page_tokens(im) for im in images])

    def process_queries(self, queries):
        return self._batch([self._query_tokens(q) for q in queries])

    def process_texts(self, texts=None):
        return self.process_queries(texts)


def _page(width: int, shade: int) -> Image.Image:
    return Image.new("RGB", (width, 16), (shade, shade, shade))


def _pairs(n: int = 6):
    """Pages of different lengths and, for each, a query made of that page's own tokens."""
    pages = [_page(40 + 8 * i, 10 * i) for i in range(n)]
    queries = [" ".join(str(t - 1) for t in TinyProcessor._page_tokens(p)[:3]) for p in pages]
    return pages, queries


def _engine(**kwargs):
    return ColEngineTokenEncoder("org/tiny", "colqwen3.5", model=TinyModel(), processor=TinyProcessor(), **kwargs)


# ----------------------------------------------------------------------- colpali-engine families


def test_padding_rows_are_dropped_and_the_width_is_learned():
    encoder = _engine(page_batch_size=3)
    pages, queries = _pairs()
    vectors = encoder.encode_images(pages)
    assert [v.shape for v in vectors] == [(5 + i, DIM) for i in range(6)]
    assert all(v.dtype == np.float32 and v.flags.c_contiguous for v in vectors)
    assert encoder.dim == DIM
    assert [q.shape[0] for q in encoder.encode_queries(queries)] == [3] * 6


def test_batched_and_single_encoding_agree():
    pages, _ = _pairs()
    one = _engine(page_batch_size=1).encode_images(pages)
    many = _engine(page_batch_size=4).encode_images(pages)
    assert all(np.array_equal(a, b) for a, b in zip(one, many, strict=True))


def test_the_query_path_resets_rope_deltas_as_the_model_cards_do():
    encoder = _engine()
    encoder.encode_queries(["1 2 3"])
    assert encoder.model.rope_deltas is None


def test_a_width_other_than_the_expected_one_is_refused():
    encoder = _engine(expected_dim=DIM + 1)
    with pytest.raises(RuntimeError, match="expected 9"):
        encoder.encode_images([_page(40, 0)])


def test_an_unknown_family_is_refused_with_the_known_ones():
    with pytest.raises(ValueError, match="colqwen3.5"):
        ColEngineTokenEncoder("org/tiny", "colqwen9")


def test_non_finite_output_is_refused():
    encoder = _engine()
    encoder.model.table[5] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        encoder.encode_queries(["4"])


def test_every_family_names_two_classes():
    assert all(len(pair) == 2 and all(pair) for pair in COLENGINE_FAMILIES.values())
    assert {"colqwen3", "colqwen3.5", "colmodernvbert", "colgemma3"} <= set(COLENGINE_FAMILIES)


# ------------------------------------------------------------------------------- remote code


def test_remote_code_is_never_loaded_without_the_callers_consent():
    with pytest.raises(ValueError, match="trust_remote_code=True"):
        RemoteCodeEncoder("org/tiny", "tomoro", revision="abc")


def test_remote_code_needs_a_pinned_revision():
    with pytest.raises(ValueError, match="pin a revision"):
        RemoteCodeEncoder("org/tiny", "tomoro", trust_remote_code=True)


def test_an_unknown_recipe_is_refused():
    with pytest.raises(ValueError, match="unknown recipe"):
        RemoteCodeEncoder("org/tiny", "nope", model=TinyModel(), processor=TinyProcessor())


def test_the_processor_style_recipe_drives_the_named_methods():
    encoder = RemoteCodeEncoder("org/tiny", "tomoro", revision="abc", model=TinyModel(wrap=True),
                                processor=TinyProcessor())
    pages, queries = _pairs(3)
    assert [v.shape for v in encoder.encode_images(pages)] == [(5, DIM), (6, DIM), (7, DIM)]
    assert [q.shape[0] for q in encoder.encode_queries(queries)] == [3, 3, 3]
    info = encoder.info()
    assert info["recipe"] == "tomoro" and info["recipe_exercised"] is True
    assert info["revision"] == "abc" and info["loader"] == "remote-code:tomoro"


def test_the_model_style_recipe_takes_what_the_model_method_returns():
    class Model:
        def encode_image(self, images=None, **kwargs):
            assert kwargs == {"return_multivector": True}
            return [torch.ones(4, DIM) / DIM**0.5 for _ in images]

        def encode_text(self, texts=None, **kwargs):
            return [np.ones((2, DIM), np.float64) / DIM**0.5 for _ in texts]

    recipe = Recipe(style="model", image_method="encode_image", query_method="encode_text",
                    image_arg="images", query_arg="texts", image_kwargs={"return_multivector": True},
                    needs_processor=False)
    encoder = RemoteCodeEncoder("org/tiny", recipe, model=Model())
    assert [v.shape for v in encoder.encode_images([_page(40, 0)] * 2)] == [(4, DIM)] * 2
    out = encoder.encode_queries(["a"])
    assert out[0].shape == (2, DIM) and out[0].dtype == np.float32
    assert encoder.info()["recipe"] == "custom" and encoder.info()["recipe_exercised"] is False


def test_a_recipe_that_does_not_fit_the_checkpoint_says_so():
    recipe = Recipe(style="processor", image_method="no_such_method", query_method="process_queries")
    encoder = RemoteCodeEncoder("org/tiny", recipe, model=TinyModel(), processor=TinyProcessor())
    with pytest.raises(TypeError, match="no_such_method"):
        encoder.encode_images([_page(40, 0)])


def test_a_recipe_style_other_than_the_two_is_refused():
    with pytest.raises(ValueError, match="style"):
        Recipe(style="other", image_method="a", query_method="b")


def test_only_recipes_whose_calls_were_run_are_marked_exercised():
    assert RECIPES["tomoro"].exercised
    assert {name for name, r in RECIPES.items() if r.exercised} == {"tomoro"}


# ------------------------------------------------------------------------------------ PyLate


class _ColBERT:
    def encode(self, sentences, is_query=False, batch_size=32, show_progress_bar=False):
        n = 2 if is_query else 5
        return [np.full((n + len(s) % 2, DIM), 1 / DIM**0.5, np.float32) for s in sentences]


def test_the_pylate_encoder_uses_the_query_flag_and_refuses_images():
    encoder = PyLateEncoder("org/tiny-colbert", model=_ColBERT())
    assert [v.shape[0] for v in encoder.encode_documents(["ab", "abc"])] == [5, 6]
    assert [v.shape[0] for v in encoder.encode_queries(["ab"])] == [2]
    with pytest.raises(TypeError, match="text model"):
        encoder.encode_images([_page(40, 0)])


# ------------------------------------------------------------------------------- the adapter


def test_the_adapter_returns_checked_corpora_with_identity_and_no_layout():
    adapter = TokenEncoderAdapter(_engine())
    pages, queries = _pairs(4)
    docs = adapter.encode_documents([(PageRef("doc", i + 1), p) for i, p in enumerate(pages)])
    qs = adapter.encode_queries(queries, ids=["a", "b", "c", "d"])
    validate_corpus(docs)
    validate_corpus(qs)
    assert docs.ids == [PageRef("doc", i + 1).page_id for i in range(4)] and qs.ids == ["a", "b", "c", "d"]
    assert docs.positions is None and docs.protected is None and docs.images is None
    identity = adapter.identity()
    assert (identity.model_id, identity.dimension, identity.modality) == ("org/tiny", DIM, "image")
    assert identity.normalized is None and identity.extra["loader"] == "colpali-engine:colqwen3.5"
    assert docs.attrs["encoder_identity"] == identity.to_dict()
    assert docs.attrs["loader"] == "colpali-engine:colqwen3.5" and "model_status" not in docs.attrs


def test_the_adapter_has_no_identity_before_the_width_is_known():
    assert TokenEncoderAdapter(_engine()).identity() is None


def test_a_text_adapter_passes_strings_to_encode_documents():
    adapter = TokenEncoderAdapter(PyLateEncoder("org/tiny-colbert", model=_ColBERT()), modality="text")
    docs = adapter.encode_documents(["one", "two two"], ids=["x", "y"])
    assert docs.ids == ["x", "y"] and adapter.identity().modality == "text"


def test_the_adapter_refuses_an_object_that_cannot_encode():
    with pytest.raises(TypeError, match="encode_queries"):
        TokenEncoderAdapter(object())
    with pytest.raises(ValueError, match="modality"):
        TokenEncoderAdapter(_engine(), modality="smell")


# ---------------------------------------------------------------------------- check_retriever


def test_check_retriever_passes_a_model_whose_queries_find_their_pages():
    pages, queries = _pairs()
    report = check_retriever(TokenEncoderAdapter(_engine()), pages, queries, expected_dim=DIM)
    assert report["ok"] and report["query_ranks_own_document_first"] == 6
    assert report["dimension"] == DIM and report["unit_norm_max_error"] < 1e-5
    assert report["vectors_per_document"] == [5, 6, 7, 8, 9, 10]


def test_check_retriever_fails_vectors_that_are_not_unit_norm():
    encoder = _engine()
    encoder.model.table = encoder.model.table * 3.0
    pages, queries = _pairs()
    report = check_retriever(TokenEncoderAdapter(encoder), pages, queries)
    assert not report["ok"] and report["unit_norm_max_error"] > 1.0


def test_check_retriever_fails_queries_that_do_not_find_their_pages():
    pages, queries = _pairs()
    report = check_retriever(TokenEncoderAdapter(_engine()), pages, list(reversed(queries)), allowed_misses=0)
    assert not report["ok"] and report["query_ranks_own_document_first"] < 6


def test_check_retriever_fails_another_width_and_refuses_unmatched_input():
    pages, queries = _pairs()
    assert not check_retriever(TokenEncoderAdapter(_engine()), pages, queries, expected_dim=DIM + 1)["ok"]
    with pytest.raises(ValueError, match="matched pairs"):
        check_retriever(TokenEncoderAdapter(_engine()), pages, queries[:-1])


# -------------------------------------------------------------------------------- load_adapter


def test_load_adapter_builds_the_token_adapter_for_the_newer_engine_families(monkeypatch):
    from optivision.encoders import late

    built = {}

    def fake_init(self, model_name, family, **kwargs):
        built.update(model=model_name, family=family, **kwargs)
        self.checkpoint, self.loader = model_name, f"colpali-engine:{family}"

    monkeypatch.setattr(late.ColEngineTokenEncoder, "__init__", fake_init)
    monkeypatch.setattr(late.ColEngineTokenEncoder, "encode_queries", lambda self, q: [], raising=False)
    adapter = load_adapter("org/some-fine-tune", loader="colpali-engine:colqwen3", device="cpu")
    assert isinstance(adapter, TokenEncoderAdapter) and adapter.info is None
    assert built == {"model": "org/some-fine-tune", "family": "colqwen3", "device": "cpu", "expected_dim": None}


def test_load_adapter_passes_the_registry_pin_and_width_to_a_remote_code_model(monkeypatch):
    from optivision.encoders import late

    built = {}

    def fake_init(self, model_name, recipe, **kwargs):
        built.update(model=model_name, recipe=recipe, **kwargs)
        self.checkpoint, self.loader = model_name, f"remote-code:{recipe}"

    monkeypatch.setattr(late.RemoteCodeEncoder, "__init__", fake_init)
    model_id = "TomoroAI/tomoro-colqwen3-embed-4b"
    adapter = load_adapter(model_id, trust_remote_code=True)
    info = MODELS[model_id]
    assert adapter.info is info and info.revision and info.dimension == 320
    assert built == {"model": model_id, "recipe": "tomoro", "trust_remote_code": True,
                     "expected_dim": 320, "known_revision": info.revision}


def test_an_unknown_loader_is_refused_with_the_known_ones():
    with pytest.raises(ValueError, match="unknown loader"):
        load_adapter("org/model", loader="telepathy")
    loaders = supported_loaders()
    assert "nemotron" in loaders and "pylate" in loaders and "remote-code:tomoro" in loaders
    assert "colpali-engine:colqwen2.5" in loaders and "colpali-engine:colqwen3.5" in loaders


def test_every_registry_loader_is_one_load_adapter_can_build():
    loaders = set(supported_loaders())
    assert {info.loader for info in MODELS.values()} <= loaders
