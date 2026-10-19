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


def test_a_pylate_encoder_without_registry_entry_defaults_to_text():
    assert TokenEncoderAdapter(PyLateEncoder("org/tiny-colbert", model=_ColBERT())).modality == "text"
    assert TokenEncoderAdapter(_engine()).modality == "image"


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


def test_load_adapter_applies_a_registry_load_option_unless_the_caller_overrides_it(monkeypatch):
    from optivision.encoders import colvlm

    built = []

    def fake_init(self, backend="colsmol", model_name=None, **kwargs):
        built.append(kwargs)
        self.name, self.checkpoint = backend, model_name
        self.max_num_visual_tokens = kwargs.get("max_num_visual_tokens")

    monkeypatch.setattr(colvlm.ColVLMEncoder, "__init__", fake_init)
    model_id = "vidore/colqwen2.5-v0.1"
    assert dict(MODELS[model_id].load_options) == {"max_num_visual_tokens": 768}
    adapter = load_adapter(model_id, dtype="float32")
    assert built[-1] == {"dtype": "float32", "max_num_visual_tokens": 768}
    assert adapter.identity() is not None and adapter.identity().extra["max_num_visual_tokens"] == "768"
    load_adapter(model_id, max_num_visual_tokens=1280)
    assert built[-1] == {"max_num_visual_tokens": 1280}
    load_adapter("vidore/colqwen2.5-v0.2")  # no option in its row: nothing added
    assert built[-1] == {}


def test_an_unknown_loader_is_refused_with_the_known_ones():
    with pytest.raises(ValueError, match="unknown loader"):
        load_adapter("org/model", loader="telepathy")
    loaders = supported_loaders()
    assert "nemotron" in loaders and "pylate" in loaders and "remote-code:tomoro" in loaders
    assert "colpali-engine:colqwen2.5" in loaders and "colpali-engine:colqwen3.5" in loaders


def test_every_registry_loader_is_one_load_adapter_can_build():
    loaders = set(supported_loaders())
    assert {info.loader for info in MODELS.values()} <= loaders


# ------------------------------------------------------------------ 0.4.1 regressions


def test_check_retriever_counts_a_tie_as_a_miss_so_a_constant_encoder_fails():
    encoder = _engine()
    encoder.model.table = encoder.model.table[:1].repeat(64, 1)  # every token the same unit vector
    pages, queries = _pairs()
    report = check_retriever(TokenEncoderAdapter(encoder), pages, queries)
    assert report["query_ranks_own_document_first"] == 0 and not report["ok"]
    two_pages, two_queries = _pairs(2)
    report = check_retriever(TokenEncoderAdapter(encoder), two_pages, two_queries, allowed_misses=0)
    assert report["query_ranks_own_document_first"] == 0 and not report["ok"]


def test_check_retriever_needs_more_pairs_than_twice_the_allowed_misses():
    pages, queries = _pairs(3)
    with pytest.raises(ValueError, match="at least 6"):
        check_retriever(TokenEncoderAdapter(_engine()), pages, queries)  # allowed_misses=2
    assert check_retriever(TokenEncoderAdapter(_engine()), pages, queries, allowed_misses=0)["ok"]


def test_a_stacked_numpy_output_is_split_into_rows():
    from optivision.encoders.late import _as_list, _rows

    emb = np.zeros((2, 3, DIM), np.float32)
    emb[:, :2, 0] = 1.0  # the third row of each item is zero padding
    rows = _as_list(emb)
    assert [r.shape for r in rows] == [(2, DIM), (2, DIM)]
    masked = _rows(emb, np.array([[1, 0, 0], [1, 1, 0]]))
    assert [r.shape[0] for r in masked] == [1, 2]


def test_non_finite_values_are_refused_only_in_the_rows_that_are_kept():
    from optivision.encoders.late import _rows

    emb = np.zeros((1, 3, DIM), np.float32)
    emb[0, :2, 0] = 1.0
    emb[0, 2, :] = np.inf  # overflow at a padded position is dropped with the padding
    assert [r.shape[0] for r in _rows(emb, np.array([[1, 1, 0]]))] == [2]
    emb[0, 1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _rows(emb, np.array([[1, 1, 0]]))


def test_the_engine_encoder_records_and_passes_its_pinned_revision(monkeypatch):
    from optivision.encoders import colvlm

    assert _engine(revision="abc123").info()["revision"] == "abc123"
    assert _engine().info()["revision"] is None
    seen = []
    monkeypatch.setattr(colvlm, "_adapter_base", lambda checkpoint, revision=None: None)
    monkeypatch.setattr(colvlm, "_from_pretrained", lambda cls, ckpt, dtype, **extra: seen.append(extra))
    colvlm._load_model(object, "org/full", None, revision="abc123")
    colvlm._load_model(object, "org/full", None)
    assert seen == [{"revision": "abc123"}, {}]


def test_load_adapter_pins_the_measured_colqwen35_revision(monkeypatch):
    from optivision.encoders import late

    built = {}

    def fake_init(self, model_name, family, **kwargs):
        built.update(model=model_name, family=family, **kwargs)
        self.checkpoint, self.loader = model_name, f"colpali-engine:{family}"

    monkeypatch.setattr(late.ColEngineTokenEncoder, "__init__", fake_init)
    monkeypatch.setattr(late.ColEngineTokenEncoder, "encode_queries", lambda self, q: [], raising=False)
    model_id = "athrael-soju/colqwen3.5-4.5B-v3"
    load_adapter(model_id)
    assert MODELS[model_id].revision == "c4ea4b92ff3322d68bfd6c65e492c69d0176f910"
    assert built["revision"] == "c4ea4b92ff3322d68bfd6c65e492c69d0176f910"
    load_adapter(model_id, revision="latest-sha")
    assert built["revision"] == "latest-sha"  # the caller can still override the pin


def test_injected_model_device_and_dtype_come_from_its_parameters():
    """M5: an injected model's inputs go where its parameters are, not to the class default "cpu"."""
    model = TinyModel()
    model.marker = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))  # first parameter decides
    enc = ColEngineTokenEncoder("org/tiny", "colqwen3.5", model=model, processor=TinyProcessor())
    first = next(model.parameters())
    assert enc.device == str(first.device)
    assert enc.torch_dtype == first.dtype
    assert enc.info()["dtype"] != "None"
    remote = RemoteCodeEncoder("org/tiny", "tomoro", model=model, processor=TinyProcessor())
    assert remote.device == str(first.device) and remote.torch_dtype == torch.float64
    bare = RemoteCodeEncoder("org/tiny", "tomoro", model=TinyModel(), processor=TinyProcessor())
    assert bare.device == "cpu"  # no parameters to read: the default stands

    class Fake:  # a stand-in that reports a GPU device without needing one
        def parameters(self):
            return iter([SimpleNamespace(device="cuda:0", dtype=torch.float16)])

    enc2 = ColEngineTokenEncoder("org/tiny", "colqwen3.5", model=Fake(), processor=TinyProcessor())
    assert enc2.device == "cuda:0" and enc2.torch_dtype == torch.float16


class _PixelProcessor(TinyProcessor):
    """Like TinyProcessor, plus float32 pixel values, as an InternVL processor returns them."""

    def process_images(self, images=None):
        batch = super().process_images(images)
        return {**batch, "pixel_values": torch.ones(len(images), 3, 2, 2)}


class _HalfModel(TinyModel):
    """Records the dtype its inputs arrive in; its one parameter is float16."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))
        self.seen = {}

    def forward(self, input_ids, attention_mask=None, pixel_values=None):
        self.seen = {"input_ids": input_ids.dtype, "attention_mask": attention_mask.dtype,
                     "pixel_values": None if pixel_values is None else pixel_values.dtype}
        return super().forward(input_ids, attention_mask)


def test_a_cast_inputs_recipe_casts_floating_inputs_to_the_model_dtype():
    """M2: the vintern recipe casts pixel values to the model dtype, as its card does by hand."""
    assert RECIPES["vintern"].cast_inputs and not RECIPES["processor"].cast_inputs
    model = _HalfModel()
    RemoteCodeEncoder("org/tiny", "vintern", model=model, processor=_PixelProcessor()).encode_images([_page(40, 0)])
    assert model.seen == {"input_ids": torch.long, "attention_mask": torch.long, "pixel_values": torch.float16}
    RemoteCodeEncoder("org/tiny", "processor", model=model, processor=_PixelProcessor()).encode_images([_page(40, 0)])
    assert model.seen["pixel_values"] == torch.float32  # the other recipes leave the batch as built


def test_info_records_the_batch_sizes_and_whether_the_mask_was_applied():
    """L1: batch size is part of what makes vectors comparable, so info() records it."""
    enc = RemoteCodeEncoder("org/tiny", "tomoro", model=TinyModel(wrap=True), processor=TinyProcessor(),
                            query_batch_size=1)
    assert enc.info()["mask_applied"] is None
    enc.encode_queries(["1 2", "3"])
    info = enc.info()
    assert info["query_batch_size"] == 1 and info["page_batch_size"] == 1 and info["mask_applied"] is True


def test_a_mask_that_does_not_fit_is_warned_about_once_and_recorded():
    """L3: a mask of another length is not applied; the encoder says so instead of staying silent."""
    from optivision.encoders.late import _rows

    owner = SimpleNamespace(mask_applied=None)
    emb = np.zeros((1, 3, DIM), np.float32)
    emb[0, :, 0] = 1.0
    with pytest.warns(RuntimeWarning, match="does not match"):
        rows = _rows(emb, np.array([[1, 1]]), owner=owner)
    assert rows[0].shape[0] == 3 and owner.mask_applied is False
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _rows(emb, np.array([[1, 1]]), owner=owner)  # once per encoder: no second warning

