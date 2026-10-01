from __future__ import annotations

import numpy as np
import pytest

from optivision.adapters import (
    MEASURED,
    MODELS,
    UNVERIFIED,
    PageEncoderAdapter,
    SentenceTransformersAdapter,
    supported_models,
)
from optivision.encoders import SyntheticEncoder
from optivision.types import PageRef

from .conftest import make_page


def test_measured_models_are_exactly_the_benchmarked_ones():
    """The registry must not claim more than the reports support."""
    measured = {m.model_id for m in supported_models(MEASURED)}
    assert measured == {"vidore/colpali-v1.3-merged", "vidore/colSmol-256M", "answerdotai/answerai-colbert-small-v1"}


def test_every_entry_says_where_its_status_comes_from():
    for info in MODELS.values():
        assert info.status in {"measured", "untested", "unverified"}
        assert info.evidence


def test_page_encoder_adapter_builds_a_corpus_with_structure():
    adapter = PageEncoderAdapter(SyntheticEncoder(dim=16, grid=8))
    docs = [(PageRef("a", 1), make_page()), (PageRef("b", 1), make_page(ink_rows=1))]
    corpus = adapter.encode_documents(docs)
    assert len(corpus) == 2 and corpus.dimension == 16
    assert corpus.protected is not None and corpus.positions is not None
    assert corpus.images is not None and len(corpus.images) == 2
    queries = adapter.encode_queries(["tax invoice", "memo"], ids=["q1", "q2"])
    assert queries.ids == ["q1", "q2"] and queries.dimension == 16


class _StubMultiVectorEncoder:
    """Stands in for sentence_transformers.MultiVectorEncoder's documented API."""

    def __init__(self, dim=8):
        self.dim = dim

    def encode_document(self, inputs):
        return [np.ones((len(str(x)) % 5 + 1, self.dim), np.float32) for x in inputs]

    def encode_query(self, texts):
        return [np.ones((3, self.dim), np.float32) for _ in texts]


def test_sentence_transformers_glue_warns_and_converts():
    with pytest.warns(UserWarning, match="unverified"):
        adapter = SentenceTransformersAdapter("some/unknown-model", model=_StubMultiVectorEncoder())
    docs = adapter.encode_documents(["a", "bbbb"], ids=["x", "y"])
    assert docs.ids == ["x", "y"] and docs.counts.tolist() == [2, 5]
    assert docs.attrs["model_status"] == UNVERIFIED
    assert adapter.encode_queries(["q"]).counts.tolist() == [3]


def test_colqwen_models_load_through_colpali_engine_and_stay_untested():
    from optivision.adapters import MODELS, UNTESTED
    from optivision.encoders.colvlm import BACKENDS

    for model_id, backend in (("vidore/colqwen2-v1.0", "colqwen2"), ("vidore/colqwen2.5-v0.2", "colqwen2.5")):
        info = MODELS[model_id]
        assert info.loader == f"colpali-engine:{backend}"
        assert info.status == UNTESTED  # until results exist (see the measured-set test above)
        assert backend in BACKENDS
    # the colqwen2 default is the pre-merged checkpoint, never the adapter-only repo
    assert BACKENDS["colqwen2"][0] == "vidore/colqwen2-v1.0-merged"


def test_adapter_only_checkpoints_are_detected(tmp_path):
    import json

    from optivision.encoders.colvlm import _adapter_base

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "org/base"}))
    assert _adapter_base(str(adapter)) == "org/base"

    full = tmp_path / "full"
    full.mkdir()
    (full / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "org/base"}))
    (full / "config.json").write_text("{}")
    assert _adapter_base(str(full)) is None  # ships full weights: load directly

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.json").write_text("{}")
    assert _adapter_base(str(plain)) is None
