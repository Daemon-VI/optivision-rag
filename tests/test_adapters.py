from __future__ import annotations

import json
from pathlib import Path

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

CAMPAIGN_OUTCOMES = Path(__file__).resolve().parents[1] / "reports" / "universal" / "campaign" / "outcomes.json"


@pytest.mark.skipif(not CAMPAIGN_OUTCOMES.exists(), reason="needs the repository's reports (not in an installed wheel)")
def test_measured_models_are_exactly_the_benchmarked_ones():
    """The registry must not claim more than the reports support."""
    measured = {m.model_id for m in supported_models(MEASURED)}
    campaign = {m for m, r in json.loads(CAMPAIGN_OUTCOMES.read_text(encoding="utf-8")).items()
                if r["outcome"] == "measured"}
    assert measured == {"vidore/colpali-v1.3-merged", "vidore/colSmol-256M", "answerdotai/answerai-colbert-small-v1",
                        "vidore/colqwen2-v1.0-merged", "vidore/colqwen2-v1.0", "vidore/colqwen2.5-v0.2",
                        "nvidia/nemotron-colembed-vl-4b-v2", "nvidia/nemotron-colembed-vl-8b-v2",
                        "athrael-soju/colqwen3.5-4.5B-v3", "TomoroAI/tomoro-colqwen3-embed-4b"} | campaign
    for model_id in campaign:  # each one points at its own campaign folder
        assert f"reports/universal/campaign/{model_id.replace('/', '--')}" in MODELS[model_id].evidence


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


def test_colqwen_models_load_through_colpali_engine():
    from optivision.adapters import MODELS
    from optivision.encoders.colvlm import BACKENDS

    for model_id, backend in (("vidore/colqwen2-v1.0", "colqwen2"), ("vidore/colqwen2-v1.0-merged", "colqwen2"),
                              ("vidore/colqwen2.5-v0.2", "colqwen2.5")):
        assert MODELS[model_id].loader == f"colpali-engine:{backend}"
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


def test_sentence_transformers_glue_encodes_in_chunks_and_passes_options():
    class Recording(_StubMultiVectorEncoder):
        def __init__(self):
            super().__init__()
            self.calls = []

        def encode_document(self, inputs, **kwargs):
            self.calls.append((len(inputs), kwargs))
            return super().encode_document(inputs)

    stub = Recording()
    with pytest.warns(UserWarning, match="unverified"):
        adapter = SentenceTransformersAdapter("some/unknown-model", model=stub, batch_size=2,
                                              encode_kwargs={"show_progress_bar": False})
    docs = adapter.encode_documents(["a", "bb", "ccc", "dddd", "eeeee"])
    assert [n for n, _ in stub.calls] == [2, 2, 1]
    assert all(kw == {"show_progress_bar": False} for _, kw in stub.calls)
    assert docs.counts.tolist() == [2, 3, 4, 5, 1] and docs.dtype == np.float32
    with pytest.raises(ValueError, match="batch_size"):
        SentenceTransformersAdapter("some/unknown-model", model=stub, batch_size=0)


def test_sentence_transformers_device_map_is_left_in_place():
    from optivision.adapters import _device_map_placement, _keep_placement

    class Inner:
        def __init__(self):
            self.hf_device_map = {"model.visual": 0, "model.layers.0": 0, "model.layers.1": 1}

    class Outer:
        moved = False

        def modules(self):
            return iter([self, Inner()])

        def to(self, *args, **kwargs):
            self.moved = True
            return self

    model = Outer()
    assert _device_map_placement(model) == {"devices": ["0", "1"], "device_map": "auto"}
    _keep_placement(model)  # sentence-transformers' encode() calls model.to(device) every time
    assert model.to("cuda:0") is model and not model.moved
    assert _device_map_placement(_StubMultiVectorEncoder()) is None


def test_a_split_models_small_heads_move_to_the_input_device():
    """ColVec1.1 multiplies its head's output by the attention mask, which is on the input device."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("accelerate")
    from optivision.adapters import _device_map_placement, _heads_on_input_device

    split = torch.nn.Module()
    split.backbone = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 64))
    split.big = torch.nn.Linear(64, 64)  # a top-level module too large to count as a head
    split.head = torch.nn.Linear(64, 2)  # under 2% of the parameters
    split.hf_device_map = {"backbone.0": "cpu", "backbone.1": 1, "big": 1, "head": 1}

    class Encoder:
        device = torch.device("cpu")

        def modules(self):
            return iter([self, split])

    assert _heads_on_input_device(Encoder()) == ["head"]
    assert split.hf_device_map == {"backbone.0": "cpu", "backbone.1": 1, "big": 1, "head": "cpu"}
    assert split.head._hf_hook.execution_device == torch.device("cpu")
    assert split.head(torch.ones(3, 64)).shape == (3, 2)
    assert _device_map_placement(Encoder(), ["head"]) == {
        "devices": ["1", "cpu"], "device_map": "auto", "heads_on_input_device": ["head"]}
    assert _heads_on_input_device(_StubMultiVectorEncoder()) == []
