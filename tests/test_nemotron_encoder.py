"""Identity checks of the wide NVIDIA ColEmbed encoder, without downloading anything."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from optivision.encoders import nemotron


class _Reached(Exception):
    """Raised by the stubbed loader: the identity checks let the load proceed."""


def _config(model_type="qwen3_vl_nemotron_embed", pooling="colbert", width=2560):
    return SimpleNamespace(model_type=model_type, pooling=pooling, text_config=SimpleNamespace(hidden_size=width))


@pytest.fixture
def stub(monkeypatch):
    holder = {"config": _config()}
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *a, **k: holder["config"])

    def fake_load(*a, **k):
        raise _Reached(k)

    monkeypatch.setattr(nemotron, "_from_pretrained", fake_load)
    return holder


def test_expected_model_reaches_the_strict_loader_with_a_pinned_revision(stub):
    with pytest.raises(_Reached) as hit:
        nemotron.NemotronColEmbedEncoder("nvidia/nemotron-colembed-vl-4b-v2", device="cpu")
    kwargs = hit.value.args[0]
    assert kwargs["revision"] == nemotron.KNOWN["nvidia/nemotron-colembed-vl-4b-v2"][1]
    assert kwargs["trust_remote_code"] is True


@pytest.mark.parametrize(("config", "match"), [
    (_config(model_type="qwen3_vl"), "expected a qwen3_vl_nemotron_embed"),
    (_config(pooling="avg"), "ColBERT-pooling"),
    (_config(width=4096), "does not match the expected 2560"),
])
def test_wrong_identity_or_width_is_refused_before_loading(stub, config, match):
    stub["config"] = config
    with pytest.raises(RuntimeError, match=match):
        nemotron.NemotronColEmbedEncoder("nvidia/nemotron-colembed-vl-4b-v2", device="cpu")


def test_unknown_checkpoint_needs_a_pinned_revision(stub):
    with pytest.raises(ValueError, match="pin a revision"):
        nemotron.NemotronColEmbedEncoder("someone/else", device="cpu")
