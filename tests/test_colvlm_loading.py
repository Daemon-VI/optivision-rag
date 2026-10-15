"""Checkpoint loading for the colpali-engine encoders (no download, tiny random models).

colpali-engine 0.3.17 under transformers 5 maps ColQwen checkpoints' `model.layers.*`
onto `language_model.layers.*` but not `model.embed_tokens` / `model.norm`, which
then load as random weights with only a log line (seen on a Kaggle T4). These
tests rebuild that situation with a tiny ColQwen2 saved in the published key
layout.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
colpali_models = pytest.importorskip("colpali_engine.models")
safetensors_torch = pytest.importorskip("safetensors.torch")

from optivision.encoders.colvlm import _from_pretrained


def _tiny_colqwen2_checkpoint(path):
    from transformers import Qwen2VLConfig

    cfg = Qwen2VLConfig(
        text_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                     "layer_types": ["full_attention"] * 2, "num_attention_heads": 4,
                     "num_key_value_heads": 2, "vocab_size": 300,
                     "rope_scaling": {"type": "mrope", "mrope_section": [2, 1, 1]}},
        vision_config={"depth": 1, "embed_dim": 32, "hidden_size": 32, "num_heads": 4, "mlp_ratio": 2},
    )
    torch.manual_seed(0)
    model = colpali_models.ColQwen2(cfg)
    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    # the published layout: the language model lives under `model.*`
    published = {("model." + k[len("language_model."):]) if k.startswith("language_model.") else k: v
                 for k, v in state.items()}
    safetensors_torch.save_file(published, str(path / "model.safetensors"))
    cfg.save_pretrained(str(path))
    return state


def test_plain_colpali_engine_load_leaves_weights_random(tmp_path):
    _tiny_colqwen2_checkpoint(tmp_path)
    _, info = colpali_models.ColQwen2.from_pretrained(str(tmp_path), output_loading_info=True)
    assert "language_model.embed_tokens.weight" in info["missing_keys"]  # the upstream bug this guards against


def test_loader_maps_every_tensor_and_refuses_partial_loads(tmp_path):
    state = _tiny_colqwen2_checkpoint(tmp_path)
    model = _from_pretrained(colpali_models.ColQwen2, str(tmp_path), torch.float32)
    loaded = model.state_dict()
    for key, value in state.items():
        assert torch.equal(loaded[key], value), key
    assert model.loading_report["unexpected"] == []

    # A checkpoint missing a weight must fail loudly, not load it as random values.
    broken = tmp_path / "broken"
    broken.mkdir()
    published = safetensors_torch.load_file(str(tmp_path / "model.safetensors"))
    del published["model.norm.weight"]
    safetensors_torch.save_file(published, str(broken / "model.safetensors"))
    (broken / "config.json").write_text((tmp_path / "config.json").read_text())
    with pytest.raises(RuntimeError, match="would be random"):
        _from_pretrained(colpali_models.ColQwen2, str(broken), torch.float32)


def test_segment_mapping_only_when_it_accounts_for_every_key():
    from optivision.encoders.colvlm import _segment_mapping

    missing = ["language_model.layers.0.w", "language_model.norm.weight"]
    unexpected = ["language_model.model.layers.0.w", "language_model.model.norm.weight"]
    assert _segment_mapping(missing, unexpected) == {r"^language_model\.model\.": "language_model."}
    # a key that maps onto nothing, or a missing key nothing maps onto: no rules
    assert _segment_mapping(missing, [*unexpected, "other.extra.bias"]) is None
    assert _segment_mapping([*missing, "custom_text_proj.weight"], unexpected) is None


def test_loader_maps_a_checkpoint_nested_one_level_deeper(tmp_path):
    # Cognitive-Lab/ColNetraEmbed stores its language model as `language_model.model.*`
    state = _tiny_colqwen2_checkpoint(tmp_path)
    nested = {("language_model.model." + k[len("language_model."):]) if k.startswith("language_model.") else k:
              v.detach().clone().contiguous() for k, v in state.items()}
    deep = tmp_path / "deep"
    deep.mkdir()
    safetensors_torch.save_file(nested, str(deep / "model.safetensors"))
    (deep / "config.json").write_text((tmp_path / "config.json").read_text())
    model = _from_pretrained(colpali_models.ColQwen2, str(deep), torch.float32)
    loaded = model.state_dict()
    for key, value in state.items():
        assert torch.equal(loaded[key], value), key
    assert model.loading_report["key_remap"] == {r"^language_model\.model\.": "language_model."}
    assert model.loading_report["unexpected"] == []


def test_multi_gpu_map_never_splits_a_decoder_layer(tmp_path):
    from optivision.encoders.colvlm import layer_device_map

    _tiny_colqwen2_checkpoint(tmp_path)
    model = _from_pretrained(colpali_models.ColQwen2, str(tmp_path), torch.float32)
    sizes = {n: sum(t.numel() * t.element_size() for t in m.parameters())
             for n, m in model.named_modules() if n.startswith("language_model.layers.") and n.count(".") == 2}
    total = sum(t.numel() * t.element_size() for t in model.parameters())
    # GPU 0 can take everything except about one decoder layer
    budgets = [total - max(sizes.values()) // 2, total]
    dm = layer_device_map(model, budgets)
    layer_keys = sorted((k for k in dm if k.startswith("language_model.layers.")), key=lambda k: int(k.rsplit(".", 1)[1]))
    assert layer_keys == [f"language_model.layers.{i}" for i in range(len(sizes))]  # whole layers only
    devices = [dm[k] for k in layer_keys]
    assert devices == sorted(devices) and devices[-1] == 1  # in order, and the split happened
    assert dm["custom_text_proj"] == 0 and dm["visual"] == 0 and dm["language_model.embed_tokens"] == 0
    assert dm["language_model.norm"] == devices[-1] and dm["language_model.rotary_emb"] == 0
    for name, _ in model.named_parameters():
        assert any(name == k or name.startswith(k + ".") for k in dm), name
    with pytest.raises(RuntimeError, match="does not fit"):
        layer_device_map(model, [total // 4, total // 4])


class _BareLoader(torch.nn.Module):
    """A remote-code class whose from_pretrained ignores output_loading_info, like ColVec1."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 2)
        self.head = torch.nn.Linear(2, 2)

    @classmethod
    def from_pretrained(cls, checkpoint, **kwargs):
        model = cls()
        state = safetensors_torch.load_file(f"{checkpoint}/model.safetensors")
        model.load_state_dict(state, strict=False)  # silently skips what does not match
        return model


def test_loader_checks_a_model_returned_without_loading_info(tmp_path):
    reference = _BareLoader()
    safetensors_torch.save_file({k: v.contiguous() for k, v in reference.state_dict().items()},
                                str(tmp_path / "model.safetensors"))
    model = _from_pretrained(_BareLoader, str(tmp_path), torch.float32)
    assert model.loading_report == {"missing_allowed": [], "unexpected": []}
    assert torch.equal(model.proj.weight, reference.proj.weight)

    partial = tmp_path / "partial"
    partial.mkdir()
    safetensors_torch.save_file({"proj.weight": reference.proj.weight.contiguous(),
                                 "proj.bias": reference.proj.bias.contiguous(),
                                 "renamed.head.weight": reference.head.weight.contiguous()},
                                str(partial / "model.safetensors"))
    with pytest.raises(RuntimeError, match="would be random"):
        _from_pretrained(_BareLoader, str(partial), torch.float32)
