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
