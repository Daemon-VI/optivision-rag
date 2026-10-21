"""optivision.easy._encode: loading, OOM halving, the wiring check, estimates (FINAL_DESIGN 4.2, 5.5, 9).

Offline: no model is downloaded and no GPU is used. Loaders are replaced by fake adapters
(``CallableAdapter``) whose "pages" carry their words, so a query finds the page whose words it
shares; ``optivision.adapters.load_adapter`` is monkeypatched to see the arguments a load gets.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import types
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from optivision import adapters
from optivision.adapters import MODELS, CallableAdapter, load_adapter
from optivision.easy import _encode
from optivision.easy._types import MissingExtra, RemoteCodeRefused, UnknownModel, WiringError

ROOT = Path(__file__).resolve().parents[1]
RATES_BUILDER = ROOT / "scripts" / "make_encode_rates.py"
SAMPLE_BUILDER = ROOT / "scripts" / "make_sample_pdf.py"
DIM = 64


# ------------------------------------------------------------------------------ fakes


@dataclass(frozen=True)
class FakeImage:
    """Stands in for a rendered page: the words a model would read off it."""

    text: str


@dataclass(frozen=True)
class Rec:
    """The fields of a ``_docs.PageRecord`` the wiring check reads."""

    id: str
    doc_id: str
    text: str
    blank: bool = False


def _word_vector(word: str) -> np.ndarray:
    v = np.random.default_rng(zlib.crc32(word.encode())).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _embed(text: str) -> np.ndarray:
    words = sorted({w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split()}) or ["empty"]
    return np.stack([_word_vector(w) for w in words])


def _text_of(item) -> str:
    return item.text if isinstance(item, FakeImage) else str(item)


def word_adapter(calls: list | None = None) -> CallableAdapter:
    """An adapter whose vectors are one unit vector per distinct word of a page or query."""

    def docs(items):
        if calls is not None:
            calls.append(len(items))
        return [_embed(_text_of(i)) for i in items]

    return CallableAdapter(docs, lambda qs: [_embed(q) for q in qs])


def constant_adapter() -> CallableAdapter:
    """Every page and query gets the same vector: nothing is ever found (a broken load)."""
    one = _word_vector("same")[None, :]
    return CallableAdapter(lambda items: [one.copy() for _ in items], lambda qs: [one.copy() for _ in qs])


def _oom_error(message: str = "CUDA out of memory. Tried to allocate 2.00 GiB") -> BaseException:
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(getattr(torch.cuda, "OutOfMemoryError", None), type):
        return torch.cuda.OutOfMemoryError(message)
    return type("OutOfMemoryError", (RuntimeError,), {})(message)


TOPICS = ["harbour buoys turbidity", "orchard russet crates", "derailleur hanger cable", "ferry gannet quay",
          "sourdough starter flour", "anemometer gust frost", "library charts pilot", "varroa brood super",
          "inverter rafter isolator", "pendulum regulator gallery", "semaphore lever rodding", "hypothesis judges marks",
          "roaster bergamot cinnamon", "pallet racking forklift"]


def user_pages(n: int = 14, *, blank_images: bool = False) -> list[tuple[Rec, FakeImage]]:
    pages = []
    for i in range(n):
        a, b, c = TOPICS[i % len(TOPICS)].split()
        text = (f"Report {i}\nThe {a} committee examined the {b} records and the {c} schedule in detail today. "
                f"Every finding about the {a} and the {c} was written into the shared ledger by the clerk.")
        rec = Rec(f"docs/file{i:02d}.pdf::p1", f"docs/file{i:02d}.pdf", text)
        pages.append((rec, FakeImage("" if blank_images else text)))
    return pages


@pytest.fixture
def fake_sample(monkeypatch):
    """The sample PDF's text is real; its rendering is replaced by images carrying that text."""
    records = {r.page: r.text for r in _encode._sample_records()}
    monkeypatch.setattr(_encode, "_render_sample", lambda pages, **kw: [FakeImage(records[p]) for p in pages])
    return records


@pytest.fixture
def loader_seen(monkeypatch):
    """load_adapter replaced by a recorder that returns a working fake adapter; no loader package needed."""
    seen: list[tuple[str, dict]] = []

    def fake_load_adapter(model_id, loader=None, **kwargs):
        seen.append((model_id, kwargs))
        adapter = word_adapter()
        adapter.info = MODELS.get(model_id)
        return adapter

    monkeypatch.setattr(adapters, "load_adapter", fake_load_adapter)
    monkeypatch.setattr(_encode, "missing_packages", lambda *a, **k: [])
    monkeypatch.setattr(_encode, "_cuda_count", lambda: 0)
    return seen


# ------------------------------------------------------------------------------ rates and estimates


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not RATES_BUILDER.exists() or not (ROOT / "reports" / "universal" / "campaign").exists(),
                    reason="needs the repository's run records (not in an installed wheel)")
def test_the_shipped_rates_equal_what_the_run_records_give():
    builder = _module(RATES_BUILDER, "make_encode_rates")
    shipped = (ROOT / "src" / "optivision" / "presets" / "encode_rates.json").read_text(encoding="utf-8")
    assert shipped.replace("\r\n", "\n") == builder.render(builder.build())


def test_every_timed_model_is_registered_and_its_rates_are_consistent():
    models = _encode.rates()["models"]
    assert len(models) >= 50
    for model_id, entry in models.items():
        assert model_id in MODELS, model_id
        lo, hi = entry["s_per_page"]
        assert 0 < lo <= hi and entry["step_s_per_page"][0] >= lo
        assert entry["unit"] == ("document" if entry["modality"] == "text" else "page")
        assert entry["gpus"] >= 1 and entry["multi_gpu"] == (entry["gpus"] > 1)
        assert entry["steps"] and all(s["pages"] > 0 for s in entry["steps"])


def test_estimate_gives_the_design_example_for_colqwen2_hf():
    est = _encode.estimate("vidore/colqwen2-v1.0-hf", 8400)
    assert est.known and est.basis == "our Kaggle run, 0.61-0.65 s/page" and est.gpu == "one T4"
    assert est.lo_s == pytest.approx(8400 * 0.6106 + 30.4) and est.hi_s == pytest.approx(8400 * 0.645 + 74.6)
    assert est.encode_s == pytest.approx((8400 * 0.6106, 8400 * 0.645))
    c = est.claim()
    assert c.kind == "estimate" and c.text == "about 1.4-1.5 h"
    assert c.tag == "[estimate: our Kaggle run, 0.61-0.65 s/page]"
    assert est.setup_claim().text == "about 1-2 min" and est.setup_claim().tag == c.tag
    assert est.dtype == "float16" and not est.multi_gpu


def test_estimate_names_a_split_run_a_text_unit_and_an_untimed_model():
    split = _encode.estimate("tencent/EVIE-8B", 100)
    assert split.gpu == "2x T4" and split.multi_gpu
    text = _encode.estimate("colbert-ir/colbertv2.0", 5183)
    assert text.unit == "document" and text.basis.endswith("s/document")
    for model in ("vidore/colpali-v1.3-merged", None, "not/registered"):
        est = _encode.estimate(model, 500)
        assert not est.known and est.claim() is None and "not timed" in est.basis


def test_recorded_dtype_comes_from_our_runs():
    assert _encode.recorded_dtype("vidore/colqwen2-v1.0-hf") == "float16"
    assert _encode.recorded_dtype("vidore/colpali-v1.3") == "float32"
    assert _encode.recorded_dtype("colbert-ir/colbertv2.0") is None  # text runs recorded no dtype
    assert _encode.recorded_dtype("vidore/colpali-v1.3-merged") is None  # no run.json of ours
    assert _encode.recorded_dtype(None) is None


def test_install_lines_name_the_extra_and_the_model_specific_requirements():
    st = 'pip install "optivision-rag[easy-st]"'
    colpali = 'pip install "optivision-rag[easy-colpali]"'
    assert _encode.install_line("vidore/colqwen2-v1.0-hf") == st
    assert _encode.install_line("vidore/colpali-v1.3") == colpali + ' "accelerate"'  # our run split it over two T4s
    assert _encode.install_line("vidore/colSmol-500M") == colpali  # spread was asked, one GPU held it
    assert _encode.install_line("Hcompany/NeoMME-260M-Retriever-ST-late") == st + ' "transformers>=5.17"'
    assert _encode.install_line("jinaai/jina-colbert-v2") == st + ' "einops"'
    assert _encode.install_line("TomoroAI/tomoro-colqwen3-embed-4b").startswith(colpali)
    assert _encode.install_line(None) == 'pip install "optivision-rag[easy]"'


# ------------------------------------------------------------------------------ refusals


@pytest.mark.parametrize("model_id", ["TomoroAI/tomoro-colqwen3-embed-4b", "TomoroAI/tomoro-colqwen3-embed-8b",
                                      "webAI-Official/webAI-ColVec1-4b"])
def test_a_remote_code_model_is_refused_without_trust_remote_code(model_id, loader_seen):
    info = MODELS[model_id]
    with pytest.raises(RemoteCodeRefused) as caught:
        _encode.load_encoder(model_id)
    text = str(caught.value)
    assert model_id in text and info.revision in text and info.license in text
    assert "trust_remote_code=True" in text and not loader_seen
    assert isinstance(caught.value, PermissionError)


def test_every_entry_whose_notes_ask_for_remote_code_is_refused_and_others_are_not():
    needs = {m for m, info in MODELS.items() if _encode.needs_remote_code(info)}
    assert {"TomoroAI/tomoro-colqwen3-embed-4b", "TomoroAI/tomoro-colqwen3-embed-8b",
            "webAI-Official/webAI-ColVec1-4b", "jinaai/jina-colbert-v2"} <= needs
    assert "ModernVBERT/colmodernvbert" not in needs  # its notes mention another route's remote code
    assert "vidore/colqwen2-v1.0-hf" not in needs


def test_trust_remote_code_reaches_the_loader(loader_seen):
    _encode.load_encoder("TomoroAI/tomoro-colqwen3-embed-4b", trust_remote_code=True, echo=False)
    model_id, kwargs = loader_seen[-1]
    assert model_id == "TomoroAI/tomoro-colqwen3-embed-4b" and kwargs["trust_remote_code"] is True
    _encode.load_encoder("webAI-Official/webAI-ColVec1.1-8b", trust_remote_code=True, echo=False)
    assert loader_seen[-1][1]["trust_remote_code"] is True  # the sentence-transformers path too


def test_a_missing_loader_package_gives_the_exact_pip_line(monkeypatch):
    real = _encode.has_module
    monkeypatch.setattr(_encode, "has_module",
                        lambda name: False if name in ("sentence_transformers", "colpali_engine") else real(name))
    with pytest.raises(MissingExtra) as caught:
        _encode.load_encoder("vidore/colqwen2-v1.0-hf")
    assert 'pip install "optivision-rag[easy-st]"' in str(caught.value)
    assert "sentence-transformers" in str(caught.value) and isinstance(caught.value, ImportError)
    with pytest.raises(MissingExtra, match=r'pip install "optivision-rag\[easy-colpali\]"'):
        _encode.load_encoder("vidore/colpali-v1.3")


def test_a_too_old_transformers_for_neomme_is_a_missing_extra(monkeypatch):
    monkeypatch.setattr(_encode, "has_module", lambda name: True)
    monkeypatch.setattr(_encode, "installed_version", lambda dist: "5.16.1" if dist == "transformers" else "1.0")
    with pytest.raises(MissingExtra, match=r"transformers>=5\.17 \(installed: 5\.16\.1\)"):
        _encode.load_encoder("Hcompany/NeoMME-800M-Retriever-ST-late")


def test_an_import_error_during_the_load_becomes_a_missing_extra(monkeypatch, loader_seen):
    def failing(model_id, loader=None, **kwargs):
        raise ImportError("This modeling file requires the following packages: flash_attn")

    monkeypatch.setattr(adapters, "load_adapter", failing)
    with pytest.raises(MissingExtra, match="flash_attn"):
        _encode.load_encoder("vidore/colqwen2-v1.0-hf")


def test_an_unknown_id_is_refused_with_close_ids(loader_seen):
    with pytest.raises(UnknownModel, match="closest ids"):
        _encode.load_encoder("no-such-org/no-such-model")


# ------------------------------------------------------------------------------ what a load is given


def test_the_sentence_transformers_path_gets_our_settings_and_the_registry_revision(loader_seen):
    enc = _encode.load_encoder("vidore/colqwen2-v1.0-hf", echo=False)
    model_id, kwargs = loader_seen[-1]
    info = MODELS[model_id]
    assert kwargs["revision"] == info.revision  # load_adapter does not add it on this path
    assert kwargs["dtype"] == "float16" and kwargs["batch_size"] == 2
    assert kwargs["encode_kwargs"] == {"show_progress_bar": False}
    assert "model_kwargs" not in kwargs and "device" not in kwargs
    rec = enc.record
    assert rec["model"] == model_id and rec["loader"] == "sentence-transformers" and rec["registered"]
    assert rec["revision"] == info.revision and rec["resolved_revision"] == info.revision
    assert rec["dtype_ours"] == "float16" and rec["dtype_source"] == "as in our run"
    assert rec["dpi"] == 150 and rec["max_side"] == 1536 and rec["batch_size"] == 2
    assert {"torch", "transformers", "sentence-transformers", "colpali-engine", "python"} <= set(rec["versions"])
    assert "device" in rec and "gpu" in rec and "attn_implementation" in rec


def test_load_adapter_itself_still_adds_no_revision_on_the_sentence_transformers_path():
    class Stub:
        device = "cpu"

    adapter = load_adapter("vidore/colqwen2-v1.0-hf", model=Stub())
    assert adapter.revision is None  # the default is unchanged; easy mode passes revision itself


def test_colvlm_backends_get_the_registry_revision_and_token_encoders_get_it_from_load_adapter(loader_seen):
    _encode.load_encoder("vidore/colpali-v1.3", echo=False)
    kwargs = loader_seen[-1][1]
    assert kwargs["revision"] == MODELS["vidore/colpali-v1.3"].revision
    assert kwargs["dtype"] == "float32" and kwargs["multi_gpu"] is False and kwargs["device"] == "auto"
    _encode.load_encoder("vultr/VultronRetrieverFlash-Qwen3.5-0.8B", echo=False)  # colpali-engine:colqwen3.5
    assert "revision" not in loader_seen[-1][1]  # ColEngineTokenEncoder: load_adapter sets the registry pin


def test_a_model_our_run_split_is_split_again_on_two_gpus(monkeypatch, loader_seen):
    monkeypatch.setattr(_encode, "_cuda_count", lambda: 2)
    monkeypatch.setattr(_encode, "_gpu_budgets", lambda: {0: 1, 1: 2})
    enc = _encode.load_encoder("tencent/EVIE-8B", echo=False)
    kwargs = loader_seen[-1][1]
    assert kwargs["model_kwargs"] == {"device_map": "auto", "max_memory": {0: 1, 1: 2}}
    assert kwargs["processor_kwargs"] == {"size": {"longest_edge": 1048576, "shortest_edge": 65536}}
    assert kwargs["batch_size"] == 1 and kwargs["dtype"] == "float16" and enc.record["multi_gpu"]


def test_one_gpu_for_a_split_run_is_noted(monkeypatch, loader_seen):
    monkeypatch.setattr(_encode, "_cuda_count", lambda: 1)
    enc = _encode.load_encoder("tencent/EVIE-8B", echo=False)
    assert "model_kwargs" not in loader_seen[-1][1]
    assert any("this machine has one GPU" in n for n in enc.notes)


def test_an_unrecorded_dtype_uses_the_loader_default_and_says_so(loader_seen):
    enc = _encode.load_encoder("colbert-ir/colbertv2.0", echo=False)
    assert "dtype" not in loader_seen[-1][1]
    assert enc.record["dtype_source"] == "loader default; our run recorded none"
    assert any("recorded no dtype" in n for n in enc.notes)
    assert loader_seen[-1][1]["batch_size"] == 32  # a text model, no recorded batch size


def test_a_dtype_that_does_not_fit_falls_back_to_half_precision(monkeypatch, loader_seen, capsys):
    calls = []

    def oom_in_float32(model_id, loader=None, **kwargs):
        calls.append(kwargs["dtype"])
        if kwargs["dtype"] == "float32":
            raise _oom_error()
        return word_adapter()

    monkeypatch.setattr(adapters, "load_adapter", oom_in_float32)
    monkeypatch.setattr(_encode, "_cuda_count", lambda: 1)
    monkeypatch.setattr(_encode, "_bf16_native", lambda: False)
    enc = _encode.load_encoder("vidore/colpali-v1.3")
    assert calls == ["float32", "float16"]
    assert enc.record["dtype_source"].startswith("fallback") and enc.record["dtype_matches_ours"] is False
    assert any(n.startswith("dtype differs from our measurement") for n in enc.notes)
    assert "dtype differs from our measurement" in capsys.readouterr().out
    assert "dtype differs from our measurement" in enc.line().render()


def test_a_dtype_given_by_hand_is_used_and_a_difference_is_noted(loader_seen):
    enc = _encode.load_encoder("vidore/colqwen2-v1.0-hf", dtype="float32", echo=False)
    assert loader_seen[-1][1]["dtype"] == "float32"
    assert any("dtype differs from our measurement" in n for n in enc.notes)


def test_an_adapter_instance_is_used_as_it_is(loader_seen):
    adapter = word_adapter()
    enc = _encode.load_encoder(adapter)
    assert enc.adapter is adapter and not loader_seen
    assert enc.record["registered"] is False and enc.info is None
    assert any("unmeasured" in n for n in enc.notes)
    assert "your adapter" in enc.line().render()
    with pytest.raises(TypeError):
        _encode.load_encoder(object())


# ------------------------------------------------------------------------------ chunking and OOM


def test_out_of_memory_halves_the_chunk_down_to_what_fits_and_records_it():
    calls: list[int] = []

    def docs(items):
        calls.append(len(items))
        if len(items) > 2:
            raise _oom_error()
        return [_embed(_text_of(i)) for i in items]

    enc = _encode.Encoder(CallableAdapter(docs, lambda qs: [_embed(q) for q in qs]), chunk_size=8)
    pages = [FakeImage(f"page {i} words {TOPICS[i]}") for i in range(10)]
    ids = [f"f.pdf::p{i + 1}" for i in range(10)]
    seen = []
    corpus = enc.encode_pages(pages, ids, progress=lambda done, total: seen.append((done, total)))
    assert calls[:2] == [8, 4] and set(calls[2:]) == {2}
    assert enc.chunk_size == 2 and enc.record["page_chunk_final"] == 2 and enc.record["oom_halvings"] == 2
    assert enc.record["page_chunk"] == 8
    assert corpus.ids == ids and seen[-1] == (10, 10)
    for i, page in enumerate(pages):
        np.testing.assert_array_equal(corpus[i].vectors, _embed(page.text))
    calls.clear()
    enc.encode_pages(pages[:4], ids[:4])  # the smaller chunk is kept for later calls
    assert calls == [2, 2]


def test_out_of_memory_on_a_single_page_is_raised_with_advice():
    def docs(items):
        raise _oom_error()

    enc = _encode.Encoder(CallableAdapter(docs, lambda qs: []), chunk_size=4)
    with pytest.raises(RuntimeError, match="half precision") as caught:
        enc.encode_pages([FakeImage("a b c")] * 3, ["a::p1", "b::p1", "c::p1"])
    assert _encode._is_oom(caught.value.__cause__)


def test_a_plain_runtime_error_saying_out_of_memory_counts_and_other_errors_pass_through():
    assert _encode._is_oom(RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB"))
    assert not _encode._is_oom(RuntimeError("shape mismatch"))

    def docs(items):
        raise ValueError("not an OOM")

    enc = _encode.Encoder(CallableAdapter(docs, lambda qs: []), chunk_size=4)
    with pytest.raises(ValueError, match="not an OOM"):
        enc.encode_pages([FakeImage("x")], ["x::p1"])
    assert enc.oom_halvings == 0


def test_queries_are_encoded_in_chunks_with_default_ids():
    enc = _encode.Encoder(word_adapter())
    corpus = enc.encode_queries([f"query {i}" for i in range(70)])
    assert len(corpus) == 70 and corpus.ids[:3] == ["0", "1", "2"]
    np.testing.assert_array_equal(corpus[5].vectors, _embed("query 5"))


def test_page_adapters_get_page_refs_and_the_images_are_not_kept(page_image):
    from optivision.adapters import PageEncoderAdapter
    from optivision.encoders.synthetic import SyntheticEncoder

    enc = _encode.Encoder(PageEncoderAdapter(SyntheticEncoder(dim=16, grid=4)), chunk_size=1)
    assert enc._pairs
    corpus = enc.encode_pages([page_image, page_image.copy()], ["my_pdfs/a.pdf::p3", "loose-id"])
    assert corpus.ids == ["my_pdfs/a.pdf::p3", "loose-id"] and corpus.images is None
    assert corpus.metadata[0]["page_ref"]["doc_id"] == "my_pdfs/a.pdf" and corpus.metadata[0]["page_ref"]["page_no"] == 3
    assert corpus.metadata[1]["page_ref"] == {**corpus.metadata[1]["page_ref"], "doc_id": "loose-id", "page_no": 1}
    assert corpus.protected is not None  # the page layout the adapter carries is kept
    assert _encode._split_id("my_pdfs/sub/r.pdf::p12") == ("my_pdfs/sub/r.pdf", 12)
    assert _encode._split_id("loose-id") == ("loose-id", 1)


def test_text_models_get_strings():
    seen = []
    adapter = CallableAdapter(lambda items: seen.extend(items) or [_embed(str(i)) for i in items],
                              lambda qs: [_embed(q) for q in qs])
    enc = _encode.Encoder(adapter, modality="text")
    enc.encode_pages(["alpha beta", "gamma delta"], ["d1", "d2"])
    assert seen == ["alpha beta", "gamma delta"]


# ------------------------------------------------------------------------------ the wiring check


def test_select_pairs_takes_one_page_per_file_with_a_sentence_no_other_page_holds():
    pairs = _encode.select_pairs(user_pages(14))
    assert len(pairs) == 12
    assert len({rec.doc_id for rec, _, _ in pairs}) == 12
    texts = [rec.text.lower() for rec, _, _ in pairs]
    for i, (_, _, sentence) in enumerate(pairs):
        holders = [j for j, t in enumerate(texts) if sentence.lower() in t]
        assert holders == [i]
        assert not any(ch.isdigit() for ch in sentence)


def test_select_pairs_spreads_over_the_pages_of_one_file_and_skips_blank_and_textless_pages():
    pages = [(Rec(f"one.pdf::p{i}", "one.pdf", f"The {t} section explains how the {t.split()[1]} work is planned "
                  f"and checked by the team."), None) for i, t in enumerate(TOPICS, start=1)]
    pages.append((Rec("one.pdf::p99", "one.pdf", "short"), None))
    pages.append((Rec("one.pdf::p98", "one.pdf", "A blank page with enough characters in its text layer.", True), None))
    pairs = _encode.select_pairs(pages)
    ids = [rec.id for rec, _, _ in pairs]
    assert len(pairs) == 12 and "one.pdf::p99" not in ids and "one.pdf::p98" not in ids


def test_wiring_passes_on_the_users_pages():
    enc = _encode.Encoder(word_adapter())
    wiring = _encode.wiring_check(enc, user_pages(14))
    assert wiring.ok and not wiring.used_sample and wiring.sample is None
    assert (wiring.hits, wiring.n, wiring.files) == (12, 12, 12)
    assert "12 pages from 12 files" in wiring.message and "12 of 12 found their own page. OK." in wiring.message
    assert wiring.message.endswith("[counted]") and wiring.line().label == "Wiring"


def test_wiring_falls_back_to_the_sample_when_the_users_pages_fail(fake_sample):
    enc = _encode.Encoder(word_adapter())
    wiring = _encode.wiring_check(enc, user_pages(14, blank_images=True))
    assert wiring.ok and wiring.used_sample
    assert wiring.user is not None and not wiring.user["ok"] and wiring.sample["ok"]
    assert (wiring.hits, wiring.n) == (12, 12)
    assert "your pages look alike; wiring is fine on our sample" in wiring.message
    assert wiring.sample["pages"][0].startswith("sample.pdf::p")


def test_wiring_with_too_few_text_pages_checks_on_the_sample(fake_sample):
    enc = _encode.Encoder(word_adapter())
    wiring = _encode.wiring_check(enc, user_pages(3))
    assert wiring.ok and wiring.used_sample and wiring.user is None
    assert "too few pages with a text layer" in wiring.message


def test_wiring_refuses_when_the_sample_fails_too(fake_sample):
    enc = _encode.Encoder(constant_adapter())
    with pytest.raises(WiringError) as caught:
        _encode.wiring_check(enc, user_pages(14))
    text = str(caught.value)
    assert "Your pages" in text and "Our sample" in text and "0 of 12 found their own page" in text
    assert "nothing was written" in text and text.rstrip().endswith("ov.doctor()")


def test_wiring_failure_names_vectors_that_are_not_unit_length(fake_sample):
    adapter = CallableAdapter(lambda items: [3.0 * _embed(_text_of(i)) for i in items],
                              lambda qs: [3.0 * _embed(q) for q in qs])
    with pytest.raises(WiringError, match="not unit length"):
        _encode.wiring_check(_encode.Encoder(adapter), user_pages(14))


def test_wiring_on_a_text_model_uses_the_page_text(fake_sample):
    enc = _encode.Encoder(word_adapter(), modality="text")
    pages = [(rec, None) for rec, _ in user_pages(14)]
    assert _encode.wiring_check(enc, pages).ok


# ------------------------------------------------------------------------------ the sample PDF


def test_the_sample_pdf_is_small_has_twenty_pages_and_twelve_distinct_sentences():
    path = _encode.sample_path()
    assert path.is_file() and path.stat().st_size < 200_000
    records = _encode._sample_records()
    assert len(records) == 20 and all(len(r.text) > 200 for r in records)
    pairs = _encode.select_pairs([(r, None) for r in records])
    assert len(pairs) == 12 and len({s for _, _, s in pairs}) == 12
    for rec, _, sentence in pairs:
        assert [r.id for r in records if sentence.lower() in " ".join(r.text.split()).lower()] == [rec.id]


def test_the_sample_pdf_matches_its_script():
    builder = _module(SAMPLE_BUILDER, "make_sample_pdf")
    assert builder.check(_encode.sample_path()) == []
    assert len(builder.PAGES) == 20


def test_the_sample_renders_like_a_users_page():
    images = _encode._render_sample([1, 20])
    assert len(images) == 2 and max(images[0].size) <= _encode.MAX_SIDE and images[0].mode == "RGB"


def test_user_facing_text_avoids_the_banned_words():
    from tests.test_docs_examples import BANNED_PHRASES, BANNED_WORDS

    source = Path(_encode.__file__).read_text(encoding="utf-8")
    texts = [" ".join(re.findall(r'"([^"\n]*)"', source)),  # the module's string literals
             " ".join(_module(SAMPLE_BUILDER, "make_sample_pdf").page_text(i) for i in range(20))]
    for text in texts:
        low = re.sub(r"\S*/\S*", " ", text.lower())  # paths and URLs are not claims
        for phrase in (*BANNED_PHRASES, "lossless", "universal", "guaranteed", "sota", "state of the art", "veto"):
            assert phrase not in low, phrase
        assert BANNED_WORDS.search(low) is None
        assert re.search(r"\bfirst\b", low) is None


# ------------------------------------------------------------------------------ the ColVLMEncoder revision


def _fake_colpali_engine(monkeypatch, seen: dict):
    import torch

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2)

    class FakeColQwen2:
        pass

    class FakeProcessor:
        image_token_id = 7

        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            seen["processor"] = ((checkpoint,), kwargs)
            return cls()

    models = types.ModuleType("colpali_engine.models")
    models.ColQwen2 = FakeColQwen2
    models.ColQwen2Processor = FakeProcessor
    package = types.ModuleType("colpali_engine")
    package.models = models
    monkeypatch.setitem(sys.modules, "colpali_engine", package)
    monkeypatch.setitem(sys.modules, "colpali_engine.models", models)
    from optivision.encoders import colvlm

    def fake_load_model(*args, **kwargs):
        seen["load"] = (args, kwargs)
        return FakeModel()

    monkeypatch.setattr(colvlm, "_load_model", fake_load_model)
    monkeypatch.setattr(colvlm, "place_model", lambda model, device, multi: (model, device, {"devices": [device]}))
    return colvlm, FakeColQwen2


def test_colvlm_encoder_default_calls_are_unchanged_and_a_revision_is_passed_on(monkeypatch):
    torch = pytest.importorskip("torch")
    seen: dict = {}
    colvlm, model_cls = _fake_colpali_engine(monkeypatch, seen)
    enc = colvlm.ColVLMEncoder(backend="colqwen2", model_name="org/ckpt", device="cpu", dtype="float32")
    assert seen["load"] == ((model_cls, "org/ckpt", torch.float32), {})  # exactly the 0.5.1 call
    assert seen["processor"] == (("org/ckpt",), {})
    assert enc.revision is None
    from optivision.adapters import PageEncoderAdapter

    assert PageEncoderAdapter(enc).identity() is None or PageEncoderAdapter(enc).identity().revision is None

    enc = colvlm.ColVLMEncoder(backend="colqwen2", model_name="org/ckpt", device="cpu", dtype="float32",
                               max_num_visual_tokens=768, revision="abc123")
    assert seen["load"] == ((model_cls, "org/ckpt", torch.float32), {"revision": "abc123"})
    assert seen["processor"] == (("org/ckpt",), {"revision": "abc123", "max_num_visual_tokens": 768})
    assert enc.revision == "abc123"


def test_colvlm_encoder_signature_only_gained_an_optional_revision():
    import inspect

    from optivision.encoders.colvlm import ColVLMEncoder

    params = inspect.signature(ColVLMEncoder.__init__).parameters
    assert list(params)[-1] == "revision" and params["revision"].default is None
    assert list(params)[:-1] == ["self", "backend", "model_name", "device", "dtype", "query_batch_size",
                                 "multi_gpu", "max_num_visual_tokens"]


# ------------------------------------------------------------------------------ light import


def test_the_module_imports_no_torch_or_loader_at_import_time():
    code =("import sys\n"
            "for name in ('torch', 'scipy', 'transformers', 'sentence_transformers', 'colpali_engine', 'pypdfium2'):\n"
            "    sys.modules[name] = None\n"
            "import optivision.easy._encode as e\n"
            "print(e.install_line('vidore/colqwen2-v1.0-hf'))\n"
            "print(e.estimate('vidore/colqwen2-v1.0-hf', 10).basis)\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120,
                            check=False)
    assert result.returncode == 0, result.stderr
    assert 'optivision-rag[easy-st]' in result.stdout and "0.61-0.65" in result.stdout
