"""Tests for the demo adapter.

These use the synthetic encoder deliberately: the point is to prove the adapter
*reports what the pipeline actually produced*, which is a property of the
plumbing and does not need a 0.5 GB checkpoint. The demo app itself refuses the
synthetic backend — that refusal is tested here too.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from optivision.config import Config
from optivision.corpus import CorpusSpec, generate_synthetic_corpus
from optivision.demo import (
    CompressionResult,
    DemoError,
    DemoPipeline,
    load_page,
    looks_blank,
    on_device,
    pdf_page_count,
    pick_device,
    store_in_qdrant,
)

from .conftest import make_page


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("democorpus")
    generate_synthetic_corpus(out, CorpusSpec(n_docs=2, pages_per_doc=2, seed=5))
    return out


@pytest.fixture
def pipeline(corpus) -> DemoPipeline:
    cfg = Config()
    cfg.encoder.backend = "synthetic"
    cfg.encoder.synthetic_dim = 64
    cfg.encoder.synthetic_grid = 16
    cfg.encoder.synthetic_layout = str(corpus / "layout.json")
    return DemoPipeline(cfg)


class TestLoadPage:
    def test_renders_a_pdf_page(self, corpus):
        pdf = next((corpus / "pdfs").glob("*.pdf"))
        image, total = load_page(pdf, 1)
        assert isinstance(image, Image.Image)
        assert total == 2
        assert max(image.size) <= 1536

    def test_page_count_matches(self, corpus):
        pdf = next((corpus / "pdfs").glob("*.pdf"))
        assert pdf_page_count(pdf) == 2

    def test_second_page_differs_from_first(self, corpus):
        pdf = next((corpus / "pdfs").glob("*.pdf"))
        a, _ = load_page(pdf, 1)
        b, _ = load_page(pdf, 2)
        assert not np.array_equal(np.asarray(a), np.asarray(b))

    def test_out_of_range_page_is_a_friendly_error(self, corpus):
        pdf = next((corpus / "pdfs").glob("*.pdf"))
        with pytest.raises(DemoError, match="does not exist"):
            load_page(pdf, 99)

    def test_unsupported_type_is_a_friendly_error(self, tmp_path):
        bad = tmp_path / "notes.txt"
        bad.write_text("hello")
        with pytest.raises(DemoError, match="Unsupported file type"):
            load_page(bad)

    def test_missing_file_is_a_friendly_error(self, tmp_path):
        with pytest.raises(DemoError, match="could not be read"):
            load_page(tmp_path / "nope.pdf")

    def test_image_file_loads(self, tmp_path):
        path = tmp_path / "scan.png"
        make_page().save(path)
        image, total = load_page(path)
        assert total == 1 and image.mode == "RGB"


class TestBlankDetection:
    def test_blank_page_is_flagged(self, blank_image):
        assert looks_blank(blank_image)

    def test_inked_page_is_not_flagged(self, page_image):
        assert not looks_blank(page_image)


class TestCompressionResult:
    def test_numbers_match_the_arrays_they_came_from(self, pipeline, corpus):
        """The whole point of the adapter: no number may drift from the tensors."""
        pdf = next((corpus / "pdfs").glob("*.pdf"))
        image, _ = load_page(pdf, 1)
        result = pipeline.compress(image, page_id=pdf.stem)

        page = result.compressed_page
        assert result.original_tokens == page.n_tokens_before
        assert result.final_tokens == page.n_vectors == page.codes.shape[0]
        assert result.embedding_dim == page.dim
        assert result.original_bytes == page.raw_nbytes()
        assert result.compressed_bytes == page.nbytes == page.codes.nbytes

    def test_storage_arithmetic_is_exactly_the_documented_formula(self, pipeline, corpus):
        pdf = next((corpus / "pdfs").glob("*.pdf"))
        image, _ = load_page(pdf, 1)
        r = pipeline.compress(image, page_id=pdf.stem)

        assert r.original_bytes == r.original_tokens * r.embedding_dim * 4
        assert r.compressed_bytes == r.final_tokens * r.embedding_dim // 8
        assert r.bits_per_dimension == pytest.approx(1.0)
        assert r.storage_reduction_factor == pytest.approx(
            r.original_bytes / r.compressed_bytes
        )
        assert r.token_reduction_percent == pytest.approx(
            (r.original_tokens - r.final_tokens) / r.original_tokens * 100
        )

    def test_token_counts_are_internally_consistent(self, pipeline, corpus):
        pdf = next((corpus / "pdfs").glob("*.pdf"))
        image, _ = load_page(pdf, 1)
        r = pipeline.compress(image, page_id=pdf.stem)

        # final = surviving patches + the tokens that are never pruned
        assert r.final_tokens == r.redundancy_tokens + r.non_grid_tokens
        assert r.original_tokens == r.grid_patches + r.non_grid_tokens
        assert r.redundancy_tokens <= r.spatial_tokens <= r.grid_patches
        assert r.final_tokens < r.original_tokens

    def test_compression_actually_shrinks(self, pipeline, corpus):
        pdf = next((corpus / "pdfs").glob("*.pdf"))
        image, _ = load_page(pdf, 1)
        r = pipeline.compress(image, page_id=pdf.stem)
        assert r.compressed_bytes < r.original_bytes
        assert r.storage_reduction_factor > 20
        assert 0 < r.token_reduction_percent < 100

    def test_visualisation_matches_the_pipeline_mask(self, pipeline, corpus):
        """The retained-regions image must be the real keep-mask, not a heatmap."""
        pdf = next((corpus / "pdfs").glob("*.pdf"))
        image, _ = load_page(pdf, 1)
        r = pipeline.compress(image, page_id=pdf.stem)

        assert r.retained_regions_image is not None
        assert r.retained_regions_image.size == image.size
        assert r.original_image is not None
        # An overlay that changed nothing would mean the mask was not applied.
        assert not np.array_equal(
            np.asarray(r.retained_regions_image), np.asarray(r.original_image)
        )

    def test_progress_messages_are_emitted(self, pipeline, corpus):
        pdf = next((corpus / "pdfs").glob("*.pdf"))
        image, _ = load_page(pdf, 1)
        seen: list[str] = []
        pipeline.compress(image, page_id=pdf.stem, progress=seen.append)
        assert any("visual tokens" in m for m in seen)
        assert seen[-1].startswith("Compression complete")

    def test_as_dict_is_scalar_only(self, pipeline, corpus):
        pdf = next((corpus / "pdfs").glob("*.pdf"))
        image, _ = load_page(pdf, 1)
        data = pipeline.compress(image, page_id=pdf.stem).as_dict()
        assert "original_image" not in data
        assert "compressed_page" not in data
        assert isinstance(data["original_tokens"], int)

    def test_timings_are_real_and_positive(self, pipeline, corpus):
        pdf = next((corpus / "pdfs").glob("*.pdf"))
        image, _ = load_page(pdf, 1)
        r = pipeline.compress(image, page_id=pdf.stem)
        assert r.encoding_seconds > 0
        assert r.pruning_ms > 0
        assert r.quantization_ms > 0


class TestDeviceHandling:
    def test_pick_device_returns_something_usable(self):
        assert pick_device() in {"cpu", "cuda", "mps"}

    def test_on_device_is_a_noop_without_a_torch_model(self, pipeline):
        encoder = pipeline.encoder
        with on_device(encoder, "cuda") as same:
            assert same is encoder  # synthetic encoder has no .model to move


class TestOptionalQdrant:
    def test_storing_a_page_reports_the_multivector(self, pipeline, corpus, tmp_path):
        pdf = next((corpus / "pdfs").glob("*.pdf"))
        image, _ = load_page(pdf, 1)
        r = pipeline.compress(image, page_id=pdf.stem)

        info = store_in_qdrant(
            r.compressed_page, collection="test_demo", path=str(tmp_path / "q")
        )
        assert info["ok"] is True
        assert info["n_points"] == 1
        assert info["n_vectors"] == r.final_tokens

    def test_failure_is_reported_not_raised(self, tmp_path):
        """The compression demo must never fail because Qdrant is unhappy.

        ``path`` must stay inside tmp_path: an embedded Qdrant client creates its
        storage directory eagerly, so a relative path here would litter the repo.
        """

        class Broken:
            dim = 8
            n_vectors = 1  # no .codes, so the upsert cannot succeed

        info = store_in_qdrant(Broken(), collection="x", path=str(tmp_path / "q"))
        assert info["ok"] is False
        assert "error" in info


class TestAppRefusesSyntheticBackend:
    def test_demo_pipeline_itself_allows_synthetic_for_tests(self, pipeline):
        # The adapter is backend-agnostic; only the app enforces the policy.
        assert pipeline.cfg.encoder.backend == "synthetic"

    def test_app_rejects_synthetic_config(self, tmp_path, monkeypatch):
        pytest.importorskip("gradio")
        import app as app_module

        cfg = Config()
        cfg.encoder.backend = "synthetic"
        cfg_path = tmp_path / "syn.yaml"
        cfg.dump(cfg_path)

        monkeypatch.setattr(app_module, "CONFIG_PATH", str(cfg_path))
        monkeypatch.setattr(app_module, "_PIPELINE", None)
        with pytest.raises(DemoError, match="synthetic"):
            app_module.get_pipeline()


def test_result_dataclass_has_the_documented_fields():
    """Guards the contract the UI renders against."""
    expected = {
        "original_tokens", "spatial_tokens", "final_tokens", "embedding_dim",
        "original_bytes", "compressed_bytes", "token_reduction_percent",
        "storage_reduction_factor", "encoding_seconds", "original_image",
        "retained_regions_image",
    }
    assert expected <= set(CompressionResult.__dataclass_fields__)
