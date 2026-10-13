"""The v0.3.0 public surface: names, import paths, registries and the CLI.

Everything here is a promise to existing users, scripts, the VS Code extension and the
research harness (docs/package/ARCHITECTURE_AUDIT.md, sections 3 and 10). New code may add
to the surface; none of these may disappear, move, or start needing an optional dependency.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

import optivision

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _words(text: str) -> list[str]:
    return text.split()


#: optivision.__all__ at v0.3.0 (audit section 3.1)
ALL_V030 = _words(
    "Config CompressionConfig EncoderConfig IndexConfig IngestConfig PruningConfig SearchConfig "
    "OptiVisionRAG IndexReport PageRef PageEncoding PrunedPage CompressedPage SearchHit "
    "SearchResult MultiVectorCorpus MultiVectorRepresentation Pipeline CompressedCorpus "
    "maxsim_matrix AdaptiveMerge HierarchicalMerge RandomPruner RedundancyPruner SpatialPruner "
    "PCAProjector RandomProjector TruncateProjector DimensionProjector Float32Quantizer "
    "Float16Quantizer Int8Quantizer Int4Quantizer Lloyd2Quantizer BinaryQuantizer "
    "calibrate CalibrationResult optimize OptimizationResult __version__"
)

#: (module -> names) imported by tests, scripts, frozen research code and apps (audit 3.2).
#: A leading underscore is a private name that some consumer still uses; names that are
#: modules (``encoders.nemotron``) are resolved the way ``from package import name`` does.
SURFACE = {
    "optivision": " ".join([*ALL_V030, "calibration"]),
    "optivision.adapters": (
        "MEASURED UNVERIFIED MODELS supported_models load_adapter PageEncoderAdapter "
        "SentenceTransformersAdapter"
    ),
    "optivision.bench": (
        "EncodedCorpus Variant default_variants keep_ratio_sweep run_variant run_benchmark "
        "_encode_queries_cached"
    ),
    "optivision.benchmark": (
        "Dataset legacy_variants load_legacy_dataset load_vector_dataset run_matrix save_result"
    ),
    "optivision.calibration": (
        "Candidate calibrate score_space score_candidate select pseudo_queries basic_search_space "
        "recommended_search_space default_search_space wide_search_space projection_targets "
        "_compress _judge _per_candidate_confidence _rank_key TRANSFORM_CACHE_BYTES _cache_put"
    ),
    "optivision.cli": "app",
    "optivision.cli_universal": "parse_pipeline",
    "optivision.compose": "Pipeline CompressedCorpus save_compressed load_compressed",
    "optivision.compression": (
        "Compressor INT8_SCALE decode code_nbytes lloyd2_code_nbytes pack_bits unpack_signs pack2 "
        "unpack2 rotation_matrix fit_lloyd2 encode_lloyd2 decode_lloyd2 maxsim_float "
        "maxsim_asymmetric maxsim_asymmetric_batch maxsim_asymmetric_lloyd2 "
        "maxsim_asymmetric_lloyd2_batch _int8_codes"
    ),
    "optivision.compression.binary": "pack_bits unpack_signs",
    "optivision.config": "Config CompressionConfig PruningConfig available_presets",
    "optivision.corpus": "CorpusSpec generate_synthetic_corpus load_queries",
    "optivision.demo": (
        "CompressionResult DemoError DemoPipeline load_page looks_blank on_device pdf_page_count "
        "pick_device store_in_qdrant"
    ),
    "optivision.encoders": "SyntheticEncoder get_encoder nemotron",
    "optivision.encoders.colvlm": (
        "ColVLMEncoder BACKENDS layer_device_map _adapter_base _from_pretrained"
    ),
    "optivision.encoders.nemotron": "NemotronColEmbedEncoder KNOWN _from_pretrained",
    "optivision.encoders.synthetic": "SyntheticEncoder",
    "optivision.evaluation": (
        "measure mean_metrics parse_metric per_query_metrics relevant_from_baseline "
        "relevant_from_qrels retention retention_lower_bound split_queries"
    ),
    "optivision.index.numpy_index": "NumpyIndex",
    "optivision.index.qdrant_index": "QdrantIndex",
    "optivision.ingest": "iter_pages",
    "optivision.metrics": (
        "evaluate ndcg_at_k mrr_at_k recall_at_k rank_correlation rank_correlation_shared"
    ),
    "optivision.optimize": "optimize OptimizationResult",
    "optivision.pareto": "choose dominates pareto_front",
    "optivision.pipeline": "OptiVisionRAG",
    "optivision.pruning": (
        "TokenPruner build_keep_mask codebook_saliency fit_codebook patch_saliency prune_redundant"
    ),
    "optivision.pruning.saliency": "patch_saliency",
    "optivision.representation": "MultiVectorCorpus MultiVectorRepresentation as_corpus",
    "optivision.scoring": "maxsim_matrix maxsim_pair rank segment_reduce",
    "optivision.stages": (
        "AdaptiveMerge HierarchicalMerge RandomPruner RedundancyPruner SpatialPruner PCAProjector "
        "RandomProjector TruncateProjector DimensionProjector Float32Quantizer Float16Quantizer "
        "Int8Quantizer Int4Quantizer Lloyd2Quantizer BinaryQuantizer stage_from_dict"
    ),
    "optivision.stages.base": "DocView FIT_SAMPLE_ROWS fit_sample_rows",
    "optivision.stages.merge": "WIDE_DIM _pool _unit _ward_input",
    "optivision.storage": "ExactIndex TieredIndex NpzStorage to_legacy_pages",
    "optivision.types": "PageRef PatchGrid PageEncoding PrunedPage CompressedPage",
    "optivision.viz": "overlay_mask overlay_saliency",
}
PAIRS = [(module, name) for module, names in SURFACE.items() for name in _words(names)]

#: registry names of the stages (audit section 10); new stages may be added, none removed
STAGE_NAMES = {
    "adaptive_merge", "binary", "float16", "float32", "hierarchical_merge", "int4", "int8",
    "lloyd2", "pca", "project", "random_projection", "random_prune", "redundancy", "spatial",
    "truncate",
}

#: modules that `import optivision` loads and therefore binds as attributes of the package
PACKAGE_SUBMODULES = _words(
    "calibration compose compression config encoders evaluation index ingest pipeline pruning "
    "representation scoring stages types"
)

#: what the VS Code extension passes to the CLI (vscode-extension/src/extension.ts, buildArgs)
EXTENSION_COMMANDS = {
    "init-config": ["--backend"],
    "make-corpus": ["--docs", "--pages", "--seed", "--code-scale"],
    "index": ["--append", "--float-cache", "--limit", "-c"],
    "search": ["--top-k", "-c"],
    "stats": ["-c"],
    "explain": ["--out", "--limit", "-c"],
    "bench": ["--out", "--limit", "--sweep", "--codebook", "--cache", "--top-k", "-c"],
    "fetch-vidore": ["--dataset", "--out", "--limit"],
}
UNIVERSAL_COMMANDS = ["inspect", "compress", "calibrate", "benchmark", "compare"]

#: optional packages that no name above may need in order to be importable
OPTIONAL_DEPENDENCIES = [
    "torch", "transformers", "qdrant_client", "colpali_engine", "peft", "accelerate",
    "huggingface_hub", "sentence_transformers", "scipy", "datasets", "reportlab",
]


def _resolve(module: str, name: str):
    """``from module import name``: an attribute, else a submodule."""
    mod = importlib.import_module(module)
    if hasattr(mod, name):
        return getattr(mod, name)
    return importlib.import_module(f"{module}.{name}")


def _env() -> dict[str, str]:
    path = os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])
    return {**os.environ, "PYTHONPATH": path}


# ------------------------------------------------------------- the package namespace


def test_the_transcribed_all_is_the_forty_names():
    assert len(ALL_V030) == len(set(ALL_V030)) == 40


def test_all_still_contains_the_forty_names_of_v030():
    assert set(ALL_V030) <= set(optivision.__all__)


@pytest.mark.parametrize("name", ALL_V030)
def test_each_name_of_all_resolves(name):
    assert hasattr(optivision, name)


def test_every_name_in_all_resolves():
    assert [name for name in optivision.__all__ if not hasattr(optivision, name)] == []


@pytest.mark.parametrize("name", PACKAGE_SUBMODULES)
def test_submodules_loaded_by_import_are_attributes_of_the_package(name):
    assert isinstance(getattr(optivision, name), types.ModuleType)
    assert getattr(optivision, name).__name__ == f"optivision.{name}"


def test_optimize_is_the_function_not_the_module():
    assert isinstance(optivision.optimize, types.FunctionType)
    assert optivision.optimize.__module__ == "optivision.optimize"
    module = sys.modules["optivision.optimize"]
    assert isinstance(module, types.ModuleType)
    assert module.optimize is optivision.optimize
    assert module.OptimizationResult is optivision.OptimizationResult


@pytest.mark.parametrize(("name", "module"), [
    ("Pipeline", "optivision.compose"), ("CompressedCorpus", "optivision.compose"),
    ("MultiVectorCorpus", "optivision.representation"),
    ("MultiVectorRepresentation", "optivision.representation"),
    ("maxsim_matrix", "optivision.scoring"), ("calibrate", "optivision.calibration"),
    ("OptiVisionRAG", "optivision.pipeline"), ("PageRef", "optivision.types"),
    ("Int8Quantizer", "optivision.stages"), ("SpatialPruner", "optivision.stages"),
    ("Config", "optivision.config"),
])
def test_the_names_in_all_are_the_same_objects_as_in_their_modules(name, module):
    assert getattr(optivision, name) is getattr(importlib.import_module(module), name)


@pytest.mark.skipif(not (SRC / "optivision" / "__init__.py").is_file(),
                    reason="the repository layout (src/) is absent")
def test_version_is_a_string_literal_in_init():
    """The publish workflow greps this line, and setuptools reads the attribute."""
    assert isinstance(optivision.__version__, str)
    text = (SRC / "optivision" / "__init__.py").read_text(encoding="utf-8")
    pattern = rf'^__version__ = "{re.escape(optivision.__version__)}"$'
    assert re.search(pattern, text, re.MULTILINE)


# ------------------------------------------------------------------- import pairs


def test_the_surface_has_no_duplicate_pairs_and_keeps_its_size():
    assert len(PAIRS) == len(set(PAIRS))
    assert len(PAIRS) >= 200  # the audit counted 201 pairs, the 40 names of __all__ among them


@pytest.mark.parametrize(("module", "name"), PAIRS, ids=[f"{m}:{n}" for m, n in PAIRS])
def test_import_pair_resolves(module, name):
    assert _resolve(module, name) is not None


def test_no_import_pair_needs_an_optional_dependency():
    """Block torch, transformers, qdrant, SciPy and the rest, then resolve every pair."""
    code = (
        "import importlib, json, sys\n"
        "surface, blocked = json.loads(sys.argv[1]), json.loads(sys.argv[2])\n"
        "for name in blocked:\n"
        "    sys.modules[name] = None\n"  # `import name` now raises ImportError
        "import optivision.compose\n"
        "registry = sorted(importlib.import_module('optivision.stages.base').STAGES)\n"
        "failures = []\n"
        "for module, names in surface.items():\n"
        "    try:\n"
        "        mod = importlib.import_module(module)\n"
        "    except Exception as exc:\n"
        "        failures.append(f'{module}: {type(exc).__name__}: {exc}')\n"
        "        continue\n"
        "    for name in names.split():\n"
        "        if hasattr(mod, name):\n"
        "            continue\n"
        "        try:\n"
        "            importlib.import_module(f'{module}.{name}')\n"
        "        except Exception as exc:\n"
        "            failures.append(f'{module}.{name}: {type(exc).__name__}: {exc}')\n"
        "print(json.dumps({'failures': failures, 'registry': registry}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, json.dumps(SURFACE), json.dumps(OPTIONAL_DEPENDENCIES)],
        capture_output=True, text=True, env=_env(), check=False, timeout=300,
    )
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout.strip().splitlines()[-1])
    assert report["failures"] == []
    # importing only optivision.compose already registered every stage (no lazy registry gap)
    assert STAGE_NAMES <= set(report["registry"])


# ------------------------------------------------------------------ registries


def test_the_stage_registry_holds_the_fifteen_names():
    from optivision.stages.base import STAGES

    assert STAGE_NAMES <= set(STAGES)
    assert len(STAGE_NAMES) == 15


@pytest.mark.parametrize("name", sorted(STAGE_NAMES))
def test_each_registered_stage_is_a_stage_class_with_its_own_name(name):
    from optivision.stages.base import STAGES, Stage

    assert issubclass(STAGES[name], Stage) and STAGES[name].name == name


def test_a_pipeline_spec_naming_any_default_stage_can_be_rebuilt():
    from optivision.compose import Pipeline

    spec = {"stages": [{"stage": "float16"}]}
    assert Pipeline.from_dict(spec).to_dict() == spec


# ---------------------------------------------------------------------- the CLI


def _commands():
    import typer.main

    from optivision.cli import app

    return typer.main.get_command(app).commands


def test_cli_app_exists():
    import optivision.cli

    assert optivision.cli.app is not None and callable(optivision.cli.app)
    assert optivision.cli.__name__ == "optivision.cli"


@pytest.mark.parametrize("command", sorted(EXTENSION_COMMANDS))
def test_the_cli_commands_and_options_the_extension_uses_exist(command):
    assert command in _commands()
    params = _commands()[command].params
    options = {opt for p in params for opt in (*p.opts, *p.secondary_opts)}
    assert set(EXTENSION_COMMANDS[command]) <= options


def test_the_universal_commands_are_registered_on_the_main_app():
    assert set(UNIVERSAL_COMMANDS) <= set(_commands())


def test_python_dash_m_optivision_cli_runs():
    done = subprocess.run(
        [sys.executable, "-m", "optivision.cli", "--version"],
        capture_output=True, text=True, env=_env(), check=False, timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == f"optivision-rag {optivision.__version__}"


@pytest.mark.skipif(not (ROOT / "pyproject.toml").is_file(),
                    reason="the repository layout (pyproject.toml) is absent")
def test_the_console_script_entry_point_is_declared():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["name"] == "optivision-rag"
    assert data["project"]["scripts"]["optivision"] == "optivision.cli:app"
