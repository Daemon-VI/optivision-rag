"""The layer namespaces ``optivision.ext.vision`` and ``optivision.legacy``.

They move no code: every name is the object it always was, importable from its old path, and
reaching a name does not pull in torch, transformers or Qdrant.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from optivision import ext, legacy
from optivision.ext import vision

SRC = Path(__file__).resolve().parents[1] / "src"

#: (namespace module, name) -> (module the object comes from, attribute there)
VISION = {
    "PageRef": "optivision.types", "PatchGrid": "optivision.types",
    "PageEncoding": "optivision.types", "PrunedPage": "optivision.types",
    "SpatialPruner": "optivision.stages",
    "TokenPruner": "optivision.pruning", "build_keep_mask": "optivision.pruning",
    "dilate_mask": "optivision.pruning", "patch_saliency": "optivision.pruning",
    "codebook_saliency": "optivision.pruning", "fit_codebook": "optivision.pruning",
    "BaseEncoder": "optivision.encoders", "SyntheticEncoder": "optivision.encoders",
    "get_encoder": "optivision.encoders",
    "ColVLMEncoder": "optivision.encoders.colvlm",
    "NemotronColEmbedEncoder": "optivision.encoders.nemotron",
    "PageEncoderAdapter": "optivision.adapters", "NemotronAdapter": "optivision.adapters",
    "iter_pages": "optivision.ingest",
    "overlay_mask": "optivision.viz", "overlay_saliency": "optivision.viz",
    "explain_page": "optivision.viz",
}
LEGACY = {
    "Config": "optivision.config", "CompressionConfig": "optivision.config",
    "EncoderConfig": "optivision.config", "IndexConfig": "optivision.config",
    "IngestConfig": "optivision.config", "PruningConfig": "optivision.config",
    "SearchConfig": "optivision.config", "available_presets": "optivision.config",
    "OptiVisionRAG": "optivision.pipeline", "IndexReport": "optivision.pipeline",
    "Compressor": "optivision.compression",
    "CompressedPage": "optivision.types", "SearchHit": "optivision.types",
    "SearchResult": "optivision.types",
    "TokenPruner": "optivision.pruning",
    "EncodedCorpus": "optivision.bench", "Variant": "optivision.bench",
    "default_variants": "optivision.bench", "keep_ratio_sweep": "optivision.bench",
    "codebook_sweep": "optivision.bench", "run_variant": "optivision.bench",
    "run_benchmark": "optivision.bench",
}
NAMESPACES = [("optivision.ext.vision", vision, VISION), ("optivision.legacy", legacy, LEGACY)]
NAMESPACE_IDS = [label for label, _, _ in NAMESPACES]
CASES = [(label, name, source) for label, _, table in NAMESPACES for name, source in table.items()]
HEAVY = ("torch", "transformers", "qdrant_client", "colpali_engine", "sentence_transformers",
         "scipy", "peft", "accelerate", "huggingface_hub")


def _run(code: str) -> subprocess.CompletedProcess:
    path = os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": path},
                          capture_output=True, text=True, check=False, timeout=300)


@pytest.mark.parametrize(("label", "name", "source"), CASES, ids=[f"{c[0]}:{c[1]}" for c in CASES])
def test_every_name_resolves_to_the_original_object(label, name, source):
    namespace = importlib.import_module(label)
    original = getattr(importlib.import_module(source), name)
    assert getattr(namespace, name) is original


@pytest.mark.parametrize(("label", "namespace", "table"), NAMESPACES, ids=NAMESPACE_IDS)
def test_all_lists_exactly_the_names_and_dir_shows_them(label, namespace, table):
    assert sorted(namespace.__all__) == sorted(table)
    assert len(namespace.__all__) == len(set(namespace.__all__))
    assert set(table) <= set(dir(namespace))
    assert set(namespace.__all__) == set(namespace._EXPORTS)
    assert {name: module for name, (module, _) in namespace._EXPORTS.items()} == table


@pytest.mark.parametrize(("label", "namespace", "table"), NAMESPACES, ids=NAMESPACE_IDS)
def test_an_unknown_name_is_an_attribute_error(label, namespace, table):
    with pytest.raises(AttributeError, match=f"module '{label}' has no attribute 'NotAThing'"):
        namespace.NotAThing  # noqa: B018
    assert not hasattr(namespace, "NotAThing")


def test_the_extension_package_names_its_namespace():
    assert ext.__all__ == ["vision"] and ext.vision is vision


def test_the_page_types_are_not_the_core():
    """The core takes arrays; the page types stay out here."""
    core = importlib.import_module("optivision.core.vectors")
    assert not any(hasattr(core, name) for name in ("PageRef", "PatchGrid", "PageEncoding"))


def test_star_import_gives_exactly_all():
    for label, _, table in NAMESPACES:
        scope: dict = {}
        exec(f"from {label} import *", scope)  # noqa: S102 -- fixed strings, a namespace under test
        assert {k for k in scope if not k.startswith("__")} == set(table)


@pytest.mark.parametrize("label", [n[0] for n in NAMESPACES])
def test_importing_and_using_a_namespace_does_not_import_optional_packages(label):
    code = (
        "import importlib, json, sys\n"
        f"ns = importlib.import_module({label!r})\n"
        "for name in ns.__all__:\n"
        "    getattr(ns, name)\n"
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in "
        f"{set(HEAVY)!r})))\n"
    )
    done = _run(code)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []


@pytest.mark.parametrize(("label", "heavy_module"), [
    ("optivision.ext.vision", "optivision.encoders.colvlm"),
    ("optivision.ext.vision", "optivision.encoders.nemotron"),
    ("optivision.ext.vision", "optivision.viz"),
    ("optivision.ext.vision", "optivision.adapters"),
    ("optivision.legacy", "optivision.bench"),
])
def test_importing_a_namespace_leaves_the_modules_behind_its_names_unimported(label, heavy_module):
    code = (
        "import importlib, sys\n"
        f"importlib.import_module({label!r})\n"
        f"print({heavy_module!r} in sys.modules)\n"
    )
    done = _run(code)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "False"


def test_a_name_is_imported_when_it_is_asked_for_and_then_kept():
    code = (
        "import sys, optivision.ext.vision as v\n"
        "assert 'optivision.encoders.nemotron' not in sys.modules\n"
        "cls = v.NemotronColEmbedEncoder\n"
        "assert 'optivision.encoders.nemotron' in sys.modules\n"
        "assert 'NemotronColEmbedEncoder' in vars(v) and v.NemotronColEmbedEncoder is cls\n"
    )
    done = _run(code)
    assert done.returncode == 0, done.stderr
