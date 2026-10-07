"""The certified-selection layer is additive: ``optimize()`` and ``calibrate()`` are what they were.

The signatures below are copied from the v0.3.0 source (``optimize.py``, ``calibration.py``, both
unchanged); the golden values were computed with that code before the new layer was wired in.
Whatever the new entry point does, it must not change these, and importing it must not change what
``import optivision`` costs or loads beyond the stage registration the design calls for.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import optivision
from optivision.compose import Pipeline
from optivision.representation import as_corpus
from optivision.stages import AdaptiveMerge, BinaryQuantizer, Float16Quantizer, Int8Quantizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

try:
    import scipy.cluster.hierarchy  # noqa: F401

    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_SCIPY = False

needs_scipy = pytest.mark.skipif(not HAVE_SCIPY, reason="the default candidate space needs SciPy")
pytestmark = [pytest.mark.filterwarnings("ignore:calibrating on"),
              pytest.mark.filterwarnings("ignore:Int8Quantizer")]

#: ``optivision.__all__`` at v0.3.0
ALL_V030 = [
    "AdaptiveMerge", "BinaryQuantizer", "CalibrationResult", "CompressedCorpus", "CompressedPage",
    "CompressionConfig", "Config", "DimensionProjector", "EncoderConfig", "Float16Quantizer",
    "Float32Quantizer", "HierarchicalMerge", "IndexConfig", "IndexReport", "IngestConfig",
    "Int4Quantizer", "Int8Quantizer", "Lloyd2Quantizer", "MultiVectorCorpus",
    "MultiVectorRepresentation", "OptiVisionRAG", "OptimizationResult", "PCAProjector",
    "PageEncoding", "PageRef", "Pipeline", "PrunedPage", "PruningConfig", "RandomProjector",
    "RandomPruner", "RedundancyPruner", "SearchConfig", "SearchHit", "SearchResult",
    "SpatialPruner", "TruncateProjector", "__version__", "calibrate", "maxsim_matrix", "optimize",
]

#: the names this branch adds to ``optivision.__all__`` and where each one lives
NEW_NAMES = {
    "CandidateSpace": "optivision.core.selection",
    "Certificate": "optivision.core.artifact",
    "CertifiedResult": "optivision.api",
    "Decision": "optivision.core.selection",
    "EncoderIdentity": "optivision.core.vectors",
    "Outcome": "optivision.core.selection",
    "certify": "optivision.api",
    "load_artifact": "optivision.core.artifact",
    "save_artifact": "optivision.core.artifact",
    "verify": "optivision.api",
    "verify_certificate": "optivision.core.artifact",
}

REQUIRED = "<required>"

#: ``(name, kind, default, annotation)`` of every parameter, copied from the v0.3.0 source
OPTIMIZE_PARAMETERS = [
    ("vectors", "POSITIONAL_OR_KEYWORD", REQUIRED, "Any"),
    ("queries", "POSITIONAL_OR_KEYWORD", None, "Any"),
    ("qrels", "POSITIONAL_OR_KEYWORD", None, "dict[str, Any] | None"),
    ("quality_target", "POSITIONAL_OR_KEYWORD", 0.97, "float"),
    ("metric", "POSITIONAL_OR_KEYWORD", "ndcg@5", "str"),
    ("search_space", "POSITIONAL_OR_KEYWORD", None, "dict[str, Sequence[Pipeline]] | None"),
    ("calibrate_kwargs", "VAR_KEYWORD", REQUIRED, "Any"),
]
CALIBRATE_PARAMETERS = [
    ("corpus", "POSITIONAL_OR_KEYWORD", REQUIRED, "MultiVectorCorpus"),
    ("queries", "POSITIONAL_OR_KEYWORD", REQUIRED, "MultiVectorCorpus"),
    ("qrels", "POSITIONAL_OR_KEYWORD", None, "dict[str, Any] | None"),
    ("quality_target", "POSITIONAL_OR_KEYWORD", 0.97, "float"),
    ("metric", "POSITIONAL_OR_KEYWORD", "ndcg@5", "str"),
    ("search_space", "POSITIONAL_OR_KEYWORD", None, "dict[str, Sequence[Pipeline]] | None"),
    ("calibration_fraction", "POSITIONAL_OR_KEYWORD", 0.5, "float"),
    ("seed", "POSITIONAL_OR_KEYWORD", 0, "int"),
    ("reference", "POSITIONAL_OR_KEYWORD", "auto", "str"),
    ("baseline_depth", "POSITIONAL_OR_KEYWORD", 1, "int"),
    ("safety", "POSITIONAL_OR_KEYWORD", "lower_ci", "str"),
    ("max_bytes_per_doc", "POSITIONAL_OR_KEYWORD", None, "float | None"),
    ("assume_monotone", "POSITIONAL_OR_KEYWORD", True, "bool"),
    ("n_boot", "POSITIONAL_OR_KEYWORD", 1000, "int"),
    ("progress", "POSITIONAL_OR_KEYWORD", None, "Any"),
    ("cache_transforms", "POSITIONAL_OR_KEYWORD", True, "bool"),
    ("confidence", "POSITIONAL_OR_KEYWORD", 0.999, "float"),
    ("multiplicity", "POSITIONAL_OR_KEYWORD", "none", "str"),
    ("margin", "POSITIONAL_OR_KEYWORD", 0.0, "float"),
]


def _parameters(fn):
    return [(p.name, p.kind.name, REQUIRED if p.default is p.empty else p.default, p.annotation)
            for p in inspect.signature(fn).parameters.values()]


def _env() -> dict[str, str]:
    path = os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])
    return {**os.environ, "PYTHONPATH": path}


def _run_python(code: str) -> str:
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=_env(), check=False, timeout=300)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip().splitlines()[-1]


# ----------------------------------------------------------------- the old signatures


def test_optimize_has_exactly_its_v030_signature():
    assert _parameters(optivision.optimize) == OPTIMIZE_PARAMETERS
    assert inspect.signature(optivision.optimize).return_annotation == "OptimizationResult"


def test_calibrate_has_exactly_its_v030_signature():
    assert _parameters(optivision.calibrate) == CALIBRATE_PARAMETERS
    assert inspect.signature(optivision.calibrate).return_annotation == "CalibrationResult"


def test_the_result_types_keep_their_fields():
    assert [f.name for f in dataclasses.fields(optivision.OptimizationResult)] == [
        "pipeline", "compressed", "calibration"]
    assert [f.name for f in dataclasses.fields(optivision.CalibrationResult)] == [
        "target", "metric", "reference", "safety", "n_calibration_queries", "n_holdout_queries",
        "selected", "holdout", "candidates", "baseline"]


def test_certify_is_not_optimize_under_another_name():
    from optivision.api import certify

    assert certify is not optivision.optimize
    assert "quality_target" not in inspect.signature(certify).parameters
    assert "calibrate_kwargs" not in inspect.signature(certify).parameters


# ------------------------------------------------------------------ the old behaviour


def _problem(seed=7, n_docs=24, dim=16):
    """Topic-structured documents and queries cut from them, as ``test_optimize.py`` builds them."""
    rng = np.random.default_rng(seed)
    topics = rng.standard_normal((30, dim))
    docs, queries, qrels = [], [], {}
    for i in range(n_docs):
        v = topics[np.repeat(rng.choice(30, 4, replace=False), 10)] + 0.3 * rng.standard_normal((40, dim))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        docs.append(v.astype(np.float32))
        q = v[rng.choice(40, 5, replace=False)] + 0.3 * rng.standard_normal((5, dim))
        queries.append((q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32))
        qrels[str(i)] = {str(i)}
    return docs, queries, qrels


def _space():
    return {"m": [Pipeline(([AdaptiveMerge(radius=r)] if r else []) + [q()])
                  for r in (None, 0.8, 0.6) for q in (Float16Quantizer, Int8Quantizer, BinaryQuantizer)]}


def _optimize(docs, queries, qrels):
    return optivision.optimize(docs, queries=queries, qrels=qrels, quality_target=0.7,
                               search_space=_space(), assume_monotone=False, n_boot=100)


def _fingerprint(res):
    return (res.pipeline.label(), res.compression_ratio, res.quality_retention, res.target_met,
            len(res.calibration.candidates),
            hashlib.sha256(res.compressed.codes.tobytes()).hexdigest())


def test_optimize_gives_the_values_v030_gave():
    docs, queries, qrels = _problem()
    res = _optimize(docs, queries, qrels)
    assert res.pipeline.label() == "adaptive_merge(0.8) > binary"
    assert res.compression_ratio == pytest.approx(245.76, rel=1e-9)
    assert res.quality_retention == pytest.approx(0.8613053963059044, rel=1e-6)
    assert res.target_met is True
    assert len(res.calibration.candidates) == 9
    assert type(res.compressed).__name__ == "CompressedCorpus"


def test_calibrate_gives_the_values_v030_gave():
    docs, queries, qrels = _problem()
    cal = optivision.calibrate(as_corpus(docs), as_corpus(queries), qrels, quality_target=0.7,
                               search_space=_space(), assume_monotone=False, n_boot=100)
    assert cal.selected.label == "adaptive_merge(0.8) > binary"
    assert (cal.n_calibration_queries, cal.n_holdout_queries, len(cal.candidates)) == (12, 12, 9)
    assert cal.holdout["retention"] == pytest.approx(0.8613053963059044, rel=1e-6)
    assert cal.selected.retention == pytest.approx(0.9108897131727828, rel=1e-6)
    assert cal.met_target_on_holdout is True


@needs_scipy
def test_optimize_with_the_default_space_gives_the_values_v030_gave():
    docs, queries, _ = _problem()
    res = optivision.optimize(docs, queries=queries, quality_target=0.9, n_boot=100)
    assert res.pipeline.label() == "int8(per_vector)"
    assert res.compression_ratio == pytest.approx(3.5555555555555554, rel=1e-9)
    assert res.quality_retention == pytest.approx(1.0)
    assert len(res.calibration.candidates) == 9


def test_optimize_is_unchanged_by_a_certify_call_between_two_runs():
    from optivision.api import certify
    from optivision.testing import make_corpus, make_queries

    docs, queries, qrels = _problem()
    before = _fingerprint(_optimize(docs, queries, qrels))
    corpus = make_corpus(30, 16, tokens=(6, 10), seed=1)
    qs, labels = make_queries(corpus, 40, noise=0.5, seed=2)
    certify(corpus, qs, labels, target=0.5, space=_space(), created="2026-10-06T00:00:00Z")
    assert _fingerprint(_optimize(docs, queries, qrels)) == before


def test_optimize_and_calibrate_do_not_load_the_certified_layer():
    """They are not re-routed: running them never imports the API or the artifact modules."""
    code = (
        "import json, sys, warnings\n"
        "warnings.simplefilter('ignore')\n"
        "import numpy as np, optivision\n"
        "from optivision.stages import Float16Quantizer\n"
        "rng = np.random.default_rng(0)\n"
        "docs = [rng.standard_normal((6, 8)).astype('float32') for _ in range(8)]\n"
        "qs = [rng.standard_normal((3, 8)).astype('float32') for _ in range(8)]\n"
        "space = {'f': [optivision.Pipeline([Float16Quantizer()])]}\n"
        "optivision.optimize(docs, queries=qs, search_space=space, n_boot=20)\n"
        "optivision.calibrate(optivision.MultiVectorCorpus.from_arrays(docs),\n"
        "                     optivision.MultiVectorCorpus.from_arrays(qs), search_space=space, n_boot=20)\n"
        "print(json.dumps(sorted(m for m in sys.modules if m in ('optivision.api',\n"
        "      'optivision.core.artifact', 'optivision.core.selection', 'optivision.core.evaluate'))))\n"
    )
    assert json.loads(_run_python(code)) == []


# ---------------------------------------------------------------------- the namespace


def test_the_forty_names_of_v030_are_still_in_all_and_resolve():
    assert len(ALL_V030) == len(set(ALL_V030)) == 40
    assert set(ALL_V030) <= set(optivision.__all__)
    assert [n for n in ALL_V030 if not hasattr(optivision, n)] == []


def test_all_holds_the_forty_names_plus_exactly_the_new_ones():
    assert set(optivision.__all__) - set(ALL_V030) == set(NEW_NAMES)
    assert len(optivision.__all__) == len(set(optivision.__all__)) == 40 + len(NEW_NAMES)


@pytest.mark.parametrize(("name", "module"), sorted(NEW_NAMES.items()))
def test_new_names_resolve_to_the_objects_of_their_modules(name, module):
    import importlib

    assert getattr(optivision, name) is getattr(importlib.import_module(module), name)
    assert name in dir(optivision)


def test_from_import_and_star_import_work_for_the_new_names():
    code = (
        f"from optivision import {', '.join(sorted(NEW_NAMES))}\n"
        "import json\n"
        "print(json.dumps({'from_import': True}))\n"
    )
    assert json.loads(_run_python(code)) == {"from_import": True}
    star = (
        "from optivision import *\n"
        "import json, optivision\n"
        "print(json.dumps([n for n in optivision.__all__ if n not in globals()]))\n"
    )
    assert json.loads(_run_python(star)) == []


def test_an_unknown_name_is_still_an_attribute_error():
    with pytest.raises(AttributeError, match="no_such_name"):
        optivision.__getattr__("no_such_name")
    assert not hasattr(optivision, "no_such_name")


def test_optimize_is_still_the_function_not_the_module():
    assert isinstance(optivision.optimize, types.FunctionType)
    assert optivision.optimize.__module__ == "optivision.optimize"
    assert sys.modules["optivision.optimize"].optimize is optivision.optimize


@pytest.mark.skipif(not (SRC / "optivision" / "__init__.py").is_file(),
                    reason="the repository layout (src/) is absent")
def test_version_is_still_0_3_0_and_a_literal():
    assert optivision.__version__ == "0.3.0"
    text = (SRC / "optivision" / "__init__.py").read_text(encoding="utf-8")
    assert re.search(r'^__version__ = "0\.3\.0"$', text, re.MULTILINE)


def test_the_core_package_exports_what_its_all_lists():
    from optivision import core

    assert sorted(core.__all__) == sorted(core._EXPORTS)
    assert len(core.__all__) == len(set(core.__all__))
    assert [n for n in core.__all__ if not hasattr(core, n)] == []
    assert "Certificate" in dir(core)
    with pytest.raises(AttributeError, match="no_such_name"):
        core.__getattr__("no_such_name")
    assert not hasattr(core, "no_such_name")


def test_the_cli_has_the_old_commands_and_the_two_new_ones():
    import typer.main

    from optivision.cli import app

    commands = set(typer.main.get_command(app).commands)
    old = {"init-config", "make-corpus", "index", "search", "stats", "explain", "bench",
           "fetch-vidore", "inspect", "compress", "calibrate", "benchmark", "compare"}
    assert old <= commands
    assert {"certify", "verify"} <= commands


# ---------------------------------------------------- what `import optivision` does


def test_a_bare_import_registers_the_rotation_stage_and_rebuilds_a_spec_naming_it():
    code = (
        "import json, optivision\n"
        "from optivision.stages.base import STAGES\n"
        "spec = {'stages': [{'stage': 'rotation', 'seed': 3}, {'stage': 'int8', 'scale': 'per_vector'}]}\n"
        "pipe = optivision.Pipeline.from_dict(spec)\n"
        "print(json.dumps({'registered': 'rotation' in STAGES, 'round_trip': pipe.to_dict() == spec,\n"
        "                  'class': type(pipe.stages[0]).__name__}))\n"
    )
    out = json.loads(_run_python(code))
    assert out == {"registered": True, "round_trip": True, "class": "RotationProjector"}


def test_the_original_stage_names_are_still_registered_to_the_same_classes():
    code = (
        "import json, optivision, optivision.stages as S\n"
        "from optivision.stages.base import STAGES\n"
        "names = ['adaptive_merge', 'binary', 'float16', 'float32', 'hierarchical_merge', 'int4', 'int8',\n"
        "         'lloyd2', 'pca', 'project', 'random_projection', 'random_prune', 'redundancy',\n"
        "         'spatial', 'truncate']\n"
        "classes = {'adaptive_merge': S.AdaptiveMerge, 'binary': S.BinaryQuantizer,\n"
        "           'float16': S.Float16Quantizer, 'float32': S.Float32Quantizer,\n"
        "           'hierarchical_merge': S.HierarchicalMerge, 'int4': S.Int4Quantizer,\n"
        "           'int8': S.Int8Quantizer, 'lloyd2': S.Lloyd2Quantizer, 'pca': S.PCAProjector,\n"
        "           'project': S.DimensionProjector, 'random_projection': S.RandomProjector,\n"
        "           'random_prune': S.RandomPruner, 'redundancy': S.RedundancyPruner,\n"
        "           'spatial': S.SpatialPruner, 'truncate': S.TruncateProjector}\n"
        "print(json.dumps({'same': all(STAGES[n] is classes[n] for n in names), 'extra': sorted(set(STAGES) - set(names))}))\n"
    )
    assert json.loads(_run_python(code)) == {"same": True, "extra": ["rotation"]}


def test_import_optivision_loads_no_optional_dependency_and_not_the_certified_layer():
    code = (
        "import json, sys, optivision\n"
        "heavy = [m for m in ('scipy', 'torch', 'qdrant_client', 'transformers', 'colpali_engine',\n"
        "                     'sentence_transformers', 'datasets', 'reportlab') if m in sys.modules]\n"
        "lazy = [m for m in ('optivision.api', 'optivision.core.artifact', 'optivision.core.selection',\n"
        "                    'optivision.core.evaluate') if m in sys.modules]\n"
        "print(json.dumps({'heavy': heavy, 'lazy': lazy}))\n"
    )
    assert json.loads(_run_python(code)) == {"heavy": [], "lazy": []}


def test_using_the_new_names_does_not_import_scipy_torch_or_qdrant_either():
    code = (
        "import json, sys, optivision\n"
        "for name in optivision.__all__:\n"
        "    getattr(optivision, name)\n"
        "heavy = [m for m in ('scipy', 'torch', 'qdrant_client', 'transformers') if m in sys.modules]\n"
        "print(json.dumps({'heavy': heavy, 'api': 'optivision.api' in sys.modules}))\n"
    )
    assert json.loads(_run_python(code)) == {"heavy": [], "api": True}


def test_the_new_layer_is_used_by_import_optivision_only_through_core_stages():
    """The one eager addition is ``optivision.core.stages``; everything else loads on demand."""
    code = (
        "import json, sys, optivision\n"
        "core = sorted(m for m in sys.modules if m.startswith('optivision.core'))\n"
        "print(json.dumps(core))\n"
    )
    loaded = json.loads(_run_python(code))
    assert "optivision.core.stages" in loaded
    assert not [m for m in loaded if m.split(".")[2:3] in (["artifact"], ["selection"], ["evaluate"], ["retrieve"])]
