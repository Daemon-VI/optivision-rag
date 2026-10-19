"""Measure a batch of models, one after the other, inside one GPU session (Kaggle or any box).

For every model of the batch file: encode the evaluation data with the package's own loader
(``scripts/encode_vectors.py`` for page-image models on ViDoRe DocVQA + InfoVQA,
``scripts/encode_text_vectors.py`` for text models on BEIR SciFact, both of which run
``check_retriever`` first and write nothing if it fails), run the fixed-pipeline study
(``s13_wide_study.py``) on each split, and, when ``--selection`` is given and the session has
time to spare, the selection study (``s7b_selection_rules.py``). A model that fails or runs out
of time is recorded and the batch goes on with the next one.

    python scripts/measure_models.py batch.json --out results --data /tmp/optdata --deadline-hours 11

The batch file is a JSON list of objects:

    {"model": "vidore/colpali-v1.2", "kind": "image", "dtype": "float32", "multi_gpu": true,
     "loader": null, "trust_remote_code": false, "revision": null, "pip": [], "batch_size": 4,
     "processor_kwargs": null}

Only ``model`` is required (``kind`` defaults to the registry's modality). ``pip`` lists
packages to install before that model (they stay installed for the rest of the session).
``processor_kwargs`` goes to a sentence-transformers page model's processor (a page budget).

Follow-up measurements use five more keys:

- ``datasets``: HF image datasets for an image model (default ViDoRe V1 DocVQA and InfoVQA),
  or BEIR names for a text model (default ``["scifact"]``).
- ``distractors`` and ``distractor_count``: unlabelled pages or documents from other datasets,
  added to every dataset's corpus (``encode_vectors.py --distractors`` /
  ``encode_text_vectors.py --distractors``).
- ``limit``: pages or documents per dataset for this model.
- ``studies``: ``false`` to encode and record the vectors' geometry only.
- ``study``: which fixed-pipeline studies run on each split, from ``STUDIES`` (default
  ``["s13"]``); ``s17`` is the int4 norm study (``s17_int4_norm.py``, into ``int4norm/``).

After every encode, ``scripts/vector_geometry.py`` writes ``geometry/<tag>.json`` (statistics
of the vectors, before they are deleted).

Every model gets ``<out>/<model short name>/`` with its logs, the study JSON files
(``wide/``, ``selection_rules/``), the qrels and ``run.json`` (what ran, how long, and the
outcome of each step). ``<out>/batch_summary.json`` lists every model's outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

IMAGE_SPLITS = {"docvqa": "vidore/docvqa_test_subsampled", "infovqa": "vidore/infovqa_test_subsampled"}
RESERVE_SECONDS = 15 * 60  # kept for packaging the results


#: the fixed-pipeline studies a spec's ``study`` key can name; ``s13`` is the default
STUDIES = {"s13": "s13_wide_study.py", "s17": "s17_int4_norm.py"}


def split_key(dataset: str) -> str:
    """The step name of an image dataset: ``docvqa`` / ``infovqa`` for the two default splits."""
    for key, name in IMAGE_SPLITS.items():
        if name == dataset:
            return key
    return dataset.split("/")[-1].removesuffix("_subsampled").removesuffix("_test")


def tag_for(short: str, dataset: str, distractor_count: int = 0) -> str:
    """The vector-file tag the encode scripts write (``encode_vectors.tag_for``, ``encode_text_vectors.tag_for``)."""
    return f"{short}_{dataset.split('/')[-1]}" + (f"_plus{distractor_count}" if distractor_count else "")


class Session:
    def __init__(self, deadline_hours: float) -> None:
        self.t0 = time.time()
        self.deadline = self.t0 + deadline_hours * 3600

    def remaining(self) -> float:
        return self.deadline - time.time() - RESERVE_SECONDS


def run_step(cmd: list[str], log: Path, session: Session, env: dict[str, str] | None = None) -> dict:
    """Run one step with its output in ``log``; never longer than the session has left."""
    start = time.time()
    budget = session.remaining()
    if budget < 120:
        return {"status": "skipped (no time left)", "seconds": 0}
    with log.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        try:
            code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=budget,
                                  env={**os.environ, **(env or {})}, check=False).returncode
            status = "ok" if code == 0 else f"failed (exit {code})"
        except subprocess.TimeoutExpired:
            status = "timed out"
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-3:]
    print(f"   {log.name}: {status} in {(time.time() - start) / 60:.1f} min | {' / '.join(tail)[-200:]}", flush=True)
    return {"status": status, "seconds": round(time.time() - start, 1)}


def check_report(log: Path) -> dict | None:
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith('{"check_retriever"'):
            return json.loads(line)["check_retriever"]
    return None


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 1e9


def measure(spec: dict, out: Path, data: Path, session: Session, selection: bool, py: str,
            limit: int | None = None) -> dict:
    from optivision.adapters import MODELS

    model = spec["model"]
    chosen = spec.get("study") or ["s13"]
    chosen = [chosen] if isinstance(chosen, str) else list(chosen)
    if unknown := [s for s in chosen if s not in STUDIES]:
        raise ValueError(f"unknown study {unknown}; known: {sorted(STUDIES)}")
    info = MODELS.get(model)
    kind = spec.get("kind") or ("text" if info is not None and info.modality == "text" else "image")
    short = model.split("/")[-1]
    folder = out / short
    (folder / "logs").mkdir(parents=True, exist_ok=True)
    record: dict = {"model": model, "kind": kind, "spec": spec, "registry_status": info.status if info else None,
                    "steps": {}, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    vectors = data / "vectors"
    vectors.mkdir(parents=True, exist_ok=True)
    print(f"== {model} ({kind}); {session.remaining() / 3600:.2f} h left; {free_gb(str(data)):.0f} GB free", flush=True)

    for package in spec.get("pip") or []:
        record["steps"][f"pip {package}"] = run_step(
            [py, "-m", "pip", "install", "-q", *package.split()], folder / "logs" / "pip.log", session)

    limit = spec.get("limit", limit)
    distractors = ",".join(spec.get("distractors") or [])
    count = int(spec.get("distractor_count") or 0)
    options: list[str] = ["--limit", str(limit)] if limit else []
    if spec.get("loader"):
        options += ["--loader", spec["loader"]]
    if spec.get("trust_remote_code"):
        options += ["--trust-remote-code"]
    if kind == "image":
        options += ["--dtype", spec.get("dtype", "auto"), "--batch-size", str(spec.get("batch_size", 4))]
        if spec.get("multi_gpu"):
            options += ["--multi-gpu"]
        if spec.get("revision"):
            options += ["--revision", spec["revision"]]
        if spec.get("processor_kwargs"):
            options += ["--processor-kwargs", json.dumps(spec["processor_kwargs"])]
        if distractors:
            options += ["--distractors", distractors, "--distractor-pages", str(count)]
        tags = []
        datasets = spec.get("datasets") or list(IMAGE_SPLITS.values())
        for i, dataset in enumerate(datasets):
            step = f"encode_{split_key(dataset)}"
            log = folder / "logs" / f"{step}.log"
            record["steps"][step] = run_step(
                [py, str(ROOT / "scripts" / "encode_vectors.py"), model, dataset, "--out", str(vectors),
                 "--check-pairs", "12" if i == 0 else "0", *options], log, session)
            if i == 0:
                record["check_retriever"] = check_report(log)
            if record["steps"][step]["status"] != "ok":
                break
            tags.append(tag_for(short, dataset, count))
    else:
        options += ["--batch-size", str(spec.get("batch_size", 32))]
        if distractors:
            options += ["--distractors", distractors, "--distractor-docs", str(count)]
        tags = []
        datasets = spec.get("datasets") or ["scifact"]
        for i, dataset in enumerate(datasets):
            step = f"encode_{dataset}"
            log = folder / "logs" / f"{step}.log"
            record["steps"][step] = run_step(
                [py, str(ROOT / "scripts" / "encode_text_vectors.py"), model, "--dataset", dataset, "--out", str(vectors),
                 "--check-pairs", "12" if i == 0 else "0", *options], log, session)
            if i == 0:
                record["check_retriever"] = check_report(log)
            if record["steps"][step]["status"] != "ok":
                break
            tags.append(tag_for(short, dataset, count))

    path = os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")]))
    env = {"OPTIVISION_DATA": str(data), "OPTIVISION_OUT": str(folder), "PYTHONUNBUFFERED": "1", "PYTHONPATH": path,
           "OPTIVISION_SCORE_DEVICE": os.environ.get("OPTIVISION_SCORE_DEVICE", "cuda"),
           "OPTIVISION_CACHE_BYTES": os.environ.get("OPTIVISION_CACHE_BYTES", str(4 * 1024**3))}
    for tag in tags:  # cheap, and the only record of the vectors themselves once they are deleted
        record["steps"][f"geometry {tag}"] = run_step(
            [py, str(ROOT / "scripts" / "vector_geometry.py"), f"vec:{tag}", "--out", str(folder / "geometry")],
            folder / "logs" / f"geometry_{tag}.log", session, env)
    if spec.get("studies", True) is False:
        tags_studied: list[str] = []
    else:
        tags_studied = tags
    for tag in tags_studied:
        shutil.copy(vectors / f"{tag}_qrels.json", folder / f"{tag}_qrels.json")
        for study in chosen:
            record["steps"][f"{study} {tag}"] = run_step(
                [py, str(ROOT / "scripts" / "universal_study" / STUDIES[study]), f"vec:{tag}"],
                folder / "logs" / f"{study}_{tag}.log", session, env)
    if selection:
        for tag in tags_studied:
            record["steps"][f"s7b {tag}"] = run_step(
                [py, str(ROOT / "scripts" / "universal_study" / "s7b_selection_rules.py"), f"vec:{tag}"],
                folder / "logs" / f"s7b_{tag}.log", session, env)
    for tag in tags:  # the vectors are not kept: a wide model's are many gigabytes
        for suffix in ("_docs.npz", "_queries.npz", "_qrels.json"):
            (vectors / f"{tag}{suffix}").unlink(missing_ok=True)
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    shutil.rmtree(hub / f"models--{model.replace('/', '--')}", ignore_errors=True)
    record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    statuses = [s["status"] for s in record["steps"].values()]
    studies = [k for k in record["steps"] if k.split(" ")[0] in chosen]
    n_sets = len(spec.get("datasets") or (IMAGE_SPLITS if kind == "image" else ["scifact"]))
    expected = n_sets * len(chosen)
    encoded_only = spec.get("studies", True) is False and len(tags) == n_sets
    record["outcome"] = ("encoded (geometry only)" if encoded_only
                         else "measured" if studies and all(record["steps"][k]["status"] == "ok" for k in studies)
                         and len(studies) == expected
                         else "check failed" if record.get("check_retriever") and not record["check_retriever"]["ok"]
                         else "incomplete: " + ", ".join(f"{k} {v}" for k, v in zip(record["steps"], statuses, strict=True)
                                                         if v != "ok"))
    (folder / "run.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    print(f"   -> {record['outcome']}", flush=True)
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("batch", type=Path, help="JSON list of model specifications")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("/tmp/optdata"))
    ap.add_argument("--deadline-hours", type=float, default=11.0)
    ap.add_argument("--selection", action="store_true", help="also run the selection study when time allows")
    ap.add_argument("--limit", type=int, default=None, help="pages / documents per split, for a dry run")
    a = ap.parse_args()

    batch = json.loads(a.batch.read_text(encoding="utf-8"))
    shorts = [spec["model"].split("/")[-1] for spec in batch]
    if len(set(shorts)) != len(shorts):  # one folder and one set of vector files per short name
        raise SystemExit(f"two models in this batch share a short name: {sorted({s for s in shorts if shorts.count(s) > 1})}")
    session = Session(a.deadline_hours)
    a.out.mkdir(parents=True, exist_ok=True)
    results = []
    spent: list[float] = []  # seconds per model so far, for the selection-study decision
    for i, spec in enumerate(batch):
        if session.remaining() < 600:
            results.append({"model": spec["model"], "outcome": "not started (no time left)"})
            continue
        # The selection study is extra: run it only if the models still to come keep their share
        # of the session (the average time of a model so far, or 2 h before the first one ends).
        left = len(batch) - i - 1
        per_model = sum(spent) / len(spent) if spent else 2 * 3600
        selection = a.selection and session.remaining() > per_model * (left + 1) + 3600
        start = time.time()
        try:
            results.append(measure(spec, a.out, a.data, session, selection, sys.executable, a.limit))
        except Exception as exc:  # noqa: BLE001 -- one model's failure must not end the batch
            results.append({"model": spec["model"], "outcome": f"error: {type(exc).__name__}: {exc}"})
            print(f"   -> error: {exc}", flush=True)
        spent.append(time.time() - start)
        (a.out / "batch_summary.json").write_text(json.dumps(
            [{"model": r["model"], "outcome": r["outcome"]} for r in results], indent=1), encoding="utf-8")
    print(json.dumps([{"model": r["model"], "outcome": r["outcome"]} for r in results], indent=1))
    print(f"session used {(time.time() - session.t0) / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
