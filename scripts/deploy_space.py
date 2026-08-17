"""Create (if needed) and upload this project to a Hugging Face repo.

Handles both targets:

    --repo-type space   a live Gradio demo (requires a PRO account)
    --repo-type model   a plain code repo, free for everyone (the default)

Uses the HF API rather than `git push` on purpose, for two reasons: git blocks on
an interactive credential prompt, and the Hub's pre-receive hook rejects plain
binary files in git history (the report figures) while the API routes them
through Xet storage automatically.

Auth, in order of preference:
    1. --token / HF_TOKEN environment variable
    2. a stored login from `hf auth login`

Usage:
    python scripts/deploy_space.py                          # model repo (free)
    python scripts/deploy_space.py --repo-type space        # needs PRO
    python scripts/deploy_space.py --name my-demo
    python scripts/deploy_space.py --repo someuser/optivision-rag
    python scripts/deploy_space.py --private
    python scripts/deploy_space.py --dry-run                # list what would upload
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Mirrors .gitignore — the Space needs source, configs and docs, not the venv,
# the generated corpus, the model cache or the benchmark indexes.
IGNORE = [
    ".venv/*", "*.pyc", "__pycache__/*", "*/__pycache__/*",
    ".git/*", ".pytest_cache/*", ".ruff_cache/*", "*.egg-info/*",
    "data/*", "reports/bench_indexes/*", ".lock", "collection/*",
]


def is_ignored(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, pat) or rel.startswith(pat.rstrip("*")) for pat in IGNORE)


def listing() -> list[str]:
    out = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if not is_ignored(rel):
            out.append(rel)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="full repo id, e.g. user/optivision-rag")
    ap.add_argument("--name", default="optivision-rag", help="space name under your account")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF write token")
    ap.add_argument("--private", action="store_true", help="create a private repo")
    ap.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "space", "dataset"],
        help="model/dataset are free; space needs a PRO account",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = listing()
    total_mb = sum((ROOT / f).stat().st_size for f in files) / 1e6
    print(f"{len(files)} files, {total_mb:.1f} MB")
    if args.dry_run:
        for f in files:
            print("  ", f)
        return 0

    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi(token=args.token)
    try:
        me = api.whoami()
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        print(
            "Not authenticated. Either run\n"
            "    huggingface-cli login\n"
            "or pass a write token:\n"
            "    python scripts/deploy_space.py --token hf_xxx\n"
            f"\n(underlying error: {type(exc).__name__}: {str(exc)[:160]})",
            file=sys.stderr,
        )
        return 2

    user = me["name"]
    repo_id = args.repo or f"{user}/{args.name}"
    print(f"authenticated as {user} -> {args.repo_type} {repo_id}")

    try:
        extra = {"space_sdk": "gradio"} if args.repo_type == "space" else {}
        url = api.create_repo(
            repo_id=repo_id,
            repo_type=args.repo_type,
            private=args.private,
            exist_ok=True,
            **extra,
        )
        print(f"{args.repo_type} ready: {url}")
    except HfHubHTTPError as exc:
        if "402" in str(exc):
            print(
                f"could not create the {args.repo_type}: Hugging Face requires a PRO "
                "subscription to host a live Gradio Space. Use --repo-type model to "
                "publish the code instead (free).",
                file=sys.stderr,
            )
        else:
            print(f"could not create the {args.repo_type}: {exc}", file=sys.stderr)
        return 3

    print("uploading...")
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=repo_id,
        repo_type=args.repo_type,
        ignore_patterns=IGNORE,
        commit_message="OptiVision RAG: Stage-I pipeline and single-page demo",
    )
    prefix = "spaces/" if args.repo_type == "space" else ""
    print(f"\ndone -> https://huggingface.co/{prefix}{repo_id}")
    if args.repo_type == "space":
        print("The Space builds, then downloads the ColSmol checkpoint on first use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
