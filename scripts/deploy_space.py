"""Create (if needed) and upload this project to a Hugging Face Space.

Uses the HF API rather than `git push` on purpose: git would block on an
interactive credential prompt, which is unusable from a non-interactive shell.

Auth, in order of preference:
    1. --token / HF_TOKEN environment variable
    2. a stored login from `huggingface-cli login`

Usage:
    python scripts/deploy_space.py                     # <your-user>/optivision-rag
    python scripts/deploy_space.py --name my-demo
    python scripts/deploy_space.py --repo someuser/optivision-rag
    python scripts/deploy_space.py --private
    python scripts/deploy_space.py --dry-run           # list what would upload
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
    ap.add_argument("--private", action="store_true", help="create a private Space")
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
    print(f"authenticated as {user} -> space {repo_id}")

    try:
        url = api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=args.private,
            exist_ok=True,
        )
        print(f"space ready: {url}")
    except HfHubHTTPError as exc:
        print(f"could not create the space: {exc}", file=sys.stderr)
        return 3

    print("uploading...")
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=IGNORE,
        commit_message="OptiVision RAG: Stage-I pipeline and single-page demo",
    )
    print(f"\ndone -> https://huggingface.co/spaces/{repo_id}")
    print("The Space will build, then download the ColSmol checkpoint on the first compression.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
