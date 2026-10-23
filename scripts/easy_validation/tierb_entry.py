"""D4 (reports/easy/DEVIATIONS.md): ``encode_unit.py`` with the Tier B entry point ``vlm_probes.py`` looks for.

    python scripts/easy_validation/tierb_entry.py jobs.json --out results/   # the same arguments as encode_unit.py

``vlm_probes.generator`` (frozen, part 2) looks up ``make_vlm_probes``, ``make_probes_vlm`` or ``make_probes`` on
``optivision.easy._probes_vlm`` (frozen, part 2). That module names its generator ``generate_vlm_probes`` and
returns a ``ProbeBatch``, so the first e5 run failed before any question was made. This module adds
``make_vlm_probes`` there. It is the same generator, called with the keywords ``vlm_probes.generate``
passes, and it returns the probes as a list. Page images are looked up by page id, as the tier asks, not
by position. Neither frozen file changes, and nothing else differs from running ``encode_unit.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent))

from optivision.easy import _probes_vlm  # noqa: E402
from optivision.easy._probes_common import page_id  # noqa: E402


def by_page_id(images: Any) -> Any:
    """``vlm_probes.LazyImages`` is indexed by position in its source; the VLM tier asks by page id."""
    src = getattr(images, "src", None)
    if src is None or callable(images) or isinstance(images, Mapping):
        return images
    where = {str(pid): i for i, pid in enumerate(src.ids)}
    return lambda page: images[where[page_id(page)]]


def make_vlm_probes(pages: Sequence[Any], *, images: Any = None, seed: int = 0, max_pages: int = 300,
                    model: str = _probes_vlm.DEFAULT_MODEL, device: str = "auto") -> list[Any]:
    """``generate_vlm_probes`` called with the keywords ``vlm_probes.generate`` passes, returning its probes."""
    batch = _probes_vlm.generate_vlm_probes(pages, images=by_page_id(images), seed=seed, max_pages=max_pages,
                                            model=model, device=device)
    return list(batch.probes)


def install() -> None:
    _probes_vlm.make_vlm_probes = make_vlm_probes


def main(argv: Sequence[str] | None = None) -> int:
    install()
    from easy_validation import encode_unit

    return encode_unit.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
