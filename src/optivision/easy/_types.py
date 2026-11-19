"""Value types and errors shared by the easy-mode modules (docs/easy/CONTRACT.md).

Nothing here imports NumPy at module level beyond typing, so ``import optivision.easy``
stays light.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: the statuses a ladder row can have at a target (FINAL_DESIGN 4.1)
ROW_STATUSES = ("pick", "qualifies", "close call", "below target", "family stopped", "not measured",
                "outside size range", "needs practice questions", "not allowed")

#: the ladder chains, lightest rung first (FINAL_DESIGN 5.1.1)
CHAINS = ("float", "8-bit", "centred 4-bit", "1-bit")


@dataclass(frozen=True)
class LayoutRow:
    """One layout of the ladder, as ``options()`` shows it and ``build()`` uses it.

    ``name`` is the catalogue label (``"hm(0.25) > int8"``), ``plain`` the beginner name
    (``"merged (1 in 4 vectors) + 8-bit"``) and ``spec`` the catalogue ``pipeline`` dict, which
    ``Pipeline.from_dict`` rebuilds (never ``parse_pipeline``). Sizes are per page and per
    dataset: ``bytes_per_page`` is ``(smallest, largest)`` over the datasets the rule used.
    ``ratio_f16`` / ``ratio_f32`` are the smallest per-dataset ratios. ``lowest_kept`` is the
    lowest 95% interval low over those datasets and ``worst_point`` the lowest point; both are
    None for a layout without rows. ``need`` is target plus margins; ``slack`` is
    ``lowest_kept - need``.
    """

    name: str
    plain: str
    spec: dict[str, Any]
    chain: str
    bytes_per_page: tuple[float, float] | None
    ratio_f16: float | None
    ratio_f32: float | None
    lowest_kept: float | None
    worst_point: float | None
    datasets: tuple[str, ...] = ()
    status: str = "not measured"
    reason: str = ""
    need: float | None = None
    slack: float | None = None
    one_bit: bool = False
    merged: bool = False


@dataclass(frozen=True)
class Selection:
    """What the query-free rule picked for (model, target, pages, checked) (FINAL_DESIGN 5.1).

    ``rows`` holds every ladder row with its status at this target, in ladder order (float rows
    first, then each chain lightest rung first). ``sequence`` is the fixed test order of the
    practice check: the pick, then each safer qualifying layout by falling ratio, ending at
    float16 (or float32). ``safer`` is the largest rung with slack >= 0.005 when the pick is a
    close call. ``basis`` says whose rows were used: ``"own rows"``, ``"measured via <id>"`` or
    ``"unmeasured: pooled minimum over <n> <modality> models"``.
    """

    model: str | None
    modality: str
    target: float
    n_pages: int | None
    checked: bool
    rows: tuple[LayoutRow, ...]
    pick: LayoutRow
    sequence: tuple[LayoutRow, ...]
    close_call: bool = False
    safer: LayoutRow | None = None
    basis: str = "own rows"
    notes: tuple[str, ...] = ()
    hand_picked: bool = False


@dataclass(frozen=True)
class Hit:
    """One search result. ``id`` is ``PageRef(doc_id, page_no).page_id``
    (``my_pdfs/sub/report.pdf::p3``); ``page`` is 1-based; ``file`` is the source path at build
    time, or None for an index built from vectors; ``text`` is the page's text layer when it was
    kept, else ``""``."""

    id: str
    file: str | None
    page: int | None
    score: float
    text: str = ""
    _index: Any = field(default=None, repr=False, compare=False)

    def image(self, dpi: int = 150) -> Any:
        """The page re-rendered from its source file, as a ``PIL.Image.Image``."""
        if self.file is None or self.page is None:
            raise NoSource(f"{self.id} was built from vectors; there is no page to render")
        from pathlib import Path

        if not Path(self.file).exists():
            raise SourceMissing(f"{self.file} is gone; Index.relink(folder) finds moved files by content")
        from . import _docs

        return _docs.render_page(self.file, self.page, dpi=dpi)


@dataclass
class SyncReport:
    """What a build or ``add()`` changed. Each list holds ``(path, reason)`` pairs."""

    added: list[tuple[str, str]] = field(default_factory=list)
    changed: list[tuple[str, str]] = field(default_factory=list)
    missing: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    aliases: int = 0
    n_pages: int = 0
    layout_still_qualifies: bool = True
    message: str = ""


# ---------------------------------------------------------------------------- errors


class EasyError(Exception):
    """Base class of the easy-mode refusals (FINAL_DESIGN 4.2, Refusals)."""


class UnknownModel(EasyError, KeyError):
    """The model id is not in the registry; the message names the three closest ids."""

    def __str__(self) -> str:  # KeyError would quote the message
        return str(self.args[0]) if self.args else ""


class AmbiguousModel(EasyError, KeyError):
    """A short name matches several registry ids; the message lists them with loader and licence."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class WiringError(EasyError, RuntimeError):
    """The wiring check failed on the user's pages and on the bundled sample PDF."""


class RemoteCodeRefused(EasyError, PermissionError):
    """A model that runs its repository's own code, without ``trust_remote_code=True``."""


class TargetMismatch(EasyError, ValueError):
    """An existing index was built at another target or layout, and ``rebuild`` was not given."""


class MissingExtra(EasyError, ImportError):
    """The loader's optional dependencies are absent; the message gives the install line."""


class NotEnoughDisk(EasyError, OSError):
    """Not enough free disk for the temporary copy or the index; names needed and free GB."""


class NoSource(EasyError, LookupError):
    """The index was built from vectors, so a hit has no page to render."""


class SourceMissing(EasyError, FileNotFoundError):
    """The source file of a hit is gone."""


class IndexLocked(EasyError, RuntimeError):
    """Another build or ``add()`` holds the index's writer lock."""


class FeatureOff(EasyError, RuntimeError):
    """A feature whose preregistered test has not passed (presets/easy_ledger.json) is switched off."""
