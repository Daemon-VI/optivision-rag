"""Two-tier retrieval: a shortlist from one source, rescored by another.

``TieredRetriever``
    The in-process pair from ``optivision.storage``: a small *hot* store (for example merged
    vectors as binary codes) shortlists ``candidates`` documents per query, a *cold* store
    (for example per-vector int8 or float) rescores only those. ``score`` and ``search`` are
    ``TieredIndex.score`` and ``TieredIndex.search``, unchanged; this class adds validation, the
    alignment check that ``TieredIndex`` does not make, and a typed :class:`Ranking`.

``TwoTierRetriever``
    The same idea for any shortlist source (an in-process retriever, or a backend behind an
    engine) and any reranker. It reproduces ``TieredIndex.search`` for the first ``k <=
    candidates`` ranks, including how ties break (see below; the class docstring names the one
    case where the two differ), and has no order to offer beyond its shortlist.

What two tiers buy. The hot tier is what has to stay in RAM; the cold tier can sit on slower
storage and is read for the shortlist only. That is a saving in resident memory. It is not a
saving in total storage (the sum of both tiers is stored), and no speed claim is made here: the
hot scan still reads every document of the hot store, ``TieredIndex.score`` still returns a dense
``[n_queries, n_docs]`` matrix, and whether the whole is quicker depends on the stack
(``docs/package/BACKEND_ARCHITECTURE.md``).

Tie-breaking (invariant I5). ``TieredIndex`` ranks the full matrix, so among shortlisted
documents with equal cold scores the one with the smaller store position comes first, whatever the
order of the shortlist. ``TwoTierRetriever`` therefore sorts the shortlist by store position
before it ranks the cold scores. Ranking in first-stage order would break such ties differently.

Quality is bounded by two things, both to be measured rather than assumed: whether the right
document reaches the shortlist, and how good the cold scores are. Agreement with a reference
ranking is not retrieval quality.

Device: nothing here selects one or reads the environment; see ``exact.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...representation import MultiVectorCorpus
from ...scoring import rank
from ...storage import TieredIndex
from .checks import as_queries, check_candidates, check_k
from .exact import ExactRetriever, _ids_array, ranking_from_positions
from .types import SCORE_MAXSIM_IP, SCORE_TIERED, TIE_INDEX, Ranking, Reranker, Retriever


def _require_aligned(a_ids: list[str], b_ids: list[str], a_name: str, b_name: str) -> None:
    if len(a_ids) != len(b_ids):
        raise ValueError(
            f"the {a_name} and {b_name} must hold the same documents: {len(a_ids)} vs {len(b_ids)}"
        )
    for position, (a, b) in enumerate(zip(a_ids, b_ids, strict=True)):
        if a != b:
            raise ValueError(
                f"the {a_name} and {b_name} must hold the same documents in the same order; "
                f"position {position} is {a!r} in the {a_name} and {b!r} in the {b_name}"
            )


class TieredRetriever:
    """Shortlist on a hot :class:`ExactRetriever`, rescore the shortlist on a cold one.

    Args:
        hot: scored in full to shortlist ``candidates`` documents per query.
        cold: rescored on the shortlist only.
        candidates: shortlist size; clamped to the number of documents. ``>= 1``.

    Raises ``ValueError`` unless both tiers hold the same ids in the same order (``TieredIndex``
    does not check this and would silently pair unrelated documents), and when the tiers are empty
    (``TieredIndex`` cannot shortlist an empty corpus). Each tier projects the raw queries with its
    own pipeline, so the two may differ in dimension.

    ``search`` returns a ``Ranking`` of kind ``"tiered"``: the first ``candidates`` ranks carry
    cold scores, the rest carry hot scores shifted below the lowest cold score. Those two groups
    are not on one scale; ``score`` is ``TieredIndex.score`` and has the same property, and it
    allocates the dense matrix plus copies of it (``hot.copy()`` and a masked temporary).

    ``last_timing`` is a copy of ``TieredIndex.last_timing``: instance state that every call
    overwrites. Do not read it across concurrent calls.
    """

    def __init__(self, hot: ExactRetriever, cold: ExactRetriever, candidates: int = 50) -> None:
        for name, tier in (("hot", hot), ("cold", cold)):
            if not isinstance(tier, ExactRetriever):
                raise TypeError(
                    f"{name} must be an ExactRetriever or CompressedRetriever, "
                    f"got {type(tier).__name__}"
                )
        candidates = check_candidates(candidates)
        _require_aligned(hot.ids, cold.ids, "hot tier", "cold tier")
        if hot.n_docs == 0:
            raise ValueError("the tiers hold no documents; TieredIndex cannot shortlist an empty corpus")
        self.hot = hot
        self.cold = cold
        self.candidates = candidates
        self.ids: list[str] = list(hot.ids)
        self._ids = _ids_array(self.ids)
        self._tiered = TieredIndex(hot.index, cold.index, candidates)

    @property
    def n_docs(self) -> int:
        return len(self.ids)

    @property
    def shortlist_size(self) -> int:
        """Documents rescored per query: ``min(candidates, n_docs)``."""
        return min(self.candidates, self.n_docs)

    @property
    def last_timing(self) -> dict[str, float]:
        return dict(self._tiered.last_timing)

    def _prepare(self, queries: Any) -> MultiVectorCorpus:
        q = self.hot._prepare(queries)
        self.cold._prepare(q)
        return q

    def score(self, queries: Any) -> np.ndarray:
        """``TieredIndex.score``: rescored shortlist on top, everything else below it."""
        return self._tiered.score(self._prepare(queries))

    def search(self, queries: Any, k: int = 10) -> Ranking:
        """``TieredIndex.search``; ``k >= 1``, and beyond ``candidates`` ranks the hot order."""
        k = check_k(k)
        top, scores = self._tiered.search(self._prepare(queries), k)
        return ranking_from_positions(top, scores, self._ids, SCORE_TIERED)


class TwoTierRetriever:
    """Any shortlist source plus any reranker.

    Args:
        first: a :class:`Retriever` that proposes documents. Its ranking may carry store
            ``indices`` (then they are used as positions in the reranker's store) or only ids
            (a backend; then ids are mapped to positions through the reranker's ``ids``).
        reranker: a :class:`Reranker`, normally an :class:`ExactRetriever` over the colder store.
        candidates: shortlist size, ``>= 1``, clamped to ``n_docs`` (the reranker's, else the first
            stage's; one of them must report it).
        score_kind: the kind of the reranker's scores; the default suits the library's own
            MaxSim stores.

    ``search(queries, k)`` asks ``first`` for ``m = min(candidates, n_docs)`` documents, sorts
    each row of the shortlist by store position, rescores it, and ranks the cold scores (ties by
    ascending position). It has no order beyond the shortlist, so ``k`` may not exceed ``m``
    unless ``m`` already covers the whole corpus, in which case ``k`` is clamped to ``n_docs`` as
    everywhere else. For ``k <= m`` it equals ``TieredIndex.search`` provided the shortlist holds
    at least ``k`` documents with a finite cold score; ``TieredIndex`` places finite documents
    outside its shortlist above a shortlisted document that is empty in the cold store.

    Rows with fewer than ``m`` proposals (a store smaller than ``m``, an engine returning less)
    are padded with ``-1`` internally, never passed to the reranker, scored ``-inf`` and returned
    as padding. A document proposed twice for one query, a position outside the reranker's
    store (``IndexError``) or an id the reranker's store does not hold (``KeyError``) is an error:
    for backends see the id-convention note in ``optivision.backends``.

    If ``first`` and ``reranker`` both expose ``ids``, they must be equal (positions are then
    shared). A first stage without ``ids`` is assumed to be id-only or aligned by its owner; when
    it returns positions and the reranker has ids, each proposed id must be the reranker's id for
    that position (``ValueError`` otherwise). The returned ids are the proposed ones. The
    reranker's ``ids`` are read once, at the first search that needs them. ``reranker.rescore``
    must return a float32 array of the candidates' shape.
    """

    def __init__(
        self,
        first: Retriever,
        reranker: Reranker,
        candidates: int = 50,
        *,
        score_kind: str = SCORE_MAXSIM_IP,
    ) -> None:
        if not callable(getattr(first, "search", None)):
            raise TypeError(f"first must provide search(queries, k), got {type(first).__name__}")
        if callable(getattr(first, "search_many", None)) and not hasattr(first, "n_docs"):
            raise TypeError(
                f"first is a {type(first).__name__}, which looks like a Backend (its search takes "
                "one query); wrap it in optivision.backends.BackendRetriever(first)")
        if not callable(getattr(reranker, "rescore", None)):
            raise TypeError(
                f"reranker must provide rescore(queries, candidates), got {type(reranker).__name__}"
            )
        self.candidates = check_candidates(candidates)
        first_ids, rerank_ids = getattr(first, "ids", None), getattr(reranker, "ids", None)
        if first_ids is not None and rerank_ids is not None:
            _require_aligned(list(first_ids), list(rerank_ids), "first stage", "reranker")
        self.first = first
        self.reranker = reranker
        self.score_kind = score_kind
        self._id_map: dict[str, int] | None = None
        self._reranker_names: np.ndarray | None = None
        self._counting_part()  # fail now, not at the first search, if nothing counts documents

    # ------------------------------------------------------------ shape

    def _counting_part(self) -> Any:
        for part in (self.reranker, self.first):
            if getattr(part, "n_docs", None) is not None:
                return part
        raise TypeError("neither the reranker nor the first stage reports n_docs")

    @property
    def n_docs(self) -> int:
        """Documents the reranker reports, else the first stage (read at every call)."""
        return int(self._counting_part().n_docs)

    # ---------------------------------------------------------- id->pos

    def _names(self) -> np.ndarray | None:
        """The reranker's ids as an object array (read once), or ``None`` if it has none."""
        if self._reranker_names is None:
            ids = getattr(self.reranker, "ids", None)
            if ids is not None:
                self._reranker_names = _ids_array(list(ids))
        return self._reranker_names

    def _positions_from_ids(self, rows: list[list[str]], width: int) -> np.ndarray:
        if self._id_map is None:
            names = self._names()
            if names is None:
                raise TypeError(
                    "the first stage returned ids without store positions, so the reranker must "
                    "expose ids (a list of document ids in store order) to map them to positions"
                )
            mapping = {doc_id: pos for pos, doc_id in enumerate(names.tolist())}
            if len(mapping) != len(names):
                raise ValueError(
                    "the reranker's document ids are not unique, so ids cannot be mapped to positions"
                )
            self._id_map = mapping
        out = np.full((len(rows), width), -1, dtype=np.int64)
        for q, row in enumerate(rows):
            for j, doc_id in enumerate(row):
                pos = self._id_map.get(doc_id)
                if pos is None:
                    raise KeyError(
                        f"the first stage returned id {doc_id!r} (query {q}, rank {j}), which is "
                        "not a document id of the reranker's store (backend page ids are "
                        "'<doc_id>::p<page_no>'; see the id-convention note in optivision.backends)"
                    )
                out[q, j] = pos
        return out

    # ------------------------------------------------------------- work

    def search(self, queries: Any, k: int = 10) -> Ranking:
        k = check_k(k)
        q = as_queries(queries)
        n_docs = self.n_docs
        m = min(self.candidates, n_docs)
        if m < n_docs and k > m:  # when m == n_docs a complete ranking exists and k just clamps
            raise ValueError(
                f"k={k} exceeds the shortlist size {m}: a two-tier ranking has no order beyond "
                "its shortlist (raise candidates, or lower k)"
            )
        if m == 0:
            return Ranking(
                ids=[[] for _ in range(len(q))],
                scores=np.zeros((len(q), 0), dtype=np.float32),
                indices=np.zeros((len(q), 0), dtype=np.int64),
                score_kind=self.score_kind,
                tie_break=TIE_INDEX,
            )
        proposal = self.first.search(q, m)
        if proposal.n_queries != len(q):
            raise ValueError(f"the first stage returned {proposal.n_queries} rows for {len(q)} queries")
        if proposal.width > m:
            raise ValueError(f"the first stage returned {proposal.width} columns for a request of {m}")
        n_q, width = proposal.scores.shape
        if proposal.indices is not None:
            positions = proposal.indices
        else:
            positions = self._positions_from_ids(proposal.ids, width)
        if (positions >= n_docs).any():
            raise IndexError(
                f"the first stage returned position {int(positions.max())} but the reranker's "
                f"store has {n_docs} documents"
            )
        listed = positions >= 0
        proposed = np.empty((n_q, width), dtype=object)
        for row, name_row in enumerate(proposal.ids):
            proposed[row, : len(name_row)] = name_row
        known = self._names()
        if known is not None and proposal.indices is not None:
            clash = listed & (proposed != known[np.where(listed, positions, 0)])
            if clash.any():
                q_, j_ = (int(a[0]) for a in np.nonzero(clash))
                raise ValueError(
                    f"the first stage calls position {int(positions[q_, j_])} "
                    f"{proposed[q_, j_]!r} (query {q_}, rank {j_}) but the reranker's store calls it "
                    f"{known[positions[q_, j_]]!r}"
                )

        # sort each shortlist by store position, padding last (invariant I5)
        pad = np.iinfo(np.int64).max
        keyed = np.where(listed, positions, pad)
        order = np.argsort(keyed, axis=1, kind="stable")
        keyed = np.take_along_axis(keyed, order, axis=1)
        names = np.take_along_axis(proposed, order, axis=1)
        listed = keyed != pad
        twice = listed[:, 1:] & (keyed[:, 1:] == keyed[:, :-1])
        if twice.any():
            bad = int(np.flatnonzero(twice.any(axis=1))[0])
            raise ValueError(f"the first stage proposed the same document twice for query {bad}")

        cold = self._rescore(q, np.where(listed, keyed, 0), (n_q, width))
        cold = np.where(listed, cold, np.float32(-np.inf)).astype(np.float32, copy=False)
        final = rank(cold, k)
        top_scores = np.take_along_axis(cold, final, axis=1)
        top_positions = np.take_along_axis(np.where(listed, keyed, -1), final, axis=1)
        top_names = np.take_along_axis(names, final, axis=1)
        counts = (top_positions >= 0).sum(axis=1)
        return Ranking(
            ids=[top_names[row, : counts[row]].tolist() for row in range(n_q)],
            scores=top_scores,
            indices=top_positions,
            score_kind=self.score_kind,
            tie_break=TIE_INDEX,
        )

    def _rescore(self, queries: Any, candidates: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        if shape[1] == 0:
            return np.zeros(shape, dtype=np.float32)
        cold = np.asarray(self.reranker.rescore(queries, candidates))
        if cold.shape != shape:
            raise ValueError(f"reranker.rescore returned shape {cold.shape}, expected {shape}")
        if cold.dtype != np.float32:
            raise TypeError(f"reranker.rescore must return float32 scores, got {cold.dtype}")
        return cold
