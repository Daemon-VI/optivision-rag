"""The frozen rule and its registered arms (FINAL_DESIGN 5.1.2 and section 8, V1-arms and arm S).

Arm A is ``optivision.easy._levels.select`` itself. The other arms change one margin and keep
everything else of the selection the rule returned (its rows, their lowest-kept values and
ratios, which rows the folder's path allows, the chain rule and the pick rule):

- B: no 1-bit margin (``need = T + size(rung)*s(N)``);
- C: no size term (``need = T + 0.025*[1-bit]``);
- D: one flat margin (``need = T + 0.01`` on every compressed rung);
- S: the size coefficients refit by arm S (``need = T + c_S(rung)*s(N) + 0.025*[1-bit]``).

:func:`arm_choice` recomputes arm A as a control; ``run_all`` records any disagreement with
``select`` and, if there is one, reports the arms as not comparable.
"""

from __future__ import annotations

from typing import Any

from easy_validation import api, units

#: row statuses of a layout the folder's path allows (the others are refused before margins)
ALLOWED = {"pick", "qualifies", "close call", "below target", "family stopped"}
ARMS = ("A", "B", "C", "D")


def need(r: api.Rung, target: float, n_pages: int, arm: str, modality: str,
         coefficients: dict[str, float] | None = None) -> float:
    """What a rung's lowest kept must reach under an arm. ``"A+0.01"`` adds 0.01 to every
    compressed rung of the frozen rule (the V1b/V1c decision rule, in-sample)."""
    if r.is_float:
        return target
    extra = 0.0
    if arm.startswith("A+"):
        extra = float(arm[2:].split("/")[0])
        arm = "A"
    if modality == "text":
        return target + units.TEXT_MARGIN + extra
    s = units.s_of_n(n_pages)
    coef = (coefficients or units.SIZE_COEFFICIENTS)[r.size_class]
    one_bit = units.ONE_BIT_MARGIN if r.one_bit else 0.0
    if arm in ("A", "S"):
        return target + coef * s + one_bit + extra
    if arm == "B":
        return target + coef * s
    if arm == "C":
        return target + one_bit
    if arm == "D":
        return target + 0.01
    raise ValueError(f"unknown arm {arm!r}")


def arm_choice(selection: Any, target: float, n_pages: int, arm: str, modality: str,
               coefficients: dict[str, float] | None = None) -> tuple[api.Rung, list[Any]]:
    """(pick, fixed test sequence) of an arm over the rows ``select`` returned."""
    rows = api.rows(selection)
    floats = [r for r in rows if r.is_float]
    allowed = [r for r in rows if not r.is_float and r.status in ALLOWED and r.lowest_kept is not None]
    if arm.endswith("/codecs"):  # the decision rule's cap: codec chains only, no merges
        allowed = [r for r in allowed if not r.merged]
    qualifies: dict[str, bool] = {}
    for chain in dict.fromkeys(r.chain for r in allowed):
        ok = True
        for r in (x for x in allowed if x.chain == chain):  # lightest rung first (selection order)
            ok = ok and r.lowest_kept >= need(r, target, n_pages, arm, modality, coefficients) - 1e-12
            qualifies[r.key] = ok
    candidates = sorted((r for r in allowed if qualifies.get(r.key)), key=lambda r: -(r.ratio or 0.0))
    f16 = next((r for r in floats if "float16" in r.name or r.spec.get("stages") == [{"stage": "float16"}]),
               floats[0] if floats else None)
    f32 = next((r for r in floats if r is not f16), f16)
    if candidates:
        pick = candidates[0]
    elif f16 is not None and f16.lowest_kept is not None and f16.lowest_kept >= target:
        pick = f16
    else:
        pick = f32
    seq = [pick] + [r for r in candidates[1:]] + ([f16] if f16 is not None and pick is not f16 and not pick.is_float
                                                     else [])
    return pick, [r.row for r in seq]


def refit_coefficients(drops: dict[str, list[tuple[float, float]]]) -> dict[str, float]:
    """Arm S: per rung class, the largest low drop per unit of s(N), rounded up to 0.0025.

    ``drops[class]`` holds ``(low(500) - low(N), s(N))`` for every unit and N > 500. A class
    with no measured drop keeps the frozen coefficient; a negative largest drop gives 0.
    """
    out = {}
    for cls, frozen in units.SIZE_COEFFICIENTS.items():
        values = [d / s for d, s in drops.get(cls, []) if s > 0]
        out[cls] = units.round_up(max(0.0, max(values))) if values else frozen
    return out
