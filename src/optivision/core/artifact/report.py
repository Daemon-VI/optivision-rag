"""The outcome of verifying a certificate: named checks, each with a status and a reason.

Verification never raises on a bad certificate, it reports. A report is a list of
:class:`Check` objects; ``ok`` means that no check failed. It does not mean that the
certificate is true: a certificate is only as good as the queries, labels and corpus it was
issued for (see ``docs/package/CERTIFICATE_SPEC.md``, sections 1 and 8).

Statuses:

``pass``      the check ran and found nothing wrong;
``fail``      the check ran and found a problem;
``skipped``   the check was not run (an input was not supplied, or it does not apply);
``differs``   the running code or libraries differ from the ones that issued the
              certificate. That is information, not tampering by itself; ``strict_code``
              turns it into a failure.

Levels: ``integrity`` (the certificate alone), ``replay`` (needs the evidence) and
``binding`` (needs the live corpus, queries, reference, pipeline or compressed corpus).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "BINDING",
    "DIFFERS",
    "FAIL",
    "INTEGRITY",
    "LEVELS",
    "PASS",
    "REPLAY",
    "SKIPPED",
    "STATUSES",
    "Check",
    "VerificationReport",
]

PASS, FAIL, SKIPPED, DIFFERS = "pass", "fail", "skipped", "differs"
STATUSES = (PASS, FAIL, SKIPPED, DIFFERS)
INTEGRITY, REPLAY, BINDING = "integrity", "replay", "binding"
LEVELS = (INTEGRITY, REPLAY, BINDING)


@dataclass(frozen=True)
class Check:
    """One named check: ``status`` is one of ``pass``, ``fail``, ``skipped``, ``differs``."""

    name: str
    status: str
    detail: str = ""
    level: str = INTEGRITY

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {self.level!r}")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail,
                "level": self.level}


@dataclass(frozen=True)
class VerificationReport:
    """Every check that was considered, in the order it was made."""

    checks: tuple[Check, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))

    @property
    def ok(self) -> bool:
        """True when no check failed (``differs`` and ``skipped`` do not fail a report)."""
        return not any(c.status == FAIL for c in self.checks)

    @property
    def levels(self) -> tuple[str, ...]:
        """The levels in which at least one check actually ran, in level order."""
        ran = {c.level for c in self.checks if c.status != SKIPPED}
        return tuple(level for level in LEVELS if level in ran)

    @property
    def level(self) -> str:
        """The highest level in which at least one check ran: ``integrity``, ``replay`` or
        ``binding``. A level can be reached while only some of its checks ran: read ``checks``
        (or ``summary()``) for which ones."""
        return self.levels[-1] if self.levels else INTEGRITY

    def failed(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.status == FAIL)

    def differing(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.status == DIFFERS)

    def get(self, name: str) -> Check:
        """The check called ``name``; ``KeyError`` if it was not part of the report."""
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(f"no check named {name!r}; the report has {[c.name for c in self.checks]}")

    def status(self, name: str) -> str:
        return self.get(name).status

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "level": self.level, "levels": list(self.levels),
                "checks": [c.to_dict() for c in self.checks]}

    def summary(self) -> str:
        """A readable multi-line account of what was checked and what was not."""
        n_failed = len(self.failed())
        head = "OK" if self.ok else f"FAILED ({n_failed} check{'s' if n_failed != 1 else ''})"
        lines = [
            f"Certificate verification: {head}",
            f"  level reached: {self.level} (levels with checks run: {', '.join(self.levels)})",
        ]
        width = max((len(c.name) for c in self.checks), default=0)
        for c in self.checks:
            lines.append(f"  [{c.status:<7}] {c.name:<{width}}  {c.detail}".rstrip())
        lines.append(
            "  ok means that no check failed. It does not make the certificate true: it is as "
            "good as the queries, labels and corpus it was issued for. 'skipped' checks were "
            "not run; 'differs' reports different code or libraries, not tampering by itself.")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
