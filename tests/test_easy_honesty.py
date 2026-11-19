"""The honesty rules of easy mode (FINAL_DESIGN 7), enforced on the source of ``optivision.easy``.

Every string that reaches a user through ``Line(...)``, ``Block.add(...)`` or ``Block(title=, next=)`` is found
by reading the modules' syntax trees (so a module that another agent adds later is scanned without a change
here) and must

* pass ``_claim.check_template``: no digit outside a claim or an allowed token (a number cannot reach a
  user without its tag);
* be ASCII;
* be free of the banned words of ``tests/_easy_lint.py``: the unambiguous ones everywhere, the ambiguous
  ones only inside claim patterns, so a path or a plain sentence passes;
* never contain "veto" (the 0.6.0 output names the practice check by that name only);
* never carry two kinds of evidence, or two sources of one kind, on a line.

The same lint runs on the user documents ``docs/easy/*.md`` (not ``CONTRACT.md``, which is for builders)
and on ``EASY.md`` wherever it is.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _easy_lint import ALWAYS, CLAIM_ONLY, banned_hits, non_ascii

from optivision.easy import _claim
from optivision.easy._claim import BareNumber, Block, Line, MixedEvidence, claim, lit

ROOT = Path(__file__).resolve().parents[1]
EASY = ROOT / "src" / "optivision" / "easy"
DOCS = ROOT / "docs" / "easy"
USER_DOC_SKIP = {"CONTRACT.md"}


# ---------------------------------------------------------------------------------------- the scanner


@dataclass
class Found:
    """One call that puts text in front of a user."""

    module: str
    lineno: int
    call: str                                   # "Line", "add" or "Block"
    template: str | None = None                 # the template, when it can be read from the source
    strings: list[str] = field(default_factory=list)          # every string constant in the call
    problems: list[str] = field(default_factory=list)
    fragments: list[str] = field(default_factory=list)        # constant pieces of a template built at run time
    kinds: list[tuple[str, str | None]] = field(default_factory=list)    # (kind, source) of constant claim() calls


def _constants(tree: ast.AST) -> dict[str, str]:
    """Module-level ``NAME = "text"`` assignments, so ``Line(label, TEMPLATE)`` can be read."""
    out: dict[str, str] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value = _text(node.value, out)[0]
            if value is not None:
                out[node.targets[0].id] = value
    return out


def _text(node: ast.AST, consts: dict[str, str]) -> tuple[str | None, list[str]]:
    """(the template text or None, problems). An f-string becomes its constant parts with ``{0}`` for each value."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, []
    if isinstance(node, ast.Name):
        return consts.get(node.id), []
    if isinstance(node, ast.JoinedStr):
        parts, problems = [], []
        for piece in node.values:
            if isinstance(piece, ast.Constant):
                parts.append(str(piece.value))
            else:
                if isinstance(piece, ast.FormattedValue) and (piece.format_spec is not None or piece.conversion != -1):
                    problems.append("an f-string template formats a value; pass the value as a claim")
                parts.append("{0}")
        return "".join(parts), problems
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, lp = _text(node.left, consts)
        right, rp = _text(node.right, consts)
        if left is not None and right is not None:
            return left + right, lp + rp
        return None, lp + rp
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return None, ["a template built with % formats values before the formatter sees them; pass claims"]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        return None, ["a template built with .format() formats values before the formatter sees them; pass claims"]
    return None, []


def _called(node: ast.Call) -> str | None:
    f = node.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def _claim_kind(call: ast.Call) -> tuple[str, str | None] | None:
    """(kind, source) of a ``claim(value, "kind", "source", ...)`` call when both are constants."""
    if _called(call) != "claim":
        return None
    kind = call.args[1] if len(call.args) > 1 else next((k.value for k in call.keywords if k.arg == "kind"), None)
    if not (isinstance(kind, ast.Constant) and isinstance(kind.value, str)):
        return None
    source = call.args[2] if len(call.args) > 2 else next((k.value for k in call.keywords if k.arg == "source"), None)
    return kind.value, (source.value if isinstance(source, ast.Constant) and isinstance(source.value, str)
                        else None if source is None else "<dynamic>")


def scan_source(source: str, module: str = "<test>") -> list[Found]:
    """Every ``Line(...)``, ``<block>.add(label, template, ...)`` and ``Block(title=, next=)`` call in a source."""
    tree = ast.parse(source)
    consts = _constants(tree)
    found: list[Found] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called(node)
        template_node: ast.AST | None = None
        if name == "Line":
            template_node = node.args[1] if len(node.args) > 1 else next(
                (k.value for k in node.keywords if k.arg == "template"), None)
        elif name == "add" and isinstance(node.func, ast.Attribute) and len(node.args) >= 2 and (
                isinstance(node.args[1], (ast.Constant, ast.JoinedStr, ast.BinOp, ast.Name))
                or (isinstance(node.args[1], ast.Call) and _called(node.args[1]) == "format")):
            template_node = node.args[1]
            if isinstance(template_node, ast.Constant) and not isinstance(template_node.value, str):
                continue
            if isinstance(template_node, ast.Name) and template_node.id not in consts:
                template_node = None
        elif name == "Block":
            pass
        else:
            continue
        item = Found(module, node.lineno, name)
        if template_node is not None:
            item.template, item.problems = _text(template_node, consts)
            if item.template is None:
                item.fragments = [c.value for c in ast.walk(template_node)
                                  if isinstance(c, ast.Constant) and isinstance(c.value, str)]
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                item.strings.append(sub.value)
            if isinstance(sub, ast.Call):
                k = _claim_kind(sub)
                if k is not None:
                    item.kinds.append(k)
        if name == "Block":
            item.strings = [s for k in node.keywords if k.arg in ("title", "next")
                            for s in [_text(k.value, consts)[0]] if s is not None]
            item.kinds = []
            if not item.strings:
                continue
        if name == "add" and not item.template and template_node is None:
            continue
        found.append(item)
    return found


def _problems(item: Found) -> list[str]:
    out = list(item.problems)
    if item.template is not None:
        try:
            _claim.check_template(item.template)
        except BareNumber as exc:
            out.append(str(exc))
        if "veto" in item.template.lower():
            out.append(f"the word 'veto' in template {item.template!r}")
    for piece in item.fragments:                  # a template built at run time: its constant pieces hold no number
        try:
            _claim.check_template(piece)
        except BareNumber as exc:
            out.append(f"a piece of a template built at run time: {exc}")
    for text in item.strings:
        if non_ascii(text):
            out.append(f"non-ASCII {non_ascii(text)} in {text!r}")
        for hit in banned_hits(text):
            out.append(f"banned wording {hit}")
        if "veto" in text.lower():
            out.append(f"the word 'veto' in {text!r}")
    kinds = {k for k, _ in item.kinds}
    if len(kinds) > 1:
        out.append(f"one line, two kinds of evidence: {sorted(kinds)}")
    for kind in kinds:
        sources = {s for k, s in item.kinds if k == kind and s not in (None, "<dynamic>")}
        if len(sources) > 1:
            out.append(f"one line, two sources of {kind}: {sorted(sources)}")
    return out


def _modules() -> list[Path]:
    return sorted(EASY.rglob("*.py"))


def _id(path: Path) -> str:
    return path.relative_to(EASY).as_posix()


# ---------------------------------------------------------------------------------------- the scan itself


@pytest.mark.parametrize("path", _modules(), ids=_id)
def test_every_user_facing_template_of_a_module_is_honest(path):
    try:
        found = scan_source(path.read_text(encoding="utf-8"), _id(path))
    except SyntaxError as exc:
        pytest.fail(f"{path} does not parse: {exc}")
    bad = [f"{_id(path)}:{item.lineno} ({item.call}): {problem}" for item in found for problem in _problems(item)]
    assert not bad, "\n".join(bad)


def test_the_scan_finds_the_templates_that_exist():
    """The scan is not vacuous: modules that build lines (the glossary, any later module) are read."""
    total = 0
    for path in _modules():
        total += len(scan_source(path.read_text(encoding="utf-8"), _id(path)))
    assert total >= 1, "no Line(...) or Block.add(...) call found under src/optivision/easy/"


def test_the_scanner_reads_what_it_should():
    src = '''
from optivision.easy._claim import Block, Line, claim, lit
T = "kept {0} of the quality"
def f(block):
    Line("a", "plain {0}", claim(0.9, "counted"))
    block.add("b", T, claim(0.9, "counted"))
    block.add("c", "x " "y {0}", lit("z"))
    block.add("d", f"{n:.1f} done")
    block.add("e", "%d files" % n)
    block.add("f", "{0}".format(n))
    block.add("g", "mix {0} {1}", claim(1, "counted"), claim(2, "probe"))
    block.add("h", "two {0} {1}", claim(1, "catalogue", "A"), claim(2, "catalogue", "B"))
    Block([], next="ov.build(...)", title="T")
    other.add("just one argument")
    seen.add(3)
'''
    found = scan_source(src)
    by_label = {f.strings[0]: f for f in found if f.strings}
    assert by_label["a"].template == "plain {0}" and by_label["a"].kinds == [("counted", None)]
    assert by_label["b"].template == "kept {0} of the quality"
    assert by_label["c"].template == "x y {0}"
    assert "formats a value" in by_label["d"].problems[0]
    assert "%" in by_label["e"].problems[0] and ".format()" in by_label["f"].problems[0]
    assert any("two kinds of evidence" in p for p in _problems(by_label["g"]))
    assert any("two sources" in p for p in _problems(by_label["h"]))
    assert any(f.call == "Block" for f in found)
    assert len(found) == 9, [f.call + ":" + str(f.lineno) for f in found]


def test_the_scanner_checks_the_pieces_of_templates_built_at_run_time():
    src = '''
def f(block, unit, rng):
    block.add("a", rng + " MB per thousand " + unit, lit("x"))
    block.add("b", rng + " kept 97.2% of " + unit, lit("x"))
'''
    found = {f.strings[0]: f for f in scan_source(src)}
    assert found["a"].template is None and found["a"].fragments and _problems(found["a"]) == []
    assert any("built at run time" in p for p in _problems(found["b"]))


def test_the_scanner_flags_bad_templates():
    src = '''
def f(block):
    block.add("n", "kept 97.2% of the quality")
    block.add("v", "the veto fired")
    block.add("a", "café {0}", lit("x"))
    block.add("l", "lossless {0}", lit("x"))
    block.add("ok", "see optivision-universal and reports/universal/ and the first pages {0}", lit("x"))
    Line("l2", "the first tool to do it")
'''
    found = {f.strings[0]: f for f in scan_source(src)}
    assert any("not a claim" in p for p in _problems(found["n"]))
    assert any("veto" in p for p in _problems(found["v"]))
    assert any("non-ASCII" in p for p in _problems(found["a"]))
    assert any("lossless" in p for p in _problems(found["l"]))
    assert _problems(found["ok"]) == [] or all("first" not in p for p in _problems(found["ok"]))
    assert any("first (claim)" in p for p in _problems(found["l2"]))


# ---------------------------------------------------------------------------------------- the formatter's rules


def test_mixed_evidence_triggers_on_mixed_lines():
    with pytest.raises(MixedEvidence):
        Line("x", "{0} and {1}", claim(0.97, "catalogue", "ViDoRe V1 DocVQA 500 pages"), claim(137, "probe"))
    with pytest.raises(MixedEvidence):
        Line("x", "{0} and {1}", claim(1, "counted"), claim(2, "estimate", "our Kaggle run"))
    with pytest.raises(MixedEvidence):                      # one kind, two sources
        Line("x", "{0} and {1}", claim(1, "catalogue", "DocVQA 500 pages"), claim(2, "catalogue", "InfoVQA 500 pages"))
    with pytest.raises(MixedEvidence):
        Block([]).add("x", "{0} {1}", claim(1, "user-queries", "agreement with full precision, not relevance"),
                      claim(2, "user-queries", "search quality"))
    # one kind and one source is fine, and literals never mix anything
    Line("x", "{0} and {1}", claim(1, "catalogue", "DocVQA 500 pages"), claim(2, "catalogue", "DocVQA 500 pages"))
    Line("x", "{0} and {1}", claim(1, "counted"), lit("a path"))


def test_a_number_cannot_reach_a_user_without_a_tag():
    with pytest.raises(BareNumber):
        Line("x", "kept 97.2% of the quality")
    with pytest.raises(BareNumber):
        Block([]).add("x", "8400 pages")
    with pytest.raises(TypeError):
        Line("x", "{0}", 0.97)                                # a raw float is not a claim or a literal
    Line("x", "the 8-bit layout of float16 vectors on a T4, nDCG@5")      # names with digits are allowed tokens


def test_the_lint_flags_claims_and_lets_paths_pass():
    for bad in ("lossless", "a universal layer", "guaranteed to keep", "state of the art", "state-of-the-art",
                "SOTA results", "the veto", "vetoed", "the first tool to compress", "first-ever", "faster than",
                "better than float16", "novel method", "simply run it", "just run it", "works with every model",
                "universal compression", "it scales to 20,000 pages"):
        assert banned_hits(bad), bad
    for good in ("pip install optivision-universal", "reports/universal/summary.json", "src/optivision/easy/_veto.py",
                 "first_page.png", "the first 50 pages are timed", "adjusted on our validation sets", "justify",
                 "Fewer bits per number; centred 4-bit subtracts the average first"):
        assert banned_hits(good) == [], good


def test_the_lint_covers_every_banned_phrase_of_the_docs_tests():
    """If ``tests/test_docs_examples.py`` gains a banned word, this fails until ``_easy_lint`` has it too."""
    import importlib

    try:
        docs = importlib.import_module("tests.test_docs_examples")
    except ImportError:
        pytest.skip("tests.test_docs_examples is not importable from here")
    for phrase in docs.BANNED_PHRASES:
        assert banned_hits(f"x {phrase} y"), f"{phrase!r} is banned by the docs tests but not by _easy_lint"
    for word in ("simply", "just"):
        assert docs.BANNED_WORDS.search(word) and banned_hits(f"it is {word} so"), word
    for word in ("lossless", "universal", "guaranteed", "SOTA", "state of the art", "veto"):
        assert banned_hits(f"a {word} thing"), word
    assert "first (claim)" in CLAIM_ONLY and {"veto", "lossless", "universal", "guaranteed"} <= set(ALWAYS)


# ---------------------------------------------------------------------------------------- the documents


def _user_docs() -> list[Path]:
    return sorted(p for p in DOCS.glob("*.md") if p.name not in USER_DOC_SKIP) if DOCS.is_dir() else []


def _easy_md() -> list[Path]:
    candidates = [ROOT / "EASY.md", DOCS / "EASY.md", ROOT / "docs" / "EASY.md", ROOT / "docs" / "package" / "EASY.md",
                  EASY / "EASY.md"]
    return sorted({p for p in candidates if p.is_file()})


@pytest.mark.parametrize("path", _user_docs() or [None], ids=lambda p: "none" if p is None else p.name)
def test_user_documents_pass_the_claim_lint(path):
    if path is None:
        pytest.skip("docs/easy has no user document yet (CONTRACT.md is for builders and is not linted)")
    text = path.read_text(encoding="utf-8")
    assert banned_hits(text) == [], path.name
    assert "veto" not in text.lower(), f"{path.name} names the practice check by an internal word"


def test_easy_md_never_says_veto_and_passes_the_lint():
    for path in _easy_md():
        text = path.read_text(encoding="utf-8")
        assert "veto" not in text.lower(), f"{path} must not use the word 'veto'"
        assert banned_hits(text) == [], str(path)


def test_the_contract_is_not_a_user_document():
    assert "CONTRACT.md" in USER_DOC_SKIP
    assert all(p.name != "CONTRACT.md" for p in _user_docs())
