"""The layering is enforced, not documented.

A layering rule that lives only in a README survives until the first deadline.
These tests parse the import graph and fail the build when a dependency points
outward, so the architecture is a property of the code rather than an intention
about it.

The rules, innermost first:

``domain``
    Standard library only. No numpy, no scipy, no framework, no other VIFLAP
    package. This is what lets the concepts a court would recognise be defined
    once and stay put through a change of database or web framework.
``analysis``
    May use the domain and scientific libraries. No I/O, no framework, no
    knowledge of the layers above.
``application``
    May use the domain and analysis. No framework, no database driver.
``infrastructure`` and ``interfaces``
    May use anything below them.

There is also a rule about vocabulary, checked the same way: no module may
contain the language of identity outside the policy that defines it.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "viflap"

#: Which VIFLAP packages each layer may import from.
ALLOWED_INTERNAL: dict[str, frozenset[str]] = {
    # Every layer may import from itself; the entries below list what else it
    # may reach.
    "domain": frozenset({"domain"}),
    "analysis": frozenset({"domain", "analysis"}),
    "evaluation": frozenset({"domain", "analysis", "evaluation"}),
    "application": frozenset({"domain", "analysis", "application"}),
    "infrastructure": frozenset({"domain", "analysis", "application", "infrastructure"}),
    "interfaces": frozenset(
        {"domain", "analysis", "evaluation", "application", "infrastructure", "interfaces"}
    ),
}

#: Third-party packages each layer may import.
ALLOWED_EXTERNAL: dict[str, frozenset[str]] = {
    "domain": frozenset(),
    "analysis": frozenset({"numpy", "scipy", "matplotlib"}),
    "evaluation": frozenset({"numpy", "scipy", "matplotlib"}),
    "application": frozenset({"numpy"}),
    # torch and speechbrain are here and nowhere else, deliberately. A borrowed
    # embedding extractor is an adapter to an external artefact: it loads a
    # checkpoint from disk and depends on a deep learning framework, and both
    # are things the analysis layer is forbidden. Admitting them to
    # infrastructure keeps the science — which sample rate a checkpoint is
    # defined against, and what feeding it anything else costs — in the layer
    # that takes arrays and returns arrays.
    "infrastructure": frozenset({"numpy", "scipy", "sqlalchemy", "torch", "speechbrain"}),
    "interfaces": frozenset({"numpy", "fastapi", "pydantic", "starlette", "uvicorn"}),
}

STANDARD_LIBRARY_PREFIXES = frozenset(
    {
        "__future__",
        "abc",
        "argparse",
        "ast",
        "base64",
        "collections",
        "contextlib",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "functools",
        "hashlib",
        "hmac",
        "importlib",
        "inspect",
        "io",
        "itertools",
        "json",
        "logging",
        "math",
        "os",
        "pathlib",
        "pickle",
        "random",
        "re",
        "secrets",
        "shutil",
        "statistics",
        "string",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "time",
        "types",
        "typing",
        "unicodedata",
        "uuid",
        "warnings",
        "wave",
        "zipfile",
    }
)


def iter_modules() -> Iterator[tuple[Path, str]]:
    """Yield every VIFLAP module with the layer it belongs to."""
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT)
        layer = relative.parts[0] if len(relative.parts) > 1 else ""
        if layer in ALLOWED_INTERNAL:
            yield path, layer


def imported_roots(path: Path) -> set[str]:
    """Top-level module names imported by a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                # Relative imports stay within the module's own package, which
                # the layer rules already permit.
                continue
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


def viflap_subpackages(path: Path) -> set[str]:
    """VIFLAP sub-packages a file imports from."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    packages: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            parts = name.split(".")
            if parts[0] == "viflap" and len(parts) > 1:
                packages.add(parts[1])
    return packages


@pytest.mark.parametrize(
    "path,layer", list(iter_modules()), ids=lambda value: str(value)[-60:]
)
def test_internal_dependencies_point_inward(path: Path, layer: str) -> None:
    """No module may import from a layer outside its own."""
    allowed = ALLOWED_INTERNAL[layer]
    violations = viflap_subpackages(path) - allowed
    assert not violations, (
        f"{path.relative_to(PACKAGE_ROOT)} is in the '{layer}' layer and imports "
        f"from {sorted(violations)}, which lies outside it. Dependencies point "
        f"inward only; move the shared concept inward or invert the dependency "
        f"with a port."
    )


@pytest.mark.parametrize(
    "path,layer", list(iter_modules()), ids=lambda value: str(value)[-60:]
)
def test_external_dependencies_are_permitted(path: Path, layer: str) -> None:
    """Each layer may only use the third-party packages it is allowed."""
    allowed = ALLOWED_EXTERNAL[layer]
    violations = {
        root
        for root in imported_roots(path)
        if root not in STANDARD_LIBRARY_PREFIXES
        and root != "viflap"
        and root not in allowed
    }
    assert not violations, (
        f"{path.relative_to(PACKAGE_ROOT)} is in the '{layer}' layer and imports "
        f"{sorted(violations)}, which that layer may not use."
    )


def test_domain_layer_uses_only_the_standard_library() -> None:
    """The domain is pure, and this is the check that keeps it that way.

    Once numpy appears in the domain, the concepts a court would recognise are
    entangled with an array library's release schedule.
    """
    offenders: dict[str, set[str]] = {}
    for path in sorted((PACKAGE_ROOT / "domain").rglob("*.py")):
        external = {
            root
            for root in imported_roots(path)
            if root not in STANDARD_LIBRARY_PREFIXES and root != "viflap"
        }
        if external:
            offenders[str(path.relative_to(PACKAGE_ROOT))] = external
    assert not offenders, f"the domain layer has third-party dependencies: {offenders}"


def test_analysis_layer_performs_no_file_io() -> None:
    """The science layer takes arrays and returns arrays.

    Model persistence and codec invocation are the two documented exceptions:
    both are genuinely about serialising or invoking a computation rather than
    about the surrounding system, and both are confined to named modules.
    """
    permitted = {
        "speaker/pipeline.py",  # model artefact serialisation
        "spoof/countermeasure.py",  # model artefact serialisation
        "channel/codec.py",  # external codec invocation
        "calibration/plots.py",  # figure rendering
    }
    offenders: list[str] = []
    for path in sorted((PACKAGE_ROOT / "analysis").rglob("*.py")):
        relative = str(path.relative_to(PACKAGE_ROOT / "analysis")).replace("\\", "/")
        if relative in permitted:
            continue
        source = path.read_text(encoding="utf-8")
        if re.search(r"\bopen\s*\(|\bPath\s*\(.*\)\s*\.\s*(write|read)", source):
            offenders.append(relative)
    assert not offenders, f"analysis modules performing file I/O: {offenders}"


def test_identity_vocabulary_appears_only_in_the_policy_that_forbids_it() -> None:
    """The system has no vocabulary for asserting identity.

    Checked over source text, so a well-meaning addition of a ``match_score``
    field or a log line saying "identified" fails the build rather than reaching
    an operator's screen.

    Only the module defining the policy may contain the words, and only the
    words themselves rather than assertions built from them.
    """
    forbidden = re.compile(
        r"(?<![\w-])(match(?:es|ed|ing)?|identified|voiceprint)(?![\w-])",
        re.IGNORECASE,
    )
    exempt = {"domain/governance.py"}
    offenders: dict[str, list[str]] = {}

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = str(path.relative_to(PACKAGE_ROOT)).replace("\\", "/")
        if relative in exempt:
            continue
        hits = [
            f"{kind} (line {line}): {text[:80]}"
            for kind, line, text in _emittable_text_and_names(path)
            if forbidden.search(text)
        ]
        if hits:
            offenders[relative] = hits

    assert not offenders, (
        f"the vocabulary of identity appears in emittable strings or in "
        f"identifiers: {offenders}. Prose explaining why the vocabulary is "
        f"prohibited is permitted; a string that could reach an operator is "
        f"not — the output policy would reject it at the boundary, turning a "
        f"wording problem into a 500."
    )


def _emittable_text_and_names(path: Path) -> Iterator[tuple[str, int, str]]:
    """Yield string literals that could be emitted, and every identifier.

    Excludes docstrings and comments, which are prose *about* the system rather
    than output *from* it — the module explaining why identity claims are
    prohibited necessarily contains the words it prohibits.

    Uses the AST rather than line heuristics, because a docstring's continuation
    lines look exactly like ordinary source to a line-based scan, and every
    explanatory paragraph in the codebase would trip it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    # Any bare string expression statement is documentation by convention: a
    # module, class or function docstring, or the attribute docstring that
    # follows a field declaration. None of them can be emitted — a string that
    # is never bound to anything cannot reach an operator — so all are excluded.
    # Restricting this to the first statement of a definition would flag every
    # attribute docstring in the codebase, of which there are many, and the
    # resulting noise would get the check disabled.
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        # Guarded: several node types carry a ``body`` that is a single
        # expression rather than a statement list — a conditional expression,
        # for instance — and iterating one raises.
        if not isinstance(body, list):
            continue
        for statement in body:
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                docstrings.add(id(statement.value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            yield "string literal", node.lineno, node.value
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield "definition name", node.lineno, node.name
        elif isinstance(node, ast.Name):
            yield "name", node.lineno, node.id
        elif isinstance(node, ast.Attribute):
            yield "attribute", node.lineno, node.attr
        elif isinstance(node, ast.arg):
            yield "parameter", node.lineno, node.arg


def test_every_layer_has_a_module_docstring() -> None:
    """Each package states what it is for and where its boundary lies."""
    missing = []
    for layer in ALLOWED_INTERNAL:
        init = PACKAGE_ROOT / layer / "__init__.py"
        if not init.exists():
            continue
        tree = ast.parse(init.read_text(encoding="utf-8"))
        if not ast.get_docstring(tree):
            missing.append(layer)
    assert not missing, f"layers without a package docstring: {missing}"
