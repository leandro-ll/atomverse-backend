"""Tests for build-time chemistry toolchain isolation and runtime seam integrity.

Validates: Requirements 3.1, 3.2, 3.3, 3.4
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path


FORBIDDEN_RUNTIME_PACKAGES = {"chempy", "sympy", "scipy"}
FORBIDDEN_APP_IMPORTS = {"chempy", "scripts.catalog"}


def test_build_dependencies_pinned_in_pyproject(backend_root: Path) -> None:
    pyproject = tomllib.loads((backend_root / "pyproject.toml").read_text(encoding="utf-8"))
    build_deps = pyproject.get("project", {}).get("optional-dependencies", {}).get("build", [])

    assert any(dep.startswith("chempy==") for dep in build_deps), (
        "ChemPy must be pinned to an exact version in [project.optional-dependencies.build]"
    )


def test_build_lock_contains_chempy(backend_root: Path) -> None:
    build_lock_path = backend_root / "requirements-build.lock"
    assert build_lock_path.is_file(), "requirements-build.lock must exist"

    content = build_lock_path.read_text(encoding="utf-8").lower()
    assert "chempy==" in content, "requirements-build.lock must contain pinned chempy"


def test_runtime_lock_excludes_build_dependencies(backend_root: Path) -> None:
    runtime_lock_path = backend_root / "requirements.lock"
    assert runtime_lock_path.is_file(), "requirements.lock must exist"

    runtime_text = runtime_lock_path.read_text(encoding="utf-8").lower()
    # Find all package names declared at the start of lines or before '=='
    declared_packages = set(re.findall(r"^([a-z0-9_-]+)==", runtime_text, flags=re.MULTILINE))

    for forbidden in FORBIDDEN_RUNTIME_PACKAGES:
        assert forbidden not in declared_packages, (
            f"Forbidden build dependency '{forbidden}' found in runtime lock requirements.lock"
        )
        assert f"\n{forbidden}==" not in runtime_text and not runtime_text.startswith(f"{forbidden}=="), (
            f"Forbidden package '{forbidden}' must not be present in runtime lock"
        )


def test_backend_app_modules_never_import_build_toolchain(backend_root: Path) -> None:
    app_root = backend_root / "app"
    if not app_root.exists():
        return

    python_files = list(app_root.rglob("*.py"))
    assert python_files, "Expected backend/app python modules to inspect"

    for py_file in python_files:
        content = py_file.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in FORBIDDEN_APP_IMPORTS:
                        assert not alias.name.startswith(forbidden), (
                            f"{py_file} imports forbidden build toolchain module '{alias.name}'"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for forbidden in FORBIDDEN_APP_IMPORTS:
                        assert not node.module.startswith(forbidden), (
                            f"{py_file} imports from forbidden build toolchain module '{node.module}'"
                        )
