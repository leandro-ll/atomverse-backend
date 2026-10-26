"""Baseline backend test-scaffolding and CI policy consistency tests.

Validates: Requirements 14.5, 24.5, 25.7
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

REQUIRED_DENY_CATEGORIES = {
    "secrets",
    "caches",
    "bytecode",
    "local-metadata",
    "semantic-review",
    "unapproved-media",
    "logs",
}


@pytest.fixture(scope="module")
def pyproject(backend_root: Path) -> dict:
    return tomllib.loads((backend_root / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def suite_registry(repository_root: Path) -> dict:
    path = repository_root / "scripts" / "ci" / "test-suites.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def release_policy(repository_root: Path) -> dict:
    path = repository_root / "scripts" / "ci" / "release-manifest-policy.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_backend_pytest_configuration_is_explicit(pyproject: dict) -> None:
    options = pyproject["tool"]["pytest"]["ini_options"]
    assert options["testpaths"] == ["tests"]
    assert "--strict-markers" in options["addopts"]
    assert "--strict-config" in options["addopts"]


def test_backend_test_directories_exist(backend_root: Path) -> None:
    assert (backend_root / "tests").is_dir()
    assert (backend_root / "tests" / "contracts").is_dir()


def test_backend_suite_fails_on_an_empty_match_set(suite_registry: dict) -> None:
    suites = {suite["id"]: suite for suite in suite_registry["suites"]}
    backend_unit = suites["backend-unit"]

    assert backend_unit["runner"] == "pytest"
    assert backend_unit["cwd"] == "backend"
    assert backend_unit["emptyPolicy"] == "fail"
    assert "emptyReason" not in backend_unit

    reserved = suites["backend-contracts"]
    assert reserved["emptyPolicy"] == "allow"
    assert len(reserved["emptyReason"]) >= 20


def test_pytest_no_tests_collected_is_never_reported_as_success(suite_registry: dict) -> None:
    runner = suite_registry["runners"]["pytest"]

    assert runner["emptyExitCode"] == 5
    assert runner["emptyExitCodeReason"]
    assert "5" not in runner["exitCodes"]
    assert runner["exitCodes"]["0"] == "passed"


def test_release_policy_denies_every_required_category(release_policy: dict) -> None:
    categories = {rule["category"] for rule in release_policy["deny"]}

    assert REQUIRED_DENY_CATEGORIES <= categories
    assert release_policy["defaultDecision"] == "unclassified"


def test_release_policy_declares_backend_exclusion_targets(release_policy: dict) -> None:
    targets = {
        target["path"]: target["category"] for target in release_policy["localExclusionTargets"]
    }

    assert targets["backend/.env"] == "secrets"
    assert targets["backend/__pycache__/main.cpython-311.pyc"] == "bytecode"
    assert targets["backend/.ruff_cache/CACHEDIR.TAG"] == "caches"

    declared_categories = {rule["category"] for rule in release_policy["deny"]}
    for path, category in targets.items():
        assert category in declared_categories, f"{path} expects undeclared category {category}"
