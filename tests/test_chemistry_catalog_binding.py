"""Backend parity tests for the generated chemistry catalog binding.

The backend reads reagent, ion, reaction, equation, capability, and inverse
candidate data from ``app.generated.chemistry_catalog``, which is generated from
the same committed artifact the frontend binding is generated from. These tests
confirm the backend binding still describes exactly the committed catalog
release, stays deeply read-only, and agrees with the frontend binding on catalog
version, digests, and supported definitions.

Byte-stable regeneration is owned by the generator and covered by
``tests/chemistry/catalog-bindings.test.js``; canonical serialization and digest
computation live in the Node toolchain, so this suite compares the embedded
digests against the committed digest artifact rather than recomputing them.

Validates: Requirements 1.1, 1.4, 27.6
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from app.generated import chemistry_catalog as binding

MANIFEST_BLOCK = "atomverse:manifest-json"
FRONTEND_BINDING_RELATIVE_PATH = "js/generated/chemistry-catalog.js"

_STRING_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')

CONTRACT_FIELDS = (
    "bindingContractVersion",
    "schemaVersion",
    "catalogVersion",
    "locale",
    "digestAlgorithm",
    "canonicalization",
    "catalogFile",
    "catalogDigest",
    "schemaFile",
    "schemaDigest",
    "digestValueFile",
    "definitionsDigest",
)


def extract_binding_manifest(source: str) -> dict[str, Any]:
    """Decodes the manifest embedded in a generated binding source.

    The manifest is read out of the marked string-literal block as text, so the
    frontend binding is inspected without a JavaScript runtime. Every escape the
    generator emits is valid JSON, which is what makes the two-stage decode
    below work for both languages.
    """
    begin = source.find(f"{MANIFEST_BLOCK}:begin")
    end = source.find(f"{MANIFEST_BLOCK}:end")
    assert begin >= 0 and end > begin, f"{MANIFEST_BLOCK} block markers are missing"

    segments = _STRING_LITERAL.findall(source[begin:end])
    assert segments, f"{MANIFEST_BLOCK} block contains no string literal"

    return json.loads(json.loads('"' + "".join(segments) + '"'))


def thaw(value: Any) -> Any:
    """Converts the read-only binding view back to plain JSON data."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def diff_contracts(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    """Names every contract field on which two bindings disagree."""
    return [field for field in CONTRACT_FIELDS if left.get(field) != right.get(field)]


@pytest.fixture(scope="module")
def catalog(repository_root: Path) -> dict[str, Any]:
    path = repository_root / "catalog" / "chemistry.catalog.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def committed_digest(repository_root: Path) -> dict[str, Any]:
    path = repository_root / "catalog" / "chemistry.digest.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frontend_manifest(repository_root: Path) -> dict[str, Any]:
    source = (repository_root / FRONTEND_BINDING_RELATIVE_PATH).read_text(encoding="utf-8")
    return extract_binding_manifest(source)


def test_binding_serves_the_committed_catalog(catalog: dict[str, Any]) -> None:
    assert thaw(binding.CATALOG) == catalog
    assert binding.CATALOG_VERSION == catalog["catalogVersion"]
    assert binding.SCHEMA_VERSION == catalog["schemaVersion"]
    assert binding.LOCALE == catalog["locale"]
    assert binding.DIGEST_ALGORITHM == catalog["digest"]["algorithm"]
    assert binding.CANONICALIZATION == catalog["digest"]["canonicalization"]


def test_binding_embeds_the_committed_catalog_and_schema_digests(
    committed_digest: dict[str, Any],
) -> None:
    assert binding.CATALOG_DIGEST == committed_digest["catalogDigest"]
    assert binding.SCHEMA_DIGEST == committed_digest["schemaDigest"]
    assert binding.BINDING_CONTRACT["catalogFile"] == committed_digest["catalogFile"]
    assert binding.BINDING_CONTRACT["schemaFile"] == committed_digest["schemaFile"]
    assert re.fullmatch(r"[0-9a-f]{64}", binding.CATALOG_DIGEST)
    assert re.fullmatch(r"[0-9a-f]{64}", binding.DEFINITIONS_DIGEST)


def test_binding_is_deeply_read_only() -> None:
    assert isinstance(binding.CATALOG, Mapping)
    assert isinstance(binding.CATALOG["reagents"]["AgNO3"]["ions"], tuple)
    assert isinstance(binding.CATALOG["reagents"]["AgNO3"]["ions"][0], Mapping)

    with pytest.raises(TypeError):
        binding.CATALOG["reagents"] = {}
    with pytest.raises(TypeError):
        binding.CATALOG["reagents"]["AgNO3"]["ions"][0]["multiplicity"] = 5
    with pytest.raises(TypeError):
        del binding.CATALOG["species"]["AgCl"]
    with pytest.raises(AttributeError):
        binding.CATALOG["reactions"]["AgNO3|NaCl"]["molecularProducts"].append({})

    assert binding.CATALOG["reagents"]["AgNO3"]["ions"][0]["multiplicity"] == 1


def test_supported_definitions_cover_the_catalog(catalog: dict[str, Any]) -> None:
    definitions = binding.SUPPORTED_DEFINITIONS

    assert list(definitions["reagentIds"]) == sorted(catalog["reagents"])
    assert list(definitions["ionIds"]) == sorted(catalog["ions"])
    assert list(definitions["speciesIds"]) == sorted(catalog["species"])
    assert [reaction["pairKey"] for reaction in definitions["reactions"]] == sorted(
        catalog["reactions"]
    )
    assert sorted(reaction["reactionId"] for reaction in definitions["reactions"]) == sorted(
        reaction["id"] for reaction in catalog["reactions"].values()
    )
    assert list(definitions["categories"]) == sorted(catalog["model"]["categories"])
    assert list(definitions["pHSupportedReactionIds"]) == sorted(
        catalog["model"]["capabilities"]["pH"]["supportedReactionIds"]
    )
    assert thaw(definitions["bounds"]) == catalog["model"]["bounds"]


def test_backend_and_frontend_bindings_agree(frontend_manifest: dict[str, Any]) -> None:
    assert diff_contracts(frontend_manifest["contract"], binding.BINDING_CONTRACT) == []
    assert frontend_manifest["supportedDefinitions"] == thaw(binding.SUPPORTED_DEFINITIONS)
    assert frontend_manifest["contract"]["definitionsDigest"] == binding.DEFINITIONS_DIGEST


def test_contract_comparison_detects_version_and_digest_divergence(
    frontend_manifest: dict[str, Any],
) -> None:
    diverged = dict(frontend_manifest["contract"])
    diverged["catalogVersion"] = "9.9.9"
    diverged["catalogDigest"] = "0" * 64

    assert diff_contracts(diverged, binding.BINDING_CONTRACT) == ["catalogVersion", "catalogDigest"]
