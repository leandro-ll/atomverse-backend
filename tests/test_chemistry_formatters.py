"""Backend display-formatter tests and frontend parity snapshot.

The backend and the frontend derive every formula, equation, and localized
model-boundary string from the same structural catalog terms, so they must
produce identical output. Both suites assert against the shared golden snapshot
``tests/fixtures/chemistry/display-formatting.json``, which is written by hand
and therefore cannot drift along with a wrong catalog value or a wrong
formatter.

Validates: Requirements 1.3, 1.5, 2.2
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from app.domain.chemistry_formatters import (
    FORMATTER,
    ChemistryFormatter,
    format_charge_superscript,
    format_formula_terms,
    format_phase,
    format_subscript,
    normalize_pair_key,
)
from app.generated.chemistry_catalog import CATALOG

FIXTURE_RELATIVE_PATH = Path("tests") / "fixtures" / "chemistry" / "display-formatting.json"


@pytest.fixture(scope="module")
def fixture(repository_root: Path) -> dict[str, Any]:
    return json.loads((repository_root / FIXTURE_RELATIVE_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def formatter() -> ChemistryFormatter:
    return FORMATTER


def test_formatter_matches_the_committed_catalog_release(
    fixture: dict[str, Any], formatter: ChemistryFormatter
) -> None:
    assert fixture["catalogVersion"] == formatter.catalog_version
    assert fixture["locale"] == formatter.locale


def test_species_and_ion_formulas_are_derived_from_structural_terms(
    fixture: dict[str, Any], formatter: ChemistryFormatter
) -> None:
    assert sorted(fixture["speciesFormulas"]) == sorted(CATALOG["species"])
    assert sorted(fixture["ionFormulas"]) == sorted(CATALOG["ions"])

    for species_id, expected in fixture["speciesFormulas"].items():
        assert formatter.format_species_formula(species_id) == expected
        assert formatter.format_formula(species_id) == expected
        assert not any(character in "0123456789" for character in expected)

    for ion_id, expected in fixture["ionFormulas"].items():
        assert formatter.format_ion_formula(ion_id) == expected
        assert formatter.format_formula(ion_id) == expected

    assert formatter.format_species_formula("Pb(NO3)2") == "Pb(NO\u2083)\u2082"
    assert formatter.format_species_formula("AgNO3") == "AgNO\u2083"
    assert formatter.format_ion_formula("SO4_2-") == "SO\u2084\u00b2\u207b"
    assert formatter.describe_ion("Pb2+")["charge"] == 2


def test_reagent_labels_and_dissociation_follow_the_catalog(
    fixture: dict[str, Any], formatter: ChemistryFormatter
) -> None:
    for reagent_id, expected in fixture["reagentLabels"].items():
        assert formatter.format_reagent_label(reagent_id) == expected
    for reagent_id, expected in fixture["dissociationEquations"].items():
        assert formatter.format_dissociation_equation(reagent_id) == expected

    reagents = formatter.list_reagents()
    assert [reagent["reagentId"] for reagent in reagents] == sorted(
        CATALOG["reagents"], key=lambda reagent_id: CATALOG["reagents"][reagent_id]["displayOrder"]
    )
    assert [ion["multiplicity"] for ion in reagents[1]["ions"]] == [1, 2]


def test_equations_are_identical_in_either_reagent_order(
    fixture: dict[str, Any], formatter: ChemistryFormatter
) -> None:
    assert len(fixture["reactions"]) == len(CATALOG["reactions"])

    for expected in fixture["reactions"]:
        reagent_a, reagent_b = expected["reagents"]
        pair_key = expected["pairKey"]
        assert normalize_pair_key(reagent_a, reagent_b) == pair_key

        assert formatter.format_molecular_equation(pair_key) == expected["molecularEquation"]
        assert (
            formatter.format_molecular_equation(expected["reactionId"])
            == expected["molecularEquation"]
        )
        assert formatter.format_net_ionic_equation(pair_key) == expected["netIonicEquation"]
        assert formatter.format_product_summary(pair_key) == expected["productSummary"]
        assert list(formatter.format_spectator_ions(pair_key)) == expected["spectatorIons"]

        forward = formatter.format_reaction_display(reagent_a, reagent_b)
        reverse = formatter.format_reaction_display(reagent_b, reagent_a)
        assert forward["molecularEquation"] == reverse["molecularEquation"]
        assert forward["netIonicEquation"] == reverse["netIonicEquation"]
        assert forward["reactionId"] == reverse["reactionId"] == expected["reactionId"]
        assert forward["productSummary"] == expected["productSummary"]
        assert forward["category"] == expected["category"]
        assert forward["energyLabel"] == expected["energyLabel"]
        assert forward["phase"]["layered"] is False

        if expected["precipitate"] is None:
            assert forward["precipitate"] is None
            assert forward["pH"]["supported"] is True
        else:
            assert forward["precipitate"]["formula"] == expected["precipitate"]["formula"]
            assert forward["precipitate"]["displayName"] == expected["precipitate"]["displayName"]
            assert forward["pH"]["supported"] is False
            assert forward["pH"]["reasonCode"] == fixture["pHUnsupportedReasonCodes"][pair_key]


def test_energy_term_is_appended_only_to_an_exothermic_equation(
    fixture: dict[str, Any], formatter: ChemistryFormatter
) -> None:
    energy_term = fixture["energyTerm"]

    for expected in fixture["reactions"]:
        pair_key = expected["pairKey"]
        molecular = formatter.format_molecular_equation(pair_key, energy_term)
        net_ionic = formatter.format_net_ionic_equation(pair_key, energy_term)

        if expected["energyLabel"] == "exothermic":
            assert molecular == expected["molecularEquationWithEnergy"]
            assert net_ionic == expected["netIonicEquationWithEnergy"]
        else:
            assert molecular == expected["molecularEquation"]
            assert net_ionic == expected["netIonicEquation"]


def test_unmatched_pairs_report_unsupported_with_no_equation(
    fixture: dict[str, Any], formatter: ChemistryFormatter
) -> None:
    expected = fixture["noModeledReaction"]

    for pair in fixture["unmodeledPairs"]:
        reagent_a, reagent_b = pair["reagents"]
        assert formatter.find_reaction(reagent_a, reagent_b) is None

        for display in (
            formatter.format_reaction_display(reagent_a, reagent_b),
            formatter.format_reaction_display(reagent_b, reagent_a),
        ):
            assert display["pairKey"] == pair["pairKey"]
            assert display["status"] == expected["status"]
            assert display["category"] == expected["category"]
            assert display["reasonCode"] == expected["reasonCode"]
            assert display["messageKey"] == expected["messageKey"]
            assert display["message"] == fixture["boundaryMessages"][expected["messageKey"]]
            assert display["molecularEquation"] is None
            assert display["netIonicEquation"] is None
            assert display["productSummary"] is None
            assert display["precipitate"] is None
            assert display["reactionId"] is None
            assert display["spectatorIons"] == ()
            assert display["phase"]["phaseModel"] == expected["phaseModel"]
            assert display["phase"]["layered"] == expected["layered"]
            assert display["pH"]["reasonCode"] == expected["pHReasonCode"]
            assert display["pH"]["pH"] is None


def test_localized_model_boundaries_are_resolved_from_the_catalog(
    fixture: dict[str, Any], formatter: ChemistryFormatter
) -> None:
    for message_key, message in fixture["boundaryMessages"].items():
        assert formatter.resolve_message(message_key) == message

    for capability, message_key in fixture["capabilityBoundaryKeys"].items():
        assert formatter.describe_capability_boundary(capability) == {
            "capability": capability,
            "messageKey": message_key,
            "message": fixture["boundaryMessages"][message_key],
        }

    for reason_code, message_key in fixture["pHUnsupportedMessageKeys"].items():
        assert formatter.describe_unsupported_ph(reason_code) == {
            "status": "unsupported",
            "capability": "pH",
            "reasonCode": reason_code,
            "messageKey": message_key,
            "message": fixture["boundaryMessages"][message_key],
            "pH": None,
        }

    outside = fixture["outsideCatalogCapability"]
    assert formatter.describe_outside_catalog_capability() == {
        **outside,
        "message": fixture["boundaryMessages"][outside["messageKey"]],
    }

    boundaries = formatter.list_model_boundaries()
    assert [entry["capability"] for entry in boundaries["capabilities"]] == sorted(
        fixture["capabilityBoundaryKeys"]
    )
    assert [entry["reasonCode"] for entry in boundaries["pHUnsupportedReasons"]] == sorted(
        fixture["pHUnsupportedMessageKeys"]
    )
    for phase_model in boundaries["phaseModels"]:
        assert phase_model["message"] == fixture["boundaryMessages"][phase_model["messageKey"]]
        assert phase_model["layered"] is False


def test_unknown_identifiers_and_malformed_terms_raise(formatter: ChemistryFormatter) -> None:
    with pytest.raises(ValueError, match="unknown species"):
        formatter.format_species_formula("AgCl2")
    with pytest.raises(ValueError, match="unknown ion"):
        formatter.format_ion_formula("Rb+")
    with pytest.raises(ValueError, match="unknown reagent"):
        formatter.format_reagent_formula("KBr")
    with pytest.raises(ValueError, match="unknown reaction"):
        formatter.format_molecular_equation("AgNO3|KBr")
    with pytest.raises(ValueError, match="unknown reagent"):
        formatter.find_reaction("AgNO3", "KBr")
    with pytest.raises(ValueError, match="reason code"):
        formatter.describe_unsupported_ph("PH_MODEL_UNKNOWN")
    with pytest.raises(ValueError, match="capability boundary"):
        formatter.describe_capability_boundary("viscosity")
    with pytest.raises(ValueError, match="phase model"):
        formatter.describe_phase_model("layered_organic")
    with pytest.raises(ValueError, match="message key"):
        formatter.resolve_message("model.unknown.key")

    with pytest.raises(TypeError, match="non-empty"):
        format_formula_terms([])
    with pytest.raises(ValueError, match="positive integer"):
        format_formula_terms([{"element": "Ag", "count": 0}])
    with pytest.raises(TypeError, match="element or a group"):
        format_formula_terms([{"count": 2}])
    with pytest.raises(ValueError, match="positive integer"):
        format_subscript(0)
    with pytest.raises(ValueError, match="integer"):
        format_charge_superscript(1.5)
    with pytest.raises(ValueError, match="unknown phase"):
        format_phase("plasma")
    with pytest.raises(TypeError, match="catalog must be a mapping"):
        ChemistryFormatter(None)


def thaw(value: Any) -> Any:
    """Converts the read-only binding view into mutable JSON data."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def test_formatters_follow_structural_terms_rather_than_the_ascii_formula() -> None:
    drifted = thaw(CATALOG)
    drifted["species"]["AgNO3"]["formulaTerms"] = [
        {"element": "Ag", "count": 1},
        {"element": "N", "count": 1},
    ]

    assert ChemistryFormatter(drifted).format_species_formula("AgNO3") == "AgN"
    assert FORMATTER.format_species_formula("AgNO3") == "AgNO\u2083"
