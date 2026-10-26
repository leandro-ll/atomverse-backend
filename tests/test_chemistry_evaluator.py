"""Backend experiment-evaluator tests.

The backend and the frontend evaluate supported experiments from the same
catalog records, so for one catalog and one input they must agree field for
field. These tests cover the same ground as
``tests/chemistry/chemistry-evaluator.test.js``: pair resolution through the
normalized catalog key, formula-unit amounts, balanced extent, limiting
reactants, excess, products, equations, phase, appearance, and energy, with
both A/B orders exercised for all six supported reactions, plus the
``no_modeled_reaction`` outcome every unmatched pair returns.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from app.domain.chemistry_evaluator import (
    EXPERIMENT_INPUT_REASON_CODES,
    MILLILITRES_PER_LITRE,
    NO_MODELED_REACTION_CATEGORY,
    VALIDATED_EXPERIMENT_FIELDS,
    ChemistryEvaluator,
    InvalidExperimentQuantityError,
    calculate_amount,
    evaluate_experiment,
    negative_log10,
    to_litres,
)
from app.domain.chemistry_formatters import ChemistryFormatter, normalize_pair_key
from app.generated.chemistry_catalog import CATALOG

SUPPORTED_PAIRS = (
    ("AgNO3", "NaCl"),
    ("AgNO3", "KI"),
    ("Pb(NO3)2", "NaCl"),
    ("Pb(NO3)2", "KI"),
    ("BaCl2", "Na2SO4"),
    ("HCl", "NaOH"),
)

#: Absolute tolerance used where double arithmetic makes equality brittle.
AMOUNT_TOLERANCE = 1e-12

#: Absolute tolerance for a cologarithm, which has no exact decimal form.
#:
#: The fixture states every pH to nine decimals, so this is stricter than the
#: catalog's own declared ``model.tolerances.pH`` of 1e-6 while still being
#: reachable by a hand-written golden value.
PH_TOLERANCE = 1e-8

PAIR_KEY_FIXTURE_PATH = Path("tests") / "fixtures" / "chemistry" / "reagent-pair-keys.json"
DISPLAY_FIXTURE_PATH = Path("tests") / "fixtures" / "chemistry" / "display-formatting.json"

#: Quantities used across the no-reaction cases, different per solution so the
#: A/B swap is a real swap.
UNMODELED_QUANTITIES_A = (1.2, 40)
UNMODELED_QUANTITIES_B = (0.5, 70)


@pytest.fixture(scope="module")
def evaluator() -> ChemistryEvaluator:
    return ChemistryEvaluator(CATALOG)


@pytest.fixture(scope="module")
def formatter() -> ChemistryFormatter:
    return ChemistryFormatter(CATALOG)


@pytest.fixture(scope="module")
def pair_key_fixture(repository_root: Path) -> dict[str, Any]:
    return json.loads((repository_root / PAIR_KEY_FIXTURE_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def display_fixture(repository_root: Path) -> dict[str, Any]:
    return json.loads((repository_root / DISPLAY_FIXTURE_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def unmodeled_pairs(pair_key_fixture: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """The unordered reagent pairs version 1 models no reaction for."""
    pairs = tuple(pair for pair in pair_key_fixture["pairs"] if pair["reactionId"] is None)
    assert len(pairs) == pair_key_fixture["unmodeledPairCount"]
    return pairs


def build_input(
    reagent_a: str,
    quantities_a: tuple[float, float],
    reagent_b: str,
    quantities_b: tuple[float, float],
) -> dict[str, Any]:
    """Input where each reagent carries its own quantities.

    Keeping the quantities with the reagent lets a test swap the A/B slots while
    both solutions stay chemically identical.
    """
    return {
        "reagentA": reagent_a,
        "concentrationA_molL": quantities_a[0],
        "volumeA_mL": quantities_a[1],
        "reagentB": reagent_b,
        "concentrationB_molL": quantities_b[0],
        "volumeB_mL": quantities_b[1],
        "temperature_C": 25,
        "pressure_atm": 1,
    }


def reactant_of(result: Mapping[str, Any], reagent_id: str) -> Mapping[str, Any]:
    for reactant in result["reactants"]:
        if reactant["reagentId"] == reagent_id:
            return reactant
    raise AssertionError(f"result declares no reactant for {reagent_id}")


def product_of(result: Mapping[str, Any], species_id: str) -> Mapping[str, Any]:
    for product in result["products"]:
        if product["speciesId"] == species_id:
            return product
    raise AssertionError(f"result declares no product for {species_id}")


def test_evaluator_reads_the_committed_catalog_release(evaluator: ChemistryEvaluator) -> None:
    assert evaluator.catalog_version == CATALOG["catalogVersion"]
    assert evaluator.model_version == CATALOG["model"]["modelVersion"]
    assert evaluator.amount_tolerance == CATALOG["model"]["tolerances"]["amount_mol"]


def test_amounts_are_concentration_times_volume_in_litres(evaluator: ChemistryEvaluator) -> None:
    assert MILLILITRES_PER_LITRE == 1000
    assert to_litres(50) == 50 / 1000
    assert calculate_amount(1.4, 35) == 1.4 * (35 / 1000)

    result = evaluator.evaluate_experiment(build_input("AgNO3", (1.4, 35), "NaCl", (0.3, 80)))

    silver = reactant_of(result, "AgNO3")
    assert silver["volume_L"] == 35 / 1000
    assert silver["amount_mol"] == 1.4 * (35 / 1000)
    assert reactant_of(result, "NaCl")["amount_mol"] == 0.3 * (80 / 1000)
    assert result["totalVolume_L"] == pytest.approx(0.115, abs=AMOUNT_TOLERANCE)


@pytest.mark.parametrize(
    "field",
    ["concentrationA_molL", "volumeA_mL", "concentrationB_molL", "volumeB_mL"],
)
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), "1", None])
def test_quantities_without_a_defined_amount_are_rejected(
    evaluator: ChemistryEvaluator, field: str, value: Any
) -> None:
    experiment_input = build_input("AgNO3", (1, 50), "NaCl", (1, 50))
    experiment_input[field] = value
    with pytest.raises(InvalidExperimentQuantityError):
        evaluator.evaluate_experiment(experiment_input)


@pytest.mark.parametrize(("reagent_a", "reagent_b"), SUPPORTED_PAIRS)
def test_supported_pairs_resolve_through_the_normalized_key(
    evaluator: ChemistryEvaluator, reagent_a: str, reagent_b: str
) -> None:
    pair_key = normalize_pair_key(reagent_a, reagent_b)
    reaction = CATALOG["reactions"][pair_key]
    assert evaluator.find_reaction(reagent_a, reagent_b) is reaction
    assert evaluator.find_reaction(reagent_b, reagent_a) is reaction

    result = evaluator.evaluate_experiment(build_input(reagent_a, (1, 50), reagent_b, (1, 50)))
    assert result["pairKey"] == pair_key
    assert result["reactionId"] == reaction["id"]
    assert result["category"] == reaction["category"]
    assert result["support"] == {
        "kind": "supported",
        "modelVersion": CATALOG["model"]["modelVersion"],
    }


@pytest.mark.parametrize(("reagent_a", "reagent_b"), SUPPORTED_PAIRS)
def test_both_ab_orders_produce_the_identical_result(
    evaluator: ChemistryEvaluator, reagent_a: str, reagent_b: str
) -> None:
    quantities_a = (1.7, 42)
    quantities_b = (0.6, 93)

    forward = evaluator.evaluate_experiment(
        build_input(reagent_a, quantities_a, reagent_b, quantities_b)
    )
    reversed_order = evaluator.evaluate_experiment(
        build_input(reagent_b, quantities_b, reagent_a, quantities_a)
    )
    assert reversed_order == forward


def evaluate_unmodeled(evaluator: ChemistryEvaluator, pair: Mapping[str, Any]) -> dict[str, Any]:
    reagent_a, reagent_b = pair["reagents"]
    return evaluator.evaluate_experiment(
        build_input(reagent_a, UNMODELED_QUANTITIES_A, reagent_b, UNMODELED_QUANTITIES_B)
    )


def test_every_unmodeled_pair_returns_a_consistent_described_result(
    evaluator: ChemistryEvaluator,
    unmodeled_pairs: tuple[dict[str, Any], ...],
    display_fixture: dict[str, Any],
) -> None:
    expected = display_fixture["noModeledReaction"]

    for pair in unmodeled_pairs:
        result = evaluate_unmodeled(evaluator, pair)
        pair_key = pair["pairKey"]

        assert result["pairKey"] == pair_key
        assert result["category"] == NO_MODELED_REACTION_CATEGORY == expected["category"]
        assert result["support"] == {
            "kind": "unsupported",
            "code": expected["reasonCode"],
            "messageKey": expected["messageKey"],
            "message": display_fixture["boundaryMessages"][expected["messageKey"]],
        }
        assert result["catalogVersion"] == CATALOG["catalogVersion"]
        assert result["modelVersion"] == CATALOG["model"]["modelVersion"]
        assert result["reactionId"] is None
        assert result["displayName"] is None
        assert result["pH"] is None
        assert result["speciesLedger"] is None


def test_an_unmodeled_pair_reports_no_product_and_no_equation(
    evaluator: ChemistryEvaluator,
    unmodeled_pairs: tuple[dict[str, Any], ...],
) -> None:
    for pair in unmodeled_pairs:
        result = evaluate_unmodeled(evaluator, pair)

        assert result["products"] == ()
        assert result["productSummary"] is None
        assert result["molecularEquation"] is None
        assert result["netIonicEquation"] is None
        assert result["appearance"]["precipitate"] is None
        assert result["spectatorIons"] == ()
        assert result["extent_mol"] == 0.0
        assert result["limitingSpecies"] == ()


def test_an_unmodeled_pair_leaves_both_solutions_in_the_mixture(
    evaluator: ChemistryEvaluator,
    unmodeled_pairs: tuple[dict[str, Any], ...],
) -> None:
    for pair in unmodeled_pairs:
        result = evaluate_unmodeled(evaluator, pair)

        assert result["totalVolume_L"] == pytest.approx(0.11, abs=AMOUNT_TOLERANCE)
        assert len(result["reactants"]) == 2
        for reactant in result["reactants"]:
            assert reactant["amount_mol"] == calculate_amount(
                reactant["concentration_molL"], reactant["volume_mL"]
            )
            assert reactant["consumed_mol"] == 0.0
            assert reactant["remaining_mol"] == reactant["amount_mol"]
            assert reactant["limiting"] is False
            # No balanced equation exists, so no stoichiometric coefficient may
            # be claimed for either solution.
            assert reactant["coefficient"] is None

        assert tuple(species["reagentId"] for species in result["excessSpecies"]) == tuple(
            reactant["reagentId"] for reactant in result["reactants"]
        )
        for species in result["excessSpecies"]:
            assert species["role"] == "excess"
            assert species["amount_mol"] == reactant_of(result, species["reagentId"])["amount_mol"]


def test_an_unmodeled_pair_stays_in_one_homogeneous_aqueous_phase(
    evaluator: ChemistryEvaluator,
    formatter: ChemistryFormatter,
    unmodeled_pairs: tuple[dict[str, Any], ...],
    display_fixture: dict[str, Any],
) -> None:
    expected = display_fixture["noModeledReaction"]
    presentation = CATALOG["model"]["unsupportedOutcomes"]["noModeledReaction"]["presentation"]

    for pair in unmodeled_pairs:
        result = evaluate_unmodeled(evaluator, pair)

        assert result["phaseModel"] == formatter.describe_phase_model(expected["phaseModel"])
        assert result["phaseModel"]["phaseModel"] == expected["phaseModel"]
        assert result["phaseModel"]["phases"] == ("aq",)
        assert result["phaseModel"]["layered"] is expected["layered"]
        assert (
            result["phaseModel"]["message"]
            == display_fixture["boundaryMessages"]["model.phase.homogeneousAqueous"]
        )

        assert result["appearance"] == {
            "clarity": presentation["appearance"],
            "color": presentation["solutionColor"],
            "precipitate": None,
        }
        assert result["appearance"]["clarity"] == "clear"
        assert result["energy"]["label"] == presentation["energyLabel"]
        assert result["energy"]["exothermic"] is False


def test_both_ab_orders_of_an_unmodeled_pair_agree(
    evaluator: ChemistryEvaluator,
    display_fixture: dict[str, Any],
) -> None:
    for pair in display_fixture["unmodeledPairs"]:
        reagent_a, reagent_b = pair["reagents"]
        forward = evaluator.evaluate_experiment(
            build_input(reagent_a, UNMODELED_QUANTITIES_A, reagent_b, UNMODELED_QUANTITIES_B)
        )
        reversed_order = evaluator.evaluate_experiment(
            build_input(reagent_b, UNMODELED_QUANTITIES_B, reagent_a, UNMODELED_QUANTITIES_A)
        )
        assert reversed_order == forward
        assert forward["pairKey"] == pair["pairKey"]


def test_no_mixture_can_be_represented_as_layered(
    evaluator: ChemistryEvaluator,
    pair_key_fixture: dict[str, Any],
) -> None:
    # A layer boundary would need an immiscibility model the catalog does not
    # declare, so no result of either kind may report a layered phase.
    assert tuple(CATALOG["model"]["immisciblePairs"]) == ()
    for phase_model in CATALOG["model"]["phaseModels"].values():
        assert phase_model["layered"] is False

    for pair in pair_key_fixture["pairs"]:
        reagent_a, reagent_b = pair["reagents"]
        result = evaluator.evaluate_experiment(
            build_input(reagent_a, UNMODELED_QUANTITIES_A, reagent_b, UNMODELED_QUANTITIES_B)
        )
        assert result["phaseModel"]["layered"] is False
        assert len(result["phaseModel"]["phases"]) <= 2
        assert "aq" in result["phaseModel"]["phases"]


def test_a_layered_no_reaction_phase_model_is_refused() -> None:
    model = CATALOG["model"]
    mutated = {
        **CATALOG,
        "model": {
            **model,
            "phaseModels": {
                **model["phaseModels"],
                "layered_pair": {
                    "phases": ("aq", "l"),
                    "layered": True,
                    "messageKey": "model.phase.homogeneousAqueous",
                },
            },
            "unsupportedOutcomes": {
                **model["unsupportedOutcomes"],
                "noModeledReaction": {
                    **model["unsupportedOutcomes"]["noModeledReaction"],
                    "phaseModel": "layered_pair",
                },
            },
        },
    }

    with pytest.raises(ValueError, match="one homogeneous phase"):
        evaluate_experiment(
            mutated, build_input("NaCl", UNMODELED_QUANTITIES_A, "KI", UNMODELED_QUANTITIES_B)
        )


def test_unknown_reagents_and_self_mixtures_are_rejected(evaluator: ChemistryEvaluator) -> None:
    with pytest.raises(ValueError, match="unknown reagent"):
        evaluator.evaluate_experiment(build_input("H2SO4", (1, 50), "NaCl", (1, 50)))
    with pytest.raises(ValueError, match="two different reagents"):
        evaluator.evaluate_experiment(build_input("NaCl", (1, 50), "NaCl", (1, 50)))


def test_one_to_one_precipitation_is_limited_by_the_scarcer_reagent(
    evaluator: ChemistryEvaluator,
) -> None:
    result = evaluator.evaluate_experiment(build_input("AgNO3", (1, 50), "NaCl", (0.4, 50)))

    assert reactant_of(result, "AgNO3")["amount_mol"] == pytest.approx(0.05, abs=AMOUNT_TOLERANCE)
    assert reactant_of(result, "NaCl")["amount_mol"] == pytest.approx(0.02, abs=AMOUNT_TOLERANCE)
    assert result["extent_mol"] == pytest.approx(0.02, abs=AMOUNT_TOLERANCE)
    assert result["limitingSpecies"] == ("NaCl",)

    assert reactant_of(result, "NaCl")["remaining_mol"] == 0.0
    assert reactant_of(result, "AgNO3")["remaining_mol"] == pytest.approx(
        0.03, abs=AMOUNT_TOLERANCE
    )
    assert len(result["excessSpecies"]) == 1
    excess = result["excessSpecies"][0]
    assert excess["speciesId"] == "AgNO3"
    assert excess["role"] == "excess"
    assert excess["amount_mol"] == pytest.approx(0.03, abs=AMOUNT_TOLERANCE)

    assert product_of(result, "AgCl")["amount_mol"] == pytest.approx(0.02, abs=AMOUNT_TOLERANCE)
    assert product_of(result, "NaNO3")["amount_mol"] == pytest.approx(0.02, abs=AMOUNT_TOLERANCE)


@pytest.mark.parametrize(
    ("halide", "precipitate"),
    [("NaCl", "PbCl2"), ("KI", "PbI2")],
)
def test_lead_halide_stoichiometry_divides_by_the_halide_coefficient(
    evaluator: ChemistryEvaluator, halide: str, precipitate: str
) -> None:
    result = evaluator.evaluate_experiment(build_input("Pb(NO3)2", (1, 50), halide, (1, 50)))

    assert reactant_of(result, "Pb(NO3)2")["coefficient"] == 1
    assert reactant_of(result, halide)["coefficient"] == 2

    # Equal amounts are not equal equivalents: two halide formula units are
    # consumed per lead formula unit, so the halide runs out first.
    assert result["extent_mol"] == pytest.approx(0.025, abs=AMOUNT_TOLERANCE)
    assert result["limitingSpecies"] == (halide,)
    assert reactant_of(result, halide)["consumed_mol"] == pytest.approx(0.05, abs=AMOUNT_TOLERANCE)
    assert result["excessSpecies"][0]["speciesId"] == "Pb(NO3)2"
    assert result["excessSpecies"][0]["amount_mol"] == pytest.approx(0.025, abs=AMOUNT_TOLERANCE)
    assert product_of(result, precipitate)["amount_mol"] == pytest.approx(
        0.025, abs=AMOUNT_TOLERANCE
    )


def test_equivalence_names_both_reactants_limiting_without_excess(
    evaluator: ChemistryEvaluator,
) -> None:
    result = evaluator.evaluate_experiment(build_input("HCl", (0.8, 60), "NaOH", (0.8, 60)))

    assert result["extent_mol"] == pytest.approx(0.048, abs=AMOUNT_TOLERANCE)
    assert sorted(result["limitingSpecies"]) == ["HCl", "NaOH"]
    assert result["excessSpecies"] == ()
    for reactant in result["reactants"]:
        assert reactant["remaining_mol"] == 0.0
        assert reactant["limiting"] is True
    assert product_of(result, "H2O")["amount_mol"] == pytest.approx(0.048, abs=AMOUNT_TOLERANCE)
    assert product_of(result, "NaCl")["amount_mol"] == pytest.approx(0.048, abs=AMOUNT_TOLERANCE)


@pytest.mark.parametrize(("reagent_a", "reagent_b"), SUPPORTED_PAIRS)
def test_consumption_never_exceeds_the_available_amount(
    evaluator: ChemistryEvaluator, reagent_a: str, reagent_b: str
) -> None:
    result = evaluator.evaluate_experiment(build_input(reagent_a, (1.9, 97), reagent_b, (0.2, 11)))

    assert result["extent_mol"] > 0
    assert len(result["limitingSpecies"]) >= 1
    for reactant in result["reactants"]:
        assert reactant["consumed_mol"] <= reactant["amount_mol"] + evaluator.amount_tolerance
        assert reactant["remaining_mol"] >= 0
        assert math.isclose(
            reactant["consumed_mol"] + reactant["remaining_mol"],
            reactant["amount_mol"],
            abs_tol=AMOUNT_TOLERANCE,
        )
    for product in result["products"]:
        assert math.isclose(
            product["amount_mol"],
            result["extent_mol"] * product["coefficient"],
            abs_tol=AMOUNT_TOLERANCE,
        )


@pytest.mark.parametrize(("reagent_a", "reagent_b"), SUPPORTED_PAIRS)
def test_projections_come_from_the_one_resolved_reaction_record(
    evaluator: ChemistryEvaluator, formatter: ChemistryFormatter, reagent_a: str, reagent_b: str
) -> None:
    result = evaluator.evaluate_experiment(build_input(reagent_a, (1, 50), reagent_b, (1, 50)))
    reaction = CATALOG["reactions"][result["pairKey"]]

    assert result["molecularEquation"] == formatter.format_molecular_equation(reaction)
    assert result["netIonicEquation"] == formatter.format_net_ionic_equation(reaction)
    assert result["productSummary"] == formatter.format_product_summary(reaction)
    assert result["spectatorIons"] == tuple(reaction["spectatorIons"])
    assert result["displayName"] == reaction["displayName"]

    assert result["phaseModel"] == formatter.describe_phase_model(reaction["phaseModel"])
    assert result["phaseModel"]["layered"] is False

    assert result["appearance"]["clarity"] == reaction["presentation"]["appearance"]
    assert result["appearance"]["color"] == reaction["presentation"]["solutionColor"]
    assert result["energy"]["label"] == reaction["presentation"]["energyLabel"]
    assert result["energy"]["quantitativeEnthalpy"] is False
    assert result["energy"]["message"] == formatter.resolve_message(
        CATALOG["model"]["capabilities"]["energy"]["messageKey"]
    )

    # The one modeled acid-base system reports a calculated pH; a salt
    # solution reports none until task 5.4 attaches its reason code.
    if reaction["pH"] is None:
        assert result["pH"] is None
        assert reaction["pHUnsupportedReasonCode"]
    else:
        assert result["pH"]["supported"] is True
        assert result["pH"]["modelId"] == reaction["pH"]["modelId"]

    # Task 6.1 owns the ion-level ledger.
    assert result["speciesLedger"] is None


def test_the_precipitate_agrees_with_the_equation_that_produced_it(
    evaluator: ChemistryEvaluator,
) -> None:
    precipitation = evaluator.evaluate_experiment(build_input("BaCl2", (1, 50), "Na2SO4", (1, 50)))
    assert precipitation["category"] == "precipitation"
    assert precipitation["appearance"]["precipitate"] == {
        "speciesId": "BaSO4",
        "formula": "BaSO\u2084",
        "displayName": "Barium Sulfat",
        "color": "#ffffff",
    }
    assert precipitation["appearance"]["clarity"] == "cloudy"
    assert precipitation["phaseModel"]["phases"] == ("aq", "s")
    assert product_of(precipitation, "BaSO4")["phase"] == "s"
    assert "BaSO\u2084(s)" in precipitation["molecularEquation"]
    assert "BaSO\u2084(s)" in precipitation["netIonicEquation"]

    neutralization = evaluator.evaluate_experiment(build_input("HCl", (1, 50), "NaOH", (1, 50)))
    assert neutralization["category"] == "strong_acid_base"
    assert neutralization["appearance"]["precipitate"] is None
    assert neutralization["appearance"]["clarity"] == "clear"
    assert neutralization["phaseModel"]["phases"] == ("aq",)
    assert neutralization["energy"]["exothermic"] is True


def test_repeated_evaluations_are_equal_and_read_the_supplied_catalog(
    evaluator: ChemistryEvaluator,
) -> None:
    experiment_input = build_input("AgNO3", (1.3, 27), "KI", (0.9, 64))
    assert evaluator.evaluate_experiment(experiment_input) == evaluator.evaluate_experiment(
        experiment_input
    )

    # A mutated coefficient must change the extent, proving the evaluator reads
    # the catalog rather than a hardcoded stoichiometry table.
    mutated = {
        **CATALOG,
        "reactions": {
            **CATALOG["reactions"],
            "AgNO3|NaCl": {
                **CATALOG["reactions"]["AgNO3|NaCl"],
                "molecularReactants": (
                    {"speciesId": "AgNO3", "coefficient": 1, "phase": "aq"},
                    {"speciesId": "NaCl", "coefficient": 2, "phase": "aq"},
                ),
            },
        },
    }
    silver_chloride_input = build_input("AgNO3", (1, 50), "NaCl", (1, 50))
    assert evaluate_experiment(mutated, silver_chloride_input)["extent_mol"] == pytest.approx(
        0.025, abs=AMOUNT_TOLERANCE
    )
    assert evaluator.evaluate_experiment(silver_chloride_input)["extent_mol"] == pytest.approx(
        0.05, abs=AMOUNT_TOLERANCE
    )


# --- golden evaluation vectors ----------------------------------------------
#
# ``tests/fixtures/chemistry/experiment-vectors.json`` states, by hand, the
# whole expected outcome of a concrete experiment: the corrected Unicode
# formulas and balanced equations, the category, the limiting reactant, the
# extent, what was consumed and what is left over, the product amounts, the
# spectator ions, the phase model, the appearance, and the energy label. The
# fixture is written independently of both the catalog and this evaluator, so a
# wrong coefficient, a dropped subscript, a reversed limiting reactant, or a
# reintroduced layer boundary cannot satisfy both the fixture and the code.
#
# The frontend suite ``tests/chemistry/chemistry-evaluator.test.js`` reads the
# same file, which is what makes it a parity contract rather than one language's
# snapshot. These vectors are the regression guard for the audited display
# defects: ``AgNO``, ``AgNO(aq)``, and ``Pb(NO)₂`` cannot appear in any
# learner-facing string, a precipitate cannot coexist with no-reaction text, and
# no mixture of any kind can be rendered as two layers.
#
# Validates: Requirements 2.3, 2.4, 2.5, 26.3

VECTOR_FIXTURE_PATH = Path("tests") / "fixtures" / "chemistry" / "experiment-vectors.json"


@pytest.fixture(scope="module")
def vector_fixture(repository_root: Path) -> dict[str, Any]:
    return json.loads((repository_root / VECTOR_FIXTURE_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def vectors(vector_fixture: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(vector_fixture["vectors"])


def vector_input(vector: Mapping[str, Any], *, reverse: bool = False) -> dict[str, Any]:
    """Experiment input for one vector, optionally with the A/B slots swapped."""
    solutions = list(vector["solutions"])
    if reverse:
        solutions.reverse()
    first, second = solutions
    return build_input(
        first["reagentId"],
        (first["concentration_molL"], first["volume_mL"]),
        second["reagentId"],
        (second["concentration_molL"], second["volume_mL"]),
    )


def evaluate_vector(
    evaluator: ChemistryEvaluator, vector: Mapping[str, Any], *, reverse: bool = False
) -> dict[str, Any]:
    return evaluator.evaluate_experiment(vector_input(vector, reverse=reverse))


def read_path(value: Any, path: str) -> Any:
    """Reads a dotted path, treating a missing branch as an absent value."""
    for key in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def display_strings(
    result: Mapping[str, Any], vector_fixture: Mapping[str, Any]
) -> tuple[str, ...]:
    """Every learner-facing string one result exposes."""
    fields = vector_fixture["regressionGuards"]["displayStringFields"]
    values = (read_path(result, path) for path in fields)
    return tuple(value for value in values if isinstance(value, str))


def test_vector_fixture_is_pinned_to_the_committed_catalog_release(
    vector_fixture: dict[str, Any],
) -> None:
    assert vector_fixture["catalogVersion"] == CATALOG["catalogVersion"]
    assert vector_fixture["modelVersion"] == CATALOG["model"]["modelVersion"]
    assert vector_fixture["locale"] == CATALOG["locale"]


def test_vectors_cover_every_supported_pair_and_no_reaction_pairs(
    vectors: tuple[dict[str, Any], ...],
) -> None:
    supported = [vector for vector in vectors if vector["support"]["kind"] == "supported"]
    unsupported = [vector for vector in vectors if vector["support"]["kind"] == "unsupported"]

    assert sorted({vector["pairKey"] for vector in supported}) == sorted(CATALOG["reactions"])

    # A pair whose two reagents happen to be at equivalence hides the limiting
    # reagent, so every supported pair also needs a vector where they are not.
    for pair_key in {vector["pairKey"] for vector in supported}:
        non_equivalent = [
            vector
            for vector in supported
            if vector["pairKey"] == pair_key and not vector["stoichiometricEquivalence"]
        ]
        assert non_equivalent, pair_key

    assert len(unsupported) >= 4
    for vector in unsupported:
        assert vector["pairKey"] not in CATALOG["reactions"]

    assert len({vector["id"] for vector in vectors}) == len(vectors)


def test_every_vector_quantity_is_inside_the_catalog_control_bounds(
    vectors: tuple[dict[str, Any], ...],
) -> None:
    concentration = CATALOG["model"]["bounds"]["concentration_molL"]
    volume = CATALOG["model"]["bounds"]["volume_mL"]

    for vector in vectors:
        assert len(vector["solutions"]) == 2, vector["id"]
        for solution in vector["solutions"]:
            assert solution["reagentId"] in CATALOG["reagents"]
            assert (
                concentration["minimum"]
                <= solution["concentration_molL"]
                <= concentration["maximum"]
            )
            assert volume["minimum"] <= solution["volume_mL"] <= volume["maximum"]

        reagent_ids = [solution["reagentId"] for solution in vector["solutions"]]
        # Canonical reagent order, the same order the result reports.
        assert reagent_ids == sorted(vector["reagents"])
        assert "|".join(reagent_ids) == vector["pairKey"]


def test_vectors_reproduce_the_stated_identity_and_support_status(
    evaluator: ChemistryEvaluator,
    formatter: ChemistryFormatter,
    vectors: tuple[dict[str, Any], ...],
    vector_fixture: dict[str, Any],
) -> None:
    for vector in vectors:
        result = evaluate_vector(evaluator, vector)

        assert result["catalogVersion"] == vector_fixture["catalogVersion"]
        assert result["modelVersion"] == vector_fixture["modelVersion"]
        assert result["pairKey"] == vector["pairKey"]
        assert result["reactionId"] == vector["reactionId"]
        assert result["category"] == vector["category"]
        assert result["displayName"] == vector["displayName"]
        assert result["support"]["kind"] == vector["support"]["kind"]

        if vector["support"]["kind"] == "supported":
            assert result["support"] == {
                "kind": "supported",
                "modelVersion": vector_fixture["modelVersion"],
            }
        else:
            assert result["support"]["code"] == vector["support"]["code"]
            assert result["support"]["messageKey"] == vector["support"]["messageKey"]
            assert result["support"]["message"] == formatter.resolve_message(
                vector["support"]["messageKey"]
            )


def test_vectors_reproduce_the_stated_amounts_extent_limiting_and_excess(
    evaluator: ChemistryEvaluator, vectors: tuple[dict[str, Any], ...]
) -> None:
    for vector in vectors:
        result = evaluate_vector(evaluator, vector)

        assert result["totalVolume_L"] == pytest.approx(
            vector["totalVolume_L"], abs=AMOUNT_TOLERANCE
        )
        assert result["extent_mol"] == pytest.approx(vector["extent_mol"], abs=AMOUNT_TOLERANCE)
        assert list(result["limitingSpecies"]) == vector["limitingSpecies"]

        for solution in vector["solutions"]:
            reactant = reactant_of(result, solution["reagentId"])
            assert reactant["concentration_molL"] == solution["concentration_molL"]
            assert reactant["volume_mL"] == solution["volume_mL"]
            assert reactant["coefficient"] == solution["coefficient"]
            assert reactant["amount_mol"] == pytest.approx(
                solution["amount_mol"], abs=AMOUNT_TOLERANCE
            )
            assert reactant["consumed_mol"] == pytest.approx(
                solution["consumed_mol"], abs=AMOUNT_TOLERANCE
            )
            assert reactant["remaining_mol"] == pytest.approx(
                solution["remaining_mol"], abs=AMOUNT_TOLERANCE
            )
            assert reactant["limiting"] is solution["limiting"]

        assert [species["speciesId"] for species in result["excessSpecies"]] == [
            species["speciesId"] for species in vector["excessSpecies"]
        ]
        for actual, expected in zip(result["excessSpecies"], vector["excessSpecies"], strict=True):
            assert actual["role"] == "excess"
            assert actual["amount_mol"] == pytest.approx(
                expected["amount_mol"], abs=AMOUNT_TOLERANCE
            )


def test_vectors_reproduce_the_stated_products_and_spectator_ions(
    evaluator: ChemistryEvaluator, vectors: tuple[dict[str, Any], ...]
) -> None:
    for vector in vectors:
        result = evaluate_vector(evaluator, vector)

        assert [product["speciesId"] for product in result["products"]] == [
            product["speciesId"] for product in vector["products"]
        ]
        for actual, expected in zip(result["products"], vector["products"], strict=True):
            assert actual["phase"] == expected["phase"]
            assert actual["coefficient"] == expected["coefficient"]
            assert actual["role"] == "product"
            assert actual["amount_mol"] == pytest.approx(
                expected["amount_mol"], abs=AMOUNT_TOLERANCE
            )

        assert list(result["spectatorIons"]) == vector["spectatorIons"]


def test_vectors_reproduce_the_stated_equations_phase_appearance_and_energy(
    evaluator: ChemistryEvaluator, vectors: tuple[dict[str, Any], ...]
) -> None:
    for vector in vectors:
        result = evaluate_vector(evaluator, vector)

        assert result["molecularEquation"] == vector["molecularEquation"]
        assert result["netIonicEquation"] == vector["netIonicEquation"]
        assert result["productSummary"] == vector["productSummary"]

        assert result["phaseModel"]["phaseModel"] == vector["phaseModel"]["name"]
        assert list(result["phaseModel"]["phases"]) == vector["phaseModel"]["phases"]
        assert result["phaseModel"]["layered"] is vector["phaseModel"]["layered"]

        appearance = vector["appearance"]
        assert result["appearance"]["clarity"] == appearance["clarity"]
        if appearance["precipitateSpeciesId"] is None:
            assert result["appearance"]["precipitate"] is None
        else:
            precipitate = result["appearance"]["precipitate"]
            assert precipitate["speciesId"] == appearance["precipitateSpeciesId"]
            assert precipitate["formula"] == appearance["precipitateFormula"]
            assert precipitate["displayName"] == appearance["precipitateDisplayName"]

        assert result["energy"]["label"] == vector["energy"]["label"]
        assert result["energy"]["exothermic"] is vector["energy"]["exothermic"]

        # Task 6.1 owns the ion-level ledger; the vectors record that it is
        # still an honest null rather than a fabricated particle count.
        assert result["speciesLedger"] is None


def test_every_vector_reproduces_its_stated_ph_or_its_stated_absence(
    evaluator: ChemistryEvaluator,
    formatter: ChemistryFormatter,
    vectors: tuple[dict[str, Any], ...],
) -> None:
    for vector in vectors:
        result = evaluate_vector(evaluator, vector)
        expected = vector["pH"]

        if expected is None:
            # Task 5.4 owns the reason code and the localized boundary text for
            # these; nothing numeric may stand in for them meanwhile.
            assert result["pH"] is None, vector["id"]
            assert not re.search(r"~\s*7", json.dumps(result, ensure_ascii=False)), vector["id"]
            continue

        ph = result["pH"]
        assert ph["supported"] is True, vector["id"]
        assert ph["equivalence"] is expected["equivalence"], vector["id"]
        assert ph["side"] == expected["side"], vector["id"]
        assert ph["excessIonId"] == expected["excessIonId"], vector["id"]
        assert math.isclose(
            ph["excessAmount_mol"], expected["excessAmount_mol"], abs_tol=AMOUNT_TOLERANCE
        ), vector["id"]
        assert math.isclose(
            ph["excessConcentration_molL"],
            expected["excessConcentration_molL"],
            abs_tol=PH_TOLERANCE,
        ), vector["id"]
        assert math.isclose(ph["value"], expected["value"], abs_tol=PH_TOLERANCE), vector["id"]
        assert formatter.format_quantity("pH", ph["value"]) == expected["display"], vector["id"]


def test_both_ab_orders_of_every_vector_agree(
    evaluator: ChemistryEvaluator, vectors: tuple[dict[str, Any], ...]
) -> None:
    for vector in vectors:
        forward = evaluate_vector(evaluator, vector)
        reversed_order = evaluate_vector(evaluator, vector, reverse=True)
        assert reversed_order == forward, vector["id"]


def test_the_audited_malformed_formulas_cannot_reach_a_display_string(
    evaluator: ChemistryEvaluator,
    vectors: tuple[dict[str, Any], ...],
    vector_fixture: dict[str, Any],
) -> None:
    patterns = vector_fixture["regressionGuards"]["malformedFormulaPatterns"]
    assert len(patterns) >= 4
    compiled = [(entry["label"], re.compile(entry["pattern"])) for entry in patterns]

    # The patterns must actually be able to fire, or the guard proves nothing.
    audited = (
        "AgNO(aq) + NaCl(aq)",
        "AgNO + NaCl",
        "Pb(NO)\u2082(aq) + 2KI(aq)",
        "Pb(NO)2 + 2KI",
        "AgNO3(aq)",
    )
    for sample in audited:
        assert any(pattern.search(sample) for _, pattern in compiled), sample

    for vector in vectors:
        # Both the evaluated result and the hand-written expectation are
        # scanned, so a malformed formula cannot be introduced by editing the
        # fixture.
        declared = (
            vector["molecularEquation"],
            vector["netIonicEquation"],
            vector["productSummary"],
            vector["appearance"]["precipitateFormula"],
            *vector["reagentFormulas"],
        )
        strings = (
            *display_strings(evaluate_vector(evaluator, vector), vector_fixture),
            *(value for value in declared if isinstance(value, str)),
        )
        assert strings, vector["id"]
        for text in strings:
            for label, pattern in compiled:
                assert not pattern.search(text), f"{vector['id']} / {label} / {text}"


def test_vectors_use_the_corrected_unicode_reagent_formulas(
    formatter: ChemistryFormatter, vectors: tuple[dict[str, Any], ...]
) -> None:
    for vector in vectors:
        for solution, formula in zip(vector["solutions"], vector["reagentFormulas"], strict=True):
            assert formatter.format_reagent_formula(solution["reagentId"]) == formula
        if vector["molecularEquation"] is not None:
            for formula in vector["reagentFormulas"]:
                assert formula in vector["molecularEquation"]


def test_a_precipitate_never_appears_beside_no_reaction_text(
    evaluator: ChemistryEvaluator,
    vectors: tuple[dict[str, Any], ...],
    vector_fixture: dict[str, Any],
) -> None:
    guards = vector_fixture["regressionGuards"]

    for vector in vectors:
        result = evaluate_vector(evaluator, vector)

        for text in display_strings(result, vector_fixture):
            assert guards["contradictoryNoReactionText"] not in text

        precipitate = result["appearance"]["precipitate"]
        if precipitate is not None:
            assert result["support"]["kind"] == "supported"
            assert "messageKey" not in result["support"]
            assert result["category"] != NO_MODELED_REACTION_CATEGORY
            assert precipitate["formula"] in result["molecularEquation"]
            assert any(
                product["speciesId"] == precipitate["speciesId"] for product in result["products"]
            )
            assert result["extent_mol"] > 0

        if result["category"] == NO_MODELED_REACTION_CATEGORY:
            assert result["support"]["messageKey"] == guards["noModeledReactionMessageKey"]
            assert result["appearance"]["precipitate"] is None
            assert result["products"] == ()
            assert result["molecularEquation"] is None
            assert result["netIonicEquation"] is None


def test_every_vector_stays_in_one_unlayered_mixture(
    evaluator: ChemistryEvaluator,
    vectors: tuple[dict[str, Any], ...],
    vector_fixture: dict[str, Any],
) -> None:
    for vector in vectors:
        result = evaluate_vector(evaluator, vector)

        assert result["phaseModel"]["layered"] is False
        assert "aq" in result["phaseModel"]["phases"]
        # The only second phase version 1 declares is a solid precipitate, which
        # is not a liquid layer; an ordinary mixture has no second phase at all.
        extra_phases = [phase for phase in result["phaseModel"]["phases"] if phase != "aq"]
        assert extra_phases == ([] if result["appearance"]["precipitate"] is None else ["s"])

        for text in display_strings(result, vector_fixture):
            for layer_label in vector_fixture["regressionGuards"]["forbiddenLayerLabels"]:
                assert layer_label not in text


def test_equal_lead_halide_amounts_are_not_equal_equivalents(
    evaluator: ChemistryEvaluator, vectors: tuple[dict[str, Any], ...]
) -> None:
    lead_vectors = [
        vector
        for vector in vectors
        if vector["support"]["kind"] == "supported" and "Pb(NO3)2" in vector["reagents"]
    ]
    assert len(lead_vectors) >= 4

    equal_amount_vectors = [
        vector
        for vector in lead_vectors
        if vector["solutions"][0]["concentration_molL"]
        == vector["solutions"][1]["concentration_molL"]
        and vector["solutions"][0]["volume_mL"] == vector["solutions"][1]["volume_mL"]
    ]
    assert len(equal_amount_vectors) == 2

    for vector in equal_amount_vectors:
        result = evaluate_vector(evaluator, vector)
        halide = next(
            solution for solution in vector["solutions"] if solution["reagentId"] != "Pb(NO3)2"
        )

        # The halide carries coefficient two, so it is limiting and half the
        # lead nitrate survives even though both amounts started equal.
        assert reactant_of(result, halide["reagentId"])["coefficient"] == 2
        assert reactant_of(result, "Pb(NO3)2")["coefficient"] == 1
        assert result["limitingSpecies"] == (halide["reagentId"],)
        assert result["extent_mol"] == pytest.approx(halide["amount_mol"] / 2, abs=AMOUNT_TOLERANCE)
        assert tuple(species["speciesId"] for species in result["excessSpecies"]) == ("Pb(NO3)2",)
        assert result["excessSpecies"][0]["amount_mol"] == pytest.approx(
            halide["amount_mol"] / 2, abs=AMOUNT_TOLERANCE
        )
        assert vector["stoichiometricEquivalence"] is False


def test_equivalence_vectors_report_no_excess_and_both_reactants_limiting(
    evaluator: ChemistryEvaluator, vectors: tuple[dict[str, Any], ...]
) -> None:
    equivalence_vectors = [
        vector
        for vector in vectors
        if vector["support"]["kind"] == "supported" and vector["stoichiometricEquivalence"]
    ]
    assert len(equivalence_vectors) >= 3

    for vector in equivalence_vectors:
        result = evaluate_vector(evaluator, vector)
        assert result["excessSpecies"] == ()
        assert list(result["limitingSpecies"]) == [
            solution["reagentId"] for solution in vector["solutions"]
        ]
        for reactant in result["reactants"]:
            assert reactant["remaining_mol"] == 0.0
            assert reactant["limiting"] is True


def test_the_four_canonical_demonstrations_are_preserved(
    evaluator: ChemistryEvaluator,
    vectors: tuple[dict[str, Any], ...],
    vector_fixture: dict[str, Any],
) -> None:
    """Requirement 26.3 and bugfix regression clause 3.1.

    The demonstrations the lab shipped with still reach their intended
    precipitation or neutralization category, now with the corrected
    quantitative detail the rest of this suite pins down.
    """
    preserved = vector_fixture["preservedDemonstrations"]
    assert preserved["requirement"] == "26.3"
    assert preserved["regressionClause"] == "3.1"
    assert len(preserved["vectors"]) == 4
    assert sorted("|".join(sorted(entry["reagents"])) for entry in preserved["vectors"]) == [
        "AgNO3|NaCl",
        "BaCl2|Na2SO4",
        "HCl|NaOH",
        "KI|Pb(NO3)2",
    ]
    assert len([vector for vector in vectors if vector["preservedDemonstration"]]) == 4

    by_id = {vector["id"]: vector for vector in vectors}
    for entry in preserved["vectors"]:
        vector = by_id[entry["vectorId"]]
        assert vector["preservedDemonstration"] is True

        result = evaluate_vector(evaluator, vector)
        assert result["support"]["kind"] == "supported"
        assert result["category"] == entry["category"]
        assert result["energy"]["label"] == entry["energyLabel"]
        assert result["extent_mol"] > 0
        assert len(result["products"]) > 0

        if entry["precipitateSpeciesId"] is None:
            assert result["category"] == "strong_acid_base"
            assert result["appearance"]["precipitate"] is None
            assert result["energy"]["exothermic"] is True
        else:
            assert result["category"] == "precipitation"
            assert result["appearance"]["precipitate"]["speciesId"] == entry["precipitateSpeciesId"]
            assert product_of(result, entry["precipitateSpeciesId"])["phase"] == "s"


# -- validated experiment input semantics -----------------------------------
#
# The audited lab presented four controls as reaction inputs while reading none
# of them for chemistry: molarity and volume only changed a capped particle
# count, temperature only changed animation speed, and pressure had no handler
# at all (defect 1.5). These tests are the backend half of the corrected
# contract stated by hand in ``experiment-vectors.json``'s
# ``controlCapabilities`` block, and they mirror the frontend suite field for
# field:
#
# - concentration and volume feed every quantitative path, at every declared
#   bound, including which reactant ends up limiting,
# - temperature produces bounded qualitative motion metadata and nothing else,
# - pressure is fixed at 1 atm, influences no calculated value, and is reported
#   with its localized boundary statement instead of being ignored,
# - no pH value is fabricated at any control position.
#
# Validates: Requirements 5.1, 5.2, 5.3


@pytest.fixture(scope="module")
def control_capabilities(vector_fixture: dict[str, Any]) -> dict[str, Any]:
    return vector_fixture["controlCapabilities"]


@pytest.fixture(scope="module")
def quantity_boundaries(control_capabilities: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(control_capabilities["quantityBoundaries"])


def case_input(
    test_case: Mapping[str, Any],
    *,
    temperature_c: float | None = 25,
    omit_pressure: bool = False,
) -> dict[str, Any]:
    """Experiment input for one control case at an explicit temperature."""
    first, second = test_case["solutions"]
    experiment_input = build_input(
        first["reagentId"],
        (first["concentration_molL"], first["volume_mL"]),
        second["reagentId"],
        (second["concentration_molL"], second["volume_mL"]),
    )
    experiment_input["temperature_C"] = temperature_c
    if omit_pressure:
        del experiment_input["pressure_atm"]
    return experiment_input


def catalog_with_capability_override(name: str, **overrides: Any) -> dict[str, Any]:
    """Catalog copy with one model capability field replaced.

    The generated binding is immutable all the way down, so a mutated catalog
    is assembled by overriding only the branch under test.
    """
    capabilities = CATALOG["model"]["capabilities"]
    return {
        **CATALOG,
        "model": {
            **CATALOG["model"],
            "capabilities": {**capabilities, name: {**capabilities[name], **overrides}},
        },
    }


def catalog_with_motion_override(**overrides: Any) -> dict[str, Any]:
    """Catalog copy with one declared motion-speed field replaced."""
    motion = CATALOG["model"]["capabilities"]["temperature"]["motionSpeedFactor"]
    return catalog_with_capability_override(
        "temperature", motionSpeedFactor={**motion, **overrides}
    )


def catalog_with_bounds_override(name: str, **overrides: Any) -> dict[str, Any]:
    """Catalog copy with one model bound replaced."""
    bounds = CATALOG["model"]["bounds"]
    return {
        **CATALOG,
        "model": {
            **CATALOG["model"],
            "bounds": {**bounds, name: {**bounds[name], **overrides}},
        },
    }


def without_motion(result: Mapping[str, Any]) -> dict[str, Any]:
    """The result without its motion metadata, temperature's only output."""
    return {key: value for key, value in result.items() if key != "motion"}


def amount_of(entries: Any, species_id: str) -> float:
    for entry in entries:
        if entry["speciesId"] == species_id:
            return entry["amount_mol"]
    raise AssertionError(f"result declares no entry for {species_id}")


def test_control_capabilities_are_pinned_to_the_declared_catalog_model(
    control_capabilities: dict[str, Any],
) -> None:
    bounds = CATALOG["model"]["bounds"]
    temperature = CATALOG["model"]["capabilities"]["temperature"]
    pressure = CATALOG["model"]["capabilities"]["pressure"]
    declared = control_capabilities["temperature"]

    assert control_capabilities["quantitativeInputs"] == ["concentration_molL", "volume_mL"]
    assert control_capabilities["qualitativeInputs"] == ["temperature_C"]
    assert control_capabilities["unsupportedInputs"] == ["pressure_atm"]

    assert declared["effect"] == temperature["effect"]
    assert declared["model"] == temperature["motionSpeedFactor"]["model"]
    assert declared["minimumSpeedFactor"] == temperature["motionSpeedFactor"]["atMinimum"]
    assert declared["maximumSpeedFactor"] == temperature["motionSpeedFactor"]["atMaximum"]
    assert (
        declared["referenceTemperature_C"]
        == temperature["motionSpeedFactor"]["referenceTemperature_C"]
    )
    assert declared["messageKey"] == temperature["messageKey"]
    for claim in ("affectsExtent", "affectsRateConstant", "affectsPH"):
        assert declared[claim] is False
        assert temperature[claim] is False

    assert control_capabilities["pressure"]["atm"] == pressure["fixed_atm"]
    assert control_capabilities["pressure"]["effect"] == pressure["effect"]
    assert control_capabilities["pressure"]["messageKey"] == pressure["messageKey"]
    assert bounds["pressure_atm"]["minimum"] == pressure["fixed_atm"]
    assert bounds["pressure_atm"]["maximum"] == pressure["fixed_atm"]

    temperatures = [entry["temperature_C"] for entry in declared["speedFactors"]]
    assert min(temperatures) == bounds["temperature_C"]["minimum"]
    assert max(temperatures) == bounds["temperature_C"]["maximum"]
    assert bounds["temperature_C"]["default"] in temperatures


def test_boundary_cases_exercise_every_declared_quantity_bound(
    quantity_boundaries: tuple[dict[str, Any], ...],
) -> None:
    exercised = {bound for case in quantity_boundaries for bound in case["boundsExercised"]}
    assert sorted(exercised) == [
        "concentration_molL.maximum",
        "concentration_molL.minimum",
        "volume_mL.maximum",
        "volume_mL.minimum",
    ]

    bounds = CATALOG["model"]["bounds"]
    for case in quantity_boundaries:
        for solution in case["solutions"]:
            assert solution["concentration_molL"] >= bounds["concentration_molL"]["minimum"]
            assert solution["concentration_molL"] <= bounds["concentration_molL"]["maximum"]
            assert solution["volume_mL"] >= bounds["volume_mL"]["minimum"]
            assert solution["volume_mL"] <= bounds["volume_mL"]["maximum"]
            # The hand-stated amount is the chemistry, not a restatement of the
            # implementation's own arithmetic.
            assert solution["amount_mol"] == pytest.approx(
                solution["concentration_molL"] * solution["volume_mL"] / MILLILITRES_PER_LITRE,
                abs=AMOUNT_TOLERANCE,
            )


def test_concentration_and_volume_feed_every_quantitative_path(
    evaluator: ChemistryEvaluator, quantity_boundaries: tuple[dict[str, Any], ...]
) -> None:
    for case in quantity_boundaries:
        result = evaluator.evaluate_experiment(case_input(case))

        assert result["pairKey"] == case["pairKey"]
        assert result["totalVolume_L"] == pytest.approx(case["totalVolume_L"], abs=AMOUNT_TOLERANCE)
        assert result["extent_mol"] == pytest.approx(case["extent_mol"], abs=AMOUNT_TOLERANCE)
        assert sorted(result["limitingSpecies"]) == sorted(case["limitingSpecies"])

        for solution in case["solutions"]:
            reactant = reactant_of(result, solution["reagentId"])
            assert reactant["amount_mol"] == pytest.approx(
                solution["amount_mol"], abs=AMOUNT_TOLERANCE
            )
            assert reactant["amount_mol"] == calculate_amount(
                solution["concentration_molL"], solution["volume_mL"]
            )
            assert reactant["consumed_mol"] == pytest.approx(
                result["extent_mol"] * reactant["coefficient"], abs=AMOUNT_TOLERANCE
            )

        assert sorted(entry["speciesId"] for entry in result["excessSpecies"]) == sorted(
            entry["speciesId"] for entry in case["excessSpecies"]
        )
        for excess in case["excessSpecies"]:
            assert amount_of(result["excessSpecies"], excess["speciesId"]) == pytest.approx(
                excess["amount_mol"], abs=AMOUNT_TOLERANCE
            )
        for product in case["products"]:
            assert amount_of(result["products"], product["speciesId"]) == pytest.approx(
                product["amount_mol"], abs=AMOUNT_TOLERANCE
            )


def test_swapping_bounded_quantities_moves_the_limiting_reactant(
    evaluator: ChemistryEvaluator, quantity_boundaries: tuple[dict[str, Any], ...]
) -> None:
    chloride_limited = evaluator.evaluate_experiment(case_input(quantity_boundaries[2]))
    silver_limited = evaluator.evaluate_experiment(case_input(quantity_boundaries[3]))

    # Identical reagents and an identical total volume: only the
    # concentration/volume assignment differs, and it decides the chemistry.
    assert chloride_limited["pairKey"] == silver_limited["pairKey"]
    assert chloride_limited["totalVolume_L"] == pytest.approx(
        silver_limited["totalVolume_L"], abs=AMOUNT_TOLERANCE
    )
    assert chloride_limited["limitingSpecies"] == ("NaCl",)
    assert silver_limited["limitingSpecies"] == ("AgNO3",)
    assert chloride_limited["excessSpecies"][0]["speciesId"] == "AgNO3"
    assert silver_limited["excessSpecies"][0]["speciesId"] == "NaCl"


def test_derived_amounts_scale_with_concentration_and_volume(
    evaluator: ChemistryEvaluator,
) -> None:
    reference = evaluator.evaluate_experiment(build_input("AgNO3", (0.5, 40), "NaCl", (0.5, 40)))
    concentrated = evaluator.evaluate_experiment(build_input("AgNO3", (1, 40), "NaCl", (1, 40)))
    diluted = evaluator.evaluate_experiment(build_input("AgNO3", (0.5, 80), "NaCl", (0.5, 80)))

    assert concentrated["extent_mol"] == pytest.approx(
        reference["extent_mol"] * 2, abs=AMOUNT_TOLERANCE
    )
    assert amount_of(concentrated["products"], "AgCl") == pytest.approx(
        amount_of(reference["products"], "AgCl") * 2, abs=AMOUNT_TOLERANCE
    )
    # Concentration changes the amount without changing the mixed volume.
    assert concentrated["totalVolume_L"] == reference["totalVolume_L"]

    assert diluted["extent_mol"] == pytest.approx(reference["extent_mol"] * 2, abs=AMOUNT_TOLERANCE)
    assert diluted["totalVolume_L"] == pytest.approx(
        reference["totalVolume_L"] * 2, abs=AMOUNT_TOLERANCE
    )


def test_every_declared_temperature_maps_to_its_bounded_motion_factor(
    evaluator: ChemistryEvaluator, control_capabilities: dict[str, Any]
) -> None:
    declared = control_capabilities["temperature"]

    factors = []
    for entry in declared["speedFactors"]:
        factor = evaluator.motion_speed_factor(entry["temperature_C"])
        assert factor == pytest.approx(entry["speedFactor"], abs=AMOUNT_TOLERANCE)
        assert evaluator.describe_motion(entry["temperature_C"])["speedFactor"] == pytest.approx(
            entry["speedFactor"], abs=AMOUNT_TOLERANCE
        )
        factors.append(factor)

    assert factors == sorted(factors)
    assert len(set(factors)) == len(factors)

    # The band is closed by rejection, not by clamping: a temperature outside
    # the control range yields no factor at all, so a selection can never land
    # on a boundary factor it did not ask for. Task 5.2 made this a rejection;
    # before it, both calls returned the clamped edge value.
    assert declared["outOfRangeIsRejectedNotClamped"] is True
    for out_of_range in (-40, 500):
        with pytest.raises(InvalidExperimentQuantityError):
            evaluator.motion_speed_factor(out_of_range)

    assert evaluator.describe_motion(25)["referenceSpeedFactor"] == pytest.approx(
        declared["referenceSpeedFactor"], abs=AMOUNT_TOLERANCE
    )

    # Both declared edges are still exactly the declared factors, so closing
    # the band by rejection did not move the band itself.
    temperature_bounds = CATALOG["model"]["bounds"]["temperature_C"]
    assert (
        evaluator.motion_speed_factor(temperature_bounds["minimum"])
        == declared["minimumSpeedFactor"]
    )
    assert (
        evaluator.motion_speed_factor(temperature_bounds["maximum"])
        == declared["maximumSpeedFactor"]
    )


def test_temperature_changes_nothing_but_the_motion_metadata(
    evaluator: ChemistryEvaluator, control_capabilities: dict[str, Any]
) -> None:
    vector = control_capabilities["temperature"]["invarianceVector"]
    results = [
        evaluator.evaluate_experiment(case_input(vector, temperature_c=temperature_c))
        for temperature_c in vector["temperatures_C"]
    ]

    for temperature_c, result in zip(vector["temperatures_C"], results, strict=True):
        # Every quantity matches the one hand-stated outcome, at every
        # temperature, with no drift in extent, limiting reactant, or excess.
        assert result["extent_mol"] == pytest.approx(vector["extent_mol"], abs=AMOUNT_TOLERANCE)
        assert list(result["limitingSpecies"]) == vector["limitingSpecies"]
        assert result["totalVolume_L"] == pytest.approx(
            vector["totalVolume_L"], abs=AMOUNT_TOLERANCE
        )
        assert amount_of(result["excessSpecies"], "HCl") == pytest.approx(
            vector["excessSpecies"][0]["amount_mol"], abs=AMOUNT_TOLERANCE
        )

        assert without_motion(result) == without_motion(results[0])
        assert result["motion"]["temperature_C"] == temperature_c

    assert vector["invariantExceptField"] == "motion"
    # The motion blocks really do differ, so the equality above is a claim
    # about the rest of the result rather than about identical inputs.
    assert len({result["motion"]["speedFactor"] for result in results}) == len(results)


def test_motion_metadata_is_non_quantitative_and_localized(
    evaluator: ChemistryEvaluator,
    formatter: ChemistryFormatter,
    control_capabilities: dict[str, Any],
) -> None:
    declared = control_capabilities["temperature"]
    result = evaluator.evaluate_experiment(
        case_input(declared["invarianceVector"], temperature_c=100)
    )
    motion = result["motion"]

    assert motion["quantitative"] is False
    assert motion["effect"] == declared["effect"]
    assert motion["model"] == declared["model"]
    assert motion["affectsExtent"] is False
    assert motion["affectsRateConstant"] is False
    assert motion["affectsPH"] is False
    assert motion["messageKey"] == declared["messageKey"]
    assert motion["message"] == formatter.resolve_message(declared["messageKey"])
    assert motion["message"] == formatter.describe_capability_boundary("temperature")["message"]


def test_unsupplied_temperature_defaults_and_undefined_temperature_is_rejected(
    evaluator: ChemistryEvaluator, control_capabilities: dict[str, Any]
) -> None:
    vector = control_capabilities["temperature"]["invarianceVector"]
    default_c = CATALOG["model"]["bounds"]["temperature_C"]["default"]

    experiment_input = case_input(vector, temperature_c=None)
    defaulted = evaluator.evaluate_experiment(experiment_input)
    assert defaulted["motion"]["temperature_C"] == default_c
    assert defaulted == evaluator.evaluate_experiment(case_input(vector, temperature_c=default_c))

    for value in (float("nan"), float("inf"), float("-inf"), "25", True):
        invalid = case_input(vector)
        invalid["temperature_C"] = value
        with pytest.raises(InvalidExperimentQuantityError):
            evaluator.evaluate_experiment(invalid)

    # Zero degrees is a valid selection, not a missing quantity.
    at_minimum = evaluator.evaluate_experiment(case_input(vector, temperature_c=0))
    assert (
        at_minimum["motion"]["speedFactor"]
        == control_capabilities["temperature"]["minimumSpeedFactor"]
    )


def test_motion_band_is_read_from_the_catalog(
    evaluator: ChemistryEvaluator, control_capabilities: dict[str, Any]
) -> None:
    widened = catalog_with_motion_override(atMaximum=7)
    assert ChemistryEvaluator(widened).motion_speed_factor(100) == 7
    assert (
        evaluator.motion_speed_factor(100)
        == control_capabilities["temperature"]["maximumSpeedFactor"]
    )


@pytest.mark.parametrize(
    ("claim", "message"),
    [
        ("affectsExtent", "affectsExtent false"),
        ("affectsRateConstant", "affectsRateConstant false"),
        ("affectsPH", "affectsPH false"),
    ],
)
def test_catalog_claiming_a_quantitative_temperature_effect_is_refused(
    claim: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ChemistryEvaluator(catalog_with_capability_override("temperature", **{claim: True}))


def test_catalog_claiming_an_unsupported_temperature_model_is_refused() -> None:
    with pytest.raises(ValueError, match="qualitative_motion_only"):
        ChemistryEvaluator(catalog_with_capability_override("temperature", effect="arrhenius_rate"))

    with pytest.raises(ValueError, match="unsupported motion speed model"):
        ChemistryEvaluator(catalog_with_motion_override(model="exponential"))


def test_pressure_reports_the_fixed_catalog_state_with_its_boundary_statement(
    evaluator: ChemistryEvaluator,
    formatter: ChemistryFormatter,
    control_capabilities: dict[str, Any],
    quantity_boundaries: tuple[dict[str, Any], ...],
) -> None:
    declared = control_capabilities["pressure"]
    result = evaluator.evaluate_experiment(case_input(quantity_boundaries[0]))
    pressure = result["pressure"]

    assert pressure["atm"] == declared["atm"]
    assert pressure["effect"] == declared["effect"]
    assert pressure["selectable"] is False
    assert pressure["acceptsAlternateValues"] is False
    assert pressure["affectsResult"] is False
    assert pressure["messageKey"] == declared["messageKey"]
    assert pressure["message"] == formatter.resolve_message(declared["messageKey"])
    assert pressure["message"] == formatter.describe_capability_boundary("pressure")["message"]


def test_no_result_value_depends_on_the_supplied_pressure_field(
    evaluator: ChemistryEvaluator,
    control_capabilities: dict[str, Any],
    quantity_boundaries: tuple[dict[str, Any], ...],
) -> None:
    assert control_capabilities["pressure"]["omittedFieldIsEquivalent"] is True

    for case in quantity_boundaries:
        # Omitting the control, or supplying the one fixed value, cannot change
        # a single reported value, which is what "no modeled effect" has to
        # mean. Task 5.2 rejects any other value outright, so this is the whole
        # space of accepted pressures rather than a silently ignored field.
        assert evaluator.evaluate_experiment(
            case_input(case, omit_pressure=True)
        ) == evaluator.evaluate_experiment(case_input(case))

    assert control_capabilities["pressure"]["alternateValueIsRejectedNotIgnored"] is True


def test_catalog_making_pressure_selectable_or_effective_is_refused() -> None:
    with pytest.raises(ValueError, match="neither selectable nor accept alternate values"):
        ChemistryEvaluator(catalog_with_capability_override("pressure", selectable=True))

    with pytest.raises(ValueError, match="neither selectable nor accept alternate values"):
        ChemistryEvaluator(
            catalog_with_capability_override("pressure", acceptsAlternateValues=True)
        )

    with pytest.raises(ValueError, match="effect none"):
        ChemistryEvaluator(catalog_with_capability_override("pressure", effect="ideal_gas"))

    with pytest.raises(ValueError, match="fixed at 1"):
        ChemistryEvaluator(catalog_with_bounds_override("pressure_atm", maximum=2))


def test_no_control_position_fabricates_a_ph_value(
    evaluator: ChemistryEvaluator,
    control_capabilities: dict[str, Any],
    quantity_boundaries: tuple[dict[str, Any], ...],
    ph_model: dict[str, Any],
) -> None:
    invariance_vector = control_capabilities["temperature"]["invarianceVector"]
    positions = [evaluator.evaluate_experiment(case_input(case)) for case in quantity_boundaries]
    positions += [
        evaluator.evaluate_experiment(
            case_input(invariance_vector, temperature_c=entry["temperature_C"])
        )
        for entry in control_capabilities["temperature"]["speedFactors"]
    ]

    for result in positions:
        assert result["speciesLedger"] is None

        # The one supported acid-base system reports a calculated pH; every
        # other pair still reports none at all.
        if result["pairKey"] == ph_model["supportedPairKey"]:
            assert result["pH"]["supported"] is True
            assert math.isfinite(result["pH"]["value"])
        else:
            assert result["pH"] is None

        # The audited engine seeded every result with "~7.0" before looking at
        # any quantity; no control position may reintroduce it.
        assert not re.search(r"~\s*7", json.dumps(result, ensure_ascii=False))


# -- rejected input semantics -----------------------------------------------
#
# Out-of-range input is rejected, never clamped and never ignored. Clamping is
# the failure mode this guards against: ``2.5 M`` quietly becoming ``2 M``
# evaluates a different experiment than the learner set up and reports it as
# though it were theirs, and the audited engine's ignored pressure control did
# the mirror-image thing by presenting an input with no effect. So a value just
# outside a bound must produce no result at all, while a value exactly at a
# bound must stay a valid experiment, and every rejection must name its field
# with a stable reason code and Indonesian text that the frontend half
# reproduces byte for byte.
#
# Validates: Requirements 5.4

#: Values that name no position on a control, by fixture token.
UNDEFINED_VALUES: Mapping[str, Any] = {
    "nan": float("nan"),
    "positive_infinity": float("inf"),
    "negative_infinity": float("-inf"),
    "zero": 0,
    "negative": -1,
    "text": "1",
    "boolean": True,
    "missing": None,
}

#: Which solution and quantity each A/B field sets.
SOLUTION_FIELDS: Mapping[str, tuple[int, str]] = {
    "concentrationA_molL": (0, "concentration_molL"),
    "volumeA_mL": (0, "volume_mL"),
    "concentrationB_molL": (1, "concentration_molL"),
    "volumeB_mL": (1, "volume_mL"),
}


@pytest.fixture(scope="module")
def input_validation(control_capabilities: dict[str, Any]) -> dict[str, Any]:
    return control_capabilities["inputValidation"]


def baseline_input(input_validation: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    """The valid baseline experiment with zero or more fields overridden."""
    baseline = input_validation["baseline"]
    first, second = baseline["solutions"]
    experiment_input = build_input(
        first["reagentId"],
        (first["concentration_molL"], first["volume_mL"]),
        second["reagentId"],
        (second["concentration_molL"], second["volume_mL"]),
    )
    experiment_input["temperature_C"] = baseline["temperature_C"]
    experiment_input["pressure_atm"] = baseline["pressure_atm"]
    experiment_input.update(overrides)
    return experiment_input


def violation_for(
    evaluator: ChemistryEvaluator,
    input_validation: Mapping[str, Any],
    field: str,
    value: Any,
) -> Mapping[str, Any]:
    """The single violation one overridden field produces."""
    try:
        evaluator.evaluate_experiment(baseline_input(input_validation, **{field: value}))
    except InvalidExperimentQuantityError as error:
        assert len(error.violations) == 1, f"{field}={value!r}"
        return error.violations[0]
    raise AssertionError(f"{field}={value!r} was accepted but should have been rejected")


def trigger_for(input_validation: Mapping[str, Any], field: str, reason_code: str) -> Any:
    """A fixture value that triggers one reason code on one field."""
    for case in input_validation["rejectedJustOutsideBounds"]:
        if case["field"] == field and case["reasonCode"] == reason_code:
            return case["value"]
    for case in input_validation["rejectedUndefinedValues"]:
        if case["field"] == field and case["reasonCode"] == reason_code:
            return UNDEFINED_VALUES[case["valueToken"]]
    raise AssertionError(f"no fixture case triggers {reason_code} on {field}")


def test_validation_is_pinned_to_the_declared_codes_fields_and_bounds(
    evaluator: ChemistryEvaluator, input_validation: dict[str, Any]
) -> None:
    assert list(EXPERIMENT_INPUT_REASON_CODES) == input_validation["reasonCodes"]
    assert [
        {
            "field": descriptor.field,
            "bound": descriptor.bound,
            "label": descriptor.label,
            "unit": descriptor.unit,
            "required": descriptor.required,
            "positive": descriptor.positive,
            "fixed": descriptor.fixed,
        }
        for descriptor in VALIDATED_EXPERIMENT_FIELDS
    ] == input_validation["fields"]

    # Validation reads the catalog's declared bounds, so a regenerated catalog
    # that widens the control range widens validation with it rather than being
    # contradicted by a number written into the evaluator.
    for descriptor in VALIDATED_EXPERIMENT_FIELDS:
        assert evaluator.bounds[descriptor.field] == CATALOG["model"]["bounds"][descriptor.bound]


def test_values_exactly_at_a_declared_bound_are_accepted_unchanged(
    evaluator: ChemistryEvaluator, input_validation: dict[str, Any]
) -> None:
    baseline = input_validation["baseline"]
    for case in input_validation["acceptedAtBounds"]:
        bound_name, edge = case["bound"].split(".")
        assert CATALOG["model"]["bounds"][bound_name][edge] == case["value"]

        result = evaluator.evaluate_experiment(
            baseline_input(input_validation, **{case["field"]: case["value"]})
        )
        target = SOLUTION_FIELDS.get(case["field"])
        if target is not None:
            index, prop = target
            reactant = reactant_of(result, baseline["solutions"][index]["reagentId"])
            assert reactant[prop] == case["value"]
        elif case["field"] == "temperature_C":
            assert result["motion"]["temperature_C"] == case["value"]
        else:
            assert result["pressure"]["atm"] == case["value"]


def test_values_just_outside_a_declared_bound_are_rejected_with_field_and_code(
    evaluator: ChemistryEvaluator, input_validation: dict[str, Any]
) -> None:
    for case in input_validation["rejectedJustOutsideBounds"]:
        violation = violation_for(evaluator, input_validation, case["field"], case["value"])
        assert violation["field"] == case["field"]
        assert violation["reasonCode"] == case["reasonCode"]
        assert violation["limit"] == case["limit"]
        assert violation["value"] == case["value"]


def test_values_that_name_no_position_on_a_control_are_rejected(
    evaluator: ChemistryEvaluator, input_validation: dict[str, Any]
) -> None:
    for case in input_validation["rejectedUndefinedValues"]:
        value = UNDEFINED_VALUES[case["valueToken"]]
        violation = violation_for(evaluator, input_validation, case["field"], value)
        assert violation["field"] == case["field"]
        assert violation["reasonCode"] == case["reasonCode"]
        assert violation["limit"] is None


def test_an_out_of_range_value_produces_no_result_at_all(
    evaluator: ChemistryEvaluator, input_validation: dict[str, Any]
) -> None:
    for case in input_validation["rejectedJustOutsideBounds"]:
        # The clamped experiment is itself perfectly evaluable, which is exactly
        # why substituting it would be invisible to a learner.
        clamped = evaluator.evaluate_experiment(
            baseline_input(input_validation, **{case["field"]: case["limit"]})
        )
        assert clamped["pairKey"] == input_validation["baseline"]["pairKey"]

        produced = None
        try:
            produced = evaluator.evaluate_experiment(
                baseline_input(input_validation, **{case["field"]: case["value"]})
            )
        except InvalidExperimentQuantityError:
            pass
        assert produced is None, f"{case['field']}={case['value']!r} produced a result"


def test_every_rejection_is_explained_in_indonesian_without_echoing_the_value(
    evaluator: ChemistryEvaluator, input_validation: dict[str, Any]
) -> None:
    for case in input_validation["localizedMessages"]:
        value = trigger_for(input_validation, case["field"], case["reasonCode"])
        violation = violation_for(evaluator, input_validation, case["field"], value)

        assert violation["reasonCode"] == case["reasonCode"]
        assert violation["message"] == case["message"]

        # Learner-facing text names the control, not the internal field, and
        # never echoes what was typed back as part of the explanation.
        assert case["field"] not in violation["message"]
        assert str(value) not in violation["message"]


def test_every_offending_field_is_reported_in_one_pass(
    evaluator: ChemistryEvaluator, input_validation: dict[str, Any]
) -> None:
    declared = input_validation["multiFieldRejection"]
    overrides = {override["field"]: override["value"] for override in declared["overrides"]}

    with pytest.raises(InvalidExperimentQuantityError) as raised:
        evaluator.evaluate_experiment(baseline_input(input_validation, **overrides))
    error = raised.value

    assert list(error.fields) == [item["field"] for item in declared["expected"]]
    assert list(error.reason_codes) == [item["reasonCode"] for item in declared["expected"]]
    assert len(error.localized_messages) == len(declared["expected"])

    # Field order follows the declared field order, so two halves of the pair
    # and two consumers of the same error read the list the same way.
    assert list(error.fields) == [
        descriptor.field
        for descriptor in VALIDATED_EXPERIMENT_FIELDS
        if descriptor.field in error.fields
    ]
    assert error.localized_message == " ".join(error.localized_messages)
    assert error.field == error.fields[0]


def test_declared_range_is_enforced_without_enforcing_declared_step_size(
    evaluator: ChemistryEvaluator, input_validation: dict[str, Any]
) -> None:
    # The step describes how a slider moves, not what chemistry is defined, so
    # an exactly calculated off-step quantity must stay valid.
    for case in input_validation["offStepValuesStayValid"]:
        bounds = evaluator.bounds[case["field"]]
        assert case["value"] % bounds["step"] != 0

        accepted = evaluator.validate_experiment_input(
            baseline_input(input_validation, **{case["field"]: case["value"]})
        )
        assert accepted[case["field"]] == case["value"]


def test_only_the_two_defaulting_controls_may_be_omitted(
    evaluator: ChemistryEvaluator, input_validation: dict[str, Any]
) -> None:
    supplied = evaluator.validate_experiment_input(baseline_input(input_validation))
    omitted = evaluator.validate_experiment_input(
        baseline_input(input_validation, temperature_C=None, pressure_atm=None)
    )

    assert omitted == supplied
    assert omitted["temperature_C"] == CATALOG["model"]["bounds"]["temperature_C"]["default"]
    assert omitted["pressure_atm"] == CATALOG["model"]["bounds"]["pressure_atm"]["default"]


def test_catalog_whose_bounds_cannot_judge_an_input_is_refused() -> None:
    with pytest.raises(ValueError, match="maximum at or above its minimum"):
        ChemistryEvaluator(catalog_with_bounds_override("concentration_molL", maximum=0.05))

    with pytest.raises(ValueError, match="default inside its range"):
        ChemistryEvaluator(catalog_with_bounds_override("volume_mL", default=5))

    bounds = {
        name: value for name, value in CATALOG["model"]["bounds"].items() if name != "volume_mL"
    }
    unbounded = {**CATALOG, "model": {**CATALOG["model"], "bounds": bounds}}
    with pytest.raises(ValueError, match=re.escape("no model.bounds.volume_mL")):
        ChemistryEvaluator(unbounded)


def test_validation_widens_with_the_catalog_rather_than_against_it(
    evaluator: ChemistryEvaluator, input_validation: dict[str, Any]
) -> None:
    permissive = ChemistryEvaluator(catalog_with_bounds_override("concentration_molL", maximum=4))
    experiment_input = baseline_input(input_validation, concentrationA_molL=3)

    assert (
        permissive.evaluate_experiment(experiment_input)["reactants"][0]["concentration_molL"] == 3
    )
    with pytest.raises(InvalidExperimentQuantityError):
        evaluator.evaluate_experiment(experiment_input)


# -- the ideal strong-acid/strong-base pH model -----------------------------
#
# Audited defect 1.1 is the reason this section exists: the engine seeded every
# result with a value near 7 and then reported exactly neutral for every
# HCl/NaOH mixture, without reading concentration, volume, the limiting
# reagent, or the excess at all. So these tests check the arithmetic itself
# rather than only the shape of the answer:
#
# - the three formulas, recomputed here from the fixture's hand-stated amounts
#   instead of from anything the evaluator produced,
# - the neutral point, which must be both ``pKw/2`` and the catalog's declared
#   ``neutralPH``, reached through the declared equivalence tolerance rather
#   than through exact float equality,
# - monotonicity, because "more acid" has to mean "lower pH" and not merely
#   "some other number",
# - the side of neutral, over the whole on-step control grid,
# - and that the pH is built from unrounded amounts, from the declared pKw
#   rather than a literal 14, and from nothing temperature does.
#
# The frontend half of the pair covers the same ground in
# ``tests/chemistry/chemistry-evaluator.test.js`` against the same fixture.
#
# Validates: Requirements 3.3, 3.4, 3.5, 3.7


@pytest.fixture(scope="module")
def ph_model(vector_fixture: dict[str, Any]) -> dict[str, Any]:
    return vector_fixture["pHModel"]


def ph_case(ph_model: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    for case in ph_model["cases"]:
        if case["id"] == case_id:
            return case
    raise AssertionError(f"pH fixture declares no case {case_id}")


def mixture_input(
    ph_model: Mapping[str, Any],
    acid: Mapping[str, Any],
    base: Mapping[str, Any],
) -> dict[str, Any]:
    """Experiment input for one acid/base mixture, acid in the A slot."""
    return build_input(
        ph_model["acidReagentId"],
        (acid["concentration_molL"], acid["volume_mL"]),
        ph_model["baseReagentId"],
        (base["concentration_molL"], base["volume_mL"]),
    )


def catalog_with_ph_override(**overrides: Any) -> dict[str, Any]:
    """Catalog copy with one declared pH capability field replaced."""
    return catalog_with_capability_override("pH", **overrides)


def catalog_with_tolerance_override(**overrides: Any) -> dict[str, Any]:
    """Catalog copy with one declared tolerance replaced."""
    tolerances = {**CATALOG["model"]["tolerances"], **overrides}
    return {**CATALOG, "model": {**CATALOG["model"], "tolerances": tolerances}}


def test_ph_fixture_is_pinned_to_the_declared_capability(
    evaluator: ChemistryEvaluator, ph_model: dict[str, Any]
) -> None:
    declared = CATALOG["model"]["capabilities"]["pH"]

    assert ph_model["modelId"] == declared["modelId"]
    assert ph_model["modelLabel"] == declared["modelLabel"]
    assert ph_model["pKw"] == declared["pKw"]
    assert ph_model["pKwSource"] == declared["pKwSource"]
    assert ph_model["temperatureCorrected"] == declared["temperatureCorrected"]
    assert ph_model["neutralPH"] == declared["neutralPH"]
    assert ph_model["displayPrecision"] == declared["displayPrecision"]
    assert [ph_model["supportedReactionId"]] == list(declared["supportedReactionIds"])
    assert ph_model["equivalenceTolerance_mol"] == CATALOG["model"]["tolerances"]["equivalence_mol"]
    assert ph_model["pHTolerance"] == CATALOG["model"]["tolerances"]["pH"]

    assert evaluator.ph_model == {
        "modelId": declared["modelId"],
        "modelLabel": declared["modelLabel"],
        "pKw": float(declared["pKw"]),
        "pKwSource": declared["pKwSource"],
        "temperatureCorrected": False,
        "neutralPH": float(declared["neutralPH"]),
        "supportedReactionIds": tuple(declared["supportedReactionIds"]),
    }
    assert evaluator.equivalence_tolerance == CATALOG["model"]["tolerances"]["equivalence_mol"]
    assert evaluator.ph_tolerance == CATALOG["model"]["tolerances"]["pH"]


def test_every_stated_case_comes_from_the_excess_amount_and_the_mixed_volume(
    evaluator: ChemistryEvaluator, formatter: ChemistryFormatter, ph_model: dict[str, Any]
) -> None:
    for case in ph_model["cases"]:
        label = case["id"]
        result = evaluator.evaluate_experiment(case_input(case))
        ph = result["pH"]

        # The amounts the formula runs on, stated by hand rather than read back.
        assert math.isclose(
            ph["acidEquivalents_mol"], case["acidEquivalents_mol"], abs_tol=AMOUNT_TOLERANCE
        ), label
        assert math.isclose(
            ph["baseEquivalents_mol"], case["baseEquivalents_mol"], abs_tol=AMOUNT_TOLERANCE
        ), label
        assert math.isclose(
            ph["equivalenceDifference_mol"],
            case["equivalenceDifference_mol"],
            abs_tol=AMOUNT_TOLERANCE,
        ), label
        assert ph["totalVolume_L"] == result["totalVolume_L"], label
        assert math.isclose(ph["totalVolume_L"], case["totalVolume_L"], abs_tol=AMOUNT_TOLERANCE), (
            label
        )

        assert ph["equivalence"] is case["equivalence"], label
        assert ph["side"] == case["side"], label
        assert ph["excessIonId"] == case["excessIonId"], label
        assert math.isclose(
            ph["excessAmount_mol"], case["excessAmount_mol"], abs_tol=AMOUNT_TOLERANCE
        ), label
        assert math.isclose(
            ph["excessConcentration_molL"], case["excessConcentration_molL"], abs_tol=PH_TOLERANCE
        ), label
        assert math.isclose(ph["value"], case["pH"], abs_tol=PH_TOLERANCE), label

        # The three formulas, recomputed here from the fixture's own numbers.
        if case["equivalence"]:
            assert ph["value"] == ph_model["pKw"] / 2, label
        elif case["side"] == "acidic":
            expected = -math.log10(case["excessAmount_mol"] / case["totalVolume_L"])
            assert math.isclose(ph["value"], expected, abs_tol=PH_TOLERANCE), label
        else:
            poh = -math.log10(case["excessAmount_mol"] / case["totalVolume_L"])
            assert math.isclose(poh, case["pOH"], abs_tol=PH_TOLERANCE), label
            assert math.isclose(ph["value"], ph_model["pKw"] - poh, abs_tol=PH_TOLERANCE), label

        # The excess the pH model used is the same excess the stoichiometry
        # reported, so the two halves of the result cannot disagree.
        if case["excessIonId"] is None:
            assert result["excessSpecies"] == (), label
        else:
            expected_species = (
                ph_model["acidReagentId"] if case["side"] == "acidic" else ph_model["baseReagentId"]
            )
            assert [entry["speciesId"] for entry in result["excessSpecies"]] == [
                expected_species
            ], label
            assert math.isclose(
                result["excessSpecies"][0]["amount_mol"],
                case["excessAmount_mol"],
                abs_tol=AMOUNT_TOLERANCE,
            ), label

        # Display precision is the formatter's alone, and the domain value keeps
        # every digit the formula produced.
        assert formatter.format_quantity("pH", ph["value"]) == case["display"], label
        assert formatter.format_experiment_quantities(result)["pH"] == case["display"], label
        assert ph["supported"] is True, label
        assert ph["reasonCode"] is None, label
        assert ph["message"] is None, label


def test_the_neutral_point_is_both_pkw_halved_and_the_declared_neutral_ph(
    evaluator: ChemistryEvaluator, formatter: ChemistryFormatter, ph_model: dict[str, Any]
) -> None:
    case = ph_case(ph_model, "ph-equivalence-exact")
    result = evaluator.evaluate_experiment(case_input(case))

    assert result["pH"]["value"] == ph_model["pKw"] / 2
    assert result["pH"]["value"] == ph_model["neutralPH"]
    assert result["pH"]["neutralPH"] == ph_model["neutralPH"]
    assert abs(result["pH"]["value"] - ph_model["neutralPH"]) <= ph_model["pHTolerance"]
    assert formatter.format_quantity("pH", result["pH"]["value"]) == ph_model["neutralDisplay"]

    # Equivalence is the limiting case of both excess branches, so neither
    # reagent is left over and the reaction consumed both completely.
    assert sorted(result["limitingSpecies"]) == sorted(
        (ph_model["acidReagentId"], ph_model["baseReagentId"])
    )
    assert result["excessSpecies"] == ()


def test_equivalence_uses_the_declared_tolerance_not_exact_equality(
    evaluator: ChemistryEvaluator, ph_model: dict[str, Any]
) -> None:
    case = ph_case(ph_model, "ph-equivalence-within-tolerance")
    result = evaluator.evaluate_experiment(case_input(case))
    difference = result["pH"]["equivalenceDifference_mol"]

    # The two equivalents are genuinely unequal as doubles, and still inside the
    # declared tolerance, which is the whole point of comparing widths.
    assert difference != 0
    assert abs(difference) <= ph_model["equivalenceTolerance_mol"]
    assert result["pH"]["equivalence"] is True
    assert result["pH"]["side"] == "neutral"
    assert result["pH"]["value"] == ph_model["neutralPH"]

    # Without the tolerance the same difference would be divided by the mixed
    # volume and reported as a basic pH for an acid-side difference, which is
    # the nonsense the tolerance exists to prevent.
    assert case["toleranceIgnoredLandsOnWrongSide"] is True
    ignored = -math.log10(difference / result["totalVolume_L"])
    assert math.isclose(ignored, case["pHIfToleranceIgnored"], abs_tol=1e-6)
    assert ignored > ph_model["neutralPH"]


def test_more_acid_lowers_the_ph_and_more_base_raises_it(
    evaluator: ChemistryEvaluator, ph_model: dict[str, Any]
) -> None:
    for series in ph_model["monotonicity"]["series"]:
        values: list[Mapping[str, Any]] = []
        for volume_ml in series["varied"]["volumes_mL"]:
            varied = {
                "concentration_molL": series["varied"]["concentration_molL"],
                "volume_mL": volume_ml,
            }
            if series["varied"]["reagentId"] == ph_model["acidReagentId"]:
                experiment_input = mixture_input(ph_model, varied, series["fixed"])
            else:
                experiment_input = mixture_input(ph_model, series["fixed"], varied)
            values.append(evaluator.evaluate_experiment(experiment_input)["pH"])

        for index, ph in enumerate(values):
            label = f"{series['id']} #{index}"
            assert math.isclose(ph["value"], series["pH"][index], abs_tol=PH_TOLERANCE), label
            assert ph["side"] == series["sides"][index], label

        # Strict, step by step: a flat stretch would mean the control stopped
        # affecting the answer, which is exactly the audited defect.
        for index in range(1, len(values)):
            label = f"{series['id']} step {index}"
            if series["direction"] == "strictly_decreasing":
                assert values[index]["value"] < values[index - 1]["value"], label
            else:
                assert values[index]["value"] > values[index - 1]["value"], label


def test_every_on_step_control_position_lands_on_the_correct_side_of_neutral(
    evaluator: ChemistryEvaluator, ph_model: dict[str, Any]
) -> None:
    grid = ph_model["sideOfNeutral"]
    counts = {"acidic": 0, "basic": 0, "neutral": 0}

    for concentration_acid in grid["grid"]["concentrations_molL"]:
        for volume_acid in grid["grid"]["volumes_mL"]:
            for concentration_base in grid["grid"]["concentrations_molL"]:
                for volume_base in grid["grid"]["volumes_mL"]:
                    result = evaluator.evaluate_experiment(
                        mixture_input(
                            ph_model,
                            {
                                "concentration_molL": concentration_acid,
                                "volume_mL": volume_acid,
                            },
                            {
                                "concentration_molL": concentration_base,
                                "volume_mL": volume_base,
                            },
                        )
                    )
                    ph = result["pH"]
                    label = (
                        f"{concentration_acid}M/{volume_acid}mL vs "
                        f"{concentration_base}M/{volume_base}mL"
                    )
                    counts[ph["side"]] += 1

                    if ph["equivalence"]:
                        assert ph["side"] == "neutral", label
                        assert ph["value"] == ph_model["neutralPH"], label
                        assert ph["excessIonId"] is None, label
                        continue

                    # Every on-step excess is far above the neutral
                    # concentration the declared pKw implies, so the ideal model
                    # has no way to land on or across the neutral point here.
                    assert (
                        ph["excessAmount_mol"]
                        >= grid["smallestNonEquivalentExcess_mol"] - evaluator.amount_tolerance
                    ), label
                    assert ph["excessConcentration_molL"] > grid["neutralConcentration_molL"], label

                    if ph["equivalenceDifference_mol"] > 0:
                        assert ph["side"] == "acidic", label
                        assert ph["excessIonId"] == ph_model["protonIonId"], label
                        assert ph["value"] < ph_model["neutralPH"], label
                        assert ph["value"] <= grid["acidMaximumPH"], label
                    else:
                        assert ph["side"] == "basic", label
                        assert ph["excessIonId"] == ph_model["hydroxideIonId"], label
                        assert ph["value"] > ph_model["neutralPH"], label
                        assert ph["value"] >= grid["baseMinimumPH"], label

    # All three branches were actually exercised, so the loop above is a claim
    # about behavior rather than about an unreached code path.
    assert all(count > 0 for count in counts.values())


def test_the_ph_is_calculated_from_unrounded_amounts(
    evaluator: ChemistryEvaluator, formatter: ChemistryFormatter, ph_model: dict[str, Any]
) -> None:
    proof = ph_model["unroundedInputs"]
    acid, base = proof["solutions"]

    unrounded = evaluator.evaluate_experiment(mixture_input(ph_model, acid, base))
    assert unrounded["pH"]["side"] == proof["unrounded"]["side"]
    assert unrounded["pH"]["excessIonId"] == proof["unrounded"]["excessIonId"]
    assert math.isclose(unrounded["pH"]["value"], proof["unrounded"]["pH"], abs_tol=PH_TOLERANCE)
    assert (
        formatter.format_quantity("pH", unrounded["pH"]["value"]) == proof["unrounded"]["display"]
    )

    # Exactly what a precision leak would do: hand the numbers the learner sees
    # back to the chemistry.
    displayed = {
        "concentration_molL": formatter.round_quantity_for_display(
            "concentration_molL", acid["concentration_molL"]
        ),
        "volume_mL": formatter.round_quantity_for_display("volume_mL", acid["volume_mL"]),
    }
    assert displayed == proof["displayRoundedAcidInput"]

    leaked = evaluator.evaluate_experiment(mixture_input(ph_model, displayed, base))
    assert math.isclose(leaked["pH"]["value"], proof["leaked"]["pH"], abs_tol=PH_TOLERANCE)
    assert formatter.format_quantity("pH", leaked["pH"]["value"]) == proof["leaked"]["display"]

    # The leak is not a last-digit difference: it moves the answer across the
    # neutral point, so a learner would be told the opposite chemistry.
    assert leaked["pH"]["side"] == proof["leaked"]["side"]
    assert leaked["pH"]["excessIonId"] == proof["leaked"]["excessIonId"]
    assert unrounded["pH"]["side"] != leaked["pH"]["side"]
    assert abs(leaked["pH"]["value"] - unrounded["pH"]["value"]) > ph_model["pHTolerance"]
    assert unrounded["pH"]["value"] < ph_model["neutralPH"]
    assert leaked["pH"]["value"] > ph_model["neutralPH"]


def test_the_ph_is_the_same_at_every_selectable_temperature(
    evaluator: ChemistryEvaluator, ph_model: dict[str, Any]
) -> None:
    case = ph_case(ph_model, "ph-acid-excess-matching-vector")
    bounds = CATALOG["model"]["bounds"]["temperature_C"]
    declared = CATALOG["model"]["capabilities"]["pH"]

    values = [
        evaluator.evaluate_experiment(case_input(case, temperature_c=temperature_c))["pH"]
        for temperature_c in (bounds["minimum"], bounds["default"], 50, bounds["maximum"])
    ]

    for ph in values:
        assert ph == values[0]
        assert ph["temperatureCorrected"] is False
        assert ph["pKwSource"] == declared["pKwSource"]
    assert CATALOG["model"]["capabilities"]["temperature"]["affectsPH"] is False


def test_both_ab_orders_report_the_identical_ph_block(
    evaluator: ChemistryEvaluator, ph_model: dict[str, Any]
) -> None:
    for case in ph_model["cases"]:
        first, second = case["solutions"]
        forward = evaluator.evaluate_experiment(
            build_input(
                first["reagentId"],
                (first["concentration_molL"], first["volume_mL"]),
                second["reagentId"],
                (second["concentration_molL"], second["volume_mL"]),
            )
        )
        reversed_result = evaluator.evaluate_experiment(
            build_input(
                second["reagentId"],
                (second["concentration_molL"], second["volume_mL"]),
                first["reagentId"],
                (first["concentration_molL"], first["volume_mL"]),
            )
        )
        assert reversed_result["pH"] == forward["pH"], case["id"]


def test_the_declared_pkw_is_read_rather_than_a_literal_fourteen(
    ph_model: dict[str, Any],
) -> None:
    shifted = ChemistryEvaluator(catalog_with_ph_override(pKw=15, neutralPH=7.5))

    neutral_case = ph_case(ph_model, "ph-equivalence-exact")
    base_case = ph_case(ph_model, "ph-base-excess-matching-vector")
    acid_case = ph_case(ph_model, "ph-acid-excess-matching-vector")

    assert shifted.evaluate_experiment(case_input(neutral_case))["pH"]["value"] == 7.5
    assert math.isclose(
        shifted.evaluate_experiment(case_input(base_case))["pH"]["value"],
        base_case["pH"] + 1,
        abs_tol=PH_TOLERANCE,
    )

    # An acid excess depends on no pKw at all, so it is unmoved.
    assert math.isclose(
        shifted.evaluate_experiment(case_input(acid_case))["pH"]["value"],
        acid_case["pH"],
        abs_tol=PH_TOLERANCE,
    )


def test_catalog_whose_ph_model_is_not_implemented_is_refused() -> None:
    with pytest.raises(ValueError, match="unsupported pH model"):
        ChemistryEvaluator(catalog_with_ph_override(modelId="debye_huckel_activity_corrected"))

    with pytest.raises(ValueError, match=re.escape("neutralPH equal to pKw/2")):
        ChemistryEvaluator(catalog_with_ph_override(neutralPH=7.4))

    with pytest.raises(ValueError, match="positive pKw"):
        ChemistryEvaluator(catalog_with_ph_override(pKw=-14))

    with pytest.raises(ValueError, match="temperatureCorrected false"):
        ChemistryEvaluator(catalog_with_ph_override(temperatureCorrected=True))

    with pytest.raises(ValueError, match=re.escape("tolerances.equivalence_mol")):
        ChemistryEvaluator(catalog_with_tolerance_override(equivalence_mol=0))

    with pytest.raises(ValueError, match=re.escape("tolerances.pH")):
        ChemistryEvaluator(catalog_with_tolerance_override(pH=0))


def test_the_monoprotic_formulas_are_not_applied_to_other_chemistry(
    ph_model: dict[str, Any],
) -> None:
    acid_case = ph_case(ph_model, "ph-acid-excess-matching-vector")
    experiment_input = case_input(acid_case)

    acid_id = ph_model["acidReagentId"]
    reagent = CATALOG["reagents"][acid_id]
    polyprotic_ions = tuple(
        {**term, "multiplicity": 2} if term["ionId"] == ph_model["protonIonId"] else dict(term)
        for term in reagent["ions"]
    )
    polyprotic = {
        **CATALOG,
        "reagents": {**CATALOG["reagents"], acid_id: {**reagent, "ions": polyprotic_ions}},
    }
    with pytest.raises(ValueError, match="monoprotic"):
        ChemistryEvaluator(polyprotic).evaluate_experiment(experiment_input)

    unlisted = ChemistryEvaluator(catalog_with_ph_override(supportedReactionIds=()))
    with pytest.raises(ValueError, match="does not list as supported"):
        unlisted.evaluate_experiment(experiment_input)


def test_no_system_outside_the_supported_model_reports_a_ph(
    evaluator: ChemistryEvaluator,
    formatter: ChemistryFormatter,
    ph_model: dict[str, Any],
    vectors: tuple[dict[str, Any], ...],
) -> None:
    unsupported = ph_model["unsupportedUntilTask54"]

    for reaction_id in unsupported["reactionIds"]:
        reaction = next(
            candidate
            for candidate in CATALOG["reactions"].values()
            if candidate["id"] == reaction_id
        )
        assert reaction["pH"] is None, reaction_id

        reagent_a, reagent_b = reaction["reactantSet"]
        result = evaluator.evaluate_experiment(build_input(reagent_a, (1, 50), reagent_b, (1, 50)))
        assert result["pH"] == unsupported["pH"], reaction_id
        assert formatter.format_experiment_quantities(result)["pH"] is None, reaction_id

    # The audited engine's seeded "~7.0" may not reappear anywhere, for a
    # supported mixture or an unsupported one.
    for vector in vectors:
        serialized = json.dumps(evaluate_vector(evaluator, vector), ensure_ascii=False)
        for forbidden in ph_model["forbiddenFabricatedValues"]:
            assert forbidden not in serialized, f"{vector['id']} {forbidden}"


def test_a_cologarithm_needs_a_positive_concentration() -> None:
    assert negative_log10(1e-7) == 7
    for value in (0, -1):
        with pytest.raises(ValueError, match="positive concentration"):
            negative_log10(value)
