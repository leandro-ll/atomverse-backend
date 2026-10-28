"""Inverse Experiment must only output reactions the Reaction Lab can simulate."""

from __future__ import annotations

import pytest

from app.services.lab_reactions import (
    LAB_REAGENTS,
    REACTIONS_BY_ID,
    SUPPORTED_REACTIONS,
    find_unsupported_terms,
    match_goal,
)
from inverse_engine import build_local_response, resolve_ai_choice


def test_every_reaction_uses_only_lab_reagents() -> None:
    for reaction in SUPPORTED_REACTIONS:
        assert reaction.reactant_a in LAB_REAGENTS, reaction.id
        assert reaction.reactant_b in LAB_REAGENTS, reaction.id
        assert reaction.reactant_a != reaction.reactant_b, reaction.id


def test_reaction_ids_and_pairs_are_unique() -> None:
    assert len(REACTIONS_BY_ID) == len(SUPPORTED_REACTIONS)
    pairs = [frozenset((r.reactant_a, r.reactant_b)) for r in SUPPORTED_REACTIONS]
    assert len(set(pairs)) == len(pairs)


@pytest.mark.parametrize(
    ("goal", "expected_type", "expected_product"),
    [
        ("Produce effervescent carbon dioxide (CO2) gas", "gas", "CO2"),
        ("Form a yellow precipitate (PbI2)", "precipitate", "PbI2"),
        ("Form a white precipitate of Silver Chloride (AgCl)", "precipitate", "AgCl"),
        ("Neutralize acid and base to reach neutral pH 7.0", "neutralization", "H2O"),
        ("Form white chalk precipitate (CaCO3)", "precipitate", "CaCO3"),
        ("Release ammonia gas (NH3)", "gas", "NH3"),
        ("mix vinegar and baking soda", "gas", "CO2"),
        ("buat endapan kuning", "precipitate", "PbI2"),
        ("menghasilkan gelembung gas CO₂", "gas", "CO2"),
        ("sodium hydrogen carbonate with hydrochloric acid", "gas", "CO2"),
    ],
)
def test_supported_goals_map_to_expected_reaction(goal: str, expected_type: str, expected_product: str) -> None:
    result = match_goal(goal)
    assert result.reaction is not None, goal
    assert result.reaction.type == expected_type
    assert result.reaction.product == expected_product


def test_vinegar_prefers_acetic_acid_pair() -> None:
    result = match_goal("vinegar and baking soda volcano")
    assert result.reaction is not None
    assert {result.reaction.reactant_a, result.reaction.reactant_b} == {"CH3COOH", "NaHCO3"}


@pytest.mark.parametrize(
    "goal",
    [
        "make hydrogen gas",
        "Form a BaBr2 precipitate",
        "form babr2",
        "blue precipitate of copper hydroxide",
        "produce oxygen",
        "make CO gas",
        "something completely random",
    ],
)
def test_unsupported_goals_return_no_reaction(goal: str) -> None:
    result = match_goal(goal)
    assert result.reaction is None, goal
    response = build_local_response(goal)
    assert response.supported is False
    assert response.configuration is None
    assert response.suggestions


def test_known_formulas_are_not_flagged() -> None:
    assert find_unsupported_terms("Mix Pb(NO3)2 with KI to get PbI2 and KNO3") == []
    assert find_unsupported_terms("Ba²⁺ + SO₄²⁻ → BaSO₄") == []
    assert find_unsupported_terms("Neutral pH 7.0 with HCl and NaOH") == []


def test_supported_response_is_built_from_database() -> None:
    response = build_local_response("Produce effervescent carbon dioxide (CO2) gas")
    assert response.supported is True
    assert response.configuration is not None
    assert response.configuration.reactant_a in LAB_REAGENTS
    assert response.configuration.reactant_b in LAB_REAGENTS
    assert response.predicted_outcome is not None
    assert response.predicted_outcome.gas == "CO2"
    assert response.reaction_id in REACTIONS_BY_ID
    assert 0.1 <= response.configuration.conc_a <= 2.0
    assert 10 <= response.configuration.vol_a <= 100


def test_ai_invented_reaction_is_rejected() -> None:
    goal = "Produce effervescent carbon dioxide (CO2) gas"
    local = match_goal(goal)
    response = resolve_ai_choice({"reaction_id": "babr2_agno3", "confidence": 95}, local, goal, None)
    assert response.source == "local"
    assert response.reaction_id in REACTIONS_BY_ID
    assert response.predicted_outcome is not None and response.predicted_outcome.gas == "CO2"


def test_ai_wrong_type_is_overridden_by_strong_local_match() -> None:
    goal = "Produce effervescent carbon dioxide (CO2) gas"
    local = match_goal(goal)
    response = resolve_ai_choice({"reaction_id": "agcl_nacl", "confidence": 95}, local, goal, None)
    assert response.reaction_id is not None
    assert REACTIONS_BY_ID[response.reaction_id].type == "gas"


def test_ai_valid_pick_is_used() -> None:
    goal = "something that makes a cake rise"
    local = match_goal(goal)
    response = resolve_ai_choice({"reaction_id": "co2_ch3cooh_nahco3", "confidence": 88}, local, goal, None)
    assert response.source == "gemini"
    assert response.reaction_id == "co2_ch3cooh_nahco3"
    assert response.confidence_score == 88


def test_ai_unsupported_without_local_match_is_unsupported() -> None:
    goal = "something completely random"
    local = match_goal(goal)
    response = resolve_ai_choice({"reaction_id": "UNSUPPORTED", "confidence": 90}, local, goal, None)
    assert response.supported is False
