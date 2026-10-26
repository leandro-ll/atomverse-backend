"""Property 33: Recommendations survive re-evaluation.

Validates: Requirements 14.5
- Every recommendation returned must have been recalculated through the pure evaluator.
- Immediate constraint re-checking: any divergence or constraint failure drops the candidate.
"""

from __future__ import annotations

import random
import pytest

from app.domain.inverse_solver import (
    Constraints,
    Goal,
    InverseCandidate,
    ReactionConfiguration,
    enumerate_candidates,
    verify_before_serialize,
)
from app.domain.chemistry_evaluator import evaluate_experiment
from app.generated.chemistry_catalog import CATALOG


# Feature: chemistry-species-expansion, Property 33: Recommendations survive re-evaluation
def test_inverse_reverification_property() -> None:
    enumeration = enumerate_candidates(CATALOG, Goal(kind="precipitate"))
    assert len(enumeration.retained) > 0, "Expected retained precipitate candidates"

    rng = random.Random(3333)
    iterations = 100

    for i in range(iterations):
        rxn = rng.choice(enumeration.retained)
        reagent_a, reagent_b = rxn["reactantIds"]

        c_a = rng.uniform(0.05, 0.5)
        c_b = rng.uniform(0.05, 0.5)
        v_a = rng.uniform(20.0, 80.0)
        v_b = rng.uniform(20.0, 80.0)

        config = ReactionConfiguration(
            reactant_a=reagent_a,
            conc_a=c_a,
            vol_a=v_a,
            reactant_b=reagent_b,
            conc_b=c_b,
            vol_b=v_b,
            temp=25.0,
            pressure=1.0,
        )

        candidate = InverseCandidate(
            reaction_id=rxn["id"],
            canonical_key=rxn["canonicalKey"],
            reagent_a_id=reagent_a,
            reagent_b_id=reagent_b,
            configuration=config,
            predicted_outcome="precipitate_forms",
            product_id=rxn.get("candidateSolidId"),
        )

        # Case 1: Compatible constraints
        constraints_ok = Constraints(
            min_concentration=0.01,
            max_concentration=2.0,
            min_volume=10.0,
            max_volume=100.0,
        )
        verified = verify_before_serialize(candidate, constraints_ok, CATALOG)
        if verified is not None:
            # Pure evaluator must agree
            eval_input = {
                "reagentAId": reagent_a,
                "reagentBId": reagent_b,
                "concentrationA_M": c_a,
                "volumeA_mL": v_a,
                "concentrationB_M": c_b,
                "volumeB_mL": v_b,
                "temperatureC": 25.0,
            }
            fresh = evaluate_experiment(CATALOG, eval_input)
            assert fresh["outcome"] == "precipitate_forms"
            assert verified.result is not None
            assert verified.result["outcome"] == "precipitate_forms"

        # Case 2: Incompatible constraint forces rejection (verify_before_serialize returns None)
        # e.g., max_concentration below current c_a
        constraints_fail = Constraints(max_concentration=min(c_a, c_b) - 0.01)
        rejected = verify_before_serialize(candidate, constraints_fail, CATALOG)
        assert rejected is None, f"Iter {i}: expected rejection when max_concentration violated"

        # Case 3: Incompatible predicted outcome forces rejection
        divergent_candidate = InverseCandidate(
            reaction_id=rxn["id"],
            canonical_key=rxn["canonicalKey"],
            reagent_a_id=reagent_a,
            reagent_b_id=reagent_b,
            configuration=config,
            predicted_outcome="neutralization",  # deliberately wrong predicted outcome
            product_id=rxn.get("candidateSolidId"),
        )
        rejected_outcome = verify_before_serialize(divergent_candidate, None, CATALOG)
        assert rejected_outcome is None, f"Iter {i}: expected rejection on outcome divergence"
