"""Property 30: Provider output cannot change chemistry.

Validates: Requirements 16.4
- The system resolves every chemistry fact from the generated catalog and
  does NOT treat provider output as authoritative.
- Provider hallucinations cannot alter the pure evaluator outcome, balanced
  equations, precipitate identities, or Ksp values.
"""

from __future__ import annotations

import random
import pytest

from app.domain.chemistry_evaluator import evaluate_experiment
from app.domain.inverse_solver import verify_before_serialize, InverseCandidate, ReactionConfiguration
from app.generated.chemistry_catalog import CATALOG


# Feature: chemistry-species-expansion, Property 30: Provider output cannot change chemistry
def test_provider_non_authority_property() -> None:
    reactions = [r for r in CATALOG["reactions"] if r.get("outcome") == "no_modeled_reaction"]
    assert len(reactions) > 0, "Expected no_modeled_reaction records in catalog"

    rng = random.Random(3030)
    iterations = 100

    hallucinated_claims = [
        {"outcome": "precipitate_forms", "product": "sub.hallucinated_ppt", "equation": "A + B -> Magic(s)"},
        {"outcome": "precipitate_forms", "product": "sub.agcl", "equation": "NaCl + KNO3 -> AgCl(s)"},
        {"outcome": "neutralization", "product": "sub.h2o", "equation": "NaCl + KI -> H2O(l)"},
    ]

    for i in range(iterations):
        rxn = rng.choice(reactions)
        reagent_a, reagent_b = rxn["reactantIds"]
        claim = rng.choice(hallucinated_claims)

        # 1. Pure evaluator check: ground-truth outcome cannot be overridden by provider claim
        eval_result = evaluate_experiment(CATALOG, {
            "reagentAId": reagent_a,
            "reagentBId": reagent_b,
            "concentrationA_M": 0.5,
            "volumeA_mL": 50.0,
            "concentrationB_M": 0.5,
            "volumeB_mL": 50.0,
            "temperatureC": 25.0,
        })
        assert eval_result["outcome"] == "no_modeled_reaction", (
            f"Iter {i}: pure evaluator altered by external context!"
        )

        # 2. Inverse solver verification check: provider claiming invalid candidate is rejected
        bogus_candidate = InverseCandidate(
            reaction_id=rxn["id"],
            canonical_key=rxn["canonicalKey"],
            reagent_a_id=reagent_a,
            reagent_b_id=reagent_b,
            configuration=ReactionConfiguration(
                reactant_a=reagent_a,
                conc_a=0.5,
                vol_a=50.0,
                reactant_b=reagent_b,
                conc_b=0.5,
                vol_b=50.0,
                temp=25.0,
            ),
            predicted_outcome=claim["outcome"],
            product_id=claim["product"],
        )
        verified = verify_before_serialize(bogus_candidate, None, CATALOG)
        assert verified is None, (
            f"Iter {i}: verify_before_serialize accepted provider hallucination '{claim['outcome']}' for non-reacting pair {rxn['canonicalKey']}!"
        )
