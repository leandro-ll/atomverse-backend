"""Property 34: Infeasibility yields a typed non-recommendation.

Validates: Requirements 14.6
- If pre-filtering leaves no feasible candidate, the system returns a typed
  non-recommendation with no configuration and no high confidence score.
- Never returns a default AgNO3/NaCl recommendation.
"""

from __future__ import annotations

import random
import pytest

from app.domain.inverse_solver import (
    Constraints,
    Goal,
    solve_inverse_experiment,
)
from app.generated.chemistry_catalog import CATALOG


# Feature: chemistry-species-expansion, Property 34: Infeasibility yields a typed non-recommendation
def test_inverse_infeasible_property() -> None:
    rng = random.Random(5555)
    iterations = 100

    all_reagents = sorted({r for rxn in CATALOG["reactions"] for r in rxn.get("reactantIds", ())})

    for i in range(iterations):
        scenario = rng.choice(["forbid_all", "impossible_product", "impossible_color", "empty_allowed"])

        if scenario == "forbid_all":
            # Forbid all reagents
            constraints = Constraints(forbidden_reagents=tuple(all_reagents))
            goal = Goal(kind="precipitate")
        elif scenario == "impossible_product":
            constraints = None
            goal = Goal(kind="precipitate", target_product_id=f"sub.nonexistent_{rng.randint(1000, 9999)}")
        elif scenario == "impossible_color":
            constraints = None
            goal = Goal(kind="precipitate", target_color=f"neon_purple_{rng.randint(1000, 9999)}")
        else:  # empty_allowed
            constraints = Constraints(allowed_reagents=())
            goal = Goal(kind="neutralization")

        solution = solve_inverse_experiment(goal, constraints, CATALOG)

        # Must be typed non-recommendation
        assert not solution.is_feasible, f"Iter {i} ({scenario}): expected infeasible solution"
        assert solution.configuration is None, f"Iter {i}: configuration must be None when infeasible"
        assert solution.predicted_outcome is None, f"Iter {i}: predicted_outcome must be None when infeasible"
        assert solution.confidence_score <= 30, f"Iter {i}: confidence must be low (got {solution.confidence_score})"

        # Must never return default AgNO3 / NaCl
        assert "Perak Klorida" not in solution.recommended_reaction, (
            f"Iter {i}: returned default AgNO3/NaCl on infeasible goal!"
        )
        assert "AgNO3" not in solution.recommended_reaction, (
            f"Iter {i}: returned default AgNO3 on infeasible goal!"
        )
        assert len(solution.exclusion_reasons) > 0, (
            f"Iter {i}: expected non-empty exclusion reasons when all candidates dropped"
        )
