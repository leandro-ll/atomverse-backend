"""Property 32: Candidate enumeration is bounded, fully accounted, and deterministic.

Validates: Requirements 14.2, 14.3, 14.4
"""

from __future__ import annotations

import random
import pytest

from app.domain.inverse_solver import (
    Constraints,
    ExclusionReason,
    Goal,
    InversePolicy,
    enumerate_candidates,
)
from app.generated.chemistry_catalog import CATALOG


# Feature: chemistry-species-expansion, Property 32: Candidate enumeration is bounded, fully accounted, and deterministic
def test_inverse_enumeration_property() -> None:
    reactions = CATALOG["reactions"]
    total_space = len(reactions)
    assert total_space > 0, "Catalog must contain reactions"

    all_reagents = sorted({r for rxn in reactions for r in rxn.get("reactantIds", ())})
    assert len(all_reagents) > 0, "Expected reagents from catalog reactions"

    rng = random.Random(4242)
    iterations = 100

    goals = [
        Goal(kind="precipitate"),
        Goal(kind="precipitate", target_color="kuning"),
        Goal(kind="precipitate", target_color="putih"),
        Goal(kind="precipitate", target_product_id="sub.agcl"),
        Goal(kind="precipitate", target_product_id="sub.pbi2"),
        Goal(kind="precipitate", target_product_id="sub.baso4"),
        Goal(kind="neutralization"),
        Goal(kind="any"),
    ]

    for i in range(iterations):
        goal = rng.choice(goals)
        budget = rng.randint(5, 60)
        policy = InversePolicy(maxInverseCandidates=budget)

        # Generate optional constraints
        forbidden = ()
        if rng.random() < 0.5:
            num_forbidden = rng.randint(1, 4)
            forbidden = tuple(rng.sample(all_reagents, num_forbidden))

        constraints = Constraints(forbidden_reagents=forbidden)

        enumeration1 = enumerate_candidates(CATALOG, goal, constraints, policy)
        enumeration2 = enumerate_candidates(CATALOG, goal, constraints, policy)

        # 1. Total accounting: retained + excluded == total space size
        assert len(enumeration1.retained) + len(enumeration1.excluded) == total_space, (
            f"Iter {i}: retained ({len(enumeration1.retained)}) + excluded ({len(enumeration1.excluded)}) "
            f"!= space size ({total_space})"
        )
        assert enumeration1.space_size == total_space

        # 2. Bounded retention
        assert len(enumeration1.retained) <= budget, (
            f"Iter {i}: retained count {len(enumeration1.retained)} exceeded budget {budget}"
        )

        # 3. Disjointness: no candidate is both retained and excluded
        retained_keys = {r["canonicalKey"] for r in enumeration1.retained}
        excluded_keys = {item[0] for item in enumeration1.excluded}
        overlap = retained_keys.intersection(excluded_keys)
        assert not overlap, f"Iter {i}: overlapping keys between retained and excluded: {overlap}"

        # 4. Valid exclusion reasons
        for key, reason in enumeration1.excluded:
            assert isinstance(reason, ExclusionReason), (
                f"Iter {i}: key {key} has non-ExclusionReason reason: {reason}"
            )
            assert reason in ExclusionReason

        # 5. Determinism: identical calls yield identical ordering and exclusions
        retained_keys1 = [r["canonicalKey"] for r in enumeration1.retained]
        retained_keys2 = [r["canonicalKey"] for r in enumeration2.retained]
        assert retained_keys1 == retained_keys2, f"Iter {i}: non-deterministic retained ordering"
        assert enumeration1.excluded == enumeration2.excluded, f"Iter {i}: non-deterministic exclusions"
