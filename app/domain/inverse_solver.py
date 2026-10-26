"""Inverse experiment solver and candidate pre-filter at expanded scale.

Builds on Requirement 14 (Audit Task 12):
- Evaluates approximately 990 canonical reagent pairs over the expanded library.
- Applies seeded solubility rules and catalog facts as a three-stage pre-filter
  before any quantitative evaluation.
- Records a stable, machine-readable ExclusionReason for every dropped candidate.
- Produces deterministic, total candidate ordering ending in canonicalKey.
- Enforces policy candidate bounds with CANDIDATE_BUDGET_EXCEEDED.
- Recalculates surviving recommendations through the pure evaluator and
  verifies them against constraints immediately before serialization (verify_before_serialize).
- Returns typed non-recommendations with no configuration and low confidence when
  infeasible — never a default AgNO3/NaCl.

Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5, 14.6
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Optional, Sequence

from app.domain.chemistry_evaluator import evaluate_experiment
from app.generated.chemistry_catalog import CATALOG


class ExclusionReason(StrEnum):
    """Closed enumeration of reasons why an inverse candidate is excluded."""

    NO_MODELED_REACTION = "no_modeled_reaction"
    UNSUPPORTED_CHEMISTRY = "unsupported_chemistry"
    NO_CITED_KSP = "no_cited_ksp"
    REAGENT_FORBIDDEN = "reagent_forbidden"
    REAGENT_NOT_ALLOWED = "reagent_not_allowed"
    GOAL_INCOMPATIBLE_PRODUCT = "goal_incompatible_product"
    GOAL_INCOMPATIBLE_COLOR = "goal_incompatible_color"
    CONCENTRATION_INFEASIBLE = "concentration_infeasible"
    VOLUME_INFEASIBLE = "volume_infeasible"
    CANDIDATE_BUDGET_EXCEEDED = "candidate_budget_exceeded"


@dataclass(frozen=True)
class Goal:
    """Target outcome for inverse solving."""

    kind: str = "precipitate"  # "precipitate", "neutralization", "color", "any"
    target_product_id: Optional[str] = None
    target_color: Optional[str] = None
    target_ph: Optional[float] = None


@dataclass(frozen=True)
class Constraints:
    """Learner or teacher constraints on acceptable reactions and conditions."""

    forbidden_reagents: tuple[str, ...] = ()
    allowed_reagents: Optional[tuple[str, ...]] = None
    min_concentration: Optional[float] = None
    max_concentration: Optional[float] = None
    min_volume: Optional[float] = None
    max_volume: Optional[float] = None


@dataclass(frozen=True)
class InversePolicy:
    """Solver budget and configuration defaults."""

    maxInverseCandidates: int = 50
    default_concentration_m: float = 0.1
    default_volume_ml: float = 50.0
    default_temperature_c: float = 25.0


@dataclass(frozen=True)
class CandidateEnumeration:
    """Result of candidate pre-filtering and bounded retention."""

    retained: tuple[Mapping[str, Any], ...]
    excluded: tuple[tuple[str, ExclusionReason], ...]
    space_size: int


@dataclass(frozen=True)
class ReactionConfiguration:
    reactant_a: str
    conc_a: float
    vol_a: float
    reactant_b: str
    conc_b: float
    vol_b: float
    temp: float
    pressure: float = 1.0


@dataclass(frozen=True)
class PredictedOutcome:
    color: str
    precipitate: Optional[str] = None
    gas: Optional[str] = None
    ph_final: str = "7.0"


@dataclass(frozen=True)
class InverseCandidate:
    reaction_id: str
    canonical_key: str
    reagent_a_id: str
    reagent_b_id: str
    configuration: ReactionConfiguration
    predicted_outcome: str
    product_id: Optional[str] = None
    color: str = "bening"
    confidence_score: int = 90
    explanation: str = ""
    result: Optional[Mapping[str, Any]] = None

    def with_result(self, fresh_result: Mapping[str, Any]) -> InverseCandidate:
        return InverseCandidate(
            reaction_id=self.reaction_id,
            canonical_key=self.canonical_key,
            reagent_a_id=self.reagent_a_id,
            reagent_b_id=self.reagent_b_id,
            configuration=self.configuration,
            predicted_outcome=self.predicted_outcome,
            product_id=self.product_id,
            color=self.color,
            confidence_score=self.confidence_score,
            explanation=self.explanation,
            result=fresh_result,
        )


@dataclass(frozen=True)
class InverseSolution:
    """Final solution response, either a verified recommendation or a typed non-recommendation."""

    is_feasible: bool
    confidence_score: int
    recommended_reaction: str
    configuration: Optional[ReactionConfiguration]
    predicted_outcome: Optional[PredictedOutcome]
    explanation: str
    exclusion_reasons: tuple[tuple[str, ExclusionReason], ...] = ()


def first_constraint_violation(
    rxn: Mapping[str, Any], constraints: Optional[Constraints]
) -> Optional[ExclusionReason]:
    """Stage 2: Checks candidate reactants against stated reactant constraints."""
    if constraints is None:
        return None

    reactant_ids = tuple(rxn.get("reactantIds", ()))
    if constraints.forbidden_reagents:
        forbidden = set(constraints.forbidden_reagents)
        if any(r in forbidden for r in reactant_ids):
            return ExclusionReason.REAGENT_FORBIDDEN

    if constraints.allowed_reagents is not None:
        allowed = set(constraints.allowed_reagents)
        if any(r not in allowed for r in reactant_ids):
            return ExclusionReason.REAGENT_NOT_ALLOWED

    return None


def goal_incompatibility(
    rxn: Mapping[str, Any], goal: Goal, catalog: Mapping[str, Any]
) -> Optional[ExclusionReason]:
    """Stage 3: Checks candidate outcome compatibility with stated goal."""
    rxn_outcome = rxn.get("outcome")

    if goal.kind == "neutralization":
        if rxn_outcome != "neutralization":
            return ExclusionReason.GOAL_INCOMPATIBLE_PRODUCT
        return None

    if goal.kind == "precipitate":
        candidate_solid = rxn.get("candidateSolidId")
        if not candidate_solid:
            return ExclusionReason.GOAL_INCOMPATIBLE_PRODUCT

        if goal.target_product_id:
            # Match against substance id or formula token
            substances = catalog.get("substances", ())
            sub_map = {s["id"]: s for s in substances} if isinstance(substances, (list, tuple)) else {}
            target_id = goal.target_product_id
            if target_id not in sub_map:
                for s in sub_map.values():
                    if s.get("formulaToken") == target_id:
                        target_id = s["id"]
                        break
            if candidate_solid != target_id:
                return ExclusionReason.GOAL_INCOMPATIBLE_PRODUCT

        if goal.target_color:
            target_color_lower = goal.target_color.lower().strip()
            sub_colors = catalog.get("presentation", {}).get("substanceColors", {})
            hex_color = (sub_colors.get(candidate_solid) or "").lower()

            known_colors = {
                "putih": {"#ffffff", "#f8fafc"},
                "kuning": {"#eab308", "#facc15", "#fef08a"},
                "hitam": {"#0f172a", "#1e293b"},
                "coklat": {"#b45309"},
                "biru": {"#38bdf8", "#0284c7"},
                "hijau": {"#10b981"},
            }
            if target_color_lower in known_colors:
                matching_hexes = known_colors[target_color_lower]
                if hex_color not in matching_hexes:
                    return ExclusionReason.GOAL_INCOMPATIBLE_COLOR
            else:
                if not hex_color or target_color_lower != hex_color:
                    return ExclusionReason.GOAL_INCOMPATIBLE_COLOR

    return None


def goal_affinity(rxn: Mapping[str, Any], goal: Goal, catalog: Mapping[str, Any]) -> int:
    """Integer affinity score for candidate sorting (higher first)."""
    score = 0
    candidate_solid = rxn.get("candidateSolidId")

    if goal.target_product_id and candidate_solid == goal.target_product_id:
        score += 100

    if goal.kind == "precipitate" and rxn.get("outcome") == "precipitate_forms":
        score += 20
    elif goal.kind == "neutralization" and rxn.get("outcome") == "neutralization":
        score += 20

    return score


def solubility_rank(rxn: Mapping[str, Any], catalog: Mapping[str, Any]) -> int:
    """Integer solubility rank based on Ksp exponent magnitude (higher rank = less soluble = stronger precipitation)."""
    ksp_ref = rxn.get("kspRef")
    if not ksp_ref or not isinstance(ksp_ref, dict):
        return 0

    val_str = str(ksp_ref.get("value", ""))
    if "e-" in val_str:
        try:
            exp_str = val_str.split("e-")[1]
            return int(exp_str)
        except (ValueError, IndexError):
            pass
    return 0


def enumerate_candidates(
    catalog: Mapping[str, Any],
    goal: Goal,
    constraints: Optional[Constraints] = None,
    policy: Optional[InversePolicy] = None,
) -> CandidateEnumeration:
    """Three-stage rule-based pre-filter over all canonical pairs in the catalog.

    Stage 1: Catalog outcome (unsupported, no_modeled_reaction, missing Ksp)
    Stage 2: Stated constraints (forbidden or allowed reagents)
    Stage 3: Goal compatibility (product identity, color, reaction class)
    """
    if policy is None:
        policy = InversePolicy()

    reactions = catalog.get("reactions", ())
    retained: list[Mapping[str, Any]] = []
    excluded: list[tuple[str, ExclusionReason]] = []

    for rxn in reactions:
        canon_key = rxn.get("canonicalKey", "")

        # Stage 1: Catalog outcome facts
        outcome = rxn.get("outcome")
        if outcome == "unsupported":
            excluded.append((canon_key, ExclusionReason.UNSUPPORTED_CHEMISTRY))
            continue
        if outcome == "no_modeled_reaction":
            excluded.append((canon_key, ExclusionReason.NO_MODELED_REACTION))
            continue
        if goal.kind == "precipitate" and rxn.get("kspRef") is None:
            excluded.append((canon_key, ExclusionReason.NO_CITED_KSP))
            continue

        # Stage 2: Stated constraints
        bad_constraint = first_constraint_violation(rxn, constraints)
        if bad_constraint:
            excluded.append((canon_key, bad_constraint))
            continue

        # Stage 3: Goal compatibility
        incompatible = goal_incompatibility(rxn, goal, catalog)
        if incompatible:
            excluded.append((canon_key, incompatible))
            continue

        retained.append(rxn)

    # Sort on integer-only total key ending in canonicalKey
    retained.sort(
        key=lambda r: (
            -goal_affinity(r, goal, catalog),
            -solubility_rank(r, catalog),
            r.get("canonicalKey", ""),
        )
    )

    # Truncate at policy bound
    if len(retained) > policy.maxInverseCandidates:
        for rxn in retained[policy.maxInverseCandidates :]:
            excluded.append(
                (rxn.get("canonicalKey", ""), ExclusionReason.CANDIDATE_BUDGET_EXCEEDED)
            )
        retained = retained[: policy.maxInverseCandidates]

    return CandidateEnumeration(
        retained=tuple(retained),
        excluded=tuple(sorted(excluded, key=lambda item: (item[0], item[1]))),
        space_size=len(reactions),
    )


def verify_before_serialize(
    candidate: InverseCandidate,
    constraints: Optional[Constraints],
    catalog: Mapping[str, Any],
) -> Optional[InverseCandidate]:
    """Recalculates candidate through the pure evaluator and checks stated constraints."""
    config = candidate.configuration
    experiment_input = {
        "reagentAId": candidate.reagent_a_id,
        "reagentBId": candidate.reagent_b_id,
        "concentrationA_M": config.conc_a,
        "volumeA_mL": config.vol_a,
        "concentrationB_M": config.conc_b,
        "volumeB_mL": config.vol_b,
        "temperatureC": config.temp,
    }

    try:
        fresh = evaluate_experiment(catalog, experiment_input)
    except Exception:
        return None

    fresh_outcome = fresh.get("outcome")
    if fresh_outcome != candidate.predicted_outcome:
        return None

    # Check concentration & volume constraints on inputs if defined
    if constraints:
        if constraints.min_concentration is not None and (
            config.conc_a < constraints.min_concentration
            or config.conc_b < constraints.min_concentration
        ):
            return None
        if constraints.max_concentration is not None and (
            config.conc_a > constraints.max_concentration
            or config.conc_b > constraints.max_concentration
        ):
            return None
        if constraints.min_volume is not None and (
            config.vol_a < constraints.min_volume or config.vol_b < constraints.min_volume
        ):
            return None
        if constraints.max_volume is not None and (
            config.vol_a > constraints.max_volume or config.vol_b > constraints.max_volume
        ):
            return None

    return candidate.with_result(fresh)


def solve_inverse_experiment(
    goal: Goal,
    constraints: Optional[Constraints] = None,
    catalog: Mapping[str, Any] = CATALOG,
    policy: Optional[InversePolicy] = None,
) -> InverseSolution:
    """Top-level deterministic inverse solver.

    Pre-filters candidates, builds test configurations, reverifies surviving recommendations,
    and returns a typed non-recommendation when no candidates are feasible.
    """
    if policy is None:
        policy = InversePolicy()

    enumeration = enumerate_candidates(catalog, goal, constraints, policy)

    if not enumeration.retained:
        return InverseSolution(
            is_feasible=False,
            confidence_score=0,
            recommended_reaction="Tidak ada reaksi yang memenuhi kriteria",
            configuration=None,
            predicted_outcome=None,
            explanation="Tidak ditemukan kombinasi pereaksi yang memenuhi tujuan dan batasan yang diberikan.",
            exclusion_reasons=enumeration.excluded,
        )

    # Walk through retained candidates in deterministic sorted order
    sub_colors = catalog.get("presentation", {}).get("substanceColors", {})

    for rxn in enumeration.retained:
        reactants = rxn.get("reactantIds", ())
        if len(reactants) != 2:
            continue
        reagent_a, reagent_b = reactants[0], reactants[1]
        solid_id = rxn.get("candidateSolidId")
        color_hex = sub_colors.get(solid_id, "#ffffff") if solid_id else "#e2e8f0"

        # Determine expected outcome
        expected_outcome = (
            "precipitate_forms"
            if goal.kind == "precipitate" or solid_id
            else "neutralization"
            if goal.kind == "neutralization" or rxn.get("outcome") == "neutralization"
            else rxn.get("outcome", "no_modeled_reaction")
        )

        conc_a = policy.default_concentration_m
        conc_b = policy.default_concentration_m

        # Adjust stoichiometry if known
        rxn_reactants = rxn.get("reactants", {})
        coeff_a = rxn_reactants.get(reagent_a, 1)
        coeff_b = rxn_reactants.get(reagent_b, 1)
        if coeff_a > coeff_b:
            conc_a = round(conc_b * (coeff_a / coeff_b), 4)
        elif coeff_b > coeff_a:
            conc_b = round(conc_a * (coeff_b / coeff_a), 4)

        config = ReactionConfiguration(
            reactant_a=reagent_a,
            conc_a=conc_a,
            vol_a=policy.default_volume_ml,
            reactant_b=reagent_b,
            conc_b=conc_b,
            vol_b=policy.default_volume_ml,
            temp=policy.default_temperature_c,
            pressure=1.0,
        )

        candidate = InverseCandidate(
            reaction_id=rxn.get("id", ""),
            canonical_key=rxn.get("canonicalKey", ""),
            reagent_a_id=reagent_a,
            reagent_b_id=reagent_b,
            configuration=config,
            predicted_outcome=expected_outcome,
            product_id=solid_id,
            color=color_hex,
            confidence_score=95,
            explanation=rxn.get("molecularEquation", "Reaksi kimia terverifikasi."),
        )

        verified = verify_before_serialize(candidate, constraints, catalog)
        if verified is not None:
            solid_name = solid_id
            substances = catalog.get("substances", ())
            if isinstance(substances, (list, tuple)):
                for s in substances:
                    if s.get("id") == solid_id:
                        solid_name = s.get("indonesianName", solid_id)
                        break

            pred_outcome = PredictedOutcome(
                color="keruh" if expected_outcome == "precipitate_forms" else "bening",
                precipitate=solid_name if expected_outcome == "precipitate_forms" else None,
                gas=None,
                ph_final="7.0",
            )

            rec_name = rxn.get("molecularEquation") or f"Reaksi {reagent_a} dan {reagent_b}"
            return InverseSolution(
                is_feasible=True,
                confidence_score=verified.confidence_score,
                recommended_reaction=rec_name,
                configuration=verified.configuration,
                predicted_outcome=pred_outcome,
                explanation=verified.explanation,
                exclusion_reasons=enumeration.excluded,
            )

    # If all candidates failed verification
    return InverseSolution(
        is_feasible=False,
        confidence_score=0,
        recommended_reaction="Tidak ada reaksi yang memenuhi kriteria",
        configuration=None,
        predicted_outcome=None,
        explanation="Kandidat reaksi tidak lolos verifikasi evaluasi kimia.",
        exclusion_reasons=enumeration.excluded,
    )
