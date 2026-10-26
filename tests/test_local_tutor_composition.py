"""Property 35: Fallback answers are composed from catalog facts about the selected pair.

Validates: Requirements 15.1, 15.2, 15.3, 15.5
"""

from __future__ import annotations

from pathlib import Path
import random
import pytest

from app.services.local_tutor import (
    FallbackIntent,
    build_chat_fallback,
    classify_intent,
)
from app.domain.chemistry_evaluator import evaluate_experiment
from app.generated.chemistry_catalog import CATALOG, CATALOG_VERSION

BACKEND_MAIN = Path(__file__).resolve().parents[1] / "main.py"


def test_three_hardcoded_strings_absent_from_backend_main() -> None:
    """Requirement 15.1, 19.4: Hardcoded explanations must be strictly absent from main.py."""
    content = BACKEND_MAIN.read_text(encoding="utf-8")

    banned_snippets = [
        "membentuk endapan putih AgCl",
        "menghasilkan endapan kuning PbI",
        "mengalami netralisasi eksoterm sehingga menghasilkan garam dan air",
    ]
    for snippet in banned_snippets:
        assert snippet not in content, f"Hardcoded explanation snippet found in main.py: {snippet}"


# Feature: chemistry-species-expansion, Property 35: Fallback answers are composed from catalog facts about the selected pair
def test_local_tutor_composition_property() -> None:
    reactions = [r for r in CATALOG["reactions"] if r.get("outcome") != "unsupported"]
    assert len(reactions) > 0

    rng = random.Random(3535)
    iterations = 100

    prompts = [
        "Apa hasil reaksi ini?",
        "Tuliskan persamaan reaksinya",
        "Siapa pereaksi pembatasnya?",
        "Apakah campuran ini sudah jenuh atau mengendap?",
        "Berapa perkiraan pH campuran?",
        "Bagaimana gerak partikelnya di dalam larutan?",
        "Apa batasan model pada simulasi ini?",
    ]

    for i in range(iterations):
        rxn = rng.choice(reactions)
        reagent_a, reagent_b = rxn["reactantIds"]
        prompt = rng.choice(prompts)

        lab_state = {
            "reagentA": reagent_a,
            "reagentB": reagent_b,
            "mA": rng.uniform(0.1, 1.0),
            "vA": rng.uniform(20.0, 80.0),
            "mB": rng.uniform(0.1, 1.0),
            "vB": rng.uniform(20.0, 80.0),
            "temp": 25.0,
        }

        eval_result = evaluate_experiment(CATALOG, {
            "reagentAId": reagent_a,
            "reagentBId": reagent_b,
            "concentrationA_M": lab_state["mA"],
            "volumeA_mL": lab_state["vA"],
            "concentrationB_M": lab_state["mB"],
            "volumeB_mL": lab_state["vB"],
            "temperatureC": lab_state["temp"],
        })

        reply = build_chat_fallback(prompt, lab_state, result=eval_result, catalog=CATALOG)

        # 1. Truthful metadata attached
        assert reply.source == "fallback"
        assert reply.scope == "answered"
        assert reply.catalog_version == CATALOG_VERSION
        assert reply.degraded is True
        assert reply.reason_code == "provider_unavailable"
        assert len(reply.text) > 10

        # 2. Tracks outcome category
        outcome = eval_result["outcome"]
        if outcome == "precipitate_forms":
            solid_id = eval_result.get("candidateSolidId") or eval_result.get("productId")
            if solid_id:
                sub_dict = {s["id"]: s for s in CATALOG.get("substances", ())}
                solid_name = sub_dict[solid_id]["indonesianName"]
                if classify_intent(prompt, eval_result) == FallbackIntent.CURRENT_OUTCOME:
                    assert solid_name.lower() in reply.text.lower()
        elif outcome == "below_saturation":
            if classify_intent(prompt, eval_result) in (FallbackIntent.SATURATION, FallbackIntent.CURRENT_OUTCOME):
                # Must state Q and Ksp comparison citing authored Ksp literal
                assert "Ksp" in reply.text
                assert "Q" in reply.text
                if eval_result.get("kspRef"):
                    assert eval_result["kspRef"]["value"] in reply.text
        elif outcome == "no_modeled_reaction":
            if classify_intent(prompt, eval_result) == FallbackIntent.CURRENT_OUTCOME:
                assert "tidak memiliki reaksi" in reply.text or "tidak menunjukkan reaksi" in reply.text
