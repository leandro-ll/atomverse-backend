import asyncio
import json
import os
import re
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import APIRouter
from google import genai
from pydantic import BaseModel, Field

load_dotenv()

router = APIRouter()
api_key = os.getenv("GEMINI_API_KEY")
genai_client = genai.Client(api_key=api_key) if api_key else None

if genai_client is None:
    print("WARNING: GEMINI_API_KEY tidak ditemukan; inverse endpoint akan memakai fallback lokal.")

CHEMICAL_KNOWLEDGE_BASE = [
    {
        "id": "rxn_001",
        "name": "Pembentukan Endapan Perak Klorida",
        "reactants": ["AgNO3", "NaCl"],
        "properties": {"color": "putih", "precipitate": "AgCl", "ph_final": "7.0"},
    },
    {
        "id": "rxn_002",
        "name": "Pembentukan Endapan Timbal Iodida",
        "reactants": ["Pb(NO3)2", "KI"],
        "properties": {"color": "kuning", "precipitate": "PbI2", "ph_final": "7.0"},
    },
    {
        "id": "rxn_003",
        "name": "Reaksi Netralisasi Asam Basa",
        "reactants": ["HCl", "NaOH"],
        "properties": {
            "color": "bening",
            "precipitate": None,
            "ph_final": "7.0",
            "energy": "Eksoterm",
        },
    },
]


class InverseRequest(BaseModel):
    goal: str = Field(min_length=1)
    constraints: Optional[Dict] = None


class ReactionConfiguration(BaseModel):
    reactant_a: str
    conc_a: float
    vol_a: float
    reactant_b: str
    conc_b: float
    vol_b: float
    temp: float
    pressure: float = 1


class PredictedOutcome(BaseModel):
    color: str
    precipitate: Optional[str] = None
    gas: Optional[str] = None
    ph_final: str


from app.domain.inverse_solver import Constraints, Goal, solve_inverse_experiment as solve_inverse_domain
from app.generated.chemistry_catalog import CATALOG


class InverseResponse(BaseModel):
    confidence_score: int = Field(ge=0, le=100)
    recommended_reaction: str
    configuration: Optional[ReactionConfiguration] = None
    predicted_outcome: Optional[PredictedOutcome] = None
    explanation: str


def build_fallback_response(goal_text: str, constraints_dict: Optional[Dict] = None) -> InverseResponse:
    goal_lower = goal_text.lower()
    kind = "precipitate"
    target_product = None
    target_color = None

    if "netral" in goal_lower or "ph" in goal_lower or ("asam" in goal_lower and "basa" in goal_lower):
        kind = "neutralization"
    if "kuning" in goal_lower:
        target_color = "kuning"
    elif "putih" in goal_lower:
        target_color = "putih"
    elif "hitam" in goal_lower:
        target_color = "hitam"
    elif "coklat" in goal_lower:
        target_color = "coklat"
    elif "biru" in goal_lower:
        target_color = "biru"

    goal = Goal(kind=kind, target_product_id=target_product, target_color=target_color)

    forbidden = ()
    allowed = None
    if constraints_dict:
        if "forbidden_reagents" in constraints_dict:
            forbidden = tuple(constraints_dict["forbidden_reagents"])
        if "allowed_reagents" in constraints_dict:
            allowed = tuple(constraints_dict["allowed_reagents"])

    constraints = Constraints(forbidden_reagents=forbidden, allowed_reagents=allowed)
    solution = solve_inverse_domain(goal, constraints, CATALOG)

    if not solution.is_feasible or solution.configuration is None or solution.predicted_outcome is None:
        return InverseResponse(
            confidence_score=solution.confidence_score,
            recommended_reaction=solution.recommended_reaction,
            configuration=None,
            predicted_outcome=None,
            explanation=solution.explanation,
        )

    cfg = solution.configuration
    out = solution.predicted_outcome
    return InverseResponse(
        confidence_score=solution.confidence_score,
        recommended_reaction=solution.recommended_reaction,
        configuration=ReactionConfiguration(
            reactant_a=cfg.reactant_a,
            conc_a=cfg.conc_a,
            vol_a=cfg.vol_a,
            reactant_b=cfg.reactant_b,
            conc_b=cfg.conc_b,
            vol_b=cfg.vol_b,
            temp=cfg.temp,
            pressure=cfg.pressure,
        ),
        predicted_outcome=PredictedOutcome(
            color=out.color,
            precipitate=out.precipitate,
            gas=out.gas,
            ph_final=out.ph_final,
        ),
        explanation=solution.explanation,
    )


def extract_json(response_text: str) -> dict:
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise ValueError("AI tidak mengembalikan objek JSON")
    return json.loads(match.group())


@router.post("/api/inverse-experiment", response_model=InverseResponse)
async def solve_inverse_experiment(request: InverseRequest) -> InverseResponse:
    fallback = build_fallback_response(request.goal, request.constraints)
    print(f"Inverse request diterima: {request.goal}")

    if genai_client is None:
        return fallback

    goal_lower = request.goal.lower()
    relevant_reactions = [
        reaction
        for reaction in CHEMICAL_KNOWLEDGE_BASE
        if any(
            keyword in json.dumps(reaction["properties"]).lower()
            for keyword in goal_lower.split()
        )
    ] or CHEMICAL_KNOWLEDGE_BASE

    prompt = f"""
Kamu adalah ATOMVERSE AI Inverse Problem Solver.
Tujuan pengguna: {request.goal}
Database reaksi: {json.dumps(relevant_reactions, ensure_ascii=False)}

Pilih reaksi paling sesuai dan kembalikan hanya JSON valid dengan struktur berikut:
{{
  "confidence_score": 92,
  "recommended_reaction": "Nama reaksi",
  "configuration": {{
    "reactant_a": "AgNO3", "conc_a": 0.1, "vol_a": 50,
    "reactant_b": "NaCl", "conc_b": 0.1, "vol_b": 50,
    "temp": 25, "pressure": 1
  }},
  "predicted_outcome": {{
    "color": "putih", "precipitate": "AgCl", "gas": null, "ph_final": "7.0"
  }},
  "explanation": "Penjelasan singkat"
}}
"""

    try:
        response = await asyncio.wait_for(
            genai_client.aio.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
            ),
            timeout=12,
        )
        result = extract_json(response.text)
        return InverseResponse.model_validate(result)
    except Exception as error:
        print(f"Gemini gagal atau timeout; memakai fallback lokal: {error}")
        return fallback
