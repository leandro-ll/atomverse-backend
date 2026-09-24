import asyncio
import json
import os
import re
from typing import Dict, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel, Field

load_dotenv()

router = APIRouter()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
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


class InverseResponse(BaseModel):
    confidence_score: int = Field(ge=0, le=100)
    recommended_reaction: str
    configuration: ReactionConfiguration
    predicted_outcome: PredictedOutcome
    explanation: str


def build_fallback_response(goal: str) -> InverseResponse:
    goal_lower = goal.lower()

    if "kuning" in goal_lower:
        return InverseResponse(
            confidence_score=95,
            recommended_reaction="Pembentukan Endapan Timbal Iodida",
            configuration=ReactionConfiguration(
                reactant_a="Pb(NO3)2",
                conc_a=0.1,
                vol_a=50,
                reactant_b="KI",
                conc_b=0.2,
                vol_b=50,
                temp=25,
            ),
            predicted_outcome=PredictedOutcome(
                color="kuning cerah", precipitate="PbI2", ph_final="7.0"
            ),
            explanation=(
                "Pb(NO3)2 dan KI menghasilkan endapan PbI2 berwarna kuning. "
                "Perbandingan konsentrasi 1:2 mengikuti stoikiometri reaksi."
            ),
        )

    if "ph" in goal_lower or "netral" in goal_lower:
        return InverseResponse(
            confidence_score=94,
            recommended_reaction="Netralisasi Asam Klorida dan Natrium Hidroksida",
            configuration=ReactionConfiguration(
                reactant_a="HCl",
                conc_a=0.1,
                vol_a=50,
                reactant_b="NaOH",
                conc_b=0.1,
                vol_b=50,
                temp=25,
            ),
            predicted_outcome=PredictedOutcome(
                color="bening", precipitate=None, ph_final="7.0"
            ),
            explanation=(
                "Volume dan konsentrasi HCl serta NaOH dibuat sama agar mol H+ dan OH- "
                "setara sehingga larutan mendekati pH 7."
            ),
        )

    return InverseResponse(
        confidence_score=95,
        recommended_reaction="Pembentukan Endapan Perak Klorida",
        configuration=ReactionConfiguration(
            reactant_a="AgNO3",
            conc_a=0.1,
            vol_a=50,
            reactant_b="NaCl",
            conc_b=0.1,
            vol_b=50,
            temp=25,
        ),
        predicted_outcome=PredictedOutcome(
            color="putih susu", precipitate="AgCl", ph_final="7.0"
        ),
        explanation=(
            "AgNO3 dan NaCl bereaksi dengan perbandingan mol 1:1 untuk membentuk "
            "endapan AgCl berwarna putih."
        ),
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
    fallback = build_fallback_response(request.goal)
    print(f"Inverse request diterima: {request.goal}")

    if not api_key:
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
        model = genai.GenerativeModel("gemini-3.1-flash-lite")
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, prompt),
            timeout=12,
        )
        result = extract_json(response.text)
        return InverseResponse.model_validate(result)
    except Exception as error:
        print(f"Gemini gagal atau timeout; memakai fallback lokal: {error}")
        return fallback
