import asyncio
import json
import os
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import APIRouter
from google import genai
from pydantic import BaseModel, Field

from app.services.lab_reactions import (
    CONC_RANGE,
    REACTIONS_BY_ID,
    SUGGESTIONS,
    TEMP_RANGE,
    VOL_RANGE,
    LabReaction,
    MatchResult,
    clamp,
    confidence_from_score,
    describe_for_prompt,
    match_goal,
)

load_dotenv()

router = APIRouter()


def _build_inverse_clients():
    raw_keys = [
        os.getenv("GEMINI_INVERSE_API_KEY"),
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GEMINI_CHAT_API_KEY"),
    ]
    seen = set()
    clients = []
    for key in raw_keys:
        clean = (key or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            clients.append(genai.Client(api_key=clean))
    return clients


inverse_clients = _build_inverse_clients()
genai_client = inverse_clients[0] if inverse_clients else None

if not inverse_clients:
    print("WARNING: GEMINI_API_KEY tidak ditemukan; inverse endpoint akan memakai fallback lokal.")

# Keep below the frontend's 15 s abort so the local fallback can still be returned.
GEMINI_TIMEOUT_SECONDS = 12
UNSUPPORTED_ID = "UNSUPPORTED"


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


class Suggestion(BaseModel):
    label: str
    prompt: str


class InverseResponse(BaseModel):
    confidence_score: int = Field(ge=0, le=100)
    recommended_reaction: str
    configuration: Optional[ReactionConfiguration] = None
    predicted_outcome: Optional[PredictedOutcome] = None
    explanation: str
    supported: bool = True
    source: str = "local"
    reaction_id: Optional[str] = None
    equation: Optional[str] = None
    unsupported_terms: List[str] = Field(default_factory=list)
    suggestions: List[Suggestion] = Field(default_factory=list)


def _filter_by_constraints(reaction: LabReaction, constraints: Optional[Dict]) -> bool:
    if not constraints:
        return True
    pair = {reaction.reactant_a, reaction.reactant_b}
    forbidden = set(constraints.get("forbidden_reagents") or [])
    allowed = constraints.get("allowed_reagents")
    if pair & forbidden:
        return False
    if allowed is not None and not pair.issubset(set(allowed)):
        return False
    return True


def build_supported_response(reaction: LabReaction, confidence: int, source: str) -> InverseResponse:
    """Build the response strictly from the lab database entry (never from AI text)."""
    return InverseResponse(
        confidence_score=max(0, min(100, int(confidence))),
        recommended_reaction=reaction.name,
        configuration=ReactionConfiguration(
            reactant_a=reaction.reactant_a,
            conc_a=clamp(reaction.conc_a, CONC_RANGE),
            vol_a=clamp(50, VOL_RANGE),
            reactant_b=reaction.reactant_b,
            conc_b=clamp(reaction.conc_b, CONC_RANGE),
            vol_b=clamp(50, VOL_RANGE),
            temp=clamp(25, TEMP_RANGE),
            pressure=1,
        ),
        predicted_outcome=PredictedOutcome(
            color=reaction.color,
            precipitate=reaction.precipitate,
            gas=reaction.gas,
            ph_final=reaction.ph_final,
        ),
        explanation=reaction.explanation,
        supported=True,
        source=source,
        reaction_id=reaction.id,
        equation=reaction.equation,
    )


def build_unsupported_response(goal_text: str, unsupported_terms: List[str], source: str) -> InverseResponse:
    if unsupported_terms:
        detail = (
            f"The Reaction Lab can't simulate {', '.join(unsupported_terms)} yet. "
        )
    else:
        detail = "This goal doesn't match any reaction the Reaction Lab can simulate yet. "
    return InverseResponse(
        confidence_score=0,
        recommended_reaction="Outcome not supported in the Reaction Lab",
        configuration=None,
        predicted_outcome=None,
        explanation=detail + "Try one of the supported outcomes below: precipitates (AgCl, AgI, PbI₂, PbCl₂, "
        "PbCO₃, BaSO₄, BaCO₃, CaCO₃), CO₂ or NH₃ gas evolution, or acid-base neutralization.",
        supported=False,
        source=source,
        unsupported_terms=unsupported_terms,
        suggestions=[Suggestion(**s) for s in SUGGESTIONS],
    )


def build_local_response(goal_text: str, constraints: Optional[Dict] = None,
                         match: Optional[MatchResult] = None) -> InverseResponse:
    match = match or match_goal(goal_text)
    if match.reaction is None or not _filter_by_constraints(match.reaction, constraints):
        return build_unsupported_response(goal_text, match.unsupported_terms, "local")
    return build_supported_response(match.reaction, confidence_from_score(match.score), "local")


def extract_json(response_text: str) -> dict:
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        raise ValueError("AI tidak mengembalikan objek JSON")
    return json.loads(match.group())


def build_prompt(goal_text: str) -> str:
    return f"""You are the ATOMVERSE Inverse Experiment selector.
A student describes an outcome they want. You must choose ONE reaction from the database below.
You are NOT allowed to invent reactions, reagents, or products that are not listed.

DATABASE (id: reactant A + reactant B -> observable outcome):
{describe_for_prompt()}

Rules:
- Return the id of the reaction that best produces the student's described outcome.
- If the goal requires any substance, color, gas, or process not in the database, return "{UNSUPPORTED_ID}".
- Reply with ONLY valid JSON: {{"reaction_id": "<id or {UNSUPPORTED_ID}>", "confidence": <integer 0-100>}}

Student goal: {json.dumps(goal_text, ensure_ascii=False)}
"""


def resolve_ai_choice(ai_result: dict, local: MatchResult, goal_text: str,
                      constraints: Optional[Dict]) -> InverseResponse:
    """Validate the AI's pick against the database and the local matcher."""
    reaction_id = str(ai_result.get("reaction_id", "")).strip()
    try:
        ai_confidence = int(ai_result.get("confidence", 85))
    except (TypeError, ValueError):
        ai_confidence = 85
    ai_confidence = max(50, min(99, ai_confidence))

    reaction = REACTIONS_BY_ID.get(reaction_id)

    if reaction is None:
        # UNSUPPORTED or an invented ID: defer to the deterministic matcher.
        if reaction_id != UNSUPPORTED_ID:
            print(f"Gemini mengembalikan reaction_id tidak dikenal: {reaction_id!r}; memakai matcher lokal.")
        return build_local_response(goal_text, constraints, local)

    # A strong local match of a different reaction type wins (e.g. AI picked a precipitate for a gas goal).
    if local.reaction is not None and local.strong and local.reaction.type != reaction.type:
        return build_local_response(goal_text, constraints, local)

    if not _filter_by_constraints(reaction, constraints):
        return build_local_response(goal_text, constraints, local)

    return build_supported_response(reaction, ai_confidence, "gemini")


@router.post("/api/inverse-experiment", response_model=InverseResponse)
async def solve_inverse_experiment(request: InverseRequest) -> InverseResponse:
    print(f"Inverse request diterima: {request.goal}")
    local = match_goal(request.goal)

    # Substances/colors the lab cannot simulate are rejected before asking the AI.
    if local.unsupported_terms:
        return build_unsupported_response(request.goal, local.unsupported_terms, "local")

    if not inverse_clients:
        return build_local_response(request.goal, request.constraints, local)

    last_error = None
    for client in inverse_clients:
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=build_prompt(request.goal),
                ),
                timeout=GEMINI_TIMEOUT_SECONDS,
            )
            ai_result = extract_json(response.text)
            return resolve_ai_choice(ai_result, local, request.goal, request.constraints)
        except Exception as error:
            last_error = error
            print(f"Gemini inverse attempt gagal; mencoba key berikutnya jika ada: {error}")

    print(f"Semua Gemini inverse key gagal atau timeout; memakai fallback lokal: {last_error}")
    return build_local_response(request.goal, request.constraints, local)
