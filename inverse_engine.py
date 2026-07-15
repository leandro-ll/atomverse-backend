# inverse_engine.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import os
import re
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# Cek API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("⚠️ PERINGATAN: GEMINI_API_KEY tidak ditemukan di file .env!")
else:
    genai.configure(api_key=api_key)

CHEMICAL_KNOWLEDGE_BASE = [
    {
        "id": "rxn_001",
        "name": "Pembentukan Endapan Perak Klorida",
        "reactants": ["AgNO3", "NaCl"],
        "properties": {"color": "putih", "precipitate": "AgCl", "ph_final": "7.0"}
    },
    {
        "id": "rxn_002",
        "name": "Pembentukan Endapan Timbal Iodida",
        "reactants": ["Pb(NO3)2", "KI"],
        "properties": {"color": "kuning", "precipitate": "PbI2", "ph_final": "7.0"}
    },
    {
        "id": "rxn_003",
        "name": "Reaksi Netralisasi Asam Basa",
        "reactants": ["HCl", "NaOH"],
        "properties": {"color": "bening", "precipitate": None, "ph_final": "7.0", "energy": "Eksoterm"}
    }
]

class InverseRequest(BaseModel):
    goal: str
    constraints: Optional[Dict] = None

class InverseResponse(BaseModel):
    confidence_score: int
    recommended_reaction: str
    configuration: Dict
    predicted_outcome: Dict
    explanation: str

@router.post("/api/inverse-experiment", response_model=InverseResponse)
async def solve_inverse_experiment(request: InverseRequest):
    try:
        print(f"📥 Menerima request: {request.goal}")
        
        # 1. Rule-Based Filtering
        relevant_reactions = []
        goal_lower = request.goal.lower()
        for rxn in CHEMICAL_KNOWLEDGE_BASE:
            props = json.dumps(rxn['properties']).lower()
            if any(keyword in props for keyword in goal_lower.split()):
                relevant_reactions.append(rxn)
        
        if not relevant_reactions:
            relevant_reactions = CHEMICAL_KNOWLEDGE_BASE[:2] 

        # 2. LLM Reasoning
        prompt = f"""
        Kamu adalah ATOMVERSE AI Inverse Problem Solver.
        TUJUAN PENGGUNA: "{request.goal}"
        DATABASE REAKSI: {json.dumps(relevant_reactions, indent=2)}
        
        RESPON HANYA DALAM FORMAT JSON MURNI (tanpa markdown ```json):
        {{
            "confidence_score": 92,
            "recommended_reaction": "Nama Reaksi",
            "configuration": {{
                "reactant_a": "AgNO3", "conc_a": 0.5, "unit_conc": "M", "vol_a": 50, "unit_vol": "mL",
                "reactant_b": "NaCl", "conc_b": 0.5, "unit_conc": "M", "vol_b": 50, "unit_vol": "mL",
                "temp": 25, "unit_temp": "°C", "pressure": 1, "unit_press": "atm"
            }},
            "predicted_outcome": {{
                "color": "...", "precipitate": "...", "gas": "...", "ph_final": "..."
            }},
            "explanation": "Penjelasan singkat..."
        }}
        """

        model = genai.GenerativeModel('gemini-3.1-flash-lite')
        response = model.generate_content(prompt)
        print(f"📤 Raw Response dari AI:\n{response.text}")
        
        # 3. Parse JSON dengan aman
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if json_match:
            try:
                result_data = json.loads(json_match.group())
                print("✅ Berhasil parse JSON!")
                return InverseResponse(**result_data)
            except json.JSONDecodeError as e:
                print(f"❌ Gagal parse JSON: {e}")
                print(f"Isi yang dicoba di-parse: {json_match.group()}")
                raise HTTPException(status_code=500, detail="AI mengembalikan format JSON yang tidak valid")
        else:
            raise HTTPException(status_code=500, detail="AI tidak mengembalikan format JSON sama sekali")

    except Exception as e:
        print(f"❌ CRITICAL ERROR di Backend: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))