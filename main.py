import asyncio
import os

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inverse_engine import router as inverse_router

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
gemini_status = "configured" if api_key else "fallback"
if api_key:
    genai.configure(api_key=api_key)
else:
    print("WARNING: GEMINI_API_KEY tidak ditemukan; fitur AI akan memakai fallback lokal.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(inverse_router)


class ChatRequest(BaseModel):
    message: str
    lab_state: dict


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "inverse_endpoint": "/api/inverse-experiment",
        "chat_endpoint": "/api/chat",
        "chat_mode": gemini_status,
    }


def build_chat_fallback(request: ChatRequest) -> str:
    reagent_a = str(request.lab_state.get("reagentA", ""))
    reagent_b = str(request.lab_state.get("reagentB", ""))
    pair = {reagent_a, reagent_b}

    if pair == {"AgNO3", "NaCl"}:
        return (
            "AgNO₃ dan NaCl membentuk endapan putih AgCl. Secara simbolik, reaksi ionnya "
            "adalah $Ag^+(aq) + Cl^-(aq) -> AgCl(s)$; pada tingkat submikroskopik, ion Ag⁺ "
            "dan Cl⁻ bergabung membentuk kisi padat."
        )
    if pair == {"Pb(NO3)2", "KI"}:
        return (
            "Pb(NO₃)₂ dan KI menghasilkan endapan kuning PbI₂. Reaksi ionnya adalah "
            "$Pb^{2+}(aq) + 2I^-(aq) -> PbI_2(s)$, ketika ion-ion membentuk kisi kristal padat."
        )
    if pair == {"HCl", "NaOH"}:
        return (
            "HCl dan NaOH mengalami netralisasi eksoterm sehingga menghasilkan garam dan air. "
            "Reaksi ion bersihnya adalah $H^+(aq) + OH^-(aq) -> H_2O(l)$."
        )

    return (
        f"Eksperimen saat ini menggunakan {reagent_a or 'larutan A'} dan "
        f"{reagent_b or 'larutan B'} pada suhu {request.lab_state.get('temp', 25)}°C. "
        "Jalankan simulasi untuk mengamati perubahan makroskopik, lalu bandingkan dengan gerak ion pada tampilan submikroskopik."
    )


@app.post("/api/chat")
async def chat_with_tutor(request: ChatRequest):
    global gemini_status

    fallback_reply = build_chat_fallback(request)
    if not api_key:
        gemini_status = "fallback"
        return {"reply": fallback_reply, "source": "fallback"}

    prompt = f"""Kamu adalah AI Adaptive Chemistry Tutor untuk platform ATOMVERSE.

KONTEKS EKSPERIMEN SAAT INI:
- Larutan A: {request.lab_state.get('reagentA', 'Tidak ada')}
- Larutan B: {request.lab_state.get('reagentB', 'Tidak ada')}
- Suhu: {request.lab_state.get('temp', 25)}°C
- Status: {request.lab_state.get('status', 'Belum dijalankan')}

Jawab pertanyaan siswa dalam bahasa Indonesia secara natural dan edukatif dalam 2-3 kalimat.
Hubungkan dengan konsep Johnstone's Triangle jika relevan. Untuk persamaan kimia gunakan
state symbols (aq), (s), (l), atau (g), dan gunakan -> sebagai tanda panah.

Pertanyaan siswa: {request.message}
"""

    try:
        model = genai.GenerativeModel("gemini-3.1-flash-lite")
        response = await asyncio.wait_for(
            asyncio.to_thread(model.generate_content, prompt),
            timeout=12,
        )
        reply = response.text.strip()
        if not reply:
            raise ValueError("Gemini mengembalikan jawaban kosong")
        gemini_status = "active"
        return {"reply": reply, "source": "gemini"}
    except Exception as error:
        gemini_status = "fallback"
        print(f"Gemini chat gagal atau timeout; memakai fallback lokal: {error}")
        return {"reply": fallback_reply, "source": "fallback"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
