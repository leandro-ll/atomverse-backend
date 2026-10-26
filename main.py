import asyncio
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel

from inverse_engine import router as inverse_router

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai_client = genai.Client(api_key=api_key) if api_key else None
gemini_status = "configured" if genai_client else "fallback"
if genai_client is None:
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


from app.services.local_tutor import build_chat_fallback


@app.post("/api/chat")
async def chat_with_tutor(request: ChatRequest):
    global gemini_status

    fallback_reply = build_chat_fallback(request.message, request.lab_state)
    if genai_client is None:
        gemini_status = "fallback"
        return fallback_reply.to_dict()

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
        response = await asyncio.wait_for(
            genai_client.aio.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
            ),
            timeout=12,
        )
        reply = response.text.strip()
        if not reply:
            raise ValueError("Gemini mengembalikan jawaban kosong")
        gemini_status = "active"
        return {"reply": reply, "source": "gemini", "scope": "answered", "degraded": False}
    except Exception as error:
        gemini_status = "fallback"
        print(f"Gemini chat gagal atau timeout; memakai fallback lokal: {error}")
        return fallback_reply.to_dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
