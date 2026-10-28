import asyncio
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel

from inverse_engine import router as inverse_router

load_dotenv()


def _build_chat_clients():
    raw_keys = [
        os.getenv("GEMINI_CHAT_API_KEY"),
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GEMINI_INVERSE_API_KEY"),
    ]
    seen = set()
    clients = []
    for key in raw_keys:
        clean = (key or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            clients.append(genai.Client(api_key=clean))
    return clients


chat_clients = _build_chat_clients()
genai_client = chat_clients[0] if chat_clients else None
gemini_status = "configured" if chat_clients else "fallback"
if not chat_clients:
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

    fallback_reply = build_chat_fallback(request.message, request.lab_state, force_en=True)
    if not chat_clients:
        gemini_status = "fallback"
        return fallback_reply.to_dict()

    prompt = f"""You are the AI Adaptive Chemistry Tutor for ATOMVERSE, an interactive virtual chemistry lab.

CURRENT EXPERIMENT CONTEXT:
- Solution A: {request.lab_state.get('reagentA', 'None')}
- Solution B: {request.lab_state.get('reagentB', 'None')}
- Temperature: {request.lab_state.get('temp', 25)}°C
- Status: {request.lab_state.get('status', 'Not Run')}

INSTRUCTIONS:
1. You must ALWAYS reply strictly in clear, fluent, and educational English, regardless of the language the student types.
2. Provide a helpful, concise explanation in 2-3 sentences explaining the chemical reaction or principles.
3. Connect with Johnstone's Triangle levels (Macroscopic visual observation, Submicroscopic particle collisions, and Symbolic formulas) where helpful.
4. For all chemical equations, include state symbols: (aq), (s), (l), or (g), using -> for reaction arrows.

Student question: {request.message}
"""

    last_error = None
    for client in chat_clients:
        try:
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=prompt,
                ),
                timeout=12,
            )
            reply = (response.text or "").strip()
            if not reply:
                raise ValueError("Gemini mengembalikan jawaban kosong")
            gemini_status = "active"
            return {"reply": reply, "source": "gemini", "scope": "answered", "degraded": False}
        except Exception as error:
            last_error = error
            print(f"Gemini chat attempt gagal; mencoba key berikutnya jika ada: {error}")

    gemini_status = "fallback"
    print(f"Semua Gemini chat key gagal atau timeout; memakai fallback lokal: {last_error}")
    return fallback_reply.to_dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
