import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# IMPORT ROUTER DARI FILE LAIN
from inverse_engine import router as inverse_router

# Load environment variables
load_dotenv()

# Configure Gemini using the STABLE package
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file! Please check your backend/.env file.")

genai.configure(api_key=api_key)

app = FastAPI()

# Allow CORS for frontend (Live Server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(inverse_router)

class ChatRequest(BaseModel):
    message: str
    lab_state: dict

@app.post("/api/chat")
async def chat_with_tutor(request: ChatRequest):
    try:
        # Build a STRONG context-aware prompt
        prompt = f"""Kamu adalah AI Adaptive Chemistry Tutor untuk platform ATOMVERSE.

**KONTEKS EKSPERIMEN SAAT INI:**
- Larutan A: {request.lab_state.get('reagentA', 'Tidak ada')}
- Larutan B: {request.lab_state.get('reagentB', 'Tidak ada')}
- Suhu: {request.lab_state.get('temp', 25)}°C
- Status: {request.lab_state.get('status', 'Belum dijalankan')}

**TUGAS KAMU:**
1. JAWAB pertanyaan siswa berdasarkan KONTEKS eksperimen di atas.
2. Gunakan bahasa Indonesia yang natural dan edukatif.
3. Berikan penjelasan spesifik tentang reaksi kimia yang sedang diamati.
4. Hubungkan dengan konsep Johnstone's Triangle (makroskopik, simbolik, submikroskopik).
5. Gunakan emoji secukupnya 🧪⚗️🔬
6. Jika pertanyaan tidak relevan dengan eksperimen, jawab dengan sopan dan arahkan siswa untuk fokus pada eksperimen.
7. Jangan membuat asumsi di luar konteks eksperimen.
8. Jangan memberikan jawaban yang terlalu panjang, cukup 2-3 kalimat.
9. Untuk persamaan kimia, gunakan format LaTeX dengan tanda $...$ untuk inline math dan $$...$$ untuk display math.
   Contoh: $Ag^+(aq) + Cl^-(aq) -> AgCl(s)$ atau $$Pb^{{2+}} + 2I^- -> PbI_2$$
10.- Untuk rightarrow selalu gunakan -> bukan \\rightarrow
   - Tulis subscript dengan benar: Pb(NO₃)₂ bukan Pb(NO_3)_2
   - Gunakan state symbols: (aq), (s), (l), (g)
**Pertanyaan Siswa:** {request.message}

Jawaban spesifik berdasarkan konteks eksperimen:"""

        print(f"📤 Sending to Gemini...")
        
        # Use the STABLE google-generativeai API
        model = genai.GenerativeModel('gemini-3.1-flash-lite')
        response = model.generate_content(prompt)
        
        reply = response.text
        print(f"💬 AI Response received successfully!")
        
        return {"reply": reply}

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
