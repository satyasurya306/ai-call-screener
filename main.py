import json
import os
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NGROK_DOMAIN = os.getenv("NGROK_DOMAIN", "")

if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env file.")

app = FastAPI()
client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """
You are a polite, concise AI call screener answering the phone for Chandra.
1. Ask the caller for their name and the reason for their call.
2. Keep every spoken response under 2 short sentences.
3. Never use markdown, bullet points, or emojis (this is spoken audio).
"""

MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
]

@app.post("/voice")
async def handle_incoming_call():
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <ConversationRelay
                url="wss://{NGROK_DOMAIN}/ws"
                welcomeGreeting="Hi, Chandra is busy right now. May I ask who is calling and what this is regarding?"
            />
        </Connect>
    </Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    history = []

    try:
        while True:
            data = json.loads(await websocket.receive_text())
            event_type = data.get("type")

            if event_type == "prompt":
                caller_text = data.get("voicePrompt", "").strip()
                if not caller_text:
                    continue

                print(f"\n[CALLER]: {caller_text}")
                history.append({"role": "user", "parts": [{"text": caller_text}]})

                response = None
                last_error = None

                for model_name in MODELS:
                    try:
                        response = await client.aio.models.generate_content(
                            model=model_name,
                            contents=history,
                            config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_PROMPT,
                            ),
                        )
                        break
                    except Exception as e:
                        last_error = e
                        if "503" in str(e) or "404" in str(e):
                            print(f"[INFO] {model_name} unavailable, trying next model...")
                            continue
                        raise

                if response is None:
                    raise last_error

                ai_reply = (response.text or "Sorry, could you please repeat that?").strip()
                history.append({"role": "model", "parts": [{"text": ai_reply}]})

                print(f"[AI SCREENER]: {ai_reply}")

                await websocket.send_text(json.dumps({
                    "type": "text",
                    "token": ai_reply,
                    "last": True
                }))

    except WebSocketDisconnect:
        print("\n--- Call ended ---")
    except Exception as e:
        print(f"\n[SERVER ERROR]: {e}")
        await websocket.send_text(json.dumps({
            "type": "text",
            "token": f"ERROR: {e}",
            "last": True
        }))