import asyncio
import json
import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

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


class CallSummary(BaseModel):
    caller_name: str = Field(default="Unknown Caller", description="Caller's name if mentioned, else Unknown Caller.")
    reason: str = Field(default="Screening in progress...", description="Short summary of why the caller is calling.")


class DecisionRequest(BaseModel):
    action: str  # "accept" or "decline"
    note: Optional[str] = ""


class CallStateManager:
    def __init__(self):
        self.dashboard_sockets: set[WebSocket] = set()
        self.active_call_sockets: set[WebSocket] = set()
        self.call_active: bool = False
        self.caller_name: str = "Unknown Caller"
        self.reason: str = "Waiting for caller..."
        self.decision: Optional[str] = None
        self.transcript: list[dict] = []
        self.history_log: list[dict] = []

    def start_new_call(self, greeting: str):
        self.call_active = True
        self.caller_name = "Unknown Caller"
        self.reason = "Screening in progress..."
        self.decision = None
        self.transcript = [
            {
                "speaker": "ai",
                "text": greeting,
                "time": datetime.now().strftime("%I:%M:%S %p"),
            }
        ]

    def snapshot(self) -> dict:
        return {
            "type": "state",
            "call_active": self.call_active,
            "caller_name": self.caller_name,
            "reason": self.reason,
            "decision": self.decision,
            "transcript": self.transcript,
            "history_log": self.history_log,
        }

    async def broadcast_dashboard(self, payload: dict):
        stale = []
        for ws in list(self.dashboard_sockets):
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.dashboard_sockets.discard(ws)

    async def send_to_active_calls(self, message_text: str):
        stale = []
        for ws in list(self.active_call_sockets):
            try:
                await ws.send_text(json.dumps({
                    "type": "text",
                    "token": message_text,
                    "last": True,
                }))
            except Exception:
                stale.append(ws)
        for ws in stale:
            self.active_call_sockets.discard(ws)


state = CallStateManager()


async def generate_with_fallback(contents: list[dict], system_instruction: str):
    response = None
    last_error = None
    for model_name in MODELS:
        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                ),
            )
            return response
        except Exception as e:
            last_error = e
            if "503" in str(e) or "404" in str(e):
                print(f"[INFO] {model_name} unavailable, trying next model...")
                continue
            raise
    raise last_error


async def extract_caller_summary(transcript_items: list[dict]) -> CallSummary:
    formatted = "\n".join(
        f"{item['speaker'].upper()}: {item['text']}" for item in transcript_items
    )
    prompt = (
        "Extract the caller's name and a concise 1-line call reason from this screening transcript.\n"
        f"{formatted}"
    )
    for model_name in MODELS:
        try:
            res = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CallSummary,
                ),
            )
            if res.text:
                return CallSummary.model_validate_json(res.text)
        except Exception:
            continue
    return CallSummary(caller_name=state.caller_name, reason=state.reason)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Chandra — Live AI Call Screener</title>
  <style>
    :root {
      --bg: #0b0f17;
      --card: #131a26;
      --card-2: #192232;
      --text: #eef2f7;
      --muted: #93a1b5;
      --accent: #3b82f6;
      --green: #16a34a;
      --red: #dc2626;
      --border: #253146;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: system-ui, -apple-system, sans-serif; }
    body { background: var(--bg); color: var(--text); min-height: 100vh; padding: 20px; }
    .container { max-width: 960px; margin: 0 auto; display: grid; gap: 16px; }
    @media (min-width: 780px) { .grid-2 { display: grid; grid-template-columns: 1.35fr 0.9fr; gap: 16px; } }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 18px; }
    .header { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
    .badge { padding: 5px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; background: var(--card-2); border: 1px solid var(--border); }
    .live { color: #4ade80; border-color: #15803d; }
    .meta { margin-top: 14px; display: grid; gap: 8px; background: var(--card-2); padding: 12px; border-radius: 10px; }
    .meta-row { font-size: 14px; }
    .meta-label { color: var(--muted); }
    .transcript { margin-top: 14px; max-height: 320px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; padding-right: 4px; }
    .msg { padding: 10px 12px; border-radius: 10px; max-width: 88%; font-size: 14px; line-height: 1.4; }
    .msg.ai { background: #1d293d; align-self: flex-start; }
    .msg.caller { background: #1e3a5f; align-self: flex-end; }
    .msg-time { font-size: 11px; color: var(--muted); margin-top: 4px; }
    .actions { display: flex; gap: 10px; margin-top: 16px; }
    button { flex: 1; padding: 12px 14px; border: none; border-radius: 10px; font-weight: 600; font-size: 14px; cursor: pointer; color: #fff; }
    .btn-accept { background: var(--green); }
    .btn-decline { background: var(--red); }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    h2 { font-size: 16px; margin-bottom: 10px; }
    .history-list { display: flex; flex-direction: column; gap: 10px; max-height: 440px; overflow-y: auto; }
    .history-item { background: var(--card-2); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; font-size: 13px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="card header">
      <div>
        <h1>AI Call Screener</h1>
        <div class="meta-label" style="font-size:13px; margin-top:2px;">Live screening console for Chandra</div>
      </div>
      <span id="connBadge" class="badge">Connecting...</span>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="header">
          <h2 style="margin:0;">Active Call</h2>
          <span id="callStatus" class="badge">Idle</span>
        </div>

        <div class="meta">
          <div class="meta-row"><span class="meta-label">Caller:</span> <strong id="callerName">Unknown Caller</strong></div>
          <div class="meta-row"><span class="meta-label">Reason:</span> <span id="callReason">Waiting for caller...</span></div>
          <div class="meta-row"><span class="meta-label">Decision:</span> <span id="callDecision">Pending</span></div>
        </div>

        <div id="transcriptBox" class="transcript"></div>

        <div class="actions">
          <button id="acceptBtn" class="btn-accept" onclick="sendDecision('accept')">Accept Call</button>
          <button id="declineBtn" class="btn-decline" onclick="sendDecision('decline')">Decline / Take Message</button>
        </div>
      </div>

      <div class="card">
        <h2>Call History</h2>
        <div id="historyBox" class="history-list">
          <div class="meta-label" style="font-size:13px;">No completed calls yet.</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/ws/dashboard`);
    const connBadge = document.getElementById("connBadge");

    ws.onopen = () => {
      connBadge.textContent = "Live Connected";
      connBadge.classList.add("live");
    };
    ws.onclose = () => {
      connBadge.textContent = "Disconnected";
      connBadge.classList.remove("live");
    };
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === "state") renderState(msg);
    };

    function renderState(s) {
      document.getElementById("callStatus").textContent = s.call_active ? "Screening Live" : "Idle";
      document.getElementById("callerName").textContent = s.caller_name || "Unknown Caller";
      document.getElementById("callReason").textContent = s.reason || "Waiting for caller...";
      document.getElementById("callDecision").textContent = s.decision ? s.decision.toUpperCase() : "Pending";

      const tBox = document.getElementById("transcriptBox");
      tBox.innerHTML = "";
      (s.transcript || []).forEach(item => {
        const div = document.createElement("div");
        div.className = `msg ${item.speaker === "ai" ? "ai" : "caller"}`;
        div.innerHTML = `<div>${escapeHtml(item.text)}</div><div class="msg-time">${item.speaker === "ai" ? "AI Screener" : "Caller"} • ${item.time || ""}</div>`;
        tBox.appendChild(div);
      });
      tBox.scrollTop = tBox.scrollHeight;

      const hBox = document.getElementById("historyBox");
      if (s.history_log && s.history_log.length > 0) {
        hBox.innerHTML = "";
        [...s.history_log].reverse().forEach(entry => {
          const item = document.createElement("div");
          item.className = "history-item";
          item.innerHTML = `<div><strong>${escapeHtml(entry.caller_name)}</strong> — <span class="badge">${escapeHtml(entry.decision)}</span></div><div style="margin-top:4px;">${escapeHtml(entry.reason)}</div><div class="msg-time">${entry.time}</div>`;
          hBox.appendChild(item);
        });
      }
    }

    async function sendDecision(action) {
      await fetch("/call/decision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action })
      });
    }

    function escapeHtml(str) {
      return String(str || "").replace(/[&<>"']/g, m => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[m]));
    }
  </script>
</body>
</html>
"""


@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    return DASHBOARD_HTML


@app.websocket("/ws/dashboard")
async def dashboard_websocket(websocket: WebSocket):
    await websocket.accept()
    state.dashboard_sockets.add(websocket)
    await websocket.send_text(json.dumps(state.snapshot()))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        state.dashboard_sockets.discard(websocket)


@app.post("/call/decision")
async def handle_decision(req: DecisionRequest):
    action = req.action.lower().strip()
    if action not in {"accept", "decline"}:
        return JSONResponse({"error": "Invalid action"}, status_code=400)

    state.decision = action
    if action == "accept":
        spoken = "Great news, Chandra is accepting your call now and will speak with you shortly."
    else:
        spoken = "Understood, Chandra is unavailable to take the call right now. Please leave any final details and I will pass them along."

    now_str = datetime.now().strftime("%I:%M:%S %p")
    state.transcript.append({"speaker": "ai", "text": spoken, "time": now_str})
    await state.send_to_active_calls(spoken)

    if action == "decline":
        state.history_log.append({
            "caller_name": state.caller_name,
            "reason": state.reason,
            "decision": "declined",
            "time": now_str,
        })
        state.call_active = False
    else:
        state.history_log.append({
            "caller_name": state.caller_name,
            "reason": state.reason,
            "decision": "accepted",
            "time": now_str,
        })

    await state.broadcast_dashboard(state.snapshot())
    return {"status": "ok", "decision": action}


@app.post("/voice")
async def handle_incoming_call():
    greeting = "Hi, Chandra is busy right now. May I ask who is calling and what this is regarding?"
    state.start_new_call(greeting)
    asyncio.create_task(state.broadcast_dashboard(state.snapshot()))

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Connect>
            <ConversationRelay
                url="wss://{NGROK_DOMAIN}/ws"
                welcomeGreeting="{greeting}"
            />
        </Connect>
    </Response>"""
    return Response(content=twiml, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.active_call_sockets.add(websocket)
    greeting = "Hi, Chandra is busy right now. May I ask who is calling and what this is regarding?"
    if not state.call_active:
        state.start_new_call(greeting)
        await state.broadcast_dashboard(state.snapshot())

    history = []

    try:
        while True:
            data = json.loads(await websocket.receive_text())
            event_type = data.get("type")

            if event_type == "prompt":
                caller_text = data.get("voicePrompt", "").strip()
                if not caller_text:
                    continue

                now_str = datetime.now().strftime("%I:%M:%S %p")
                print(f"\n[CALLER]: {caller_text}")
                history.append({"role": "user", "parts": [{"text": caller_text}]})
                state.transcript.append({"speaker": "caller", "text": caller_text, "time": now_str})
                await state.broadcast_dashboard(state.snapshot())

                response = await generate_with_fallback(history, SYSTEM_PROMPT)
                ai_reply = (response.text or "Sorry, could you please repeat that?").strip()
                history.append({"role": "model", "parts": [{"text": ai_reply}]})

                ai_time = datetime.now().strftime("%I:%M:%S %p")
                state.transcript.append({"speaker": "ai", "text": ai_reply, "time": ai_time})

                summary = await extract_caller_summary(state.transcript)
                if summary.caller_name and summary.caller_name != "Unknown Caller":
                    state.caller_name = summary.caller_name
                if summary.reason:
                    state.reason = summary.reason

                await state.broadcast_dashboard(state.snapshot())
                print(f"[AI SCREENER]: {ai_reply}")

                await websocket.send_text(json.dumps({
                    "type": "text",
                    "token": ai_reply,
                    "last": True,
                }))

    except WebSocketDisconnect:
        print("\n--- Call ended ---")
        state.call_active = False
        await state.broadcast_dashboard(state.snapshot())
    except Exception as e:
        print(f"\n[SERVER ERROR]: {e}")
        await websocket.send_text(json.dumps({
            "type": "text",
            "token": f"ERROR: {e}",
            "last": True,
        }))
    finally:
        state.active_call_sockets.discard(websocket)