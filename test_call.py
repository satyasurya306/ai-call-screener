import asyncio
import json
import websockets

async def simulate_call():
    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri, ping_interval=None) as ws:
        print("📞 Connected to AI Call Screener! Type as the caller (type 'quit' to hang up).\n")
        print("[AI SCREENER]: Hi, Chandra is busy right now. May I ask who is calling and what this is regarding?")

        while True:
            caller_input = await asyncio.to_thread(input, "\n[YOU (as Caller)]: ")
            if caller_input.lower() == "quit":
                break

            await ws.send(json.dumps({
                "type": "prompt",
                "voicePrompt": caller_input
            }))

            response = json.loads(await ws.recv())
            print(f"[AI SCREENER]: {response.get('token')}")

asyncio.run(simulate_call())