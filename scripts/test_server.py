"""
Run this AFTER the server is up:
    PYTHONPATH=. python scripts/run_server.py

In a second terminal:
    PYTHONPATH=. python scripts/test_server.py
"""
import requests
import asyncio
import websockets
import json

BASE = "http://localhost:8000"
PROMPT = "The capital of France is"


def test_post():
    print("=== POST /generate ===")
    resp = requests.post(f"{BASE}/generate", json={"prompt": PROMPT, "max_new_tokens": 30})
    print(f"Status: {resp.status_code}")
    print(f"Output: {resp.json()['output']}")


async def test_websocket():
    print("\n=== WS /stream ===")
    uri = "ws://localhost:8000/stream"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"prompt": PROMPT, "max_new_tokens": 30}))
        print("Streaming: ", end="", flush=True)
        async for message in ws:
            print(message, end="", flush=True)
    print()


if __name__ == "__main__":
    test_post()
    asyncio.run(test_websocket())
