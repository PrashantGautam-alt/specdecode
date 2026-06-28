"""
Run this AFTER the server is up:
    PYTHONPATH=. python scripts/run_server.py

In a second terminal:
    PYTHONPATH=. python scripts/test_server.py

Smoke-tests both endpoints with no browser needed. The medusa stream should
print tokens tagged a=True (draft accepted, blue in the UI) / a=False
(backbone token, red), and end with an acceptance summary.
"""
import requests
import asyncio
import websockets
import json

BASE = "http://localhost:8000"
PROMPT = "Explain the theory of relativity in simple terms:"


def test_post():
    print("=== POST /generate (naive) ===")
    resp = requests.post(f"{BASE}/generate", json={"prompt": PROMPT, "max_new_tokens": 30})
    print(f"Status: {resp.status_code}")
    print(f"Output: {resp.json()['output']}\n")


async def test_websocket(mode):
    print(f"=== WS /stream (mode={mode}) ===")
    uri = "ws://localhost:8000/stream"
    total = 0
    accepted = 0
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"prompt": PROMPT, "max_new_tokens": 60, "mode": mode}))
        async for message in ws:
            data = json.loads(message)
            total += 1
            if data["accepted"] is True:
                accepted += 1
            tag = {True: "+", False: ".", None: " "}[data["accepted"]]  # + accepted, . backbone
            print(f"{tag}{data['text']}", end="", flush=True)
    print()
    if mode == "medusa" and total:
        print(f"-> {accepted}/{total} tokens were accepted drafts "
              f"({100 * accepted / total:.0f}% acceptance)\n")
    else:
        print()


if __name__ == "__main__":
    test_post()
    asyncio.run(test_websocket("naive"))
    asyncio.run(test_websocket("medusa"))
