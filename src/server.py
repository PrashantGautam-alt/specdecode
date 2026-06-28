import asyncio
import json
import threading

import torch
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

app = FastAPI()

# Injected by run_server.py so both endpoints share one loaded model (dependency injection).
backbone = None
tokenizer = None
medusa = None          # MedusaModel (backbone + trained heads); None -> only naive available
WIDTH = 2              # tree branching factor used by the fused decoder


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    mode: str = "naive"


@app.post("/generate")
def generate(request: GenerateRequest):
    from src.sampler import naive_generate
    output = naive_generate(backbone, tokenizer, request.prompt, max_new_tokens=request.max_new_tokens)
    return {"output": output}


@app.websocket("/stream")
async def stream(websocket: WebSocket):
    """
    Stream generated tokens live over a WebSocket, one JSON message per token:
        {"text": "...", "accepted": true|false|null}
      true  -> a draft token the heads guessed and the backbone confirmed (the speed win)
      false -> a backbone-produced token (the 1 guaranteed token per round / a correction)
      null  -> naive generation, no acceptance concept
    mode="medusa" uses the fused tree decoder; anything else falls back to naive.
    """
    await websocket.accept()
    data = await websocket.receive_json()
    prompt = data["prompt"]
    max_new_tokens = data.get("max_new_tokens", 100)
    mode = data.get("mode", "naive")

    if mode == "medusa" and medusa is not None:
        await _stream_medusa(websocket, prompt, max_new_tokens)
    else:
        await _stream_naive(websocket, prompt, max_new_tokens)

    await websocket.close()


async def _stream_medusa(websocket, prompt, max_new_tokens):
    # The fused decoder is synchronous and GPU-bound, so run it in a worker thread.
    # Its on_token callback fires on that thread; we hop back onto the event loop with
    # loop.call_soon_threadsafe (asyncio.Queue is NOT safe to put to from another thread).
    from src.medusa import medusa_decode_tree_fused

    loop = asyncio.get_event_loop()
    out_queue = asyncio.Queue()
    DONE = object()  # sentinel marking end of stream
    K = len(medusa.heads)

    def on_token(token_id, accepted):
        text = tokenizer.decode([token_id], skip_special_tokens=True)
        loop.call_soon_threadsafe(out_queue.put_nowait, (text, accepted))

    def run_decode():
        try:
            medusa_decode_tree_fused(
                medusa, tokenizer, prompt,
                max_new_tokens=max_new_tokens, K=K, width=WIDTH,
                accept_mode="greedy", on_token=on_token,
            )
        finally:
            loop.call_soon_threadsafe(out_queue.put_nowait, DONE)

    threading.Thread(target=run_decode, daemon=True).start()

    while True:
        item = await out_queue.get()
        if item is DONE:
            break
        text, accepted = item
        await websocket.send_text(json.dumps({"text": text, "accepted": accepted}))


async def _stream_naive(websocket, prompt, max_new_tokens):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(backbone.device)
    generated = input_ids.clone()
    cache = None
    for _ in range(max_new_tokens):
        with torch.no_grad():
            # feed only the newest token once a cache exists -> O(1) per step, not O(n)
            step_in = generated if cache is None else generated[:, -1:]
            outputs = backbone(input_ids=step_in, past_key_values=cache, use_cache=True)
        cache = outputs.past_key_values
        next_token = outputs.logits[0, -1, :].argmax().item()
        generated = torch.cat([generated, torch.tensor([[next_token]], device=backbone.device)], dim=1)
        token_text = tokenizer.decode([next_token], skip_special_tokens=True)
        await websocket.send_text(json.dumps({"text": token_text, "accepted": None}))
        if next_token == tokenizer.eos_token_id:
            break
