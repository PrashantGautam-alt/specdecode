from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from typing import Optional
import torch

app = FastAPI()

backbone = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    mode: str = "naive"

@app.post("/generate")
def generate(request: GenerateRequest):
    from src.sampler import naive_generate
    output = naive_generate(backbone,tokenizer,request.prompt, max_new_tokens=request.max_new_tokens)
    return {"output":output}

@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    data = await websocket.receive_json()
    prompt = data["prompt"]
    max_new_tokens = data.get("max_new_tokens",100)
    #tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(backbone.device)
    generated = input_ids.clone()
    #generate one toekn at a time and send each over the websock
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = backbone(input_ids=generated)

        next_token = outputs.logits[0,-1,:].argmax().item()
        generated = torch.cat([generated,torch.tensor([[next_token]], device = backbone.device)], dim=1)
        token_text = tokenizer.decode([next_token], skip_special_tokens=True)
        # send JSON so the frontend can color tokens (accepted=None means naive, no color)
        import json
        await websocket.send_text(json.dumps({"text": token_text, "accepted": None}))
        if next_token == tokenizer.eos_token_id:
            break
    await websocket.close()