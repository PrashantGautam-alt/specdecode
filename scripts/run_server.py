import torch
import uvicorn

import src.server as server
from src.models import ModelLoader
from src.medusa import MedusaModel

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CHECKPOINT = "medusa_heads_8b_epoch4.pt"   # 4-head checkpoint (matches K below)
K = 4
DEVICE = "cuda:0"
HOST = "0.0.0.0"
PORT = 8000

if __name__ == "__main__":
    print(f"Loading {MODEL_NAME}...")
    loader = ModelLoader(MODEL_NAME, device=DEVICE)
    loader.load()

    print(f"Attaching {K} Medusa heads from {CHECKPOINT}...")
    medusa = MedusaModel(loader.model, num_heads=K)
    # map_location="cpu" then .to(): avoids holding a 2nd GPU copy of the heads while loading
    medusa.heads.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    medusa.heads.to(device=DEVICE, dtype=torch.float16)
    medusa.heads.eval()

    # inject into the server module so both endpoints share the same loaded model
    server.backbone = loader.model
    server.tokenizer = loader.tokenizer
    server.medusa = medusa
    print("Model + heads loaded. Starting server on "
          f"http://{HOST}:{PORT}  (naive: mode='naive', Medusa: mode='medusa')")

    uvicorn.run(server.app, host=HOST, port=PORT)
