import uvicorn
import src.server as server
from src.models import ModelLoader

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEVICE = "cuda:0"
HOST = "0.0.0.0"
PORT = 8000

if __name__ == "__main__":
    print(f"Loading {MODEL_NAME}...")
    loader = ModelLoader(MODEL_NAME, device=DEVICE)
    loader.load()

    # inject into server module so both endpoints share the same loaded model
    server.backbone = loader.model
    server.tokenizer = loader.tokenizer
    print("Model loaded. Starting server...")

    uvicorn.run(server.app, host=HOST, port=PORT)
