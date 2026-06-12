
import time
from src.models import ModelLoader


loader = ModelLoader("meta-llama/Llama-3.2-1B")
loader.load()

model = loader.model
tokenizer = loader.tokenizer

prompt = "The capital of India is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")


start_time = time.time()
output_ids = model.generate(**inputs, max_new_tokens=20)
elapsed = time.time() - start_time


text = tokenizer.decode(output_ids[0])
print(text)
print(f"Generated 20 tokens in {elapsed:.2f} seconds")

