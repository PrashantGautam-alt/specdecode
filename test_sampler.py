
from src.models import ModelLoader
from src.sampler import naive_generate


loader = ModelLoader("meta-llama/Llama-3.2-1B")
loader.load()

model = loader.model
tokenizer = loader.tokenizer


output = naive_generate(model, tokenizer, "The capital of India is", max_new_tokens=20,temperature=1.0)
print(output)
