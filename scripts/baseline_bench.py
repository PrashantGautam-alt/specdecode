

import time
import torch
from src.models import ModelLoader
from src.sampler import naive_generate

def measure_tokens_per_sec(model, tokenizer, prompt: str, max_new_tokens: int = 100, runs: int = 3) -> float:
    """
    Runs naive_generate() multiple times and returns average tokens/sec.
    Multiple runs average out GPU warm-up variance.
    """

    times = []

    for _ in range(runs):
        torch.cuda.synchronize() # wait for GPU to finish any pending work
        start = time.perf_counter()

        naive_generate(model, tokenizer, prompt, max_new_tokens = max_new_tokens, temperature=1.0)

        torch.cuda.synchronize()  # wait for GPU to finish before stopping timer
        end = time.perf_counter()


        times.append(max_new_tokens/(end-start))


    return sum(times)/len(times)


PROMPTS = [
    "Who is gonna carry the boat",
    "Explain the theory of relativity in simple terms:",
    "Write a Python function that sorts a list:",
    "The history of the Roman Empire began",
    "The best way to learn machine learning is",
]

if __name__ == "__main__":
    loader = ModelLoader("meta-llama/Llama-3.2-1B")
    loader.load()
    model = loader.model
    tokenizer = loader.tokenizer

    print(f"\n{'='*50}")
    print(f"Baseline Benchmark — Llama 3.2 1B")
    print(f"{'='*50}")

    all_speeds = []

    for prompt in PROMPTS:
        tok_per_sec = measure_tokens_per_sec(model, tokenizer, prompt)

        all_speeds.append(tok_per_sec)

        print(f"{tok_per_sec:.1f} tok/s | {prompt[:40]}...")

    print(f"{'='*50}")
    print(f"Average: {sum(all_speeds)/len(all_speeds):.1f} tok/s")























