

import time
import torch
from src.models import ModelLoader
from src.sampler import naive_generate, speculative_decode

def measure(fn, runs=3):
    times = []
    for _ in range(runs):
        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")
        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)

MAX_NEW_TOKENS = 100
PROMPT = "Explain the theory of relativity in simple terms:"

if __name__ == "__main__":
    draft_loader = ModelLoader("meta-llama/Llama-3.2-1B", device="cuda:0")
    draft_loader.load()

    target_loader = ModelLoader("meta-llama/Llama-3.1-8B-Instruct", device="cuda:1")
    target_loader.load()

    draft_model = draft_loader.model
    target_model = target_loader.model
    tokenizer = target_loader.tokenizer

    print("\nWarming up...")
    naive_generate(target_model, tokenizer, PROMPT, max_new_tokens=10)

    print("Benchmarking naive (8B only)...")
    naive_time = measure(lambda: naive_generate(target_model, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS))
    naive_tps = MAX_NEW_TOKENS / naive_time

    print("Benchmarking speculative (1B draft + 8B target)...")
    spec_time = measure(lambda: speculative_decode(draft_model, target_model, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, K=4))
    spec_tps = MAX_NEW_TOKENS / spec_time

    print(f"\n{'='*50}")
    print(f"Naive 8B:        {naive_tps:.1f} tok/s")
    print(f"Speculative K=4: {spec_tps:.1f} tok/s")
    print(f"Speedup:         {spec_tps/naive_tps:.2f}x")
    print(f"{'='*50}")


