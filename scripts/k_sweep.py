import time
import torch
from src.models import ModelLoader
from src.sampler import naive_generate, speculative_decode

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
    
    print("WARMING UP.....")
    naive_generate(target_model, tokenizer, PROMPT, max_new_tokens=10)

    print("Benchmarking naive 8B baseline...")
    naive_time = measure(lambda: naive_generate(target_model, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS))
    naive_tps = MAX_NEW_TOKENS / naive_time

    results = []
    for K in [1, 2, 4, 6, 8]:
        print(f"Benchmarking K={K}...")
        spec_time = measure(lambda: speculative_decode(draft_model, target_model, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, K=K))
        spec_tps = MAX_NEW_TOKENS / spec_time
        speedup = spec_tps / naive_tps
        results.append((K, spec_tps, speedup))

    print(f"\n{'='*45}")
    print(f"Naive 8B baseline: {naive_tps:.1f} tok/s")
    print(f"{'='*45}")
    print(f"{'K':<6} {'tok/s':<12} {'speedup'}")
    print(f"{'-'*45}")
    for K, tps, speedup in results:
        print(f"{K:<6} {tps:<12.1f} {speedup:.2f}x")
    print(f"{'='*45}")


