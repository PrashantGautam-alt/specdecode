import time
import torch
from src.models import ModelLoader
from src.medusa import MedusaModel, medusa_decode
from src.sampler import naive_generate

CHECKPOINT = "medusa_heads_8b_epoch1.pt"
MAX_NEW_TOKENS = 100
K = 4
PROMPT = "Explain the theory of relativity in simple terms:"

# SpecDecode K=4 result from Day 5 — measured against same baseline, same prompt family
SPEC_TPS = 44.4
SPEC_SPEEDUP = 1.17


def measure(fn, runs=3):
    """
    Run fn() `runs` times, synchronize CUDA before/after each run so GPU work
    is actually complete before we stop the clock. Return the mean wall time.
    """
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    return sum(times) / len(times)


if __name__ == "__main__":
    loader = ModelLoader("meta-llama/Llama-3.1-8B-Instruct", device="cuda:0")
    loader.load()
    backbone = loader.model
    tokenizer = loader.tokenizer

    medusa = MedusaModel(backbone, num_heads=K)
    medusa.heads.to(device="cuda:1")
    medusa.heads.load_state_dict(torch.load(CHECKPOINT, map_location="cuda:1"))
    medusa.heads.eval()
    print(f"Loaded Medusa heads from {CHECKPOINT}")

    print("\nWarming up...")
    naive_generate(backbone, tokenizer, PROMPT, max_new_tokens=10)
    medusa_decode(medusa, tokenizer, PROMPT, max_new_tokens=10, K=K)

    print("Benchmarking naive 8B...")
    naive_time = measure(lambda: naive_generate(backbone, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS))
    naive_tps = MAX_NEW_TOKENS / naive_time

    print("Benchmarking Medusa 4-head...")
    medusa_time = measure(lambda: medusa_decode(medusa, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, K=K))
    medusa_tps = MAX_NEW_TOKENS / medusa_time

    # One verbose run to capture acceptance rate
    print("\nAcceptance rate (verbose pass):")
    medusa_decode(medusa, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, K=K, verbose=True)

    print(f"\n{'='*62}")
    print(f"{'Config':<28} {'tok/s':>8}  {'speedup':>8}  {'notes'}")
    print(f"{'-'*62}")
    print(f"{'Naive 8B (baseline)':<28} {naive_tps:>8.1f}  {'1.00x':>8}")
    print(f"{'SpecDecode K=4 (Day 5)':<28} {SPEC_TPS:>8.1f}  {SPEC_SPEEDUP:>7.2f}x  1B draft, instruct")
    print(f"{'Medusa 4-head':<28} {medusa_tps:>8.1f}  {medusa_tps/naive_tps:>7.2f}x  epoch1 checkpoint")
    print(f"{'='*62}")
