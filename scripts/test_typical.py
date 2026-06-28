"""
Early read on typical acceptance, using the CURRENT 4-head checkpoint (while 6-head trains).

Runs the fused decoder two ways and compares acceptance (tokens/round), speed, and output:
  - greedy  (accept_mode="greedy")  -> lossless, our verified baseline
  - typical (accept_mode="typical") -> Medusa-paper acceptance, temperature sampling

We can't check typical against greedy token-for-token (it samples), so the checks are:
  1. typical output is coherent text (not garbage) -> the accept logic is sane
  2. typical acceptance (tokens/round) is HIGHER than greedy -> deeper trees will pay off
  3. speed: does the higher acceptance translate to higher tok/s?

This is the 4-head preview. The real test is 6 heads (depth-5 tree) once training finishes.
"""

import time
import torch
from src.models import ModelLoader
from src.medusa import MedusaModel, medusa_decode_tree_fused
from src.sampler import naive_generate

CHECKPOINT = "medusa_heads_8b_epoch4.pt"
MAX_NEW_TOKENS = 100
K = 4
WIDTH = 2
TEMPERATURE = 1.0
PROMPT = "Explain the theory of relativity in simple terms:"


def measure(fn, runs=3):
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


if __name__ == "__main__":
    loader = ModelLoader("meta-llama/Llama-3.1-8B-Instruct", device="cuda:0")
    loader.load()
    backbone = loader.model
    tokenizer = loader.tokenizer

    medusa = MedusaModel(backbone, num_heads=K)
    medusa.heads.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    medusa.heads.to(device="cuda:0", dtype=torch.float16)
    medusa.heads.eval()
    print(f"Loaded Medusa heads from {CHECKPOINT} (float16, on cuda:0)\n")

    naive_generate(backbone, tokenizer, PROMPT, max_new_tokens=10)  # warmup
    naive_tps = MAX_NEW_TOKENS / measure(lambda: naive_generate(backbone, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS))
    print(f"naive 8B: {naive_tps:.1f} tok/s\n")

    for mode in ("greedy", "typical"):
        print(f"{'='*60}\nACCEPT MODE: {mode}\n{'='*60}")
        torch.manual_seed(0)  # reproducible sampling for the typical run
        out = medusa_decode_tree_fused(
            medusa, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, K=K, width=WIDTH,
            verbose=True, accept_mode=mode, temperature=TEMPERATURE,
        )
        print(f"output: {out!r}")
        medusa_decode_tree_fused(medusa, tokenizer, PROMPT, max_new_tokens=10, K=K, width=WIDTH, accept_mode=mode)  # warmup
        tps = MAX_NEW_TOKENS / measure(
            lambda m=mode: medusa_decode_tree_fused(medusa, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, K=K, width=WIDTH, accept_mode=m, temperature=TEMPERATURE)
        )
        print(f"speed: {tps:.1f} tok/s  ({tps/naive_tps:.2f}x)\n")
