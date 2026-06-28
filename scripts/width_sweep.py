"""
Width sweep for the Medusa tree decoder.

The point: on a memory-bound 8B model, a backbone pass mostly pays to load the weights
from GPU memory — pushing more candidate tokens through that same pass is nearly free.
So widening the tree (more candidates per head) should raise acceptance (tokens/round)
at almost no extra backbone cost. The thing to watch is whether the PER-ROUND OVERHEAD
(building a bigger tree, walking more leaves in Python) eventually grows faster than the
acceptance gain — that's where the tok/s stops improving even as acceptance keeps rising.

For each width we report: acceptance (tokens/round), tok/s, and speedup vs naive.
"""

import time
import torch
from src.models import ModelLoader
from src.medusa import MedusaModel, medusa_decode_tree
from src.sampler import naive_generate

CHECKPOINT = "medusa_heads_8b_epoch4.pt"
MAX_NEW_TOKENS = 100
K = 4
WIDTHS = [2, 3, 4]
PROMPT = "Explain the theory of relativity in simple terms:"


def measure(fn, runs=3):
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
    # heads in float16 on cuda:0 (same setup as the main benchmark — no cross-GPU transfer)
    medusa.heads.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    medusa.heads.to(device="cuda:0", dtype=torch.float16)
    medusa.heads.eval()
    print(f"Loaded Medusa heads from {CHECKPOINT} (float16, on cuda:0)\n")

    # Greedy reference, computed once — it does not depend on width. Every tree width must
    # prefix-match this, otherwise the width's speed number is meaningless.
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        greedy_ids = backbone.generate(input_ids, max_new_tokens=40, do_sample=False, num_beams=1)
    greedy_out = tokenizer.decode(greedy_ids[0], skip_special_tokens=True)

    # Naive baseline, measured fresh on this machine so the speedup ratio is honest.
    naive_generate(backbone, tokenizer, PROMPT, max_new_tokens=10)  # warmup
    naive_time = measure(lambda: naive_generate(backbone, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS))
    naive_tps = MAX_NEW_TOKENS / naive_time
    print(f"Naive 8B baseline: {naive_tps:.1f} tok/s\n")

    results = []  # (width, num_nodes, acceptance, tok/s, passed)

    for width in WIDTHS:
        # number of nodes in a full Cartesian tree: width + width^2 + ... + width^K
        num_nodes = sum(width**d for d in range(1, K + 1))
        print(f"{'='*60}")
        print(f"WIDTH = {width}  ({num_nodes} tree nodes)")
        print(f"{'='*60}")

        # Correctness: tree output must be a prefix of (or prefixed by) greedy.
        tree_out = medusa_decode_tree(medusa, tokenizer, PROMPT, max_new_tokens=40, K=K, width=width)
        shorter, longer = sorted([greedy_out, tree_out], key=len)
        passed = longer.startswith(shorter)
        print(f"Correctness: {'PASSED' if passed else 'MISMATCH — speed number not trustworthy'}")

        # Acceptance (verbose prints the avg tokens/round line) over the full length.
        print("Acceptance:")
        medusa_decode_tree(medusa, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, K=K, width=width, verbose=True)

        # Speed.
        medusa_decode_tree(medusa, tokenizer, PROMPT, max_new_tokens=10, K=K, width=width)  # warmup
        tree_time = measure(lambda w=width: medusa_decode_tree(medusa, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, K=K, width=w))
        tree_tps = MAX_NEW_TOKENS / tree_time
        print(f"Speed: {tree_tps:.1f} tok/s  ({tree_tps/naive_tps:.2f}x)\n")

        results.append((width, num_nodes, tree_tps, passed))

    print(f"{'='*60}")
    print(f"{'width':>6} {'nodes':>7} {'tok/s':>8} {'speedup':>9} {'ok':>4}")
    print(f"{'-'*60}")
    print(f"{'naive':>6} {'-':>7} {naive_tps:>8.1f} {'1.00x':>9}")
    for width, num_nodes, tree_tps, passed in results:
        ok = "yes" if passed else "NO"
        print(f"{width:>6} {num_nodes:>7} {tree_tps:>8.1f} {tree_tps/naive_tps:>8.2f}x {ok:>4}")
    print(f"{'='*60}")
