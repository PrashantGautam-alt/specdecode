import time
import torch
from src.models import ModelLoader
from src.medusa import MedusaModel, medusa_decode, medusa_decode_tree
from src.sampler import naive_generate

CHECKPOINT = "medusa_heads_8b_epoch1.pt"
MAX_NEW_TOKENS = 100
K = 4
WIDTH = 2
PROMPT = "Explain the theory of relativity in simple terms:"

NAIVE_TPS = 37.9
SPEC_TPS = 44.4
SPEC_SPEEDUP = 1.17


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
    medusa.heads.to(device="cuda:1")
    medusa.heads.load_state_dict(torch.load(CHECKPOINT, map_location="cuda:1"))
    medusa.heads.eval()
    print(f"Loaded Medusa heads from {CHECKPOINT}")

    # --- Correctness check ---
    # medusa_decode_tree must produce identical output to plain greedy decoding.
    # If it doesn't, the accept/reject logic or cache handling has a bug.
    print("\n=== Correctness Check ===")
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        greedy_ids = backbone.generate(input_ids, max_new_tokens=40, do_sample=False, num_beams=1)
    greedy_out = tokenizer.decode(greedy_ids[0], skip_special_tokens=True)

    tree_out = medusa_decode_tree(medusa, tokenizer, PROMPT, max_new_tokens=40, K=K, width=WIDTH, verbose=True)

    # tree decode appends a whole accepted path per round, so it can overshoot
    # max_new_tokens by a few tokens while greedy stops exactly at 40. that makes the
    # outputs different LENGTHS even when correct. the real test: does the shorter
    # output match the longer one character-for-character up to its length (is it a prefix)?
    shorter, longer = sorted([greedy_out, tree_out], key=len)
    prefix_match = longer.startswith(shorter)

    print(f"\nGreedy output: {greedy_out}")
    print(f"Tree output:   {tree_out}")
    print(f"Outputs match (prefix): {prefix_match}")
    if not prefix_match:
        print("MISMATCH — do not trust the benchmark numbers below.")
    else:
        print("PASSED — tree output matches greedy decoding (tree overshot by a few tokens, which is expected).")

    # --- Benchmark ---
    print("\nWarming up...")
    naive_generate(backbone, tokenizer, PROMPT, max_new_tokens=10)
    medusa_decode(medusa, tokenizer, PROMPT, max_new_tokens=10, K=K)
    medusa_decode_tree(medusa, tokenizer, PROMPT, max_new_tokens=10, K=K, width=WIDTH)

    print("Benchmarking naive 8B...")
    naive_time = measure(lambda: naive_generate(backbone, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS))
    naive_tps = MAX_NEW_TOKENS / naive_time

    print("Benchmarking Medusa greedy (3-pass)...")
    greedy_time = measure(lambda: medusa_decode(medusa, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, K=K))
    greedy_tps = MAX_NEW_TOKENS / greedy_time

    print("Benchmarking Medusa tree attention...")
    tree_time = measure(lambda: medusa_decode_tree(medusa, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, K=K, width=WIDTH))
    tree_tps = MAX_NEW_TOKENS / tree_time

    print("\nAcceptance rate — tree (verbose pass):")
    medusa_decode_tree(medusa, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, K=K, width=WIDTH, verbose=True)

    print(f"\n{'='*68}")
    print(f"{'Config':<32} {'tok/s':>8}  {'speedup':>8}  {'notes'}")
    print(f"{'-'*68}")
    print(f"{'Naive 8B (baseline)':<32} {NAIVE_TPS:>8.1f}  {'1.00x':>8}")
    print(f"{'SpecDecode K=4':<32} {SPEC_TPS:>8.1f}  {SPEC_SPEEDUP:>7.2f}x  1B draft, instruct")
    print(f"{'Medusa greedy (3-pass)':<32} {greedy_tps:>8.1f}  {greedy_tps/NAIVE_TPS:>7.2f}x  epoch1")
    print(f"{'Medusa tree (width=2)':<32} {tree_tps:>8.1f}  {tree_tps/NAIVE_TPS:>7.2f}x  epoch1")
    print(f"{'='*68}")
