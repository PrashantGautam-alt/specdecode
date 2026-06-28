"""
Extension A — calibrated tree attention.

The naive tree spends its node budget uniformly: top-`width` candidates at every head,
every combination. But heads differ in accuracy, and rank-0 of a confident head is worth far
more than rank-1 of a weak one. This script:

  1. CALIBRATE: measure a[d][i] = P(tree-depth-d head's i-th-ranked guess is correct), on
     in-distribution text (the model's own greedy output).
  2. SELECT: greedily fill a fixed node budget with the highest path-accuracy nodes
     (path accuracy = product of a[d][i] along the path). This is Prim's algorithm — grow the
     tree by repeatedly attaching the best-connected node — maximizing acceptance, not min cost.
  3. SAVE the topology to JSON, then COMPARE Cartesian vs calibrated (same budget) on test prompts.

Run on the GPU box:
    PYTHONPATH=. python scripts/calibrate_tree.py
"""
import json
import time

import torch

from src.models import ModelLoader
from src.medusa import (
    MedusaModel,
    medusa_decode_tree_fused,
)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CHECKPOINT = "medusa_heads_8b_epoch4.pt"
K = 4
WIDTH = 2          # Cartesian baseline width -> defines the node budget we must match
R = 4              # calibrate the top-R candidates per head (lets the smart tree pick rank 0..3)
DEVICE = "cuda:0"
OUT = "tree_spec_4head.json"
GEN_TOKENS = 150   # how many tokens of in-distribution text to generate per calibration prompt

CALIBRATION_PROMPTS = [
    "Explain how a neural network learns from data:",
    "Describe the process of photosynthesis step by step:",
    "Write a short story about a traveler lost in a desert:",
    "List the main causes of the French Revolution:",
]
TEST_PROMPTS = [
    "Explain the theory of relativity in simple terms:",
    "List the steps to make a good cup of tea:",
]


def greedy_topology(a, tree_depth, budget):
    """
    Prim's-style greedy node selection. Repeatedly add the candidate node with the highest
    cumulative path accuracy; a node becomes a candidate only once its parent is in the tree
    (so parents always precede children -> the output order is valid for the decoder).

    a:          a[d][i] accuracy table, tree_depth rows x R cols
    budget:     number of nodes to select
    Returns:    list of [depth, rank, parent_idx], one per selected node
    """
    import heapq
    R_ = len(a[0])
    # frontier entries: (-path_accuracy, tiebreak, depth, rank, parent_idx)
    frontier = []
    counter = 0
    for rank in range(R_):
        heapq.heappush(frontier, (-a[0][rank], counter, 0, rank, -1))
        counter += 1

    selected = []
    while frontier and len(selected) < budget:
        neg_acc, _, depth, rank, parent = heapq.heappop(frontier)
        node_idx = len(selected)
        selected.append([depth, rank, parent])
        path_acc = -neg_acc
        if depth + 1 < tree_depth:
            for r in range(R_):
                heapq.heappush(frontier, (-(path_acc * a[depth + 1][r]), counter, depth + 1, r, node_idx))
                counter += 1
    return selected


def avg_tokens_per_round(medusa, tokenizer, prompt, tree_spec):
    """Run the fused greedy decoder once and return (avg tokens/round, tok/s)."""
    n_committed = [0]
    n_rounds = [0]

    # count via on_token: each round emits exactly one backbone token (accepted=False), plus a
    # single trailing False for the final pending -> rounds = (False count) - 1.
    def counter(_tok, accepted):
        n_committed[0] += 1
        if accepted is False:
            n_rounds[0] += 1

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    medusa_decode_tree_fused(
        medusa, tokenizer, prompt, max_new_tokens=100, K=K, width=WIDTH,
        accept_mode="greedy", tree_spec=tree_spec, on_token=counter,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    rounds = max(n_rounds[0] - 1, 1)
    return n_committed[0] / rounds, n_committed[0] / elapsed


if __name__ == "__main__":
    print(f"Loading {MODEL} + {K} heads...")
    loader = ModelLoader(MODEL, device=DEVICE)
    loader.load()
    backbone, tokenizer = loader.model, loader.tokenizer
    medusa = MedusaModel(backbone, num_heads=K)
    medusa.heads.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    medusa.heads.to(device=DEVICE, dtype=torch.float16)
    medusa.heads.eval()
    head_device = medusa.heads[0].W1.weight.device
    head_dtype = medusa.heads[0].W1.weight.dtype

    from src.sampler import naive_generate

    tree_depth = K - 1
    correct = [[0] * R for _ in range(tree_depth)]
    total = [[0] * R for _ in range(tree_depth)]

    # 1. CALIBRATE on the model's own greedy continuations (in-distribution text)
    print("Calibrating on in-distribution text...")
    for prompt in CALIBRATION_PROMPTS:
        text = naive_generate(backbone, tokenizer, prompt, max_new_tokens=GEN_TOKENS)
        ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
        seq = ids.shape[1]
        with torch.no_grad():
            out = backbone(input_ids=ids, output_hidden_states=True)
        h = out.hidden_states[-1].to(device=head_device, dtype=head_dtype)  # [1, seq, dim]
        for d in range(tree_depth):
            # head_logits[d] uses medusa.heads[d+1]; it predicts the token at offset d+2
            logits = medusa.heads[d + 1](h)[0]                  # [seq, vocab]
            topk_idx = logits.topk(R, dim=-1).indices           # [seq, R]
            offset = d + 2
            if seq <= offset:
                continue
            pred = topk_idx[: seq - offset]                     # [seq-offset, R]
            truth = ids[0, offset:]                             # [seq-offset]
            for i in range(R):
                correct[d][i] += (pred[:, i] == truth).sum().item()
                total[d][i] += (seq - offset)

    a = [[correct[d][i] / max(total[d][i], 1) for i in range(R)] for d in range(tree_depth)]

    print("\nPer-head accuracy a[depth][rank] (depth d uses head d+1):")
    for d in range(tree_depth):
        print(f"  depth {d} (head {d+1}): " + "  ".join(f"r{i}={a[d][i]:.3f}" for i in range(R)))

    # 2. SELECT — budget = the Cartesian node count for this width/depth, for a fair comparison
    budget = sum(WIDTH ** (d + 1) for d in range(tree_depth))
    spec = greedy_topology(a, tree_depth, budget)
    print(f"\nNode budget (matches Cartesian width={WIDTH}, depth={tree_depth}): {budget}")
    depth_counts = [sum(1 for s in spec if s[0] == d) for d in range(tree_depth)]
    print(f"Calibrated allocation per depth: {depth_counts}   (Cartesian was {[WIDTH**(d+1) for d in range(tree_depth)]})")

    json.dump({"K": K, "R": R, "width": WIDTH, "budget": budget, "accuracy": a, "tree_spec": spec},
              open(OUT, "w"), indent=2)
    print(f"Saved topology -> {OUT}")

    # 3. COMPARE Cartesian vs calibrated (same budget) on test prompts
    print("\n=== Cartesian vs Calibrated (greedy, lossless) ===")
    for prompt in TEST_PROMPTS:
        cart_tpr, cart_tps = avg_tokens_per_round(medusa, tokenizer, prompt, tree_spec=None)
        cal_tpr, cal_tps = avg_tokens_per_round(medusa, tokenizer, prompt, tree_spec=spec)
        print(f"\nprompt: {prompt!r}")
        print(f"  Cartesian : {cart_tpr:.2f} tok/round   {cart_tps:.1f} tok/s")
        print(f"  Calibrated: {cal_tpr:.2f} tok/round   {cal_tps:.1f} tok/s   "
              f"({100*(cal_tpr-cart_tpr)/cart_tpr:+.0f}% acceptance)")
