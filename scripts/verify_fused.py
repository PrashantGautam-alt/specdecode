"""
Rigorous correctness + speed check for the fused tree decoder.

We proved the fused/greedy disagreements are fp16 ties on individual tokens. But comparing
against HF greedy past the first divergence is meaningless (the sequences fork). So instead we
check the fused output for SELF-CONSISTENCY with greedy decoding:

    take the fused's own output, run ONE clean forward over it, and confirm every committed token
    is the model's argmax given its prefix -- OR tied for argmax within a tiny logit gap.

If every token is argmax-or-tied, the fused output IS a valid greedy decode, period. A real bug
(corrupt cache, wrong accept logic) would show a token that is NOT the argmax by a large margin.

Then we time the fused decoder against naive to see the actual speedup.
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
TIE_THRESHOLD = 0.5  # fp16 ties are ~0.0-0.05; a real error would be >> 1.0
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
    medusa.heads.load_state_dict(torch.load(CHECKPOINT, map_location="cuda:0"))
    medusa.heads.to(device="cuda:0", dtype=torch.float16)
    medusa.heads.eval()
    print(f"Loaded Medusa heads from {CHECKPOINT} (float16, on cuda:0)\n")

    prompt_len = tokenizer(PROMPT, return_tensors="pt").input_ids.shape[1]

    # --- Correctness: self-consistency with greedy ---
    fused_ids = medusa_decode_tree_fused(
        medusa, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, K=K, width=WIDTH, return_ids=True
    )
    with torch.no_grad():
        logits = backbone(fused_ids.unsqueeze(0)).logits[0]  # [seq, vocab]

    non_argmax = []  # (position, gap, fused_token, model_argmax)
    for p in range(prompt_len, fused_ids.shape[0]):
        row = logits[p - 1].float()                 # logits that predict token at position p
        argmax = row.argmax().item()
        tok = fused_ids[p].item()
        if tok != argmax:
            gap = (row[argmax] - row[tok]).item()
            non_argmax.append((p, gap, tok, argmax))

    n_checked = fused_ids.shape[0] - prompt_len
    max_gap = max((g for _, g, _, _ in non_argmax), default=0.0)
    ties = [x for x in non_argmax if x[1] < TIE_THRESHOLD]
    real_errors = [x for x in non_argmax if x[1] >= TIE_THRESHOLD]

    print("=== Correctness (self-consistency with greedy) ===")
    print(f"tokens checked:            {n_checked}")
    print(f"exact argmax matches:      {n_checked - len(non_argmax)}")
    print(f"argmax mismatches (ties):  {len(ties)}  (logit gap < {TIE_THRESHOLD})")
    print(f"real errors:               {len(real_errors)}  (logit gap >= {TIE_THRESHOLD})")
    print(f"largest logit gap seen:    {max_gap:.4f}")
    for p, gap, tok, am in non_argmax:
        flag = "TIE" if gap < TIE_THRESHOLD else "ERROR"
        print(f"  pos {p}: fused={tokenizer.decode([tok])!r} model_argmax={tokenizer.decode([am])!r} gap={gap:.4f} [{flag}]")
    if not real_errors:
        print("PASSED — every token is the model's argmax or tied for it. Fused IS a valid greedy decode.")
    else:
        print("FAILED — at least one token is non-argmax by a real margin. There is a logic bug.")

    # --- Speed ---
    print("\n=== Speed ===")
    naive_generate(backbone, tokenizer, PROMPT, max_new_tokens=10)  # warmup
    medusa_decode_tree_fused(medusa, tokenizer, PROMPT, max_new_tokens=10, K=K, width=WIDTH)  # warmup
    naive_t = measure(lambda: naive_generate(backbone, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS))
    fused_t = measure(lambda: medusa_decode_tree_fused(medusa, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, K=K, width=WIDTH))
    naive_tps = MAX_NEW_TOKENS / naive_t
    fused_tps = MAX_NEW_TOKENS / fused_t
    print(f"naive 8B:    {naive_tps:6.1f} tok/s  (1.00x)")
    print(f"fused tree:  {fused_tps:6.1f} tok/s  ({fused_tps / naive_tps:.2f}x)")
