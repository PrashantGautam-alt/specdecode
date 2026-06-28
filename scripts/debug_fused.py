"""
Localize the correctness bug in medusa_decode_tree_fused.

Generates a plain-greedy reference, then runs the fused decoder in debug mode against it.
The fused decoder stops at the FIRST token that disagrees with greedy and prints that round's
full state, so we can see whether the bug is in the proposal, the verification, or the commit.
"""

import torch
from src.models import ModelLoader
from src.medusa import MedusaModel, medusa_decode_tree_fused

CHECKPOINT = "medusa_heads_8b_epoch4.pt"
N_TOKENS = 60
K = 4
WIDTH = 2
PROMPT = "Explain the theory of relativity in simple terms:"


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

    # Greedy reference. ref_ids is ONLY the generated continuation (prompt stripped), so it lines
    # up index-for-index with what the fused decoder appends after the prompt.
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    prompt_len = input_ids.shape[1]
    with torch.no_grad():
        greedy_ids = backbone.generate(input_ids, max_new_tokens=N_TOKENS, do_sample=False, num_beams=1)
    ref_ids = greedy_ids[0, prompt_len:].tolist()
    print(f"Greedy reference ({len(ref_ids)} tokens):")
    print(f"  {tokenizer.decode(ref_ids)!r}\n")

    print("Running fused decoder against the reference...")
    out = medusa_decode_tree_fused(
        medusa, tokenizer, PROMPT, max_new_tokens=N_TOKENS, K=K, width=WIDTH,
        debug=True, ref_ids=ref_ids,
    )
    print(f"\nFused output so far:\n  {out!r}")
