"""
Profile one full generation of the Medusa tree decoder, phase by phase.

A round takes ~67 ms but two backbone passes don't account for it. This runs the real
medusa_decode_tree with profile=True, which syncs the GPU at each phase boundary and reports
average ms/round for: propose, build_tree, verify, find_path, cache_update.

Read it like this:
  - propose + verify   = the two backbone passes (actual model compute)
  - build_tree         = Python: topk per head, building the node list + attention mask
  - find_path          = Python: walking the leaves to pick the best accepted path
  - cache_update       = KV extraction / cache rebuild (tensor copies that grow with sequence)

Whichever non-backbone phase is large is the overhead to attack for Lever B. The phase syncs
inflate absolute ms a little, so trust the PERCENTAGES, not the raw totals.
"""

import torch
from src.models import ModelLoader
from src.medusa import MedusaModel, medusa_decode_tree

CHECKPOINT = "medusa_heads_8b_epoch4.pt"
MAX_NEW_TOKENS = 100
K = 4
WIDTH = 2
PROMPT = "Explain the theory of relativity in simple terms:"


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

    # Warm up so we don't profile one-time CUDA kernel compilation / allocator growth.
    medusa_decode_tree(medusa, tokenizer, PROMPT, max_new_tokens=10, K=K, width=WIDTH)

    print(f"Profiling {MAX_NEW_TOKENS} tokens at width={WIDTH}...")
    medusa_decode_tree(
        medusa, tokenizer, PROMPT, max_new_tokens=MAX_NEW_TOKENS, K=K, width=WIDTH, profile=True
    )
