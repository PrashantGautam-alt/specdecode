# Fused Tree Decoder — Debug Journal

A running log of how we hunted and fixed the correctness bug in `medusa_decode_tree_fused`.
Written so you can read it top-to-bottom afterwards and understand **what broke, why, how we
found it, and how we fixed it** — plain language, but with the real technical details.

---

## 0. Background — why this function exists and why we care

**The goal:** make the Medusa tree decoder faster, target ≥ 1.5x over plain decoding.

**What we measured (the profiler, `scripts/profile_tree.py`):** a generation round of the
*working* tree decoder takes ~66 ms, and **94% of that is two backbone passes**:

```
propose       33.07 ms   49.8%   <- backbone pass #1 (run model on last token, heads guess)
verify_fwd    29.31 ms   44.1%   <- backbone pass #2 (run model on the guessed tree)
everything else ~4 ms      6%    <- cache bookkeeping + Python (NOT the bottleneck)
```

Key fact behind this: the 8B model is **memory-bound**. A pass mostly pays to haul ~16 GB of
weights out of GPU memory (~30 ms), and that cost is the same whether 1 token or 30 tokens ride
through. So two passes per round = two memory hauls = the waste.

**The fix idea (fusing):** the VERIFY pass already computes a hidden state for every token it
processes. The hidden state we need to seed *next* round's head-guesses is sitting in *this*
round's VERIFY output — we were throwing it away. If we reuse it, we don't need a separate
PROPOSE pass. That collapses **two passes into one → ~halves the round → ballpark ~2x**.

**The problem:** we already wrote `medusa_decode_tree_fused`, but its output does not match
plain greedy decoding (`MISMATCH`). Somewhere in the "carry leftover state between rounds"
bookkeeping, something is off. This journal is the hunt for that bug.

---

## 1. What the fused decoder is supposed to do (the invariant)

Each round carries three things forward instead of re-deriving them:

- `pending`  — the one token we already know is correct (position P), not yet written to the cache
- `seed_h`   — the hidden state at position P-1, which the heads read to guess the tree
- `cache`    — the model's saved keys/values for all committed tokens

A round feeds `[pending] ++ [tree_nodes]` through the model in ONE pass:
- `pending`'s own logits give `backbone_pred_0` = the true token at P+1 (for free)
- the tree nodes get verified against it
- we accept the longest correct path, commit those tokens, and grab the new `pending` + `seed_h`
  out of *this* pass's outputs — no second pass.

Because head 0 predicts the same position the prepended `pending` already covers, the tree is
built from heads 1..K-1 (depth K-1 = 3 instead of 4). One tree level traded for one fewer pass.

---

## 2. The plan

1. **Diagnose, don't guess.** Add a round-by-round check: compare the fused output against plain
   greedy as it generates, and stop at the *first* token that disagrees — dumping that round's
   full state (pending, true next token, the tree's guesses, what got accepted, what got
   committed, the new pending). That pins the bug to one round and one phase.
2. **Read the dump** to classify the bug: is it in the proposal (seed_h wrong), the verification
   (mask/positions wrong), or the commit/carry-forward (off-by-one in tokens/cache)?
3. **Fix** the one spot, re-run the diagnostic until no divergence.
4. **Confirm** correctness (`PASSED`) and **benchmark** the speedup.

---

## 3. Findings

### Step 1 — diagnostic added
- Added `debug` + `ref_ids` params to `medusa_decode_tree_fused` (guarded; off by default the
  hot path is unchanged). `scripts/debug_fused.py` generates a greedy reference, then runs the
  fused decoder against it and prints the first divergence with full round state.
- _(results to be filled in after the run)_
