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

### Step 1 result — first divergence at output position 14 (round 10)
```
expected (greedy): ' explains'
fused produced:    ' describes'
pending_in (token at position 25): ' that'
backbone_pred_0 (true token at 26): ' describes'   <-- the model's OWN prediction, and it's wrong
```
The fused output matched greedy exactly up to "...modern physics that". Then the model's own
next-token prediction (`backbone_pred_0`) was ' describes' where greedy gives ' explains'.

**The deduction (important):** the committed token IDs matched greedy *exactly* up to ' that'.
Same model + identical token prefix + greedy = must give the identical next token — UNLESS the
model is reading a corrupted context. The model attends to the **KV cache**, not to the
`generated` list we print for ourselves. So by round 10 the cache had drifted from the true
sequence even though the tokens we recorded looked correct. It surfaced at round 10 (not round 1)
because the error accumulated quietly and only flipped a close argmax here.

Note: the *non-fused* tree reuses KV the same way and is correct, so generic KV reuse is not the
bug — it's something **fused-specific** in how the cache is carried between rounds.

### Step 2 — pin the exact round the cache first goes wrong
Added a stronger debug check: each round, recompute the next token with a FRESH, cacheless
forward over the true prefix (`generated ++ pending`) and compare it to the cache-based
`backbone_pred_0`. The first round they disagree is the first corrupted cache. Also print the
incoming cache length vs `generated` length to catch an off-by-one directly.

### Step 2 result — the cache is NOT corrupt (surprise)
`CACHE WRONG` never fired. Every round: incoming cache length == generated length, and a fresh
cacheless recompute predicted the SAME token as the cache-based `backbone_pred_0`. So the cache
is faithful. The disagreement is between the **fused decoder (' describes')** and **HF greedy
(' explains')** on an identical prefix.

**New hypothesis — prefill vs decode fp16 numerics.** HF greedy decodes *incrementally*: one new
token at a time against a cache ("decode mode"). The fused decoder gets `backbone_pred_0` from
the `pending` token sitting inside a *parallel* pass over `[pending] + tree` ("prefill mode").
In fp16, prefill and decode do their matmuls in different shapes/reduction orders, so logits
differ in the 3rd-4th decimal. On a near-tie (' explains' vs ' describes') that flips the argmax.
The non-fused tree avoids this because it computes `backbone_pred_0` from a *separate single-token*
pass (decode mode, identical to greedy) — exactly the pass fusion removes.

### Step 3 — prove or refute it
At the diverging round, compute the token after `pending` two ways and print top-2 logits + gap:
- PARALLEL = pending inside the fused multi-query pass (what we use)
- INCREMENTAL = pending as a lone token vs the cache (decode mode == HF greedy)
If INCREMENTAL gives ' explains' and the gap is tiny, the hypothesis is confirmed.

### Step 3 result — CONFIRMED: it's an fp16 tie, not a bug
```
PARALLEL (fused pass)    top2=[(' describes', 19.6406), (' explains', 19.625)]  gap=0.0156
INCREMENTAL (decode)     top2=[(' describes', 19.625),  (' explains', 19.625)]  gap=0.0
```
' describes' and ' explains' have essentially identical logits (19.64 vs 19.625). The gap is
0.0156 in the parallel pass and a literal 0.0 tie in incremental decode. The model considers the
two words equally good. fp16 rounding differences between the parallel and incremental matmul
paths decide the argmax tie differently → the "MISMATCH" is a coin flip on a dead tie.

**Conclusion: the fused decoder was never broken.** It faithfully greedy-decodes. Our correctness
test (exact bit-match with HF greedy) was too strict — no fp16 tree decoder can match greedy
across ties. The non-fused tree only "passed" because it happened not to hit a tie on this prompt.

### Step 4 — rigorous proof across the whole generation + real speed
One divergence being a tie doesn't prove all are. `scripts/verify_fused.py` checks SELF-CONSISTENCY:
take the fused's own output, run one clean forward, and confirm every committed token is the
model's argmax or tied for it (gap < 0.5; real ties are ~0.0-0.05, a real bug would be >> 1.0).
This proves the fused output is a valid greedy decode without depending on HF's tie-breaks. Same
run times the fused decoder vs naive.

### Step 4 result — PASSED, and the honest speed verdict
```
tokens checked:            103
exact argmax matches:      102
argmax mismatches (ties):  1   (gap < 0.5)
real errors:               0
largest logit gap seen:    0.0000   (pos 26: ' describes' vs ' explains', a dead tie)
PASSED — every token is the model's argmax or tied for it. Fused IS a valid greedy decode.

naive 8B:    37.9 tok/s  (1.00x)
fused tree:  47.1 tok/s  (1.24x)
```

---

## 4. Conclusion — what we found, and what it means

**What broke:** nothing. The fused decoder was correct all along. The `MISMATCH` came from an
exact-bit-match test that can't survive fp16 ties (two tokens with identical logits, where the
parallel and incremental matmul paths round the tie-break differently). 102/103 tokens are the
exact argmax; the one exception has a 0.0000 logit gap.

**What we fixed:** our *understanding* and our *test*. `scripts/verify_fused.py` now proves
correctness the right way — self-consistency (every token is argmax-or-tied), independent of HF's
tie-breaks.

**The speed reality (be honest):** fused = 1.24x, the working tree = 1.19x. Fusing one backbone
pass away did NOT roughly double speed, even though the profiler said passes are 94% of a round.
Why: fusing the PROPOSE pass into VERIFY costs a tree level (head 0 is spent on the prepended
`pending`, so the fused tree is depth K-1=3 instead of 4). A shallower tree accepts fewer tokens
per round. On a memory-bound model a pass ≈ a fixed ~30 ms cost and a tree level ≈ acceptance —
and the two roughly cancel. Net: ~1.19x → ~1.24x, a small win, not a doubling.

**Implication for the 1.5x goal:** neither the tree nor the fused decoder reaches it at K=4.
Every speed lever we tried is now mapped:
- Wider tree (Lever A): dead — acceptance is flat in width.
- Cut overhead (Lever B): dead — overhead is only ~6% of a round.
- Fuse 2 passes -> 1 (Lever B'): marginal — trades a pass for a tree level.
- **Higher acceptance (Lever C): the only lever left with headroom.**

The structural way to raise acceptance is a **deeper tree, which needs more heads**. With more
heads (K=6), the *fused* decoder (1 pass/round) could run a depth-5 tree: one cheap pass, but
many tokens accepted per round. That combination — fused's single pass + a deep tree from more
heads — is the principled path to 1.5x and beyond. Cost: training a new, larger head set.
