# Post-Training Runbook — 6-Head Medusa + Typical Acceptance

**Read this when `scripts/train_medusa_6head.py` finishes.** Goal of this phase: push the fused
tree decoder to **≥ 1.5x** using 6 heads (a depth-5 fused tree) and typical acceptance.

All commands run **on passpoli**, in `~/specdecode`, with the venv active
(`source venv/bin/activate`). Prefix GPU runs with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Background context (why we're here) is in `DAILY_LOG.md` (Session 2026-06-27) and the bug story is
in `FUSED_DEBUG_LOG.md`.

---

## Step 0 — Confirm training finished and is healthy
```
tmux attach -t train6          # detach again with: Ctrl-b then d
ls -la medusa_heads_8b_6head_epoch*.pt
```
- Expect `epoch0 .. epoch3`. Newest = **`medusa_heads_8b_6head_epoch3.pt`**.
- In `train6.log`: loss should have **fallen each epoch** and never printed `nan`.
  - If NaN / spiked: lower the LR (`5e-4 → 2e-4`) in `scripts/train_medusa_6head.py` and retrain.
  - If passpoli rebooted: resume from the newest saved epoch (the script saves every epoch).

## Step 1 — Sync the code
```
git pull
```

## Step 2 — Sanity-check typical acceptance on the OLD 4-head model first
This validates the typical-acceptance code before we trust it on 6 heads.
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. python scripts/test_typical.py
```
Look for: **typical acceptance (tokens/round) HIGHER than greedy**, and typical output still
coherent (not garbage). If typical is garbage → stop, tell Claude (likely a temperature/threshold
issue, tune `temperature` / `epsilon` / `delta` in `scripts/test_typical.py`).

## Step 3 — Point the test scripts at the 6-head checkpoint
The scripts hardcode `CHECKPOINT` and `K`. Update both:
```
sed -i 's#medusa_heads_8b_epoch4.pt#medusa_heads_8b_6head_epoch3.pt#' scripts/test_typical.py scripts/verify_fused.py
sed -i 's/^K = 4/K = 6/' scripts/test_typical.py scripts/verify_fused.py
```
Confirm:
```
grep -nE '^(CHECKPOINT|K) ' scripts/test_typical.py scripts/verify_fused.py
```
(Each should show the 6-head checkpoint and `K = 6`.)
Note: the 6-head checkpoint was trained in **bf16**; the scripts load it and cast to fp16 for
inference — that's fine.

## Step 4 — THE 1.5x ATTEMPT: 6 heads + fused + typical (depth-5 tree)
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. python scripts/test_typical.py
```
- K=6 → fused tree depth 5. With typical acceptance, tokens/round should rise well above 3.
- **Target: ≥ 56.9 tok/s (= 1.5x over the 37.9 naive baseline).**
- Compare the `greedy` vs `typical` rows it prints: how far does each get?

## Step 5 — Lossless ceiling check (greedy, 6 heads)
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. python scripts/verify_fused.py
```
- Confirms the 6-head **greedy** fused decoder is still a valid greedy decode (`PASSED`, all ties),
  and shows the lossless speedup. This is the number to quote if you want to stay lossless.

## Step 6 — If ≥ 1.5x: bank it
- Update `DAILY_LOG.md` key-numbers table, `README.md`, and the report tables with the 6-head numbers.
- Do NOT commit `.pt` files (gitignored, 8 GB+). They live on passpoli only.

---

## Memory caveat (inference, 6 fp16 heads on one card)
6 fp16 heads (~6.5 GB) + backbone (~16 GB) ≈ 22.5 GB on a 24 GB A5000 — fits, but tight. If you hit
OOM during inference, either (a) shorten the prompt / `max_new_tokens`, or (b) move heads to `cuda:1`
(`.to("cuda:1")` in the script) — but that reintroduces the cross-GPU transfer that cost us ~0.1x.

## If anything breaks — how to UNDO
- `accept_mode` defaults to `"greedy"` everywhere, so the **verified strict-greedy path is untouched**
  and remains the fallback. Just don't pass `accept_mode="typical"`.
- To remove typical acceptance entirely: `git log --oneline | grep typical`, then
  `git revert <that-hash>`. The greedy decoder is unaffected.
- The original 4-head checkpoint and all 4-head results are unchanged and still valid.

## What "success" looks like
- **Best case:** typical + 6 heads ≥ 1.5x (ideally approaching ~2x). Ship typical, document the
  quality tradeoff.
- **Acceptable:** greedy + 6 heads gives a lossless bump over 1.24x even if < 1.5x; typical clears
  1.5x. Offer both, let the use case decide.
- **If still short:** the next levers are a calibrated sparse tree (CONCEPTS.md "Calibrated optimal
  tree") and/or more training epochs for the new heads.
