# SpecDecode

Built this to understand speculative decoding from first principles, not just run someone else's implementation.
Implements three inference strategies on Llama models, measures the speedup, and visualizes what is actually happening at the token level.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red)

---

![Architecture](assets/architecture.png)
*Implementation uses K=4.*

---

## What it does

**Speculative decoding** runs a small draft model to propose K tokens at once, then verifies all K in a single pass of the big target model. If the draft was right, you get K tokens for roughly the price of one. If it was wrong, you get one corrected token and try again. The output is mathematically identical to running the big model alone.

**Medusa** takes this further by replacing the separate draft model with small MLP heads attached directly to the frozen backbone. The heads predict future tokens from the same hidden state the backbone already computed, so there is no second model to run. With tree attention (implemented here), the heads propose a tree of candidate paths that get verified in one pass instead of three.

---

## Results

Hardware: 2x NVIDIA RTX A5000 (24GB each), CUDA 12.4
Target model: Llama-3.1-8B-Instruct
Draft model: Llama-3.2-1B-Instruct

**Headline (two numbers, honest about the tradeoff):**
- **1.24–1.56x — lossless.** 4-head Medusa with a fused tree decoder under strict-greedy acceptance
  is provably bit-identical to standard greedy decoding (verified by self-consistency, 101/101 tokens
  argmax-or-tied). The **calibrated tree** (Extension A) lifts the base 1.24x to 1.34–1.56x by
  spending the node budget on high-probability candidates — still zero quality cost.
- **~1.47x — with typical acceptance (T=0.8).** Trades exact-greedy for a looser accept rule.
  Coherent across prompt types but *lossy* (temperature sampling, not identical to greedy).

| Config | tok/s | Speedup | notes |
|---|---|---|---|
| Naive 8B baseline | 37.9 | 1.00x | |
| SpecDecode K=4 (separate 1B draft) | 44.4 | 1.17x | 1B draft model, instruct |
| Medusa greedy, epoch-1 heads (3-pass) | 24.6 | 0.65x | early, undertrained heads |
| **Medusa 4-head, fused tree, greedy** | **47.0** | **1.24x** | **lossless — bit-identical to greedy decoding** |
| Medusa 6-head, fused tree, greedy | 43.3 | 1.14x | *slower*: far heads rejected under strict greedy (optimal-K overshoot) |
| **Medusa 4-head, fused tree, typical (T=0.8)** | **~55** | **~1.47x** | coherent, lossy; 1.27–1.67x across prompt types |
| **Medusa 4-head, calibrated tree, greedy** | **50.7–58.8** | **1.34–1.56x** | **lossless**; +6–15% acceptance over Cartesian, same node budget |

Two findings worth understanding from this table:
- **More heads can make greedy slower.** 6 heads (1.14x) underperform 4 heads (1.24x) under strict
  greedy: the extra far-future heads are almost always rejected, so they add tree-verification cost
  every round with near-zero accepted tokens — a live demonstration of the optimal-K cost model.
- **Typical acceptance is the only way deeper trees pay off**, but it is lossy and its safe
  temperature must be tuned against the *worst-case* prompt (a list prompt garbled at T=0.92 while
  prose stayed clean). T=0.8 is the robust operating point; speed is reported as a range across
  prompt types because acceptance depends on how predictable the text is.

---

## Architecture

```
Speculative decoding:

  [Llama 1B draft]  -->  K candidate tokens
                               |
  [Llama 8B target] -->  verify all K in one pass
                               |
                    rejection sampling (Leviathan et al.)
                               |
                    accepted tokens (output = exact 8B greedy)


Medusa (tree attention):

  [Llama 8B backbone] -->  hidden state h
                               |
              +----------------+----------------+
              |                |                |
           head 0           head 1           head 2  ...
         (top-2 at t+1)  (top-2 at t+2)  (top-2 at t+3)
              |
     tree of candidates (30 nodes, 16 paths for K=4, width=2)
              |
     one backbone pass with custom attention mask
              |
     longest matching path wins
```

---

## How to run

Install dependencies:
```bash
pip install -r requirements.txt
```

You need HuggingFace access to `meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct`.

**Naive baseline:**
```bash
PYTHONPATH=. python scripts/baseline_bench.py
```

**SpecDecode K sweep:**
```bash
PYTHONPATH=. python scripts/k_sweep.py
```

**Three-way benchmark (naive vs SpecDecode vs Medusa):**
```bash
PYTHONPATH=. python scripts/benchmark_medusa.py
```

**Tree attention benchmark:**
```bash
PYTHONPATH=. python scripts/test_medusa_tree.py
```

**FastAPI server + frontend:**
```bash
PYTHONPATH=. python scripts/run_server.py   # terminal 1
# open frontend/index.html in browser, update the server IP if running remotely
```

---

## File structure

```
src/
  models.py       model loader (handles float16, device placement, eval mode)
  sampler.py      naive_generate, speculative_decode, rejection sampling
  medusa.py       MedusaHead, MedusaModel, medusa_decode, medusa_decode_tree,
                  build_tree_candidates, build_tree_mask
  server.py       FastAPI server (POST /generate, WS /stream)

scripts/
  baseline_bench.py       measure naive tok/s
  compare_speed.py        naive vs speculative head to head
  k_sweep.py              sweep K from 1 to 8
  benchmark_medusa.py     three-way benchmark table
  test_medusa_tree.py     tree attention correctness check + benchmark
  train_medusa.py         toy training script (1B, for sanity checks)
  train_medusa_8b.py      real training (8B, UltraChat 10k/25k)
  run_server.py           load model and start uvicorn
  test_server.py          test both server endpoints

frontend/
  index.html      token visualization UI (vanilla JS, no framework)
```

---

## Training the Medusa heads

Backbone stays frozen. Only the 4 heads train.

```bash
# toy run on 5 paragraphs (sanity check, runs in minutes)
PYTHONPATH=. python scripts/train_medusa.py

# real run on UltraChat 25k, 2 epochs (takes a few hours on A5000)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. python scripts/train_medusa_8b.py
```

Heads are saved as `medusa_heads_8b_epoch{n}.pt`. Not included in the repo (too large), but the training script reproduces them.

---

## References

- Leviathan et al. (2023) - Fast Inference from Transformers via Speculative Decoding. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Cai et al. (2024) - Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
- Dao et al. (2022) - FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
