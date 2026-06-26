# SpecDecode

Built this to understand speculative decoding from first principles, not just run someone else's implementation.
Implements three inference strategies on Llama models, measures the speedup, and visualizes what is actually happening at the token level.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red)

---

## What it does

**Speculative decoding** runs a small draft model to propose K tokens at once, then verifies all K in a single pass of the big target model. If the draft was right, you get K tokens for roughly the price of one. If it was wrong, you get one corrected token and try again. The output is mathematically identical to running the big model alone.

**Medusa** takes this further by replacing the separate draft model with small MLP heads attached directly to the frozen backbone. The heads predict future tokens from the same hidden state the backbone already computed, so there is no second model to run. With tree attention (implemented here), the heads propose a tree of candidate paths that get verified in one pass instead of three.

---

## Results

Hardware: 2x NVIDIA RTX A5000 (24GB each), CUDA 12.4
Target model: Llama-3.1-8B-Instruct
Draft model: Llama-3.2-1B-Instruct

| Config | tok/s | Speedup |
|---|---|---|
| Naive 8B baseline | 37.9 | 1.00x |
| SpecDecode K=4 (1B draft) | 44.4 | 1.17x |
| Medusa 4-head greedy | 26.4 | 0.69x |
| Medusa 4-head tree (width=2) | pending | pending |

Medusa greedy is slower than naive because our implementation does 3 backbone passes per round (propose + verify + cache update). Tree attention collapses those into one. Benchmark pending on restored server.

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
