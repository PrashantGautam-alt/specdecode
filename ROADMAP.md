# SpecDecode — Project Roadmap

**Owner:** Prashant Gautam, IEOR IIT Bombay (Roll: 24B4526, Batch: 2024–2028)  
**Goal:** Build a speculative decoding inference accelerator from scratch on NVIDIA A5000 GPUs.  
**Duration:** 18 days core (extended from original 14)  
**The standard:** Every line defensible in an interview. Every number measured on real hardware.

> "Do not start this until you can explain rejection sampling out loud without notes.
> One finished project with a live demo beats two half-finished projects every time."

---

## What You Will Build

By Day 18, you will have:

1. **Core speculative sampling loop** — draft-verify-accept cycle implemented from scratch using PyTorch, no shortcuts.
2. **Fixed-K speculative decoding** — the original Leviathan et al. 2023 algorithm, measured and benchmarked.
3. **Adaptive-K speculative decoding** — your original engineering contribution. K adjusts dynamically based on sliding-window acceptance rate, framed as a throughput optimization problem. This is the novel piece.
4. **Medusa heads variant** — lightweight MLP heads attached to the base model, predicting multiple future tokens simultaneously.
5. **FastAPI streaming server** — REST + WebSocket API that any client can call.
6. **Real-time token visualization** — browser UI showing accepted tokens (green) and rejected tokens (red) live.
7. **Comprehensive benchmark suite** — four-way comparison with real numbers on real hardware.
8. **2-page technical report** — written like an engineering paper: problem, method, results, analysis.
9. **Clean GitHub repository** — documented, reproducible, something you are proud to show.

---

## Resume Bullet (Earn This)

> "Built a speculative decoding inference accelerator from scratch on NVIDIA A5000 GPUs achieving
> [FILL IN]× throughput improvement over autoregressive baseline. Implemented adaptive draft length
> via sliding-window acceptance rate optimization (original contribution), reducing unnecessary draft
> compute by [FILL IN]% vs fixed-K. Includes real-time token visualization dashboard and REST API.
> Zero quality degradation verified on MT-Bench."
>
> *(Fill in the bracketed numbers on Day 16 once benchmarks are complete.)*

---

## Final Benchmark Table (Fill In As You Go)

| Config | Draft Model | Target Model | tok/s | Speedup | Accept Rate | Notes |
|---|---|---|---|---|---|---|
| Baseline | — | Llama 3.1 8B Instruct | | 1× | — | Autoregressive |
| SpecDecode Fixed-K | Llama 3.2 1B | Llama 3.1 8B Instruct | | | | Best K from sweep |
| SpecDecode Adaptive-K | Llama 3.2 1B | Llama 3.1 8B Instruct | | | | Dynamic K |
| Medusa 4-head | 4 heads | Llama 3.1 8B Instruct | | | | Single model |

---

## Project File Structure

```
specdecode/
  src/
    models.py          # ModelLoader class — loads draft + target models
    sampler.py         # naive_generate(), speculative_decode(), adaptive_speculative_decode()
    adaptive_k.py      # AdaptiveKController — the novel contribution
    medusa.py          # MedusaHead, MedusaModel, medusa_decode()
    benchmark.py       # Timing, throughput, acceptance rate measurement utilities
    server.py          # FastAPI + WebSocket server
    utils.py           # Logging, timing helpers
  frontend/
    index.html         # Token visualization UI
    stream.js          # WebSocket client
  scripts/
    baseline_bench.py  # Measure naive autoregressive speed
    compare_speed.py   # Fixed-K vs naive comparison
    adaptive_bench.py  # Adaptive-K vs Fixed-K comparison
    run_bench.py       # Full four-way benchmark suite
    train_medusa.py    # Fine-tune Medusa heads
    eval_mtbench.py    # MT-Bench quality evaluation
  notebooks/
    01_explore_generation.ipynb
    02_rejection_sampling_proof.ipynb
    03_adaptive_k_analysis.ipynb
    04_benchmark_analysis.ipynb
  report/
    report.md          # 2-page technical report
    figures/           # All benchmark plots as PNGs
  requirements.txt
  README.md
  ROADMAP.md           # This file
  DAILY_LOG.md         # Daily progress (always up to date)
```

---

## System Architecture

```
Browser (JavaScript UI)
        | WebSocket: token stream
        | (green = accepted, red = rejected, stats bar live)
        v
FastAPI Server (Python) — src/server.py
        | calls
        v
Speculative Decoder — src/sampler.py
        |                    |
        v                    v
  Draft Model          Target Model
  (Llama 3.2 1B)       (Llama 3.1 8B Instruct)
  HuggingFace          HuggingFace
        |
        v
  AdaptiveKController — src/adaptive_k.py
  (sliding window, adjusts K each round)
        |
        v
  Benchmark Suite — scripts/run_bench.py
```

---

## Week 1 (Days 1–7): Foundations and Core Sampling Loop

**Goal:** By end of Day 7, speculative decoding is working and producing a measurable speedup.  
**Time commitment:** 6–8 hours per day.  
**Iron rule:** Do not move to the next day until today's deliverable runs and you can explain it.

---

### Day 1 — Environment Setup and First Generation

**Goal:** GPU confirmed working. Can load and run Llama 3.2 1B. Understand what a forward pass returns.

**Deliverables:**
- [ ] `test_gpu.py` — prints CUDA available, GPU name, VRAM
- [ ] `src/models.py` — ModelLoader class written and tested
- [ ] `test_models.py` — loads Llama 3.2 1B, generates 20 tokens, prints output and time

**Concepts to understand before moving on:**
- What a forward pass takes as input and returns
- Why we use float16 (VRAM math: 1B params × 2 bytes = 2GB)
- What `model.eval()` does and why

---

### Day 2 — Writing the Naive Generation Loop from Scratch

**Goal:** Implement autoregressive generation manually without calling `.generate()`.

**Why this day exists:** You cannot understand speculative decoding without first building the thing it replaces. Every step of the naive loop has a direct counterpart in the speculative loop.

**Deliverables:**
- [ ] `src/sampler.py` — `naive_generate()` written with explicit forward pass, softmax, sampling
- [ ] `scripts/baseline_bench.py` — measures tokens/sec for both 1B and 8B models on 5 prompts
- [ ] Baseline numbers recorded in DAILY_LOG.md

**Concepts to understand before moving on:**
- What logits are (shape: [batch, seq_len, vocab_size])
- What softmax does (converts raw scores to probabilities that sum to 1)
- Why temperature changes output diversity
- What the KV cache is and why it matters for speed

---

### Day 3 — Rejection Sampling: The Math into Code

**Goal:** Translate the mathematical proof into working Python. Prove it is correct.

> READ BEFORE CODING: Leviathan et al. 2023 (arXiv:2211.17192) — Abstract, Algorithm 1, Section 2 only. (~4 pages. Do this first, before writing any code.)

**The math (understand this before writing a line):**

Draft model produces token x with probability q(x).  
Target model assigns that token probability p(x).  
Accept x with probability min(1, p(x)/q(x)).  
If rejected: sample from normalize(max(0, p − q)).  

This guarantees the output distribution equals p(x) exactly — not approximately.

**Deliverables:**
- [ ] `src/sampler.py` — `speculative_sample_one_step()` written
- [ ] `tests/test_sampler.py` — 100K trial test: empirical frequencies within 1% of target distribution p
- [ ] Can derive the acceptance criterion on paper: why min(1, p(x)/q(x))?

---

### Day 4 — The Full Speculative Decoding Loop

**Goal:** Combine draft model, target model, and rejection sampling into one working loop.

**The four phases (memorize these):**
1. **DRAFT:** Run small model autoregressively for K steps. Save K token IDs and their logit distributions.
2. **TARGET:** Run large model ONCE on (context + K draft tokens). Get K+1 positions of logits in parallel.
3. **ACCEPT:** For each draft token i, run rejection sampling: accept or reject. Stop at first rejection.
4. **BONUS:** If all K draft tokens accepted, sample one free token from target_logits[K].

**Deliverables:**
- [ ] `src/sampler.py` — `speculative_decode()` written and working
- [ ] `scripts/compare_speed.py` — naive vs speculative comparison on same prompt, prints speedup ratio
- [ ] First speedup number recorded in DAILY_LOG.md

---

### Day 5 — K Sweep and First Fixed-K Benchmark

**Goal:** Find the optimal fixed K. Record the data. This data will be the baseline for comparing adaptive K later.

**Morning:** K sweep over K ∈ {1, 2, 4, 6, 8} — record tokens/sec and acceptance rate for each.

**Why K has an optimal value:**  
Too small → not enough parallelism gained.  
Too large → draft model compounds errors, acceptance rate drops, wasted compute.  
The sweet spot depends on how similar the draft and target model distributions are.

**Afternoon:** Debug session — common bugs to fix:
- KV cache invalidation (passing wrong `past_key_values` tensor)
- Off-by-one errors in draft/target logit alignment
- Temperature applied to draft but not target (distribution mismatch)

**Deliverables:**
- [ ] K sweep data table recorded in DAILY_LOG.md
- [ ] `sampler.py` cleaned up, no debug prints
- [ ] Best fixed K identified and noted

---

### Day 6 — Medusa Heads: Architecture and Implementation

**Goal:** Understand and implement the Medusa head architecture.

**What a Medusa head is:**  
Instead of a separate draft model, you attach K small neural networks (MLPs) directly to the base model's final hidden state. Each head i predicts the token at position t+i. In a single forward pass, you get the base model's prediction AND K future predictions simultaneously.

**The math:**  
head_i(h) = W2_i · SiLU(W1_i · h)

Where h is the final hidden state (4096-dim for Llama 8B). Each head is a 2-layer MLP projecting to vocab size (128,256).

**Deliverables:**
- [ ] `src/medusa.py` — `MedusaHead` and `MedusaModel` classes written
- [ ] Training script `scripts/train_medusa.py` started on passpoli (run overnight)
- [ ] Can explain: what SiLU activation does, why base model weights are frozen

---

### Day 7 — Medusa Decoding and First Three-Way Benchmark

**Goal:** Medusa decoding working. First complete system benchmark with three configurations.

**Deliverables:**
- [ ] `src/medusa.py` — `medusa_decode()` written
- [ ] Three-way benchmark: Baseline vs SpecDecode Fixed-K vs Medusa
- [ ] Numbers written in DAILY_LOG.md benchmark table

---

## Week 2 (Days 8–13): API, Visualization, and Production Polish

**Goal:** By end of Day 13, you have a working API, a live visualization, and MT-Bench quality verified.

---

### Day 8 — FastAPI Server and Streaming

**Goal:** API server running. Can generate text via HTTP and stream tokens via WebSocket.

**Deliverables:**
- [ ] `src/server.py` — POST /generate and WS /stream both working
- [ ] GET /health returns GPU status and VRAM usage
- [ ] Tested with curl from a second terminal

---

### Day 9 — Token Visualization Frontend

**Goal:** Browser UI working. Tokens appear one by one. Green = accepted, red = rejected. Stats updating live.

**Deliverables:**
- [ ] `frontend/index.html` + `frontend/stream.js` — complete
- [ ] Three-way method selector (Naive / SpecDecode / Medusa)
- [ ] Stats bar: tokens/sec and acceptance rate updating live
- [ ] Tested in browser end-to-end

---

### Day 10 — MT-Bench Quality Evaluation

**Goal:** Prove that speculative decoding produces identical quality to baseline. This is your correctness guarantee.

**Why this matters:** The rejection sampling proof guarantees identical output distribution in theory. MT-Bench verifies it empirically on real prompts.

**Deliverables:**
- [ ] `scripts/eval_mtbench.py` — generates responses for naive and speculative on same prompts
- [ ] Manual spot-check of 10–20 responses: qualitatively identical
- [ ] Results saved to `results/mtbench_naive.json` and `results/mtbench_speculative.json`

---

### Day 11 — Comprehensive Fixed-K Benchmark

**Goal:** Complete benchmark for all original configurations. This is the baseline data before adaptive K.

**Configs to benchmark:**

| Config | Draft | Target | Measure |
|---|---|---|---|
| Baseline | — | Llama 3.1 8B Instruct | tokens/sec, GPU memory |
| SpecDecode-Small | Llama 3.2 1B | Llama 3.1 8B Instruct | tokens/sec, acceptance rate |
| Medusa 4-head | 4 heads | Llama 3.1 8B Instruct | tokens/sec, acceptance rate |

Run 3 trials each. Report mean and standard deviation.

**Deliverables:**
- [ ] `results/benchmark_fixed_k.json` — all raw numbers
- [ ] `report/figures/throughput_fixed_k.png` — bar chart
- [ ] `report/figures/acceptance_rate_vs_k.png` — K sweep curve

---

### Day 12 — HuggingFace Spaces Deployment

**Goal:** Live public URL. This is what goes on your resume and GitHub.

**Note:** HuggingFace Spaces free tier has limited GPU. The demo may use 1B as target instead of 8B. Benchmark numbers come from passpoli — the demo just shows the visualization working. Be explicit about this in the README.

**Deliverables:**
- [ ] Live public URL on HuggingFace Spaces
- [ ] README with benchmark numbers and architecture diagram
- [ ] Demo working end-to-end in browser

---

### Day 13 — Code Cleanup and GitHub README

**Goal:** Clean repository that anyone can clone and run.

**Deliverables:**
- [ ] Docstrings on every function (what it does, inputs, outputs)
- [ ] GitHub README: project description, architecture diagram, benchmark table, how to reproduce
- [ ] Every script runnable end-to-end from a fresh clone
- [ ] "What I Would Do Next" section in README

---

## Week 3 (Days 14–18): Adaptive K — The Original Contribution

**Goal:** Implement, benchmark, and document adaptive K. This is what separates this project from a tutorial reimplementation.

**The problem with fixed K:**  
Fixed K assumes the draft model's quality is constant across all prompts and positions. It is not. On easy, predictable text (code, lists, repetitive patterns), a large K is efficient because acceptance rate is high. On complex, creative, or ambiguous text, a large K wastes compute because the draft model is often wrong.

**The adaptive K idea (your IEOR framing):**  
Treat K as a control variable in a throughput optimization problem. Define a sliding window of the last W acceptance decisions. If the empirical acceptance rate in the window is above threshold α_high, increase K. If below α_low, decrease K. This is a bandit-style feedback control loop — a concept you know from operations research.

**The formal objective:**  
Maximize E[tokens accepted per target model call] subject to K ∈ {K_min, ..., K_max}.

---

### Day 14 — Adaptive K: Theory and Design

**Goal:** Understand the algorithm deeply before writing any code.

**Morning:** Work through the math on paper.
- Define: W (window size), α_high, α_low, K_min, K_max
- Derive: what acceptance rate threshold maximizes expected accepted tokens per call?
- Draw: a state diagram of how K changes over time on an example sequence

**Afternoon:** Design the `AdaptiveKController` class interface before writing it.
- What state does it need to track?
- What does it need as input each step?
- What does it output?

**Deliverable:**
- [ ] Hand-drawn state diagram and parameter analysis (photograph and save to report/figures/)
- [ ] Class interface designed and written as a docstring stub in `src/adaptive_k.py`

---

### Day 15 — Adaptive K: Implementation

**Goal:** `AdaptiveKController` implemented and unit tested.

**Deliverables:**
- [ ] `src/adaptive_k.py` — `AdaptiveKController` class complete
- [ ] `src/sampler.py` — `adaptive_speculative_decode()` written, uses controller
- [ ] Unit tests: controller increases K correctly, decreases K correctly, respects bounds
- [ ] Smoke test: run adaptive decode on 3 prompts, print K values over time to confirm it's adapting

---

### Day 16 — Adaptive K vs Fixed K: Full Comparison

**Goal:** Produce the definitive benchmark that proves adaptive K is better than fixed K.

**The complete four-way benchmark:**

| Config | tok/s | Speedup | Accept Rate | Draft Compute Wasted |
|---|---|---|---|---|
| Baseline | | 1× | — | — |
| SpecDecode Fixed-K (best K) | | | | |
| SpecDecode Adaptive-K | | | | |
| Medusa 4-head | | | | |

Run each on: easy prompts (code completion), medium prompts (factual Q&A), hard prompts (creative writing). Show that adaptive K gains more on easy prompts (higher K naturally selected) and loses less on hard prompts (lower K selected when acceptance rate drops).

**Deliverables:**
- [ ] `scripts/adaptive_bench.py` — full comparison script
- [ ] `results/benchmark_final.json` — all final numbers
- [ ] `report/figures/adaptive_vs_fixed_comparison.png`
- [ ] `report/figures/k_over_time.png` — K value as function of position for example prompts

---

### Day 17 — Technical Report

**Goal:** A 2-page written report that explains the project, the contribution, and the results.

**Structure:**
1. **Abstract** (3 sentences): what the problem is, what you built, what you measured
2. **Background** (4 sentences): speculative decoding, why fixed K is suboptimal
3. **Method** (1 paragraph): the adaptive K algorithm, parameter choices, why this is an optimization problem
4. **Results** (the benchmark table + 2 charts)
5. **Analysis** (1 paragraph): when does adaptive K win? When does it not?
6. **What I Would Do Next** (bullet list): tree decoding, batch speculative decoding, learned K policy

This document is what you hand to a senior engineer in an interview who asks "can you walk me through the project?"

**Deliverable:**
- [ ] `report/report.md` — complete, 2 pages, all numbers filled in

---

### Day 18 — Final Polish and Ship

**Goal:** Everything is clean, tested, and ready to show anyone.

**Checklist:**
- [ ] Every script runs from `git clone` with no errors
- [ ] README has the live demo URL, benchmark table, and architecture diagram
- [ ] GitHub repository is public
- [ ] HuggingFace Spaces demo is live
- [ ] `DAILY_LOG.md` is complete and honest
- [ ] Can answer all 9 interview questions in Section 8 of the original roadmap without notes

---

## Key Metrics Targets

| Metric | Target | Why |
|---|---|---|
| Baseline (8B naive) tok/s | Measure and record | Your denominator for all speedups |
| SpecDecode Fixed-K speedup | > 2× | Standard result from paper |
| SpecDecode Adaptive-K speedup | > 2.5× | Your contribution must beat fixed K |
| Adaptive K draft compute saved | > 15% vs fixed best-K | Quantify the efficiency gain |
| MT-Bench quality delta | < 1% | Correctness guarantee |
| Acceptance rate (good prompts) | > 0.70 | Sign of healthy draft model |

---

## Prerequisite Concepts Checklist

Check these off only when you can explain them out loud without notes in under 90 seconds.

- [ ] What a token is and why models work with tokens not words
- [ ] What a forward pass returns (shape: [batch, seq_len, vocab_size], meaning of logits)
- [ ] What the KV cache is and why it matters for speed
- [ ] What softmax does and why we need it (converts scores to probabilities)
- [ ] Temperature sampling and how T changes output
- [ ] The rejection sampling proof: why min(1, p/q) gives us distribution p exactly
- [ ] Why verification can be done in parallel but generation cannot (the key insight)
- [ ] The 4 phases of speculative decoding: Draft, Target, Accept, Bonus
- [ ] What a Medusa head is and how it differs from a separate draft model
- [ ] Why adaptive K is a throughput optimization problem (your framing)
- [ ] Why FastAPI + WebSockets for streaming (not plain HTTP)

---

## Common Mistakes (Read Before You Hit Them)

### Mistake 1: Forgetting `torch.no_grad()` During Inference
**Symptom:** Out of memory error during inference; memory usage is 3–4× higher than expected.  
**Why:** Without `torch.no_grad()`, PyTorch stores all intermediate computations for gradient calculation. You do not need gradients during inference.  
**Fix:** Wrap all forward passes with `with torch.no_grad():`

### Mistake 2: KV Cache Shape Mismatch
**Symptom:** `RuntimeError: shape mismatch` when passing `past_key_values`.  
**Why:** When you feed (context + draft tokens) to the target model, you must pass the cached KV from the context forward call, not recompute from scratch.

### Mistake 3: Off-By-One in Draft/Target Logit Alignment
**Why:** Draft model's logit at position i is predicting position i+1. Target model's logit at position j is predicting position j+1. Making sure these align is easy to get wrong.

### Mistake 4: Temperature Applied Inconsistently
**Why:** If you apply temperature in the draft model but not the target model (or vice versa), the acceptance rates will be unexpectedly low because the distributions are artificially misaligned.

### Mistake 5: Medusa Heads Not Frozen Correctly
**Why:** If the base model weights are not frozen (`requires_grad=False`), you will accidentally update them during Medusa training, corrupting the model.

---

## Interview Questions (The Real Test)

You must be able to answer all of these without notes. If you cannot, go back.

**Q1:** How does speculative decoding work? (Explain in 60 seconds to a non-ML interviewer.)  
**Q2:** Prove that speculative decoding produces the same output distribution as the target model alone.  
**Q3:** What is the acceptance rate and why does it matter? What happens when it's low?  
**Q4:** How does Medusa differ from standard speculative decoding? What are the tradeoffs?  
**Q5:** What is the KV cache and how did you use it?  
**Q6:** Walk me through your system architecture end to end.  
**Q7:** Why did you choose FastAPI and WebSockets?  
**Q8:** What would you do next to improve this system?  
**Q9:** How did you verify your implementation is correct? (This is the trap question — most people say "it gave reasonable output." Wrong answer. The right answer involves the 100K trial statistical test.)  
**Q10 (your addition):** Why is adaptive K an optimization problem? What objective are you maximizing? What are the constraints?

---

## References

1. Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. arXiv:2211.17192. ← Read before Day 3.
2. Cai, T. et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. arXiv:2401.10774. ← Read before Day 6.
3. Original roadmap PDF: `specdecode_roadmap.pdf` (preserved for reference)
