# SpecDecode — Daily Progress Log

**Project start date:** 2026-06-06  
**Target end date:** 2026-06-28 (18 days core — extended roadmap with Adaptive K)  
**Current day:** DAY 1 COMPLETE (2026-06-13). Next up: Day 2 — write the naive generation loop by hand in `src/sampler.py`

---

## How to Use This File

- Update this file at the END of every session.
- Be honest about what is DONE vs what is PARTIALLY DONE.
- "Done" means: code runs, output is correct, I can explain it.
- Write the "Starting Point" for tomorrow so the next Claude session has context.

---

## Pre-Work Checklist (Before Day 1)

- [x] Read the full roadmap PDF (`specdecode_roadmap.pdf`)
- [ ] Understand the 7 prerequisite concepts (Section 2 of roadmap)
- [x] Get GPU access confirmed on IEOR server ✓ passpoli: 2x NVIDIA RTX A5000, 24GB VRAM each, CUDA 12.4
- [ ] Install Python environment on server (blocked: home directory not yet created on passpoli)

**Session notes (2026-06-09):**
- SSH access confirmed: Mac → login.ieor.iitb.ac.in → marol → passpoli
- GPU machine is **passpoli** (10.119.2.11). Kanjur has no GPU.
- Home directory exists on marol but NOT on passpoli or kanjur
- Sysadmin email found: systems@ieor.iitb.ac.in (Faculty: Prof. Ashutosh Mahajan, TA: Anuj Birani)
- Email sent to systems@ieor.iitb.ac.in on 2026-06-09 requesting /home/24b4526 on passpoli
- Decision: do all setup directly on passpoli once home directory is ready, not on marol

---

## Week 1: Foundations and Core Sampling Loop

### Day 1 — Environment Setup and First Generation
**Goal:** GPU confirmed working. Can load and run Llama 3.2 1B. Understand what a forward pass returns.

**Status:** DONE (completed 2026-06-13)

**Deliverable:**
- [x] `test_gpu.py` passes — CUDA available: True, GPU: NVIDIA RTX A5000, VRAM: 23.7 GB
- [x] `src/models.py` — ModelLoader class written (float16, device_map, model.eval())
- [x] `test_models.py` — WORKS: 20 tokens in 1.45 s (~13.8 tok/s, Llama 3.2 1B, includes warm-up — not the official baseline). Output: "The capital of India is New Delhi, and it is the largest city in the country..."

**What I learned today:**
- Project folder structure created on passpoli: src/, frontend/, scripts/, notebooks/, report/figures/
- Python 3.13.5 on passpoli; PyTorch 2.12.0 installed with CUDA 12.4 support (cu124)
- Initial PyTorch install used cu130 — wrong version for CUDA 12.4 driver. Fixed by reinstalling with --index-url cu124.
- NVim installed without sudo using curl + tar, added to PATH via .bashrc
- NVim config pulled from GitHub (PrashantGautam-alt/my_nvim_config)
- Logged into HuggingFace CLI using `hf auth login`
- HuggingFace Llama 3.2 access request submitted — pending approval
- Roadmap extended to 18 days. New ROADMAP.md created with Adaptive K as core Week 3 contribution.
- Understood: why float16 saves VRAM (2 bytes vs 4 bytes per parameter), what model.eval() does, what a forward pass returns

**Session 2026-06-12/13 (Day 1 finished):**
- HuggingFace Llama 3.2 1B access APPROVED (email arrived 2026-06-10 19:30)
- Learned tokenization: model sees integer IDs, never text. Vocabulary = 128,256 word fragments.
  Why fragments not words: COVERAGE (fragments compose unseen words, e.g. un+believable) +
  FREQUENCY (common words earn whole-token slots so sequences stay short — Huffman-coding logic)
- Wrote test_models.py myself: tokenize prompt → .to("cuda") → timed model.generate(max_new_tokens=20) → decode.
  generate() treated as a black box — Day 2 opens it.
- Debugged three real errors solo-ish:
  1. SyntaxError — missing comma between from_pretrained() arguments
  2. IndentationError — docstring one space off; Python blocks demand exact column alignment (nvim: gg=G re-indents)
  3. AttributeError: torch has no 'float8_e8m0fnu' — transformers too new for our torch.
     torch is capped by the cu124 wheel index (driver = CUDA 12.4, no sudo) → fixed by pinning the
     cheap unpinned library instead: transformers==4.56.2. Lesson: change the low-risk variable.
- Why ModelLoader is a class (explained back, verified): write load logic once, import everywhere;
  one point of change when loading logic evolves
- `<|begin_of_text|>` in output = special token prepended by tokenizer; skip_special_tokens=True hides it
- Git/GitHub workflow set up on passpoli: repo PrashantGautam-alt/specdecode (private), PAT auth,
  .gitignore excludes venv/ and caches. Daily ritual: git add . && git commit && git push

**Blockers / Questions:**
- ~~HuggingFace Llama 3.2 approval~~ → approved, working
- ~~8B target model access~~ → all three models approved (confirmed 2026-06-13):
  - meta-llama/Llama-3.2-1B ✓ (approved 2026-06-10)
  - meta-llama/Llama-3.1-8B ✓ (approved 2026-06-12)
  - meta-llama/Llama-3.1-8B-Instruct ✓ (confirmed 2026-06-13, access already granted)

**Starting Point for Next Session (Day 2):**
1. ssh → passpoli, `source ~/specdecode/venv/bin/activate`, `cd ~/specdecode`
2. FIRST: confirm 8B model access on HuggingFace (request meta-llama/Llama-3.1-8B-Instruct if not yet done)
3. Goal: `src/sampler.py` → `naive_generate()` — the autoregressive loop written BY HAND, no `.generate()`
4. New concepts queued: logits (shape [batch, seq_len, 128256]), softmax → probabilities,
   greedy vs temperature sampling, why each new token requires a full forward pass
5. Record official baseline tokens/sec for 1B (and 8B if access granted) — every speedup in this
   project is measured against these numbers
6. Housekeeping: `pip freeze > requirements.txt`, commit and push
   (GitHub is LIVE: repo PrashantGautam-alt/specdecode pushed 2026-06-13; Claude has gh CLI access
   on the Mac — start next session by cloning/pulling and reviewing the Day 1 code)
7. Before Day 3: read Leviathan et al. 2023 (arXiv:2211.17192) — Abstract + Section 2 + Algorithm 1

---

### Day 2 — Writing the Naive Generation Loop from Scratch
**Goal:** Implement autoregressive generation manually without calling `.generate()`.

**Status:** DONE (completed 2026-06-13)

**Deliverable:**
- [x] `src/sampler.py` — `naive_generate()` written and working
- [x] Baseline tokens/sec number recorded for 1B model (8B deferred — not needed until Day 4)
- [x] I can explain: what a forward pass returns, what logits are, what softmax does

**What I learned today:**
- Forward pass returns logits: shape [batch, seq_len, vocab_size] = [1, seq_len, 128256]
- To get the next token, slice logits[0, -1, :] — last position only
- Softmax converts raw logits to probabilities (all positive, sum to 1)
- Temperature T divides logits before softmax: T<1 sharpens, T>1 flattens, T→0 is greedy
- torch.multinomial samples one token ID from the probability distribution
- KV cache: store past_key_values and pass back each step — avoids recomputing K/V for seen tokens
- generated_ids must be initialized OUTSIDE the loop — classic bug caught and fixed
- Scripts must be run with PYTHONPATH=. from project root: `PYTHONPATH=. python scripts/baseline_bench.py`

**Baseline Numbers (official):**
| Model | tok/s | Notes |
|---|---|---|
| Llama 3.2 1B | 83.3 | 5 prompts × 3 runs, 100 new tokens each |

**Blockers / Questions:**
- None. All three model accesses confirmed (Llama 3.2 1B, 3.1 8B, 3.1 8B Instruct)

**Starting Point for Next Session (Day 3):**
1. ssh → passpoli, `source ~/specdecode/venv/bin/activate`, `cd ~/specdecode`
2. READ FIRST: Leviathan et al. 2023 (arXiv:2211.17192) — Abstract, Algorithm 1, Section 2 only (~4 pages). Do this before writing any code.
3. Goal: `src/sampler.py` — add `speculative_sample_one_step()` function
4. The math to understand before coding: accept token x with probability min(1, p(x)/q(x)). If rejected, sample from normalize(max(0, p−q))
5. Deliverable: 100K trial test in `tests/test_sampler.py` — empirical frequencies within 1% of target distribution p

---

### Day 3 — Rejection Sampling: The Math into Code
**Goal:** Translate the mathematical proof into working Python. Prove it's correct.

**Status:** DONE (completed 2026-06-13)

> ⚠️ **READ BEFORE CODING TODAY:** Read Leviathan et al. 2023 (arXiv:2211.17192)
> — just the Abstract, Algorithm 1, and the proof sketch in Section 2. (~4 pages)
> You promised yourself you'd do this on Day 3. Do it before writing any code.

**Deliverable:**
- [x] `src/sampler.py` — `speculative_sample_one_step()` written
- [x] `tests/test_sampler.py` — 100K trial test passes, output within 1% of target distribution p
- [x] I can explain the acceptance criterion: why min(1, p(x)/q(x))?

**What I learned today:**
- Read Leviathan et al. 2023 — Abstract, Algorithm 1, Section 2
- Acceptance criterion: accept draft token x with probability min(1, p(x)/q(x))
  - If p ≥ q → always accept (target model agrees or prefers this token)
  - If p < q → accept with probability p/q (draft model was overconfident)
- On rejection: sample from normalize(max(0, p − q))
  - torch.clamp(p - q, min=0) is the vectorized elementwise max(0, p−q)
  - Divide by sum to normalize
- This guarantees output distribution equals p EXACTLY — not approximately
- 100K trial result: max deviation 0.27% — well within 1% threshold

**Test Results:**
```
Target p:    [0.400, 0.300, 0.150, 0.100, 0.050]
Empirical:   [0.400, 0.297, 0.149, 0.102, 0.051]
Difference:  [0.0004, 0.0027, 0.0005, 0.0016, 0.0012]
PASSED
```

**Blockers / Questions:**
- None

**Starting Point for Next Session (Day 4):**
1. ssh → passpoli, `source ~/specdecode/venv/bin/activate`, `cd ~/specdecode`
2. Goal: `speculative_decode()` — combine draft model + target model + rejection sampling into one loop
3. The 4 phases to implement: DRAFT → TARGET → ACCEPT → BONUS TOKEN
4. Need to load both models: Llama 3.2 1B (draft) + Llama 3.1 8B Instruct (target)
5. First speedup number gets measured today — compare against 83.3 tok/s baseline
6. Run with: `PYTHONPATH=. python scripts/compare_speed.py`

---

### Day 4 — The Full Speculative Decoding Loop
**Goal:** Combine draft model, target model, and rejection sampling into one working loop.

**Status:** DONE (completed 2026-06-14)

**Deliverable:**
- [x] `src/sampler.py` — `speculative_decode()` written with persistent KV cache
- [x] First speedup number measured: **1.04x** (Naive 8B: 37.9 tok/s, Speculative K=4: 39.4 tok/s)
- [x] I can explain the 4 phases: Draft → Target → Accept → Bonus Token

**What I learned today:**
- Understood why target verification is one forward pass: transformer processes all positions in parallel
- Understood the 4 phases: DRAFT (K steps small model) → TARGET (1 pass big model) → ACCEPT (rejection sampling) → BONUS (free token if all accepted)
- First attempt: 0.89x speedup (slower than naive) — caused by resetting KV cache every iteration
- Cache reset bug: target model was reprocessing the ENTIRE growing context from scratch each iteration → O(n²) cost
- Fix: maintain persistent context caches for both models across iterations. Target only processes K new draft tokens per iteration, not the full sequence
- New architecture: prime both models on prompt once → each iteration extend K steps from cached context → update cache with only accepted tokens after each iteration
- target_next_logit must be saved each iteration to verify draft_token[0] (the logit at the last accepted position, needed before the new draft tokens are processed)

**Blockers / Questions:**
- GPU 0 and GPU 1 on passpoli both occupied by Ollama server (PID 1993590, ~20GB each)
- Cannot load both 1B + 8B models simultaneously until GPU frees up
- Contact: systems@ieor.iitb.ac.in if still blocked tomorrow

**Starting Point for Next Session (Day 4 continued → Day 5):**
1. ssh → passpoli, check GPU: `nvidia-smi` — wait until Ollama process is gone
2. `source ~/specdecode/venv/bin/activate`, `cd ~/specdecode`
3. Run: `PYTHONPATH=. python scripts/compare_speed.py` — get first real speedup number
4. Record speedup in DAILY_LOG.md key numbers table
5. If speedup looks good, move straight to Day 5: K sweep over K ∈ {1, 2, 4, 6, 8}

---

### Day 5 — Tuning K and Debugging
**Goal:** Find the optimal K. Fix any bugs caught during the K sweep.

**Status:** NOT STARTED

**Deliverable:**
- [ ] K sweep data recorded: K ∈ {1, 2, 4, 6, 8} vs tokens/sec and acceptance rate
- [ ] `sampler.py` cleaned up, no debug prints
- [ ] Peak speedup number identified and written down

**What I learned today:**
*(fill in)*

**Blockers / Questions:**
*(fill in)*

**Starting Point for Next Session (Day 5):**
1. ssh → passpoli, `source ~/specdecode/venv/bin/activate`, `cd ~/specdecode`
2. Write `scripts/k_sweep.py` — loops over K ∈ {1, 2, 4, 6, 8}, records tok/s and speedup for each
3. Run the sweep and record all numbers in this log
4. Identify peak K and explain why that K wins
5. Key question to answer: why does 1.04x feel low? Is it acceptance rate or overhead?

---

### Day 6 — Medusa Heads: Architecture and Implementation
**Goal:** Understand and implement the Medusa head architecture.

**Status:** NOT STARTED

**Deliverable:**
- [ ] `src/medusa.py` — `MedusaHead` and `MedusaModel` classes written
- [ ] Training script started on IEOR server (let it run overnight)
- [ ] I can explain: what a Medusa head is, why it's attached to the base model, what SiLU does

**What I learned today:**
*(fill in)*

**Blockers / Questions:**
*(fill in)*

**Starting Point for Next Session:**
*(fill in)*

---

### Day 7 — Medusa Decoding and First Complete System Test
**Goal:** Three-way benchmark: baseline vs SpecDecode vs Medusa.

**Status:** NOT STARTED

**Deliverable:**
- [ ] `src/medusa.py` — `medusa_decode()` written
- [ ] Three-way benchmark table: tokens/sec for all three configurations
- [ ] Numbers written down in this log

**Benchmark Numbers (fill in):**
| Config | Draft | Target | tok/s | Notes |
|---|---|---|---|---|
| Baseline | — | Llama 8B | | |
| SpecDecode-Small | Llama 1B | Llama 8B | | |
| Medusa 4-head | 4 heads | Llama 8B | | |

**Starting Point for Next Session:**
*(fill in)*

---

## Week 2: API, Visualization, Benchmarks, Deployment

### Day 8 — FastAPI Server and Streaming
**Status:** NOT STARTED  
**Deliverable:** FastAPI server running. POST /generate and WS /stream both work. Tested with curl.

---

### Day 9 — Token Visualization Frontend
**Status:** NOT STARTED  
**Deliverable:** Browser UI working. Tokens appear one by one. Green = accepted, Red = rejected. Stats bar updating live.

---

### Day 10 — MT-Bench Quality Evaluation
**Status:** NOT STARTED  
**Deliverable:** MT-Bench responses generated for naive and speculative. Manual quality check done.

---

### Day 11 — Comprehensive Benchmarking
**Status:** NOT STARTED  
**Deliverable:** All 4 configs benchmarked. Two charts saved as PNGs. Summary table printed.

---

### Day 12 — HuggingFace Spaces Deployment
**Status:** NOT STARTED  
**Deliverable:** Live public URL on HuggingFace Spaces. README with benchmark numbers.

---

### Day 13 — Code Cleanup and GitHub README
**Status:** NOT STARTED  
**Deliverable:** Clean GitHub repo. README with architecture diagram, benchmark table, how-to-run.

---

### Day 14 — Report Writing and Wrap-up
**Status:** NOT STARTED  
**Deliverable:** All 9 report sections written. Numbers match benchmarks. Ready to ship.

---

## Week 3: Optional Extensions

### Extension A (Days 15–17) — Tree-based Speculative Decoding
**Status:** NOT STARTED

### Extension B (Days 18–19) — Batch Speculative Decoding
**Status:** NOT STARTED

### Extension C (Days 20–21) — Dynamic Draft Length
**Status:** NOT STARTED

---

## Key Numbers (fill in as you go)

| Metric | Value | Date measured |
|---|---|---|
| Baseline (1B naive) tok/s | 83.3 | 2026-06-13 |
| Baseline (8B naive) tok/s | 37.9 | 2026-06-14 |
| SpecDecode-Small tok/s | 39.4 | 2026-06-14 |
| SpecDecode-Small speedup | 1.04x (K=4) | 2026-06-14 |
| Best acceptance rate | | |
| Best K value | | |
| Medusa tok/s | | |
| Medusa speedup | | |

---

## Concepts I Can Now Explain Without Notes

*(Check these off as you truly understand them — be honest)*

- [x] What a token is and why models work with tokens not words *(verified 2026-06-13 — explained coverage + frequency without notes)*
- [x] What a forward pass returns (shape, meaning of logits) *(verified 2026-06-13)*
- [x] What the KV cache is and why it matters *(verified 2026-06-13)*
- [x] What softmax does and why we need it *(verified 2026-06-13)*
- [x] Temperature sampling and how T changes output *(verified 2026-06-13)*
- [x] The rejection sampling proof: why min(1, p/q) gives us p *(verified 2026-06-13, 100K test passed)*
- [x] Why verification can be done in parallel but generation cannot *(verified 2026-06-14)*
- [x] The 4 phases of speculative decoding *(verified 2026-06-14)*
- [ ] What a Medusa head is and how it differs from a separate draft model
- [ ] Why FastAPI + WebSockets for streaming (not HTTP)
