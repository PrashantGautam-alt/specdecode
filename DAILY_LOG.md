# SpecDecode — Daily Progress Log

**Project start date:** 2026-06-06  
**Target end date:** 2026-06-28 (18 days core — extended roadmap with Adaptive K)  
**Current day:** DAY 6 DONE (2026-06-17). `forward` finished; init trick proven on GPU (`Init trick holds: True`); full Medusa training pipeline built + validated on 1B (loss 26.5 → 0.063). DAY 7 STARTED — `medusa_decode()` written (correctness-first, no KV cache yet; explain-back owed). Headline still stands: SpecDecode beats naive — **1.17x at K=4** (instruct draft).

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

**Status:** DONE (completed 2026-06-14)

**Deliverable:**
- [x] K sweep data recorded: K ∈ {1, 2, 4, 6, 8} vs tokens/sec and acceptance rate
- [~] `sampler.py` cleaned up — one print KEPT on purpose (the `avg tokens/round` acceptance counter; it's instrumentation, not debug noise). Remove before final report.
- [x] Peak speedup number identified: **1.17x at K=4** (instruct draft)

**K Sweep Results (Naive 8B baseline: 38.0 tok/s):**

Draft = Llama-3.2-1B-Instruct, Target = Llama-3.1-8B-Instruct, 100 tokens, 3 runs/K.

| K | tok/s | speedup | avg tokens/round (acceptance) |
|---|---|---|---|
| 1 | 27.8 | 0.73x | 1.00 |
| 2 | 38.4 | 1.01x | 1.83 |
| **4** | **44.4** | **1.17x** | **3.05** |
| 6 | 39.0 | 1.03x | 3.65 |
| 8 | 34.0 | 0.89x | 3.82 |

(Earlier sweep with the BASE draft `Llama-3.2-1B` peaked at only 0.97x — see "what I learned".)

**What I learned today:**
- **Removed the bonus token.** It forced 2 extra forward passes (draft + target) on every fully-accepted round to refresh the caches/lookahead. Cost more than the free token was worth. Deleted it; M now ranges 1..K only.
- **Built an acceptance counter.** Added `rounds` + print of `avg tokens/round = len(generated_ids) / rounds`. Key insight: `len(generated_ids)` is ALREADY the running sum of M (each round does `extend(accepted)` which adds M tokens), so no separate total needed.
- **Derived a cost model that fits every data point:** `speedup ≈ acceptance / (0.40 × K + 1)`. Each draft pass costs ~40% of a target pass; +1 is the single verify pass.
  - Predicted K=4: 3.05 / 2.6 = 1.17x — matched the measured 1.17x exactly.
- **The cross-GPU overhead was NOT the problem.** The model had no leftover constant term → o ≈ 0. Moving both models to one GPU would buy almost nothing. (My earlier hypothesis; the data refuted it.)
- **The draft was the real tax — 40% of target, not the 15% its size suggests.** Single-token forward passes are dominated by fixed per-call overhead, and we run K of them serially.
- **Base-vs-Instruct mismatch was capping acceptance.** Draft was the BASE 1B, target was INSTRUCT 8B → different "dialects" (base continues raw text, instruct answers like an assistant). Switching draft to `Llama-3.2-1B-Instruct` raised acceptance ~+0.6 at K=4 (2.45 → 3.05) at the SAME draft cost — that's what crossed us over 1.0x.
- **Why K=4 is the peak:** cost grows linear and GUARANTEED (+0.40 per draft token, accepted or not); acceptance grows sub-linear and CONDITIONAL (the first wrong guess wastes every later draft pass that round). Past K=4 the cost outruns the diminishing acceptance gains.

**Blockers / Questions:**
- None. GPUs were free this session (Ollama gone). Draft on cuda:0, target on cuda:1.
- Note: the lone `print` in `speculative_decode` should be made optional/removed before the final report (Day 13 cleanup).

**Starting Point for Next Session (Day 6 — Medusa):**
1. ssh → passpoli, `source ~/specdecode/venv/bin/activate`, `cd ~/specdecode`, `git pull origin main`
2. Headline to remember: **SpecDecode beats naive — 1.17x at K=4** with the instruct draft. The 0.40-per-draft-token cost model explains the whole K curve.
3. Goal: begin Medusa. READ FIRST before any code — what a Medusa head is and WHY it can be cheaper than a separate draft model (no second model to run K times → attacks that 0.40 draft tax directly).
4. New concepts queued: Medusa head (small MLP on top of the base model's last hidden state), SiLU activation, why multiple heads predict multiple future positions at once.
5. Connect it back: Medusa's whole appeal is killing the `0.40 × K` term — the heads ride the target's own forward pass instead of running a separate draft model serially.

---

### Day 6 — Medusa Heads: Architecture and Implementation
**Goal:** Understand and implement the Medusa head architecture.

**Status:** DONE (2026-06-17) — `forward` finished, init trick proven on GPU, training pipeline built + validated on the 1B (loss 26.5 → 0.063).

**Deliverable:**
- [x] `src/medusa.py` — `MedusaHead` written and verified: `W2(SiLU(W1(h)) + h)`, dims parametrized
- [x] `src/medusa.py` — `MedusaModel.__init__` DONE: backbone stored + frozen + heads in `nn.ModuleList` + **init trick complete** (W1 weight+bias zeroed with `is not None` guards, W2 = LM-head clone, W2 bias zeroed). Used `.data.zero_()` / `.data.copy_()` — equivalent to `nn.init.zeros_`.
- [x] `src/medusa.py` — `MedusaModel.forward` DONE (output_hidden_states → h = hidden_states[-1] → run all heads → return list; + dtype-cast of h to head dtype for mixed precision)
- [x] Training script `scripts/train_medusa.py` — DONE + validated on 1B (loss 26.5 → 0.063, 100 epochs on 5 medical paragraphs)
- [x] I can explain: what a Medusa head is, why it's attached to the base model, what SiLU does

**What I learned today (all captured in CONCEPTS.md):**
- Medusa head = small MLP on the final hidden state `h`, sitting PARALLEL to the LM head (not on top). Head k predicts position t+k+1. Formula `W2(SiLU(W1·h) + h)`.
- SiLU `= x·sigmoid(x)`; why non-linearity is needed (linear-of-linear collapses → can't do XOR); why SiLU beats ReLU (smooth, no dead neurons, Llama uses it).
- Residual `+ h`: the head learns a small adjustment to `h`, not the whole map.
- Init trick: start each head as an EXACT clone of the LM head (W1=0, W2=LM head copy); the `+ h` is what makes zero-init degrade to "exactly the LM head" instead of garbage.
- What `h` is: the 4096-dim last hidden state = the model's compressed understanding; the `logits` we used since Day 2 come one step AFTER `h`.
- Training economics: frozen backbone = no gradients/optimizer for the 8B → cheap; weights untouched → safe. QLoRA = quantize + LoRA to train big models on one GPU.
- PyTorch mechanics: nn.Module (init declares weighted parts, forward = the math; CALL `head(h)` not `head.forward(h)`); nn.Linear is CALLED not multiplied (`W1(h)`, not `W1*h`); nn.ModuleList (a plain list would hide the heads from PyTorch); `requires_grad=False` to freeze.

**MedusaModel progress so far (`src/medusa.py`):**
- `__init__(self, backbone, num_heads)`: stores backbone; freezes it; creates `num_heads` MedusaHeads in an `nn.ModuleList`, sized from `backbone.config.hidden_size` / `vocab_size`.
- **init trick DONE** — for each head: `head.W1.weight.data.zero_()` + guarded `head.W1.bias.data.zero_()` (half 1: kills the SiLU branch so the bracket reduces to `h`), then `head.W2.weight.data.copy_(lm_head.weight.data)` + guarded `head.W2.bias.data.zero_()` (half 2: `W2` alone reproduces the LM head). Both biases guarded with `if ... is not None`.
- Result: each head is BORN as an exact clone of the LM head. Verified by algebra only — queue a runtime sanity check for later: a fresh head's output on a sample `h` should equal `lm_head(h)`.
- THEN `MedusaModel.forward`: run backbone with `output_hidden_states=True` to get `h`, feed `h` to every head, return the K predictions.

**Blockers / Questions:**
- None.
- **RESOLVED this session** — the Day 6 "fuzzy concept" the user flagged last time was **`nn.ModuleList` vs a plain list**. Cleared up and explained back: a plain Python list isn't an `nn.Module`, so PyTorch never registers the heads inside it → they're absent from `.parameters()` (optimizer never receives them → never trains), `.to(device)` (device mismatch crash), and `state_dict()` (can't save). `nn.ModuleList` registers each head as a real submodule. (Memory note deleted.)
- Also explained back: the init-trick bias point — zeroing only `W1.weight` leaves `W1(h) = bias ≠ 0`, so `SiLU(bias) ≠ 0` and the head would NOT start as the LM head.

**Session 2026-06-17 (Day 6 FINISHED + Day 7 started):**
- Finished `MedusaModel.forward` (fixed indentation, filled the 4 placeholders by hand). Explained back: heads eat `hidden_states[-1]` not `logits` because logits are a lossy projection at ONE position; heads aim at DIFFERENT future positions, so they need the full `h`.
- **Init trick proven at runtime** — `scripts/sanity_medusa.py` printed `Init trick holds: True` (head(h) == lm_head(h)). Learned `torch.allclose` and why `==` fails on floats (same value via different computation paths rounds differently); loosened tolerance to 1e-3 for float16.
- **Built `scripts/train_medusa.py` from scratch** (every line, mostly by hand): the target SHIFT (drop last `shift` logits, first `shift` labels), weighted CE `λk=0.8^k`, optimizer over `medusa.heads.parameters()` only, the `zero_grad→backward→step` heartbeat, `.item()`, `state_dict`, `torch.save`. Toy data = 5 medical paragraphs (overfit test).
- **Hit + fixed the float16 NaN** — fp16's narrow range (max ~65k, min ~6e-5) overflows the 128k-vocab softmax / underflows gradients → Adam's divide → NaN from epoch 0. Fix: train heads in **float32**, keep backbone float16, cast `h` to head dtype in `forward`. (Full mechanism now in INTERVIEW_PREP.md §3.11.)
- **Validation run succeeded:** loss **26.49 → 0.063** over 100 epochs — proves the whole pipeline (forward, shift, weighted CE, backward, step) is correct.
- **Documentation:** built `INTERVIEW_PREP.md` (private, in `.gitignore`) — full Days 1–6 interview-defense deep-dive. Markdown is the master; LaTeX-on-demand for an Overleaf PDF. (TODO: top up CONCEPTS.md with today's concepts.)
- **Day 7 started:** wrote `medusa_decode()` (correctness-first, greedy verification, **no KV cache yet**). I generated it because the user asked; user understands it **"not fully"** → explain-back OWED before moving on.

**Compute plan locked (for the real 8B training):**
- fp8 impossible on A5000 (Ampere has no fp8 HW); fp16 NaNs AND doesn't fit (~33 GB). Must be fp32 heads.
- Route A (paid, simple): cloud H100/A100-80GB ~₹150–200. RunPod via MasterCard (₹840 min top-up though), OR IndiaAI/E2E via UPI (needs father's Aadhaar OTP → do when he's awake). Hard budget cap ₹300.
- Route B (free, more engineering): 2× A5000 (48 GB) + 8-bit Adam (bitsandbytes), backbone on GPU0 / heads on GPU1. ₹0, good interview skill.
- Dataset target: ~10–15k ShareGPT (instruct) examples, 2 epochs.

---

### Day 7 — Medusa Decoding and First Complete System Test
**Goal:** Three-way benchmark: baseline vs SpecDecode vs Medusa.

**Status:** DONE (2026-06-25) — three-way benchmark complete. Medusa greedy 0.65x; bottleneck is 3-pass structure, not training quality (acceptance 2.22 tokens/round is healthy). Tree attention (Extension A) is the fix.

**Deliverable:**
- [x] Explain-back on `medusa_decode()` — passed (2026-06-20)
- [x] `scripts/test_medusa_decode.py` — runs on A5000, `Outputs match: True`, 1.38 tokens/round
- [x] `src/medusa.py` — `medusa_decode()` rewritten with persistent KV cache (O(n) per round)
- [x] Three-way benchmark table: tokens/sec for all three configurations
- [x] Numbers written down in this log

**What was built this session (2026-06-20):**
- Rewrote `medusa_decode()` from O(n²) to O(n): PRIME step builds starting cache; PROPOSE feeds 1 token; VERIFY feeds K-1 candidates; CACHE UPDATE re-feeds accepted[:-1] to advance cache cleanly.
- Debugged `DynamicCache` in-place mutation bug (transformers 4.56.2): `past_key_values` is a mutable object — VERIFY was contaminating `full_cache` before CACHE UPDATE could use it. Fix: `snap()` helper creates a fresh `DynamicCache` from `to_legacy_cache()` + `from_legacy_cache()` at each step. Two snapshots per round: `full_cache` (for VERIFY, expendable) and `update_cache` (clean n-token snapshot for CACHE UPDATE).
- Correctness verified: `Outputs match: True`, acceptance rate 1.38 tokens/round (identical to O(n²) version — confirms cache is correct).
- Route B training set up: backbone on cuda:0 float16, heads on cuda:1 float32, 8-bit Adam (bitsandbytes). Fixed medusa.py forward to transfer h device+dtype in one line. Fixed train_medusa_8b.py: correct model name, typo, device mismatch in labels_k, save filename.
- Toy run on 8B (5 paragraphs, 100 epochs) completed: loss 36.1 → 0.09. Pipeline proven.
- Real training launched: UltraChat 10k, 2 epochs, max_length=512, expandable_segments=True. Running in tmux session `medusa_train` on passpoli as of 2026-06-20 ~20:45 IST.
- CONCEPTS.md and INTERVIEW_PREP.md fully updated: KV cache fix, DynamicCache bug, 8-bit Adam, multi-GPU placement, compression table.

**Concepts taught and verified today:**
- 8-bit Adam: what m and v are, why we compress those not parameters, the quantize-what-you-USE rule
- Multi-GPU device placement: backbone cuda:0, heads cuda:1, h crosses the boundary once per forward pass
- combined device+dtype transfer: h.to(device=..., dtype=...)
- tmux: why it's needed for long remote jobs, Ctrl+B D to detach, tmux attach to resume
- .pt files: binary pickle, not for GitHub, belongs on HuggingFace Hub

**Benchmark Numbers (measured 2026-06-25):**
| Config | tok/s | speedup | Notes |
|---|---|---|---|
| Baseline (Naive 8B) | 37.9 | 1.00x | measured Day 5 |
| SpecDecode K=4 | 44.4 | 1.17x | 1B instruct draft, measured Day 5 |
| Medusa 4-head (greedy) | 24.6 | 0.65x | epoch1 checkpoint, 2.22 tokens/round |

**Why Medusa is 0.65x despite 2.22 tokens/round acceptance:**
- Our greedy implementation does 3 backbone passes per round: PROPOSE (1 token) + VERIFY (K-1=3 tokens) + CACHE UPDATE (accepted[:-1] tokens)
- Expected speedup = 2.22 / 3 ≈ 0.74x theoretical; gap to 0.65x is cross-GPU PCIe latency (h travels cuda:0→cuda:1 every round)
- Fix: tree attention (Extension A) collapses 3 passes into ~1 pass → same acceptance, ~2x speedup

**Quick wins applied (2026-06-26):**
- Same-GPU placement (heads → cuda:0): NOT FEASIBLE. 8B backbone uses 22GB on cuda:0, heads need ~4.3GB (W2 alone = 4096×128256 per head × 4 heads). Only 1.51GB free. OOM.
- float16 heads (`medusa.heads.half()`): DONE. Heads stay on cuda:1 but now float16. Result: 0.65x → 0.69x (small gain — heads are tiny vs backbone, 3-pass structure still dominates).

**Training runs completed / in progress:**
- Run 1 (10k examples, 2 epochs, DONE): epoch 0: 55.4905, epoch 1: 48.0302. Saved as `medusa_heads_8b_10k.pt`.
- Run 2 (25k examples, 2 epochs, CRASHED at epoch 1): server crashed mid-epoch 1. Only `medusa_heads_8b_epoch0.pt` saved. Epoch 0 avg loss = 11 (random baseline ≈ 34.7, so heads learned meaningfully).
- Run 3 (resume from epoch 0, CRASHED again before saving): server went down again. `medusa_heads_8b_epoch1.pt` was NOT saved — confirmed by `ls` (2026-06-25). Only `medusa_heads_8b_epoch0.pt` remains.
- Run 4 (resume from epoch 0, IN PROGRESS as of 2026-06-25): network back on passpoli. Launched with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. python scripts/train_medusa_8b.py` in tmux session `medusa_resume`. Script already configured: START_EPOCH=1, loads epoch0 checkpoint, scheduler resumes at correct LR point, saves `medusa_heads_8b_epoch1.pt` when done.

**Session 2026-06-24 notes:**
- Discovered passpoli network down: port 443 blocked (GitHub + HuggingFace both unreachable). Used TRANSFORMERS_OFFLINE=1 + HF_DATASETS_OFFLINE=1 to use local cache.
- Modified `scripts/train_medusa_8b.py` to support checkpoint resuming (START_EPOCH, CHECKPOINT variables at top; scheduler last_epoch fix). Committed to Mac repo; manually copied to passpoli via cat (git pull broken due to network).
- Day 8 concepts taught and verified:
  - Why a server is needed (decouple model from client, browser can't call Python directly)
  - HTTP POST vs GET (GET = read no body, POST = send data with body; POST is right for /generate)
  - WebSocket vs HTTP for streaming (HTTP = all-or-nothing response; WS = persistent push connection)
  - FastAPI + uvicorn + pydantic roles explained

**Session 2026-06-25 notes:**
- Confirmed Run 3 crashed before saving epoch 1 (`medusa_heads_8b_epoch1.pt` absent from `ls`).
- Understood the command flags before running: TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE (no network needed), PYTORCH_CUDA_ALLOC_CONF (flexible GPU memory allocator), PYTHONPATH=. (find src/ from project root).
- Network restored on passpoli — dropped offline flags, launched Run 4 in tmux `medusa_resume`.
- Run 4 DONE: `medusa_heads_8b_epoch1.pt` saved. Three-way benchmark run.
- Debugged device mismatch bug in `medusa_decode`: line 110 cast only dtype, not device. Fixed by adding `device=medusa.heads[0].W1.weight.device` to the `.to()` call. User diagnosed and wrote the fix themselves.
- Medusa 0.65x result understood and explained: healthy acceptance (2.22 tokens/round) but 3-pass overhead. Path to 1.5-2x is Extension A (tree attention).
- Rule re-established: never write code without explaining it first, stating recommendation (you write / I write), and waiting for user answer.

**Starting Point for Next Session (Extension A — Tree Attention):**
1. Goal: get Medusa from 0.65x → 1.5-2x+ by replacing 3 backbone passes per round with ~1
2. Plan: tree of candidates (width 2-3 per position, depth K=4) verified in one pass with custom attention mask
3. Candidate selection: torch.topk(head_logits, k=2) per head — NOT temperature sampling (temperature randomizes; we want the most likely candidates, which is top-k)
4. Concept to teach FIRST: why normal attention breaks on a tree, and what the tree attention mask does
5. After tree attention built: re-run benchmark, target >2x for resume
6. Training verdict: 25k × 2 epochs is decent but not the bottleneck — tree attention is the unlock. More training (50k+ examples, 3-4 epochs) adds ~0.3-0.5x on top after tree attention is in.
7. Days 8 + 9 (FastAPI server + frontend) still pending — do after Extension A
8. All three days targeted for 2026-06-26.

**Implementation decision (2026-06-26):**
- Start with **simple Cartesian product tree** (width=2, depth=4, 30 nodes, 16 paths). Build and benchmark first.
- **TODO after benchmarking:** implement calibrated optimal tree (Medusa paper Section 2.1.2). Greedy node selection using per-head top-k accuracies from calibration data — identical mask-building logic, better tree topology. Expected gain: ~10–20% better acceptance for same node budget. See CONCEPTS.md "Extension A — Tree Attention implementation decisions" for full details.

---

## Week 2: API, Visualization, Benchmarks, Deployment

### Day 8 — FastAPI Server and Streaming
**Status:** DONE (2026-06-26) — server written, not yet tested on passpoli (server still down).

**Deliverable:**
- [x] `src/server.py` — FastAPI app: `POST /generate` + `WS /stream`
- [x] `scripts/run_server.py` — loads model, injects into server module, starts uvicorn
- [x] `scripts/test_server.py` — tests both endpoints (POST + WebSocket)
- [ ] Live test on passpoli — pending server coming back online

**What was built:**
- `POST /generate`: receives `GenerateRequest` (prompt, max_new_tokens, mode), calls `naive_generate`, returns `{"output": text}`
- `WS /stream`: accepts WebSocket, receives JSON prompt, streams tokens one by one as `{"text": token, "accepted": null}` (null = naive, no color)
- Dependency injection pattern: `run_server.py` loads the model and sets `server.backbone` + `server.tokenizer` — `server.py` is model-agnostic
- Pydantic validates all incoming requests — type mismatch returns 422 before generation runs

**Concepts verified:**
- Why a server is needed (browser can't call Python directly — different environments)
- POST vs GET (POST = send data + process, GET = retrieve)
- uvicorn's role (handles HTTP protocol + network connections; FastAPI handles routing/logic)
- Pydantic validation (422 on type mismatch, before any code runs)
- Dependency injection (server.py defines HOW to serve; run_server.py decides WHICH model and WHEN)

**To test when passpoli is back:**
```bash
pip install fastapi uvicorn websockets
PYTHONPATH=. python scripts/run_server.py   # Terminal 1
PYTHONPATH=. python scripts/test_server.py  # Terminal 2
```

---

### Day 9 — Token Visualization Frontend
**Status:** DONE (2026-06-26) — frontend written, not yet tested end-to-end (pending passpoli).

**Deliverable:**
- [x] `frontend/index.html` — single-file UI, no framework, no build step
- [ ] Live end-to-end test — pending passpoli

**What was built:**
- Prompt textarea + Generate button
- Token display area — tokens stream in live via WebSocket, colored: `#4361ee` (accepted), `#780000` (rejected), white (naive)
- Stats bar: tokens/sec + token count, updating every token via `onmessage`
- `onclose` fires final stats update when stream ends
- Acceptance coloring will activate once speculative/Medusa mode is wired into the WebSocket endpoint — for now naive runs show white tokens

**Key JavaScript mechanics:**
- `ws.onopen` fires when connection established → send prompt (not before — connection not ready yet)
- `ws.onmessage` fires per token → parse JSON, create colored `<span>`, append to output div
- `ws.onclose` fires when server closes connection → final stats update

**Starting Point for Next Session:**
1. When passpoli comes back: `git pull` or manually copy `src/server.py`, `scripts/run_server.py`, `scripts/test_server.py`, `frontend/index.html`
2. `pip install fastapi uvicorn websockets`
3. Run server + test script → confirm `POST /generate` and `WS /stream` both work
4. Open `frontend/index.html` in browser (update `ws://localhost:8000` → `ws://<passpoli-ip>:8000`)
5. Next: wire speculative decode + Medusa into the stream endpoint so tokens get colored

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
| Baseline (8B naive) tok/s | 38.0 | 2026-06-14 |
| SpecDecode-Small tok/s | 44.4 (K=4, instruct draft) | 2026-06-14 |
| SpecDecode-Small speedup | 1.17x (K=4, instruct draft) | 2026-06-14 |
| Best acceptance rate | 3.05 tokens/round (K=4) | 2026-06-14 |
| Best K value | 4 | 2026-06-14 |
| Medusa tok/s (greedy, float32 heads) | 24.6 | 2026-06-25 |
| Medusa speedup (greedy, float32 heads) | 0.65x | 2026-06-25 |
| Medusa tok/s (greedy, float16 heads) | 26.4 | 2026-06-26 |
| Medusa speedup (greedy, float16 heads) | 0.69x | 2026-06-26 |
| Medusa acceptance rate | 2.22 tokens/round | 2026-06-25 |

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
- [x] The cost model: why speedup ≈ acceptance / (0.40 × K + 1) *(verified 2026-06-14)*
- [x] Why acceptance rate (avg tokens/round) is THE metric that decides the speedup *(verified 2026-06-14)*
- [x] Why matching draft/target training style (base vs instruct) raises acceptance *(verified 2026-06-14)*
- [x] Why an optimal K exists and why going past it hurts (linear certain cost vs diminishing conditional acceptance) *(verified 2026-06-14)*
- [ ] What a Medusa head is and how it differs from a separate draft model
- [ ] Why FastAPI + WebSockets for streaming (not HTTP)
