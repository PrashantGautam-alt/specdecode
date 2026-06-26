# SpecDecode — Concepts Reference

A living study sheet. Each concept = the clean core idea + the detailed math/steps.
Read it, then **close it and explain each one out loud from memory** — that is what
fights forgetting. The doc prevents loss; self-testing prevents forgetting. Both, or neither works.

> Companion to `DAILY_LOG.md` (which records *what* was done). This records *what I understand*.

---

## Day 6 — Medusa

### 1. The Medusa head

**Core idea:** Instead of a separate draft model, attach small neural networks ("heads")
to the *frozen* target model's final hidden state. Each head predicts a token *further*
in the future, so a single forward pass yields several future-token guesses for free.

**Where it attaches:** On the final hidden state `h` (a vector — 4096 numbers for Llama-8B),
in **parallel with** the original LM head, NOT on top of it. Both read the same `h`.

**What each predicts (the indexing is the whole trick):**
- Original LM head -> token at **t+1** (the immediate next token — the only thing the base
  model was ever trained to do).
- Medusa head 1 -> token at **t+2**
- Medusa head 2 -> token at **t+3**
- (head k -> t+k+1)

The heads don't redo t+1 (already covered); they reach into the future the base model is blind to.

**Why it matters:** No second model to run K times -> it eliminates the per-draft-token cost
(the `0.40 x K` "draft tax" measured on Day 5). The drafts ride the target's own forward pass.

---

### 2. SiLU activation

**Core idea:** An activation function adds **non-linearity**. Without one, stacking linear
layers collapses into a single linear layer (linear-of-linear is linear) — so it could never
solve a non-linear problem (the classic example: XOR, which lives in just 2 dimensions yet
no straight line can separate it). The enemy is *curvature*, not dimension count.

**Math:** `SiLU(x) = x * sigmoid(x)`, where `sigmoid(x) = 1 / (1 + e^(-x))` is a smooth
S-curve squashing any number into the range 0 to 1.
- Big positive x -> sigmoid ~ 1 -> SiLU ~ x   (passes through, like ReLU)
- Big negative x -> sigmoid ~ 0 -> SiLU ~ 0   (suppresses, like ReLU)
- Near zero      -> smooth transition, dips slightly below zero
- `SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0`

**Why SiLU over ReLU:**
1. Smooth everywhere -> smooth gradients, no "dead neuron" zero-gradient cliff that ReLU has.
2. The slight dip below zero (non-monotonic) empirically helps deep models learn richer functions.
3. Llama itself uses SiLU -> building the head with SiLU keeps it in the same family as the backbone.

---

### 3. Residual connection (the "+ h")

**Core idea:** After transforming `h` through `W1` then SiLU, add the original `h` back in.
The head then only has to learn a small **adjustment** to `h`, not the whole mapping from scratch.
`h` is already a rich summary of the context, so we start from it and pencil in a correction.

**In the formula:** `head(h) = W2 * ( SiLU(W1 * h) + h )`. The `+ h` is the residual / skip connection.

**Why:** learning a small correction is easier and more stable than learning the full function;
it also lets gradients flow straight through `h`.

---

### 4. Initialization trick

**Core idea:** Start each head as an **exact clone of the original LM head** (a warm start),
not from random numbers.

**The init:** `W2 = copy of the LM head`, `W1 = all zeros`.

**Why it works — walk it at step zero:**
1. `W1 = 0`  ->  `W1 * h = 0`
2. `SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0`  ->  the SiLU branch is zero
3. `SiLU(W1*h) + h = 0 + h = h`   <- the residual `+ h` is what survives
4. `head(h) = softmax(W2 * h)`
5. `W2 = LM head copy`  ->  `W2 * h` = exactly the LM head's prediction
=> at step zero the head **is** the LM head: a coherent predictor of t+1.

**Why clever:**
- Warm start: begins as a competent predictor, not random garbage.
- Stable training: small initial loss -> gentle gradients.
- Only a small shift to learn: from predicting t+1 to predicting t+2.

**The matched pair:** the `+ h` residual is exactly what makes zero-init work. Without it,
`W1 = 0` would give `W2 * SiLU(0) = W2 * 0 = 0` -> `softmax(0)` -> uniform garbage. The `+ h`
keeps `h` alive when the W1 branch is zeroed, so the head degrades into "exactly the LM head"
instead of into noise. Residual and zero-init were designed for each other.

---

### What is `h` (the hidden state)?

**Core idea:** `h` is the **last hidden state** — a vector of numbers (4096 for Llama-8B) that
is the model's compressed "understanding" of the context at that position. Not text, not a
token — a list of floats.

**The journey of a token:**
```
token ID -> embedding -> [ Layer 1 -> ... -> Layer 32 ] -> h -> LM head -> logits -> softmax -> next token
 (1234)    (4096 #s)          transformer stack          (4096 #s)        (128256 #s)
```
`h` sits right after the final layer, just *before* the LM head. The `logits` we used since
Day 2 (`output.logits[0, -1, :]`) are what comes one step *after* `h`. The Medusa heads tap
`h` directly — the model's finished thought — to look further ahead.

---

## Day 6 — PyTorch mechanics (how the concepts become code)

### nn.Module — the base class for any model component
- Subclass it. Define two methods:
  - `__init__`: declare the parts that HAVE weights (the layers). First line must be `super().__init__()`.
  - `forward`: the actual math / computation.
- You CALL the module like a function — `head(h)`, NOT `head.forward(h)`. PyTorch routes the call to `forward`.
- Why it exists: nn.Module auto-tracks every weight inside it, so the optimizer can find them,
  `.to(device)` moves them, and save/load works.

### nn.Linear — one linear layer (your W1, W2)
- `nn.Linear(in_features, out_features)` computes `y = x · W^T + b` and HOLDS the weight matrix for you.
- Apply it by CALLING it: `self.W1(h)`. Do NOT write `self.W1 * h` — a module is not a matrix you
  multiply; `*` raises a TypeError. The `·` in the math = a function call in code.

### nn.ModuleList — a list PyTorch can see into
- Holds a list of sub-modules (e.g. the K heads) and properly REGISTERS them.
- A plain Python list does NOT register: the modules' weights become invisible to `.parameters()`,
  are never moved by `.to(device)`, and are never trained. Always use nn.ModuleList for a list of layers.

### Freezing — requires_grad = False
- Every learnable weight is a Parameter with a flag `requires_grad`.
- True (default): a gradient is computed in the backward pass and the optimizer can update it (costs compute + memory).
- False: no gradient, no update — frozen. The expensive backward work for that weight is skipped.
- Freeze a whole model: `for p in model.parameters(): p.requires_grad = False`.
- Same family as the `with torch.no_grad():` used for inference — but permanent and per-weight.

### nn.init.zeros_ — in-place weight init
- `nn.init.zeros_(tensor)` fills a tensor with zeros, in place.
- Trailing `_` = PyTorch convention for "modify in place" (returns nothing new).
- For the init trick, zero BOTH `W1.weight` and `W1.bias`: `W1(h) = weight·h + bias`, so both must be
  zero for the SiLU branch to vanish exactly.

---

## Day 6 — Training the heads

### `forward`: why `hidden_states[-1]`, not `logits`
- A default model call returns only `.logits`. Pass `output_hidden_states=True` to expose `.hidden_states`
  (a tuple: embeddings + one per layer). Take `[-1]` = final layer = the cooked `h`.
- Heads eat `h`, NOT `logits`: logits are a LOSSY projection of `h` aimed at ONE position (t+1). Each head
  aims at a DIFFERENT future position, so it needs the full `h` — the future is already discarded in logits.

### Device + dtype must match
- Any op needs both tensors on the same DEVICE and in the same DTYPE, or it crashes.
- Heads are born on CPU/float32; backbone is on cuda/float16. `.to("cuda:0", dtype=...)` fixes both.
- This is why `nn.ModuleList` mattered: one `.to(...)` sweeps all heads (a plain list wouldn't).

### `torch.allclose` (the init-trick runtime proof)
- `==` on floats can be False even for "equal" values: the same number reached by a different ORDER of
  operations rounds differently in the last bit (e.g. `0.1+0.2 != 0.3`).
- `allclose(a, b, rtol, atol)` allows a tolerance: `|a−b| ≤ atol + rtol·|b|`. Float16 is coarse → use ~1e-3.
- Sanity check printed `True`: a fresh head's output == `lm_head(h)`. Init trick proven, not just on paper.

### The target SHIFT (THE Medusa training idea)
- Head k predicts t+k+1, so its labels are the input shifted left by `shift = k+1`.
- Align: `logits_k = head_logits[k][:, :-shift, :]` (drop last `shift` — no answer past the end);
  `labels_k = input_ids[:, shift:]` (drop first `shift`). Both end up length `seq_len − shift`.
- Logits are 3-D `[batch, pos, vocab]`; labels are 2-D `[batch, pos]` (one token ID, no vocab dim).
- WHY different shifts: same shift → all heads learn the identical "next token" → redundant. Different
  shifts force each head to own a future slot, so together they propose a CHAIN.

### Weighted loss `λk = 0.8^k`
- `L_total = Σ 0.8^k · L_k`. Weights decay: 1, 0.8, 0.64, 0.512.
- WHY down-weight far heads: (1) far future is harder → big noisy loss; equal weights let it DOMINATE the
  gradient and starve the near heads. (2) acceptance is sequential — head 0 wrong wastes the round, so near
  heads are worth more. Weight each head by its real contribution → a decreasing schedule.

### Cross-entropy + the reshape
- CE = negative log-likelihood; minimizing it = maximum likelihood.
- `F.cross_entropy` wants flat shapes: predictions `[N, C]`, answers `[N]`. Our slices are `[1, P, vocab]`
  and `[1, P]` → `logits.reshape(-1, vocab)` and `labels.reshape(-1)`. `-1` = "compute this dim to fit".

### The optimizer + the training heartbeat
- `AdamW(medusa.heads.parameters(), lr=1e-3)` — ONLY the heads (frozen backbone = no decision variables).
- Order: `zero_grad()` → `backward()` → `step()`. zero FIRST because PyTorch ACCUMULATES gradients by
  default — without clearing, this step's gradient piles onto the last one → wrong descent direction.
- `.item()` = pull the plain Python number out of a 1-value tensor (for logging only; never feed back).

### float16 NaN → mixed precision (float32 heads)
- fp16 window: max ~65,504, min normal ~6e-5. Training leaves it: softmax `exp(12)≈163k` OVERFLOWS → inf;
  tiny gradients UNDERFLOW → 0. Adam's divide by `sqrt(v)` turns inf/inf or 0/0 → NaN, from epoch 0.
- Fix (mixed precision): frozen backbone stays fp16 (no gradients, saves memory); trainable heads → fp32;
  cast `h` to the heads' dtype in `forward`. float32's range (~3e38) holds the same values safely.

### Saving the heads: `state_dict` + `torch.save`
- `state_dict()` = dict of `name → weight tensor`. Save `medusa.heads.state_dict()` ONLY (backbone unchanged).
- `torch.save(obj, "medusa_heads.pt")` writes it; reload with `load_state_dict(torch.load(...))`.
- It's the SAVED SOLUTION of the optimization — train once, reuse forever (Day 7 decoding loads it).

---

## Day 7 — Medusa decoding

### `medusa_decode` — the 4 phases
- Speculative decoding with the heads AS the draft (no separate model). One round:
  1. **PROPOSE** — run model once; each head's argmax at the last position → a chain of K candidate tokens.
  2. **VERIFY** — feed `[context + first K−1 candidates]` through the backbone once; its argmax at the last
     K positions = the big model's own picks (ground truth). K−1 because the last real token already
     predicts candidate 0, and each appended candidate unlocks the next.
  3. **ACCEPT** — walk left to right, ALWAYS emit the backbone's token; match = head right (continue);
     first mismatch = emit the correction and stop.
  4. **APPEND** and loop.
- **Correctness guarantee:** every emitted token is the backbone's own argmax → output is IDENTICAL to plain
  greedy decoding. Medusa changes SPEED, not the result. Worst case 1 token/round; best case K.
- **Greedy verification** means we always accept the backbone's argmax. The exact rejection-sampling
  version (Day 3's `min(1, p/q)`) is the refinement for sampled (temperature > 0) decoding.

---

## Day 7 — Persistent KV cache in `medusa_decode`

### Why the naive version was O(n²)

The first `medusa_decode` had no KV cache. Every round, it fed the entire growing sequence through the backbone from scratch. After M tokens were accepted, the next round re-read M tokens again — then M+1, then M+2. Total work grows quadratically with length. Same Day-4 trap.

**Fix — four phases with a persistent cache:**

1. **PRIME** — run the whole prompt (minus the last token) through the backbone once; save the resulting `DynamicCache`. Never recomputed.
2. **PROPOSE** — feed only the last token through using the cache (O(1)). Gets `h` for the heads AND the backbone's own prediction for candidate 0. Heads argmax on `h` → K candidate tokens.
3. **VERIFY** — feed the first K-1 candidates through using a snapshot of the PROPOSE cache. Backbone's argmax at positions 1..K-1, combined with PROPOSE's position-0 prediction = full K opinions.
4. **UPDATE CACHE** — re-feed `accepted[:-1]` into a clean PROPOSE snapshot to advance the cache. At most K-1 tokens, regardless of total sequence length.

**Why VERIFY's cache can't be reused for UPDATE:** if a mismatch occurred at position i, the tokens at i+1 onward were wrong. The KV vectors for those positions are contaminated. The clean PROPOSE snapshot is the last known-good state.

---

### The DynamicCache in-place mutation bug

`DynamicCache` is **mutable and modified in place** during every forward pass — the backbone appends the new K/V vectors into whatever object you pass. Passing the same cache object to both VERIFY and UPDATE meant VERIFY contaminated it before UPDATE could use it.

**Fix — `snap()`:**

```python
def snap(kv):
    legacy = kv.to_legacy_cache()
    return DynamicCache.from_legacy_cache(legacy)
```

Round-trips the cache through the legacy tuple format, creating a **fresh, independent object** with the same K/V contents. Two snapshots are made after PROPOSE: `full_cache` (expendable, given to VERIFY) and `update_cache` (clean, reserved for UPDATE). Mutations to one can't reach the other.

---

## Day 7 / Route B — Multi-GPU training and 8-bit Adam

### 8-bit Adam — what it compresses and why

Adam stores **three things per trainable parameter:**
- The parameter itself (float32)
- **m** (first moment): running average of recent gradients — which direction has the gradient been pointing?
- **v** (second moment): running average of squared gradients — how noisy/consistent has the gradient been?

Update rule: `param -= lr * m / (sqrt(v) + epsilon)`. The ratio adapts the step size per parameter.

**The memory cost:** parameter + m + v = 3× storage per trainable weight.

For 4 Medusa heads (~2.2B params): 8.8 GB params + 17.6 GB (m + v) + 16 GB backbone = ~42 GB before activations — spills over 48 GB (2× A5000).

**8-bit Adam** (`bitsandbytes.optim.Adam8bit`) quantizes **m and v** to 8-bit integers (4× compression): 17.6 GB → 4.4 GB. Parameters stay float32.

**Why we can compress m/v but not parameters:**
- m and v are **estimates** used as multipliers. A small quantization error = a slightly imprecise step — the optimizer self-corrects over many iterations.
- Parameters are **accumulators**. Each update adds a tiny delta (maybe `1e-5`). In 8-bit (step ~0.008), that delta rounds to zero and is never applied — training silently stalls.

Rule: **quantize what you USE (optimizer stats), not what you ACCUMULATE (parameters).**

**What is actually compressed vs not — the full picture:**

| Thing | Precision | Why |
|---|---|---|
| Backbone weights | float16 | Saves memory; frozen so no gradient risk |
| Head weights (W1, W2) | **float32** | Accumulate tiny updates — can't quantize |
| Adam m and v | **8-bit** | Estimates used as multipliers — small error is fine |
| Saved `.pt` file | float32 | Head weights only, full precision |

The heads themselves are **not** compressed — trained and saved in full float32. 8-bit Adam compresses Adam's internal notebook (m and v), not the building it's renovating (the weights).

---

### Multi-GPU device placement (Route B)

Backbone on `cuda:0` (float16), heads on `cuda:1` (float32).

```python
medusa = MedusaModel(backbone, num_heads=4)
medusa.heads.to("cuda:1")
optimizer = bnb.optim.Adam8bit(medusa.heads.parameters(), lr=1e-3)
```

**The tensor boundary:** `h` comes off `cuda:0` (backbone). Heads are on `cuda:1`. Fix in `medusa.py` forward:

```python
h = h.to(device=self.heads[0].W1.weight.device, dtype=self.heads[0].W1.weight.dtype)
```

Reads device and dtype from the heads' own weights — correct by construction regardless of placement. Hardcoding `"cuda:1"` would be fragile.

**`.pt` files:** binary (Python pickle). Not human-readable — never open in nvim (exit with `:q!`). Not for GitHub (can't diff, bloat history, hit 100 MB limits). Real checkpoints go on HuggingFace Hub; toy checkpoints excluded via `*.pt` in `.gitignore`.

---

## Day 7 — Why Medusa is slow despite good acceptance (the 3-pass problem)

### The bottleneck

Measured: 2.22 tokens/round accepted, yet only 0.65x speedup. The math doesn't add up — unless you count how many backbone passes each round actually costs.

**Our greedy implementation does 3 backbone passes per round:**

| Pass | What it does | Why we can't skip it |
|---|---|---|
| **PROPOSE** | Feed last token → get `h` → heads propose K candidates | Need `h` for the heads AND backbone's opinion at position 0 |
| **VERIFY** | Feed K−1 candidates → backbone's opinions at positions 1..K−1 | Need to know which candidates the big model agrees with |
| **CACHE UPDATE** | Re-feed `accepted[:-1]` → advance cache to accepted state | Cache is contaminated by VERIFY; must restore to clean state |

**The cost model:**

```
speedup ≈ tokens_accepted_per_round / backbone_passes_per_round
        = 2.22 / 3
        ≈ 0.74x   (theoretical ceiling)
```

We measured 0.65x, not 0.74x — the gap is cross-GPU PCIe latency: `h` travels `cuda:0 → cuda:1` every single round (backbone on GPU 0, heads on GPU 1). That crossing has a fixed cost per round that doesn't shrink with better acceptance.

**The key insight:** acceptance rate is NOT the bottleneck here. Even if the heads were perfect (K=4 tokens accepted every round), we'd get at best 4/3 ≈ 1.33x. The denominator (3 passes) is the problem, not the numerator.

---

### Why 3 passes? (the VERIFY + CACHE UPDATE split)

Recall the `DynamicCache` in-place mutation bug: VERIFY mutates the cache as it runs. After VERIFY, the cache reflects a world where ALL K−1 candidates were appended — but some were rejected. We can't use that as the starting point for the next round.

So CACHE UPDATE re-feeds only the *accepted* tokens into a clean snapshot of the PROPOSE cache to advance it correctly. This is correct, but it's a third backbone call.

**Equation tying everything together:**

```
passes per round = 1 (PROPOSE) + 1 (VERIFY) + 1 (CACHE UPDATE)
                 = 3

speedup = acceptance / passes = 2.22 / 3 ≈ 0.74x
measured = 0.65x   (gap = ~0.09x from cross-GPU latency)
```

---

### The fix: tree attention (Extension A)

Tree attention collapses all 3 passes into approximately 1 by verifying a **tree of candidates** (not a chain) in a single forward pass with a custom attention mask.

- Instead of the chain `[c0, c1, c2, c3]` (1 path), the tree holds multiple candidate paths simultaneously (e.g. top-2 per head = up to 2^4 = 16 paths, pruned to ~8).
- One forward pass verifies all paths at once using a mask that controls which positions can attend to which.
- CACHE UPDATE is absorbed: the tree pass already advances the cache to wherever the winner path lands.

Projected speedup after tree attention: **1.5–2x+**, same acceptance rate, same trained weights, no new training needed.

**The concept to learn before Extension A:** why does normal causal attention break on a tree, and what does the mask need to do differently?

---

## Day 8 — FastAPI Server

### Why a server is needed
The browser and Python run in completely different environments — different processes, different machines, different languages. A server is the bridge: it speaks HTTP/WebSocket (what browsers understand) on one side and calls Python functions on the other.

### POST vs GET
- **GET** = "give me something" — no body, used for reading data
- **POST** = "here's data, process it" — has a body, used when you're sending something to be acted on
- `/generate` is POST: you're sending a prompt and asking the server to do work with it

### FastAPI + uvicorn + pydantic — three separate jobs
- **Pydantic**: validates incoming JSON → converts it to a typed Python object. If a field is the wrong type, it returns a 422 error before your code runs.
- **FastAPI**: routing and request handling — maps URLs to Python functions, reads Pydantic models, serializes responses to JSON
- **uvicorn**: the actual HTTP server — handles network connections, TCP, protocol parsing. FastAPI delegates all of that to uvicorn.

### Dependency injection pattern
`server.py` defines HOW to serve requests. `run_server.py` decides WHICH model to load and WHEN. `server.py` never imports or loads a model itself — it just uses whatever was injected into `server.backbone` and `server.tokenizer`. This means swapping the model (1B → 8B → Medusa) requires changing only `run_server.py`.

---

## Day 9 — WebSocket streaming frontend

### How JavaScript talks to a WebSocket
```javascript
const ws = new WebSocket("ws://localhost:8000/stream");

ws.onopen = () => {
    ws.send(JSON.stringify({prompt: "...", max_new_tokens: 100}));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);  // {text, accepted}
    // render token
};

ws.onclose = () => { /* final cleanup */ };
```
- `onopen` fires once when connection is established → send prompt here, not before (connection not ready before this)
- `onmessage` fires once per token the server sends → update UI here
- `onclose` fires when the server closes the connection → final stats update

### Why tokens stream via WebSocket not HTTP
HTTP is request-response: the client sends one request and waits for the full response. The entire generated text would have to be ready before anything is shown. WebSocket is a persistent two-way connection: the server can push each token the moment it's generated. The user sees text appearing live instead of waiting for the full output.

### Token coloring
Each token arrives as `{"text": "...", "accepted": true/false/null}`:
- `true` → `#4361ee` (blue) — draft token accepted by backbone
- `false` → `#780000` (dark red) — backbone correction
- `null` → white — naive generation, no acceptance concept

---

## Extension A — Tree Attention implementation decisions

### Simple Cartesian product tree (what we implement first)

Width=2, depth=4 (4 heads). Every head contributes top-2 candidates, all combinations enumerated.

- Nodes per level: 2, 4, 8, 16 → **30 total nodes**
- Paths (root-to-leaf): 2^4 = **16**
- Tree is built fresh each decode step from `torch.topk(head_logits, k=2)`
- Mask is fixed structure — same shape every round, pre-computable

### Calibrated optimal tree (TODO — implement after simple tree is benchmarked)

The Medusa paper shows the Cartesian product tree is suboptimal. With a fixed node budget, some nodes (low-probability paths) waste slots that could go to high-probability ones.

**The math:** define `a_k(i)` = accuracy of the i-th top prediction of head k (from calibration data). Assuming independence, accuracy of path `[i1, i2, i3, i4]` = `a_1(i1) × a_2(i2) × a_3(i3) × a_4(i4)`. The contribution of any node = its path accuracy.

**The algorithm:** greedy node selection — start from root, repeatedly add the unselected node connected to the current tree with highest path accuracy, until node budget is exhausted. Identical in structure to Prim's MST algorithm (maximize accuracy instead of minimize cost).

**Why deferred:** requires a calibration pass over data to measure per-head top-k accuracies. The mask-construction logic is the same — only the tree topology changes. Implement after simple tree confirms the speedup gain from tree attention itself.

**Expected gain over simple tree:** ~10–20% better acceptance length for the same node budget.
