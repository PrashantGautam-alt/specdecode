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
