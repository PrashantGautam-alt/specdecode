import time
import torch
import torch.nn as nn
from transformers import DynamicCache


class MedusaHead(nn.Module):
    """
    One Medusa head: a small two-layer MLP that sits parallel to the LM head
    and predicts a token further in the future than the base model does.

    Formula: W2(SiLU(W1(h)) + h)
    The residual '+h' means at init (W1=0) the head reduces exactly to the LM head.
    Each head is initialized this way so training starts from a sane prediction,
    not random noise.
    """

    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, vocab_size)
        self.act = nn.SiLU()

    def forward(self, h):
        return self.W2(self.act(self.W1(h)) + h)


class MedusaModel(nn.Module):
    """
    Wraps a frozen backbone with num_heads Medusa heads.

    The backbone is never updated. The heads train on top of its hidden states
    to predict tokens at positions t+2, t+3, ... t+num_heads+1.
    At inference, one backbone pass produces the hidden state h, and all heads
    run on that same h in parallel to propose multiple future tokens at once.
    """

    def __init__(self, backbone, num_heads):
        super().__init__()

        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_dim = self.backbone.config.hidden_size
        vocab_size = self.backbone.config.vocab_size

        self.heads = nn.ModuleList(
            [MedusaHead(hidden_dim, vocab_size) for _ in range(num_heads)]
        )

        lm_head = backbone.lm_head

        for head in self.heads:
            # Zero W1 so the SiLU branch vanishes: SiLU(W1*h) = SiLU(0) = 0.
            # With '+h' surviving, W2*h then needs to equal lm_head(h).
            # Zeroing the bias too, otherwise W1(h) = bias != 0 and the branch survives.
            head.W1.weight.data.zero_()
            if head.W1.bias is not None:
                head.W1.bias.data.zero_()
            # Copy the LM head weights into W2 so the head starts as an exact clone.
            head.W2.weight.data.copy_(lm_head.weight.data)
            if head.W2.bias is not None:
                head.W2.bias.data.zero_()

    def forward(self, input_ids, attention_mask=None):
        """
        One backbone pass with hidden states exposed, then all heads run on h.

        Returns a list of K tensors, each shape [batch, seq_len, vocab_size].
        Head k predicts the token at position t+k+1.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        h = outputs.hidden_states[-1]
        # backbone is float16, heads may be float32 (needed to avoid NaN during training)
        h = h.to(device=self.heads[0].W1.weight.device, dtype=self.heads[0].W1.weight.dtype)

        return [head(h) for head in self.heads]

def medusa_decode(medusa, tokenizer, prompt, max_new_tokens=100, K=4, verbose=False):
    """
    Generate with Medusa: each round the heads PROPOSE K future tokens,
    the backbone VERIFIES them in one pass, and we keep the longest correct
    prefix (greedy verification = match the backbone's own argmax).
    """
    device = next(medusa.backbone.parameters()).device

    def snap(kv):
        legacy = kv.to_legacy_cache() if hasattr(kv, 'to_legacy_cache') else kv
        return DynamicCache.from_legacy_cache(legacy)

    generated = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    start_len = generated.shape[1]  # where the prompt ends; we stop max_new_tokens past it
    total_accepted = 0
    rounds = 0

    # PRIME: run all prompt tokens except the last through the backbone once.
    # This builds the starting cache so the loop never recomputes the full context.
    # We exclude the last token because PROPOSE will process it first each round.
    with torch.no_grad():
        prime_out = medusa.backbone(input_ids=generated[:, :-1], use_cache=True)
    cache = snap(prime_out.past_key_values)

    while generated.shape[1] < start_len + max_new_tokens:
        # PHASE 1 — PROPOSE: process only the last token using the cache (O(1) cost).
        # The backbone returns the hidden state h at that position — the heads read h
        # to propose K candidates. backbone_pred_0 is what the backbone itself would
        # say next, obtained for free from the same pass.
        with torch.no_grad():
            propose_out = medusa.backbone(
                input_ids=generated[:, -1:],
                past_key_values=cache,
                output_hidden_states=True,
                use_cache=True,
            )
        full_cache = snap(propose_out.past_key_values)
        update_cache = snap(full_cache)   # clean n-token snapshot for PHASE 4; full_cache will be mutated by VERIFY
        backbone_pred_0 = propose_out.logits[0, -1, :].argmax().item()
        h = propose_out.hidden_states[-1].to(
            device=medusa.heads[0].W1.weight.device,
            dtype=medusa.heads[0].W1.weight.dtype,
        )
        candidates = [medusa.heads[k](h)[0, -1, :].argmax().item() for k in range(K)]

        # PHASE 2 — VERIFY: feed only the K-1 candidates through the backbone using
        # full_cache (O(K) cost, independent of sequence length).
        # backbone_preds[0] already came from PROPOSE; verify gives us [1..K-1].
        cand_tensor = torch.tensor([candidates[:K-1]], device=device, dtype=torch.long)
        with torch.no_grad():
            verify_out = medusa.backbone(
                input_ids=cand_tensor,
                past_key_values=full_cache,
                use_cache=True,
            )
        verify_preds = verify_out.logits[0, :, :].argmax(dim=-1).tolist()
        backbone_preds = [backbone_pred_0] + verify_preds

        # PHASE 3 — ACCEPT: walk left to right. We always emit the backbone's token,
        # never the candidate — output stays identical to plain greedy decoding.
        # Stop at the first mismatch: every later candidate used a wrong prefix.
        new_tokens = []
        for i in range(K):
            new_tokens.append(backbone_preds[i])
            if backbone_preds[i] != candidates[i]:
                break

        # PHASE 4 — UPDATE CACHE then APPEND accepted tokens.
        # We can't reuse verify's KV: it was computed with candidates, and a mismatch
        # means a wrong token was fed in — contaminating the cache past that point.
        # Re-feeding accepted[:-1] is always correct and costs at most K-1 tokens.
        m = len(new_tokens)
        if m == 1:
            cache = update_cache
        else:
            update_input = torch.tensor([new_tokens[:-1]], device=device, dtype=torch.long)
            with torch.no_grad():
                update_out = medusa.backbone(
                    input_ids=update_input,
                    past_key_values=update_cache,
                    use_cache=True,
                )
            cache = snap(update_out.past_key_values)
        new_tensor = torch.tensor([new_tokens], device=device, dtype=torch.long)
        generated = torch.cat([generated, new_tensor], dim=1)
        rounds += 1
        total_accepted += m
        if verbose:
            print(f"  round {rounds}: accepted {m}/{K} tokens")

    if verbose:
        print(f"  avg tokens/round: {total_accepted / rounds:.2f} over {rounds} rounds")
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def build_tree_candidates(head_logits, width=2):
    """
    Build the flat node list for a Cartesian product candidate tree.

    head_logits: list of K tensors, each shape [vocab_size]
    width:       number of top candidates to keep per head

    Returns:
        tokens:         list of token IDs, one per node (length = width * (width^K - 1) / (width - 1))
        parent_indices: list of ints, parent_indices[i] = index of node i's parent in tokens
                        (-1 for root-level nodes, meaning their parent is the context)
    """
    tokens = []

    parent_indices = []

    # level_node_indices tracks the flat indices of all nodes at the CURRENT level
    # at depth 1, these are indices 0, 1 (the two root nodes)
    # at depth 2, these are indices 2, 3, 4, 5 (children of the two root nodes)
    level_node_indices = []

    for depth, logits in enumerate(head_logits):
        topk_tokens = torch.topk(logits, width).indices.tolist()  # get top-`width` token IDs from this head's logits

        if depth == 0:
            # no parents yet — root level
            for tok in topk_tokens:
                tokens.append(tok)
                parent_indices.append(-1)
            level_node_indices = list(range(len(tokens)))  # e.g. [0, 1]

        else:
            next_level_indices = []
            for parent_idx in level_node_indices:
                for tok in topk_tokens:
                    tokens.append(tok)
                    parent_indices.append(parent_idx)
                    next_level_indices.append(len(tokens) - 1)
            level_node_indices = next_level_indices

    return tokens, parent_indices


def build_tree_mask(parent_indices):
    """
    Build the boolean attention mask for the candidate tree.

    parent_indices: list of ints from build_tree_candidates
                    (-1 means root node, no parent)

    Returns:
        mask: BoolTensor shape [num_nodes, num_nodes]
              mask[i][j] = True means node i is allowed to attend to node j
    """
    n = len(parent_indices)
    mask = torch.zeros(n, n, dtype=torch.bool)

    for i, parent in enumerate(parent_indices):
        if parent == -1:
            # root node — attends only to itself
            mask[i][i] = True
        else:
            # copy everything the parent could see, then add self
            mask[i] = mask[parent]
            mask[i][i] = True

    return mask

def medusa_decode_tree(medusa, tokenizer, prompt, max_new_tokens=100, K=4, width=2, verbose=False, profile=False):
    device = next(medusa.backbone.parameters()).device

    # profile=False -> the timing branches below never run, so the hot path is unchanged.
    # When True, we sync the GPU at each phase boundary and accumulate per-phase time so we
    # can see where a round's ~67 ms actually goes (compute vs cache bookkeeping vs Python).
    timings = {"propose": 0.0, "build_tree": 0.0, "verify_setup": 0.0, "verify_fwd": 0.0,
               "verify_post": 0.0, "find_path": 0.0, "cache_update": 0.0}

    def snap(kv):
        legacy = kv.to_legacy_cache() if hasattr(kv, 'to_legacy_cache') else kv
        return DynamicCache.from_legacy_cache(legacy)

    generated = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    start_len = generated.shape[1]
    total_accepted = 0
    rounds = 0

    with torch.no_grad():
        prime_out = medusa.backbone(input_ids=generated[:, :-1], use_cache=True)
    cache = snap(prime_out.past_key_values)

    while generated.shape[1] < start_len + max_new_tokens:

        if profile:
            torch.cuda.synchronize(); t0 = time.perf_counter()

        # PHASE 1 — PROPOSE
        with torch.no_grad():
            propose_out = medusa.backbone(
                input_ids=generated[:, -1:],
                past_key_values=cache,
                output_hidden_states=True,
                use_cache=True,
            )
        propose_cache = snap(propose_out.past_key_values)
        h = propose_out.hidden_states[-1].to(
            device=medusa.heads[0].W1.weight.device,
            dtype=medusa.heads[0].W1.weight.dtype,
        )
        head_logits = [medusa.heads[k](h)[0, -1, :] for k in range(K)]
        backbone_pred_0 = propose_out.logits[0,-1,:].argmax().item()

        if profile:
            torch.cuda.synchronize(); t1 = time.perf_counter()

        # PHASE 2 — BUILD TREE
        tokens, parent_indices = build_tree_candidates(head_logits=head_logits, width=width)
        tree_mask = build_tree_mask(parent_indices=parent_indices)

        if profile:
            torch.cuda.synchronize(); t2 = time.perf_counter()

        # PHASE 3 — TREE VERIFY
        # we want past_key_values (reuse context KV) + custom 4D tree mask.
        # transformers _update_causal_mask overwrites custom masks when past_key_values is active.
        # fix: temporarily replace _update_causal_mask with a passthrough for this call only.
        num_nodes = len(tokens)
        context_len = generated.shape[1]
        node_tensor = torch.tensor([tokens], device=device, dtype=torch.long)

        # 4D additive mask [1, 1, num_nodes, context_len + num_nodes]
        full_mask = torch.full(
            (1, 1, num_nodes, context_len + num_nodes), float('-inf'), dtype=torch.float16, device=device
        )
        full_mask[:, :, :, :context_len] = 0.0  # all tree nodes attend to all context
        tree_allow = tree_mask.to(device)
        full_mask[0, 0, :, context_len:] = torch.where(
            tree_allow,
            torch.zeros_like(tree_allow, dtype=torch.float16),
            torch.full_like(tree_allow, float('-inf'), dtype=torch.float16),
        )

        # depth-based position_ids: all nodes at same depth share the same RoPE rotation.
        # without this, sibling nodes get different positions and produce wrong logits.
        node_depths = []
        for i in range(num_nodes):
            d, node = 0, i
            while parent_indices[node] != -1:
                node = parent_indices[node]
                d += 1
            node_depths.append(d)
        tree_pos_ids = torch.tensor(
            [context_len + d for d in node_depths], device=device
        ).unsqueeze(0)  # [1, num_nodes]

        # use forward_pre_hook on every attention layer to inject our tree mask directly.
        # this bypasses all of transformers' internal mask building (_update_causal_mask,
        # sdpa slicing, etc.) regardless of transformers version.
        # the hook runs just before LlamaAttention.forward, after which the KV cache update
        # happens inside forward so key_states.shape[-2] = context_len + num_nodes = correct.
        _captured_mask = full_mask  # closed over by the hook below

        def _inject(module, args, kwargs):
            kwargs['attention_mask'] = _captured_mask
            return args, kwargs

        hooks = [
            layer.self_attn.register_forward_pre_hook(_inject, with_kwargs=True)
            for layer in medusa.backbone.model.layers
        ]
        # DynamicCache mutates in place: the verify pass appends the 30 tree nodes into
        # whatever cache we pass. Give it a fresh snapshot so propose_cache stays clean
        # (context_len tokens) for the m==1 case in Phase 5. Without this, the next round
        # reuses a cache polluted with 30 phantom nodes → mask/KV length mismatch.
        verify_input_cache = snap(propose_cache)
        if profile:
            torch.cuda.synchronize(); t2a = time.perf_counter()
        try:
            with torch.no_grad():
                verify_out = medusa.backbone(
                    input_ids=node_tensor,
                    past_key_values=verify_input_cache,
                    position_ids=tree_pos_ids,
                    use_cache=True,
                )
        finally:
            for h in hooks:
                h.remove()

        if profile:
            torch.cuda.synchronize(); t2b = time.perf_counter()

        verify_cache = snap(verify_out.past_key_values)
        verify_logits = verify_out.logits[0]  # [num_nodes, vocab_size]
        backbone_preds = verify_logits.argmax(dim=-1).tolist()

        if profile:
            torch.cuda.synchronize(); t3 = time.perf_counter()

        # PHASE 4 — FIND BEST PATH
        # also track best_path_nodes: the tree node indices for accepted tokens.
        # needed in Phase 5 to extract their KV from verify_cache without a backbone forward.
        leaf_start = num_nodes - width**K
        best_accepted = []
        best_path_nodes = []

        for leaf in range(leaf_start, num_nodes):
            path = []
            node = leaf
            while node != -1:
                path.append(node)
                node = parent_indices[node]
            path.reverse()

            accepted = [backbone_pred_0]
            path_nodes = []

            if backbone_pred_0 == tokens[path[0]]:
                path_nodes.append(path[0])
                for j in range(len(path)):
                    if j + 1 < len(path):
                        accepted.append(backbone_preds[path[j]])
                        if backbone_preds[path[j]] != tokens[path[j + 1]]:
                            break
                        path_nodes.append(path[j + 1])
                    else:
                        accepted.append(backbone_preds[path[j]])  # bonus token

            if len(accepted) > len(best_accepted):
                best_accepted = accepted
                best_path_nodes = path_nodes

        if not best_accepted:
            best_accepted = [backbone_pred_0]
            best_path_nodes = []

        if profile:
            torch.cuda.synchronize(); t4 = time.perf_counter()

        # PHASE 5 — CACHE UPDATE via KV extraction (no backbone forward)
        # accepted tokens at t+1..t+m-1 correspond to tree nodes best_path_nodes[0..m-2].
        # their KV already exists in verify_cache at positions context_len + node_index.
        # index them out directly — saves an entire backbone forward pass vs the old approach.
        m = len(best_accepted)
        if m == 1:
            cache = propose_cache
        else:
            kv_idx = list(range(context_len)) + [context_len + n for n in best_path_nodes]
            idx = torch.tensor(kv_idx, device=device)
            legacy = verify_cache.to_legacy_cache()
            trimmed = tuple((kv[0][:, :, idx, :], kv[1][:, :, idx, :]) for kv in legacy)
            cache = DynamicCache.from_legacy_cache(trimmed)

        if profile:
            torch.cuda.synchronize(); t5 = time.perf_counter()
            timings["propose"] += t1 - t0
            timings["build_tree"] += t2 - t1
            timings["verify_setup"] += t2a - t2
            timings["verify_fwd"] += t2b - t2a
            timings["verify_post"] += t3 - t2b
            timings["find_path"] += t4 - t3
            timings["cache_update"] += t5 - t4

        new_tensor = torch.tensor([best_accepted], device=device, dtype=torch.long)
        generated = torch.cat([generated, new_tensor], dim=1)
        rounds += 1
        total_accepted += m
        if verbose:
            print(f"  round {rounds}: accepted {m}/{K+1} tokens")

    if verbose:
        print(f"  avg tokens/round: {total_accepted / rounds:.2f} over {rounds} rounds")
    if profile:
        total = sum(timings.values())
        print(f"\n--- per-round timing breakdown (avg over {rounds} rounds) ---")
        for name, t in timings.items():
            print(f"  {name:<13} {1000*t/rounds:>7.2f} ms/round  {100*t/total:>5.1f}%")
        print(f"  {'measured':<13} {1000*total/rounds:>7.2f} ms/round  (sum of phases)")
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def medusa_decode_tree_fused(medusa, tokenizer, prompt, max_new_tokens=100, K=4, width=2, verbose=False, debug=False, ref_ids=None, return_ids=False,
                             accept_mode="greedy", temperature=1.0, epsilon=0.3, delta=0.09):
    """
    Tree decode with PROPOSE folded into VERIFY: ONE backbone pass per round instead of two.

    The standalone PROPOSE pass existed only to compute a fresh hidden state h for the heads.
    But the VERIFY pass already produces a hidden state for every node it processes — we were
    throwing them away. Here we keep them: the hidden state at this round's last matched node
    is exactly what the heads need to build the NEXT round's tree.

    Each round feeds [bonus_token] ++ [tree_nodes] in a single pass:
      - the prepended bonus token (last round's guaranteed token) gets its KV computed here,
        and its LM logits give backbone_pred_0 (the true next token) for free
      - the tree nodes get verified against backbone_pred_0 and each other

    Because head 0 predicts t+1 (the same token the LM head gives), it is redundant with the
    prepended bonus. So the tree is built from heads 1..K-1 → depth K-1 instead of K. We trade
    one tree level for eliminating a whole backbone pass — a good deal on a memory-bound model.

    Invariant held at the top of each round:
      - generated   = all committed tokens [0..P-1]
      - cache       = KV for exactly those committed tokens
      - pending     = the guaranteed token at position P (NOT yet in generated or cache)
      - seed_h      = hidden state at position P-1 (predicts pending via head 0, the tree via 1..K-1)
    """
    device = next(medusa.backbone.parameters()).device
    head_device = medusa.heads[0].W1.weight.device
    head_dtype = medusa.heads[0].W1.weight.dtype
    tree_depth = K - 1  # head 0 is spent on the prepended bonus; heads 1..K-1 build the tree

    def snap(kv):
        legacy = kv.to_legacy_cache() if hasattr(kv, 'to_legacy_cache') else kv
        return DynamicCache.from_legacy_cache(legacy)

    generated = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    start_len = generated.shape[1]
    total_new = 0
    rounds = 0

    # COLD START — one ordinary forward over the prompt to bootstrap the invariant.
    # cache holds all prompt tokens; pending is the first generated token; seed_h is the
    # hidden state at the last prompt token (which predicts pending and seeds the first tree).
    with torch.no_grad():
        cold = medusa.backbone(input_ids=generated, use_cache=True, output_hidden_states=True)
    cache = snap(cold.past_key_values)
    seed_h = cold.hidden_states[-1][0, -1, :].to(device=head_device, dtype=head_dtype)
    pending = cold.logits[0, -1, :].argmax().item()

    while generated.shape[1] < start_len + max_new_tokens:
        context_len = generated.shape[1]  # cache length = number of committed tokens
        pending_in = pending  # snapshot for the debug dump (pending gets overwritten below)
        if debug:
            # incoming cache length should equal generated length (one KV per committed token)
            cache_len_in = cache.to_legacy_cache()[0][0].shape[2]

        # BUILD TREE from seed_h using heads 1..K-1 (head 0 predicts the already-known pending).
        head_logits = [medusa.heads[k](seed_h)[:].reshape(-1) for k in range(1, K)]
        tokens, parent_indices = build_tree_candidates(head_logits=head_logits, width=width)
        tree_mask = build_tree_mask(parent_indices=parent_indices)
        num_nodes = len(tokens)

        # FUSED INPUT: [pending] ++ tree_nodes. The prepend sits at index 0 (absolute pos P).
        seq = [pending] + tokens
        seq_tensor = torch.tensor([seq], device=device, dtype=torch.long)
        q_len = 1 + num_nodes
        kv_len = context_len + 1 + num_nodes

        # node depths (1-based): depth-1 node is a candidate for position P+1.
        node_depths = []
        for i in range(num_nodes):
            d, node = 0, i
            while parent_indices[node] != -1:
                node = parent_indices[node]
                d += 1
            node_depths.append(d + 1)  # +1 because the prepend occupies the P slot

        # position_ids: prepend at P (=context_len), node at P + its depth.
        pos_ids = torch.tensor(
            [context_len] + [context_len + d for d in node_depths], device=device
        ).unsqueeze(0)

        # 4D additive mask [1, 1, q_len, kv_len].
        mask = torch.full((1, 1, q_len, kv_len), float('-inf'), dtype=torch.float16, device=device)
        mask[:, :, :, :context_len] = 0.0       # everyone attends to all committed context
        mask[:, :, :, context_len] = 0.0        # everyone attends to the prepend (ancestor of all)
        tree_allow = tree_mask.to(device)
        mask[0, 0, 1:, context_len + 1:] = torch.where(  # node-to-node follows ancestor mask
            tree_allow,
            torch.zeros_like(tree_allow, dtype=torch.float16),
            torch.full_like(tree_allow, float('-inf'), dtype=torch.float16),
        )

        _captured = mask

        def _inject(module, args, kwargs):
            kwargs['attention_mask'] = _captured
            return args, kwargs

        hooks = [
            layer.self_attn.register_forward_pre_hook(_inject, with_kwargs=True)
            for layer in medusa.backbone.model.layers
        ]
        verify_input_cache = snap(cache)  # keep `cache` clean; the pass mutates this copy
        try:
            with torch.no_grad():
                out = medusa.backbone(
                    input_ids=seq_tensor,
                    past_key_values=verify_input_cache,
                    position_ids=pos_ids,
                    output_hidden_states=True,
                    use_cache=True,
                )
        finally:
            for h in hooks:
                h.remove()

        out_cache = snap(out.past_key_values)
        hidden = out.hidden_states[-1]            # [1, q_len, dim], on backbone device
        backbone_pred_0 = out.logits[0, 0, :].argmax().item()         # true token at P+1
        backbone_preds = out.logits[0, 1:, :].argmax(dim=-1).tolist()  # prediction after each node

        # DEBUG — does the CACHE produce the same next token as a fresh cacheless pass?
        # backbone_pred_0 comes from pending attending to the incoming cache. If that disagrees
        # with a clean forward over the true prefix (generated ++ pending), the cache is corrupt.
        # The first round this fires is the first bad cache → pins the bug to the prior round's
        # cache construction.
        if debug:
            with torch.no_grad():
                true_prefix = torch.cat([generated, torch.tensor([[pending]], device=device)], dim=1)
                fresh = medusa.backbone(input_ids=true_prefix, use_cache=False)
            fresh_pred = fresh.logits[0, -1, :].argmax().item()
            if fresh_pred != backbone_pred_0 or cache_len_in != context_len:
                dec = lambda t: tokenizer.decode([t])
                print(f"\n*** CACHE WRONG at round {rounds + 1} ***")
                print(f"  incoming cache length: {cache_len_in}   generated length: {context_len}"
                      f"   {'(MISMATCH!)' if cache_len_in != context_len else '(ok)'}")
                print(f"  true prefix length (generated + pending): {true_prefix.shape[1]}")
                print(f"  pending (prepended at pos {context_len}): {pending:>6} {dec(pending)!r}")
                print(f"  backbone_pred_0 (FROM CACHE):   {backbone_pred_0:>6} {dec(backbone_pred_0)!r}")
                print(f"  fresh recompute (NO CACHE):     {fresh_pred:>6} {dec(fresh_pred)!r}")
                print(f"  last 6 committed tokens: {generated[0, -6:].tolist()} -> {tokenizer.decode(generated[0, -6:])!r}")
                return tokenizer.decode(generated[0], skip_special_tokens=True)

        # FIND BEST PATH
        leaf_start = num_nodes - width**tree_depth

        if accept_mode == "greedy":
            # STRICT GREEDY (default, verified): emit the backbone's argmax; accept a tree branch
            # only while each candidate exactly equals that argmax. Output == plain greedy decode.
            best_accepted = []
            best_path_nodes = []
            for leaf in range(leaf_start, num_nodes):
                path = []
                node = leaf
                while node != -1:
                    path.append(node)
                    node = parent_indices[node]
                path.reverse()

                accepted = [backbone_pred_0]
                path_nodes = []
                if backbone_pred_0 == tokens[path[0]]:
                    path_nodes.append(path[0])
                    for j in range(len(path)):
                        if j + 1 < len(path):
                            accepted.append(backbone_preds[path[j]])
                            if backbone_preds[path[j]] != tokens[path[j + 1]]:
                                break
                            path_nodes.append(path[j + 1])
                        else:
                            accepted.append(backbone_preds[path[j]])  # bonus token

                if len(accepted) > len(best_accepted):
                    best_accepted = accepted
                    best_path_nodes = path_nodes

            if not best_accepted:
                best_accepted = [backbone_pred_0]
                best_path_nodes = []

        else:
            # TYPICAL ACCEPTANCE (Medusa paper): emit the CANDIDATE token (not the argmax) when the
            # backbone considers it "typical" — p(candidate) > min(epsilon, delta * exp(-H(p))).
            # When the model is unsure (high entropy H) the threshold drops, so more candidates pass
            # -> a deeper/wider tree finally pays off. NOT lossless: this is temperature sampling.
            # row 0 of out.logits is the distribution after `pending`; row 1+n is after tree node n.
            logits_all = out.logits[0].float() / temperature       # [q_len, vocab]
            probs = torch.softmax(logits_all, dim=-1)
            H = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)      # entropy per row
            thresh = torch.minimum(torch.full_like(H, epsilon), delta * torch.exp(-H))  # [q_len]

            best_cands = []      # accepted candidate tokens (no bonus yet)
            best_path_nodes = []
            for leaf in range(leaf_start, num_nodes):
                path = []
                node = leaf
                while node != -1:
                    path.append(node)
                    node = parent_indices[node]
                path.reverse()

                cands = []
                path_nodes = []
                row = 0  # start from the distribution after `pending`
                for node in path:
                    cand = tokens[node]
                    if probs[row, cand].item() > thresh[row].item():
                        cands.append(cand)
                        path_nodes.append(node)
                        row = 1 + node  # next candidate is judged by the distribution after this node
                    else:
                        break
                if len(cands) > len(best_cands):
                    best_cands = cands
                    best_path_nodes = path_nodes

            # bonus: SAMPLE from the distribution after the last accepted node (or after pending).
            # always produced, so we make progress even if no candidate was typical (m==1).
            last_row = 1 + best_path_nodes[-1] if best_path_nodes else 0
            bonus = torch.multinomial(probs[last_row], 1).item()
            best_accepted = best_cands + [bonus]

        # COMMIT: the old pending (guaranteed) plus every tree token EXCEPT the new bonus.
        # the new bonus becomes next round's pending (KV deliberately left out of the cache).
        new_bonus = best_accepted[-1]
        committed_this_round = [pending] + best_accepted[:-1]
        new_tensor = torch.tensor([committed_this_round], device=device, dtype=torch.long)
        generated = torch.cat([generated, new_tensor], dim=1)

        # DEBUG — compare cumulative output against the greedy reference; on the first token that
        # disagrees, dump this round's full state and stop. Because each round only appends, the
        # first mismatch is always introduced in the round that produced it, so this round's state
        # is the relevant one. Off by default; ref_ids is the greedy continuation (prompt removed).
        if debug and ref_ids is not None:
            gen_new = generated[0, start_len:].tolist()
            for i in range(min(len(gen_new), len(ref_ids))):
                if gen_new[i] != ref_ids[i]:
                    dec = lambda t: tokenizer.decode([t])
                    print(f"\n*** FIRST DIVERGENCE at generated position {i} (round {rounds + 1}) ***")
                    print(f"  expected (greedy): {ref_ids[i]:>6} {dec(ref_ids[i])!r}")
                    print(f"  fused produced:    {gen_new[i]:>6} {dec(gen_new[i])!r}")
                    print(f"  --- this round's state ---")
                    print(f"  context_len (committed before round): {context_len}")
                    print(f"  pending_in (token at position {context_len}): {pending_in:>6} {dec(pending_in)!r}")
                    print(f"  backbone_pred_0 (true token at {context_len + 1}): {backbone_pred_0:>6} {dec(backbone_pred_0)!r}")
                    print(f"  tree depth-1 candidates (head 1 top-{width}): {tokens[:width]} -> {[dec(t) for t in tokens[:width]]}")
                    print(f"  best_accepted (true tokens kept): {best_accepted}")
                    print(f"  best_path_nodes (matched tree nodes): {best_path_nodes}")
                    print(f"  committed_this_round: {committed_this_round} -> {tokenizer.decode(committed_this_round)!r}")
                    print(f"  new pending (bonus -> next round): {new_bonus:>6} {dec(new_bonus)!r}")

                    # Confirm/refute the prefill-vs-decode hypothesis: compute the token after
                    # pending_in three ways and show top-2 logits + gap.
                    #   parallel    = pending inside the fused multi-query pass (what we use)
                    #   incremental = pending as a lone token vs the cache (decode mode == HF greedy)
                    def top2(logits):
                        v, t = torch.topk(logits.float(), 2)
                        return [(dec(t[j].item()), round(v[j].item(), 4)) for j in range(2)], round((v[0] - v[1]).item(), 4)
                    par_top2, par_gap = top2(out.logits[0, 0, :])
                    with torch.no_grad():
                        inc = medusa.backbone(input_ids=torch.tensor([[pending_in]], device=device),
                                              past_key_values=snap(cache), use_cache=True)
                    inc_top2, inc_gap = top2(inc.logits[0, -1, :])
                    print(f"  --- prediction after pending, two ways ---")
                    print(f"  PARALLEL (fused pass)    top2={par_top2}  gap={par_gap}")
                    print(f"  INCREMENTAL (decode)     top2={inc_top2}  gap={inc_gap}")
                    return tokenizer.decode(generated[0], skip_special_tokens=True)

        # NEW CACHE = context + prepend + matched nodes (excludes the new bonus).
        kv_idx = list(range(context_len)) + [context_len] + [context_len + 1 + n for n in best_path_nodes]
        idx = torch.tensor(kv_idx, device=device)
        legacy = out_cache.to_legacy_cache()
        trimmed = tuple((kv[0][:, :, idx, :], kv[1][:, :, idx, :]) for kv in legacy)
        cache = DynamicCache.from_legacy_cache(trimmed)

        # NEW SEED_H = hidden state at the last matched node (or the prepend if nothing matched).
        # this is what the heads read next round — no PROPOSE pass needed.
        if best_path_nodes:
            seed_h = hidden[0, 1 + best_path_nodes[-1], :]
        else:
            seed_h = hidden[0, 0, :]
        seed_h = seed_h.to(device=head_device, dtype=head_dtype)
        pending = new_bonus

        rounds += 1
        new_count = len(committed_this_round)
        total_new += new_count
        if verbose:
            print(f"  round {rounds}: {new_count} new tokens")

    # the final pending is committed but was held out of `generated` — append it now.
    generated = torch.cat([generated, torch.tensor([[pending]], device=device)], dim=1)

    if verbose:
        print(f"  avg new tokens/round: {total_new / rounds:.2f} over {rounds} rounds")
    # return_ids lets a verifier check the raw token sequence (decode->retokenize isn't lossless).
    if return_ids:
        return generated[0]
    return tokenizer.decode(generated[0], skip_special_tokens=True)
