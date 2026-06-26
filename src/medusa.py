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
        h = propose_out.hidden_states[-1].to(medusa.heads[0].W1.weight.dtype)
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

def medusa_decode_tree(medusa, tokenizer, prompt, max_new_tokens=100, K=4, width=2, verbose=False):
    device = next(medusa.backbone.parameters()).device

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

        # PHASE 2 — BUILD TREE
        tokens, parent_indices = build_tree_candidates(head_logits=head_logits, width=width)
        tree_mask = build_tree_mask(parent_indices=parent_indices)

        # PHASE 3 — TREE VERIFY
        # transformers _update_causal_mask in 4.44+ overwrites any 4D custom mask we pass
        # when past_key_values is active, replacing it with a [1,1,q,past_len] default.
        # fix: run a full-sequence forward (context + tree nodes) with no past_key_values.
        # transformers handles full-sequence passes predictably and uses our 4D mask as-is.
        num_nodes = len(tokens)
        context_len = generated.shape[1]
        node_tensor = torch.tensor([tokens], device=device, dtype=torch.long)
        full_input = torch.cat([generated, node_tensor], dim=1)  # [1, context_len + num_nodes]
        total_len = context_len + num_nodes

        # 4D additive mask [1, 1, total_len, total_len]: 0.0 = attend, -inf = block
        attn_mask = torch.full(
            (1, 1, total_len, total_len), float('-inf'), dtype=torch.float16, device=device
        )
        # context rows: standard lower-triangular causal
        ctx_causal = torch.tril(torch.ones(context_len, context_len, dtype=torch.bool, device=device))
        attn_mask[0, 0, :context_len, :context_len] = torch.where(
            ctx_causal,
            torch.zeros_like(ctx_causal, dtype=torch.float16),
            torch.full_like(ctx_causal, float('-inf'), dtype=torch.float16),
        )
        # tree rows: all tree nodes see all context
        attn_mask[0, 0, context_len:, :context_len] = 0.0
        # tree rows: tree-to-tree follows ancestor mask
        tree_allow = tree_mask.to(device)
        attn_mask[0, 0, context_len:, context_len:] = torch.where(
            tree_allow,
            torch.zeros_like(tree_allow, dtype=torch.float16),
            torch.full_like(tree_allow, float('-inf'), dtype=torch.float16),
        )

        with torch.no_grad():
            verify_out = medusa.backbone(
                input_ids=full_input,
                attention_mask=attn_mask,
                use_cache=False,
            )
        verify_logits = verify_out.logits[0, context_len:, :]  # [num_nodes, vocab_size]
        backbone_preds = verify_logits.argmax(dim=-1).tolist()  # [num_nodes]

        # PHASE 4 — FIND BEST PATH
        # leaves are the last width^K nodes
        leaf_start = num_nodes - width**K
        best_accepted = []

        for leaf in range(leaf_start, num_nodes):
            # trace path from leaf back to root using parent_indices
            path = []
            node = leaf
            while node != -1:
                path.append(node)
                node = parent_indices[node]
            path.reverse()  # now [root_node, ..., leaf_node]

            # walk the path checking backbone agreement
            accepted = [backbone_pred_0]
            if backbone_pred_0 != tokens[path[0]]:
                pass  # mismatch at position 0 — keep just [backbone_pred_0]
            else:
                for j in range(len(path)):
                    if j + 1 < len(path):
                        accepted.append(backbone_preds[path[j]])
                        if backbone_preds[path[j]] != tokens[path[j+1]]:
                            break
                    else:
                        accepted.append(backbone_preds[path[j]])  # bonus token

            if len(accepted) > len(best_accepted):
                best_accepted = accepted

        if not best_accepted:
            best_accepted = [backbone_pred_0]

        # PHASE 5 — CACHE UPDATE + APPEND
        m = len(best_accepted)
        if m == 1:
            cache = propose_cache
        else:
            update_input = torch.tensor([best_accepted[:-1]], device=device, dtype=torch.long)
            with torch.no_grad():
                update_out = medusa.backbone(
                    input_ids=update_input,
                    past_key_values=propose_cache,
                    use_cache=True,
                )
            cache = snap(update_out.past_key_values)

        new_tensor = torch.tensor([best_accepted], device=device, dtype=torch.long)
        generated = torch.cat([generated, new_tensor], dim=1)
        rounds += 1
        total_accepted += m
        if verbose:
            print(f"  round {rounds}: accepted {m}/{K+1} tokens")

    if verbose:
        print(f"  avg tokens/round: {total_accepted / rounds:.2f} over {rounds} rounds")
    return tokenizer.decode(generated[0], skip_special_tokens=True)
    