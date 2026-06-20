import torch
import torch.nn as nn
from transformers import DynamicCache


class MedusaHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, vocab_size)
        self.act = nn.SiLU()

    def forward(self, h):
        return self.W2(self.act(self.W1(h)) + h)


class MedusaModel(nn.Module):   
    def __init__(self, backbone, num_heads):
        super().__init__()

        # store the backbone
        self.backbone = backbone
        # freeze it
        for param in self.backbone.parameters():
            param.requires_grad = False

        hidden_dim = self.backbone.config.hidden_size   # 4096 for the 8B, 2048 for the 1B
        vocab_size = self.backbone.config.vocab_size     # 128256

        self.heads = nn.ModuleList(
            [
                MedusaHead(hidden_dim, vocab_size)
                for _ in range(num_heads)
            ]
        )

        lm_head = backbone.lm_head

        for head in self.heads:
            # Half 1: zero W1 so its SiLU branch collapses to nothing.
            # W1(h) = 0  ->  SiLU(0) = 0, leaving just the residual h in the bracket.
            # The bias must go too, or W1(h) = bias != 0 and the branch survives.
            head.W1.weight.data.zero_()
            if head.W1.bias is not None:
                head.W1.bias.data.zero_()
            # Half 2: the bracket is now exactly h, so W2 alone must reproduce the
            # LM head. Copy its weight; zero our bias since the LM head carries none.
            head.W2.weight.data.copy_(lm_head.weight.data)
            if head.W2.bias is not None:
                head.W2.bias.data.zero_()
        
    def forward(self, input_ids, attention_mask=None):
        # Step 1: one backbone pass, asking it to also return hidden states
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,)
        

        # Step 2: pull out the final-layer hidden state h
        h = outputs.hidden_states[-1]
        # Backbone runs in float16, but the heads may be float32 (training needs the
        # wider range to avoid NaNs). Cast h to the heads' dtype so head(h) agrees.
        h = h.to(device=self.heads[0].W1.weight.device, dtype=self.heads[0].W1.weight.dtype)

        # Step 3: run EVERY head on the SAME h, collect each head's logits
        head_logits = []
        for head in self.heads:
            head_logits.append(head(h))

        # Step 4: return the K predictions
        return head_logits

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
        # to propose K candidates. backbone_preds_0 is what the backbone itself would
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
        backbone_preds_0 = propose_out.logits[0, -1, :].argmax().item()
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
        backbone_preds = [backbone_preds_0] + verify_preds

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
