import torch
import torch.nn as nn

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
        h = h.to(self.heads[0].W1.weight.dtype)

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
    generated = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    start_len = generated.shape[1]  # where the prompt ends; we stop max_new_tokens past it
    total_accepted = 0
    rounds = 0

    while generated.shape[1] < start_len + max_new_tokens:
        # PHASE 1 — PROPOSE: each head's argmax at the LAST position is its guess for the
        # next K positions. This is the "draft" — but it rides the backbone's own forward
        # pass (inside medusa()), so there is no separate draft model and no 0.40*K tax.
        with torch.no_grad():
            head_logits = medusa(generated)
        candidates = [head_logits[k][0, -1, :].argmax().item() for k in range(K)]

        # PHASE 2 — VERIFY: feed [context + the first K-1 candidates] through the backbone
        # once. We append only K-1: the backbone's logits at the last real token already
        # predict candidate 0, and each appended candidate lets it predict the next one.
        # The backbone's argmax at the final K positions = what the big model itself would
        # say at those positions, which is our ground truth.
        cand_tensor = torch.tensor([candidates[:-1]], device=device, dtype=torch.long)
        verify_input = torch.cat([generated, cand_tensor], dim=1)
        with torch.no_grad():
            verify_logits = medusa.backbone(input_ids=verify_input).logits
        backbone_preds = verify_logits[0, -K:, :].argmax(dim=-1)  # [K] true next tokens

        # PHASE 3 — ACCEPT: walk left to right. We always emit the BACKBONE's token, never
        # the candidate — that is why the output stays byte-for-byte identical to plain
        # greedy decoding of the backbone (Medusa changes the SPEED, not the result). A
        # matching candidate means the head guessed right, so we continue; the first
        # mismatch is where the heads were wrong, so we emit the backbone's correction and
        # stop — every later candidate was built on a now-wrong prefix and is worthless.
        new_tokens = []
        for i in range(K):
            new_tokens.append(backbone_preds[i].item())
            if backbone_preds[i].item() != candidates[i]:
                break

        # PHASE 4 — APPEND the accepted tokens (plus the correction) and loop.
        new_tensor = torch.tensor([new_tokens], device=device, dtype=torch.long)
        generated = torch.cat([generated, new_tensor], dim=1)
        rounds += 1
        total_accepted += len(new_tokens)
        if verbose:
            print(f"  round {rounds}: accepted {len(new_tokens)}/{K} tokens")

    if verbose:
        print(f"  avg tokens/round: {total_accepted / rounds:.2f} over {rounds} rounds")
    return tokenizer.decode(generated[0], skip_special_tokens=True)