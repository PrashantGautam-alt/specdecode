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

        # Step 3: run EVERY head on the SAME h, collect each head's logits
        head_logits = []
        for head in self.heads:
            head_logits.append(head(h))

        # Step 4: return the K predictions
        return head_logits
