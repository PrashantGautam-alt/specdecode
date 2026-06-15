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

        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(MedusaHead(hidden_dim, vocab_size))

        nn.init.zeros_(some_tensor)
        