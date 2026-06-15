import torch.nn as nn

class MedusaHead(nn.Module):
    def __init__(self, hidden_dim, vocab_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, vocab_size)
        self.act = nn.SiLU()

    def forward(self, h):
        return self.W2(self.act(self.W1(h)) + h)
