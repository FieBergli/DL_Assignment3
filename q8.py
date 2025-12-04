from torch import nn
import torch
from models import BaselineClassifier
import math
import torch.nn.functional as F

class MultiHeadSelfAttentionClassifier(BaselineClassifier):
    """
    Baseline + full multi-head self-attention layer.
    Steps:
    1. Embedding: x -> x_emb
    2. Q, K, V linear projections
    3. Split into heads, compute scaled dot-product attention per head
    4. Merge heads back, final linear projection
    5. Select pooling (first token) + final classification layer
    """
    def __init__(self, vocab_size, emb=300, num_classes=2, num_heads=6, max_len=256):
        super().__init__(vocab_size=vocab_size, emb_dim=emb, num_classes=num_classes, pool='first')
        assert emb % num_heads == 0, "Embedding dim must be divisible by num_heads"
        self.emb = emb
        self.num_heads = num_heads
        self.head_dim = emb // num_heads
        # position embedding layer
        self.pos_embedding = nn.Embedding(max_len, emb)
        #Q, K, V linear layers
        self.to_q = nn.Linear(emb, emb)
        self.to_k = nn.Linear(emb, emb)
        self.to_v = nn.Linear(emb, emb)
        # final linear layer after combining all heads
        self.out_proj = nn.Linear(emb, emb)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        B, T = x.size() # B = batch size, T = sequence length
        E = self.emb # embedding size
        H = self.num_heads # number of heads
        D = self.head_dim # per-head dim
        # embed tokens
        x_emb = self.embedding(x)
        # embed position
        pos_idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embedding(pos_idx)
        x_emb = x_emb + pos_emb
        # linear projections to Q, K, V
        q = self.to_q(x_emb) # queries
        k = self.to_k(x_emb) # keys
        v = self.to_v(x_emb) # values
        # multi-head
        q = q.view(B, T, H, D).permute(0, 2, 1, 3)
        k = k.view(B, T, H, D).permute(0, 2, 1, 3)
        v = v.view(B, T, H, D).permute(0, 2, 1, 3)
        # scaled dot-product attention (per head)
        scores = torch.matmul(q, k.transpose(-2, -1))
        # scale by sqrt(D)
        scores = scores / math.sqrt(D)
        # softmax over last dim
        attention = F.softmax(scores, dim=-1)
        # use attention weights to mix values
        7
        z = torch.matmul(attention, v)
        # merge heads back
        z = z.permute(0, 2, 1, 3).contiguous()
        z = z.view(B, T, E)
        # final linear layer to mix information across heads
        attended = self.out_proj(z)
        # first pooling
        out = attended[:, 0, :]
        # classification layer from BaselineClassifier
        output = self.fc(out)
        return output