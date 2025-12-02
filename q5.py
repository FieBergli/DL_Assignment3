import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import BaselineClassifier

class MultiHeadSelfAttentionClassifier(BaselineClassifier):
    """
    Baseline + full multi-head self-attention layer.
    
    Steps:
    1. Embedding: x -> x_emb (B, T, E)
    2. Q, K, V linear projections
    3. Split into heads, compute scaled dot-product attention per head
    4. Merge heads back, final linear projection
    5. Select pooling (first token) + final classification layer
    """

    def __init__(self, vocab_size, emb=300, num_classes=2, num_heads=6):
        # We keep pool='first' for conceptual consistency, but
        # we’ll do the pooling manually in forward().
        super().__init__(vocab_size=vocab_size, emb_dim=emb, num_classes=num_classes, pool='first')

        assert emb % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.emb = emb
        self.num_heads = num_heads
        self.head_dim = emb // num_heads  # d_h = E / H

        # --- Point 2: Q, K, V linear layers (emb -> emb) ---
        self.to_q = nn.Linear(emb, emb)
        self.to_k = nn.Linear(emb, emb)
        self.to_v = nn.Linear(emb, emb)

        # Final linear layer after combining all heads
        self.out_proj = nn.Linear(emb, emb)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        x: (B, T) with token indices
        Returns: logits of shape (B, num_classes)
        """
        B, T = x.size()          # B = batch size, T = sequence length
        E = self.emb             # embedding size
        H = self.num_heads       # number of heads
        D = self.head_dim        # per-head dim, so E = H * D

        # 1) Embed tokens: (B, T) -> (B, T, E)
        x_emb = self.embedding(x)    # (batch, time, emb)

        # 2) Linear projections to Q, K, V
        # All still shape (B, T, E)
        q = self.to_q(x_emb)         # queries
        k = self.to_k(x_emb)         # keys
        v = self.to_v(x_emb)         # values

        # 3) Multi-head: split E into (H, D) and move H out as its own dimension
        #
        # First: (B, T, E) -> (B, T, H, D) by reshaping
        # Then:  (B, T, H, D) -> (B, H, T, D) by permuting dims
        #
        # Why? We want heads as a separate dimension so we can do attention
        # independently per head: (B, H, T, D).
        q = q.view(B, T, H, D).permute(0, 2, 1, 3)  # (B, H, T, D)
        k = k.view(B, T, H, D).permute(0, 2, 1, 3)  # (B, H, T, D)
        v = v.view(B, T, H, D).permute(0, 2, 1, 3)  # (B, H, T, D)

        # 4) Scaled dot-product attention (per head)
        #
        # We want scores for all query-key pairs:
        #   scores[b, h, t_q, t_k] = <q[b, h, t_q, :], k[b, h, t_k, :]>
        #
        # q: (B, H, T, D)
        # k.transpose(-2, -1): (B, H, D, T)
        # matmul over last two dims -> (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)

        # --- Point 1: scale by sqrt(D) ---
        scores = scores / math.sqrt(D)

        # Softmax over last dim: for each (b, h, t_q) we get a distribution over t_k
        attn = F.softmax(scores, dim=-1)              # (B, H, T, T)

        # Use attention weights to mix values:
        # attn: (B, H, T, T)
        # v:    (B, H, T, D)
        # Result: (B, H, T, D)
        z = torch.matmul(attn, v)                     # (B, H, T, D)

        # 5) Merge heads back: (B, H, T, D) -> (B, T, H, D) -> (B, T, E)
        z = z.permute(0, 2, 1, 3).contiguous()        # (B, T, H, D)
        z = z.view(B, T, E)                           # (B, T, E)

        # Final linear layer to mix information across heads
        attended = self.out_proj(z)                   # (B, T, E)

        # 6) Select pooling: take the first token as sequence representation
        # This is the "select pooling" / "first token pooling" the assignment talks about.
        out = attended[:, 0, :]                       # (B, E)

        # 7) Classification layer from BaselineClassifier
        logits = self.fc(out)                         # (B, num_classes)
        return logits
