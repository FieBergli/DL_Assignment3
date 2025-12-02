from q1 import pad_batch
import torch
import torch.nn as nn


class BaselineClassifier(nn.Module):
    """
    Baseline sequence classifier:
    - Embedding layer (vocab_size -> emb)
    - Global pooling over time (mean / max / first)
    - Linear projection to num_classes
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 300,
        num_classes: int = 2,
        pool: str = "mean",
    ):
        """
        pool: 'mean', 'max', or 'first'
        """
        super().__init__()
        self.emb = emb_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes)
        self.pool = pool

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # x: (batch, time), dtype=torch.long
        # 1) embed -> (batch, time, emb)
        x_emb = self.embedding(x)  # shape: (B, T, E)

        # 2) global pooling -> (batch, emb)
        if self.pool == "mean":
            # mean over time dimension, ignoring padding is more correct but here
            # we assume padding indices have embeddings that don't bias too much.
            out = x_emb.mean(dim=1)
        elif self.pool == "max":
            # max over time dimension
            out, _ = x_emb.max(dim=1)
        else:  # first
            # take first token embedding (e.g., x_emb[:, 0, :])
            out = x_emb[:, 0, :]

        # 3) linear projection -> (batch, num_classes)
        output = self.fc(out)
        return output


class SimpleSelfAttentionClassifier(BaselineClassifier):
    """Baseline with one simple self-attention layer (single head, no q/k/v projections)"""
    def __init__(self, vocab_size, emb=300, num_classes=2, pool='max'):
        super().__init__(vocab_size, emb, num_classes, pool)
        # no extra parameters for simple attention; we just do operations in forward

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # x: (B, T)
        x_emb = self.embedding(x) # (B, T, E)

        # compute raw attention logits: (B, T, E) @ (B, E, T) -> (B, T, T)
        # We use torch.matmul which broadcasts batch dims properly.
        attention_output = torch.matmul(x_emb, x_emb.transpose(1, 2)) # (B, T, T)

        # optional: we might want to mask padding positions here if we have them

        # attention distribution over time dimension (softmax over last dim)
        attention_weights = F.softmax(attention_output, dim=-1) # (B, T, T)

        # weighted sum over values (here values == x_emb): (B, T, T) @ (B, T, E) -> (B, T, E)
        attended = torch.matmul(attention_weights, x_emb) # (B, T, E)

        # now global max pooling over time
        out, _ = attended.max(dim=1) # (B, E)
        output = self.fc(out) # (B, num_classes)
        return output