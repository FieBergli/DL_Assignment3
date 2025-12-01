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
