import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from models import BaselineClassifier
from q4 import train_epochs, evaluate
from data import load_xor


class TransformerBlock(nn.Module):
    def __init__(self, emb, num_heads, ff_dim):
        super().__init__()
        assert emb % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.emb = emb
        self.num_heads = num_heads
        self.head_dim = emb // num_heads

        # --- multi-head self-attention parameters (Q, K, V, out_proj) ---
        self.to_q = nn.Linear(emb, emb)
        self.to_k = nn.Linear(emb, emb)
        self.to_v = nn.Linear(emb, emb)
        self.out_proj = nn.Linear(emb, emb)

        # layer norms
        self.layernorm1 = nn.LayerNorm(emb)
        self.layernorm2 = nn.LayerNorm(emb)

        # feed forward
        self.ffn = nn.Sequential(
            nn.Linear(emb, ff_dim), nn.ReLU(), nn.Linear(ff_dim, emb)
        )

    def forward(self, x):
        B, T, E = x.size()
        H = self.num_heads
        D = self.head_dim

        y = self.layernorm1(x)

        # Q, K, V projections
        q = self.to_q(y)
        k = self.to_k(y)
        v = self.to_v(y)

        # Split into heads: (B, T, E) -> (B, H, T, D)
        q = q.view(B, T, H, D).permute(0, 2, 1, 3)  # (B, H, T, D)
        k = k.view(B, T, H, D).permute(0, 2, 1, 3)  # (B, H, T, D)
        v = v.view(B, T, H, D).permute(0, 2, 1, 3)  # (B, H, T, D)

        # Scaled dot-product attention: (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        attn = F.softmax(scores, dim=-1)  # (B, H, T, T)

        # Weighted sum over values: (B, H, T, D)
        z = torch.matmul(attn, v)

        # Merge heads back: (B, T, E)
        z = z.permute(0, 2, 1, 3).contiguous()  # (B, T, H, D)
        z = z.view(B, T, E)

        attention_out = self.out_proj(z)

        # residual connection
        x = x + attention_out

        # feed forward + layernorm 2 + residual
        y = self.layernorm2(x)
        y = self.ffn(y)
        x = x + y
        return x


class MultiHeadSelfAttentionClassifier(BaselineClassifier):
    """
    Baseline + Transformer block (multi-head self-attention + FFN) + select pooling.
    """

    def __init__(self, vocab_size, emb=300, num_classes=2, num_heads=6, max_len=256):
        super().__init__(
            vocab_size=vocab_size, emb_dim=emb, num_classes=num_classes, pool="first"
        )

        self.emb = emb
        self.num_heads = num_heads

        # position embedding layer
        self.pos_embedding = nn.Embedding(max_len, emb)

        # transoformer block
        ff_dim = 4 * emb
        self.block = TransformerBlock(emb, num_heads, ff_dim)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        x: (B, T) with token indices
        Returns: logits of shape (B, num_classes)
        """
        B, T = x.size()  # B = batch size, T = sequence length
        E = self.emb  # embedding size

        # 1) Embed tokens: (B, T) -> (B, T, E)
        x_emb = self.embedding(x)  # (batch, time, emb)

        # 2) Embed Position
        pos_idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # (B, T)
        pos_emb = self.pos_embedding(pos_idx)  # (B, T, E)

        # 3) Add them together
        x_emb = x_emb + pos_emb

        # 4) Transformer block (this is where the attention + FFN live)
        x_emb = self.block(x_emb)  # (B, T, E)

        # 5) Select pooling: take the first token as sequence representation
        out = x_emb[:, 0, :]  # (B, E)

        # 6) Classification layer from BaselineClassifier
        output = self.fc(out)
        return output


def grid_search_attention(
    train_data,
    val_data,
    vocab_size,
    num_classes,
    pad_idx,
    num_epochs,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    lrs = [1e-3, 5e-3, 1e-4]
    batch_sizes = [32, 64, 128]
    results = []
    with open("results_q8.txt", "a") as f:
        f.write(f"\n Dataset: XOR \n")

    for lr, batch_size in itertools.product(lrs, batch_sizes):
        print(f"\n=== Training with lr={lr}, batch={batch_size} ===")
        model = MultiHeadSelfAttentionClassifier(
            vocab_size, emb=300, num_classes=num_classes, num_heads=6
        )
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        _, train_acc = train_epochs(
            model, train_data, batch_size, pad_idx, optimizer, num_epochs=num_epochs, device=device
        )
        _, val_acc = evaluate(model, val_data, batch_size, pad_idx, device=device)

        with open("results_q8.txt", "a") as f:
            f.write("\n New model:  \n")
            f.write(
                f"\n lr={lr}, batch={batch_size} | train_acc={train_acc:.3f}, val_acc={val_acc:.3f}"
            )

        print(
            f"lr={lr}, batch={batch_size} | train_acc={train_acc:.3f}, val_acc={val_acc:.3f}"
        )
        results.append((lr, batch_size, train_acc, val_acc))

    return results


if __name__ == "__model__":

    (x_train_3, y_train_3), (x_val_3, y_val_3), (i2w_3, w2i_3), numcls_3 = load_xor()
    train_data3 = (x_train_3, y_train_3)
    val_data3 = (x_val_3, y_val_3)

    pad_idx3 = w2i_3[".pad"]

    results3 = grid_search_attention(
        train_data3,
        val_data3,
        vocab_size=len(i2w_3),
        num_classes=numcls_3,
        pad_idx=pad_idx3,
        num_epochs=100,
    )
