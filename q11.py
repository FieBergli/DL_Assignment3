import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlockCausal(nn.Module):
    def __init__(
        self, emb, num_heads, ff_dim, max_len, dropout=0.4, rot_emb: bool = False
    ):
        super().__init__()
        assert emb % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.emb = emb
        self.num_heads = num_heads
        self.head_dim = emb // num_heads
        self.rot_emb = rot_emb

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
            nn.Linear(emb, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb),
        )

        # dropout for residual connections
        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len), diagonal=1)
            .bool()
            .view(1, 1, max_len, max_len),
        )

    def apply_rotary_emb(self, xq: torch.Tensor, xk: torch.Tensor, T: int):
        device = xq.device
        # Generate RoPE embeddings dynamically based on T
        seq_pos = torch.arange(T, device=device)  # Shape: (T)
        freqs = torch.outer(seq_pos, self.inv_freq.to(device))  # Shape: (T, dim // 2)
        pos_emb = (
            torch.repeat_interleave(freqs, 2, -1).unsqueeze(0).unsqueeze(0)
        )  # Shape: (1, 1, T, dim)

        # Split pos into sin and cos components, repeating each to match xq and xk dimensions
        pos_sin = torch.sin(pos_emb)
        pos_cos = torch.cos(pos_emb)

        # Apply RoPE transformation: pair and rotate dimensions
        # Rotate query and key tensors
        xq_rot = (
            xq * pos_cos
            + torch.stack((-xq[..., 1::2], xq[..., ::2]), dim=-1).reshape_as(xq)
            * pos_sin
        )

        xk_rot = (
            xk * pos_cos
            + torch.stack((-xk[..., 1::2], xk[..., ::2]), dim=-1).reshape_as(xk)
            * pos_sin
        )
        return xq_rot, xk_rot

    def forward(self, x):
        B, T, E = x.size()
        H = self.num_heads
        D = self.head_dim

        y = self.layernorm1(x)

        # Q, K, V projections
        q = self.to_q(y)
        k = self.to_k(y)
        v = self.to_v(y)

        # Split into heads:
        q = q.view(B, T, H, D).permute(0, 2, 1, 3)  # (B, H, T, D)
        k = k.view(B, T, H, D).permute(0, 2, 1, 3)  # (B, H, T, D)
        v = v.view(B, T, H, D).permute(0, 2, 1, 3)  # (B, H, T, D)

        if self.rot_emb:
            q, k = self.apply_rotary_emb(q, k, T)

        # Scaled dot-product attention: (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)

        # causal masking

        mask = self.causal_mask[..., :T, :T]  # (T, T)
        scores = scores.masked_fill(mask, float("-inf"))

        attention = F.softmax(scores, dim=-1)  # (B, H, T, T)

        # Weighted sum over values: (B, H, T, D)
        z = torch.matmul(attention, v)

        # Merge heads back: (B, T, E)
        z = z.permute(0, 2, 1, 3).contiguous()  # (B, T, H, D)
        z = z.view(B, T, E)

        attention_out = self.out_proj(z)

        # residual connection and dropout
        x = x + self.dropout_attention(attention_out)

        # feed forward + layernorm 2 + residual + dropout
        y = self.layernorm2(x)
        y = self.ffn(y)
        x = x + self.dropout_ff(y)
        return x


class AutoRegressiveTransformer(nn.Module):
    """
    Predicts the next token for every position in the input sequence.
    """

    def __init__(
        self,
        vocab_size,
        emb=300,
        num_heads=6,
        max_len=256,
        num_layers=6,
        rot_emb: bool = False,
    ):
        super().__init__()

        self.emb = emb
        self.vocab_size = vocab_size
        self.max_len = max_len

        # token + position embeddings
        self.token_embedding = nn.Embedding(vocab_size, emb)
        self.pos_embedding = nn.Embedding(max_len, emb)

        self.rot_emb = rot_emb

        ff_dim = 4 * emb
        self.blocks = nn.ModuleList(
            [
                TransformerBlockCausal(emb, num_heads, ff_dim, max_len, rot_emb=rot_emb)
                for _ in range(num_layers)
            ]
        )

        self.final_ln = nn.LayerNorm(emb)
        self.fc_out = nn.Linear(emb, vocab_size)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        x: (B, T) integer token indices
        returns: logits of shape (B, T, vocab_size)
        """
        B, T = x.size()
        assert T <= self.max_len, "sequence length exceeds max_len"

        # 1) token + position embeddings
        tok_emb = self.token_embedding(x)  # (B, T, E)

        if not self.rot_emb:
            pos_idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            pos_emb = self.pos_embedding(pos_idx)  # (B, T, E)
            tok_emb = tok_emb + pos_emb  # (B, T, E)

        h = tok_emb
        # 2) transformer blocks with *causal masking*
        for block in self.blocks:
            h = block(h)

        # 3) final layer norm and projection to logits
        h = self.final_ln(h)  # (B, T, E)
        output = self.fc_out(h)  # (B, T, vocab_size)

        return output
