import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from models import BaselineClassifier
from q4 import train_epochs, iterate_batches, evaluate
from data import load_imdb, load_imdb_synth, load_xor


class TransformerBlock(nn.Module):
    def __init__(self, emb, num_heads, ff_dim):
        super.__init__()
        self.attention = MultiHeadSelfAttention(emb, num_heads)

        self.layernorm1 = nn.LayerNorm(emb)
        self.layernorm2 = nn.LayerNorm(emb)

        self.ffn = nn.Sequential(nn.Linear(emb, ff_dim), 
                                 nn.ReLU(),
                                 nn.Linear(ff_dim, emb))
    
    def forward(self, x):
        y = self.layernorm1(x)
        y = self.attention(y)
        x += y

        y = self.layernorm2(x)
        y = self.ffn(y)
        x += y
        return x
        

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

    def __init__(self, vocab_size, emb=300, num_classes=2, num_heads=6, max_len=256):
        # We keep pool='first' for conceptual consistency, but
        # we’ll do the pooling manually in forward().
        super().__init__(vocab_size=vocab_size, emb_dim=emb, num_classes=num_classes, pool='first')

        assert emb % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.emb = emb
        self.num_heads = num_heads
        self.head_dim = emb // num_heads  # d_h = E / H

        # position embedding layer
        self.pos_embedding = nn.Embedding(max_len, emb)

        #tranformer block 
        ff_dim = 4 * emb
        self.block = TransformerBlock(emb, num_heads, ff_dim)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        x: (B, T) with token indices
        Returns: logits of shape (B, num_classes)
        """
        B, T = x.size()          # B = batch size, T = sequence length
        E = self.emb             # embedding size

        # 1) Embed tokens: (B, T) -> (B, T, E)
        x_emb = self.embedding(x)    # (batch, time, emb)

        # 2) Embed Position 
        pos_idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)  # (B, T)
        pos_emb = self.pos_embedding(pos_idx)  # (B, T, E)

        # 3) Add them together
        x_emb = x_emb + pos_emb

        #4) pass through transformer block
        x_emb = self.block(x_emb) 

        # 5) Select pooling: take the first token as sequence representation
        out = x_emb[:, 0, :]                     

        # 6) Classification layer from BaselineClassifier
        output = self.fc(out)                         
        return output
    
def grid_search_attention(train_data, val_data, vocab_size, num_classes, pad_idx, num_epochs):
    lrs = [1e-3, 5e-3, 1e-4]
    batch_sizes = [32, 64, 128]
    results = []

    for lr, batch_size in itertools.product(lrs, batch_sizes):
        print(f"\n=== Training with lr={lr}, batch={batch_size} ===")
        model = MultiHeadSelfAttentionClassifier(vocab_size, emb=300, num_classes=num_classes, num_heads=6)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_loss, train_acc = train_epochs(model, train_data, batch_size, pad_idx, optimizer, num_epochs=num_epochs)
        val_loss, val_acc = evaluate(model, val_data, batch_size, pad_idx)

        print(f'lr={lr}, batch={batch_size} | train_acc={train_acc:.3f}, val_acc={val_acc:.3f}')
        results.append((lr, batch_size, train_acc, val_acc))

    return results

(x_train_1, y_train_1), (x_val_1, y_val_1), (i2w_1, w2i_1), numcls_1 = load_imdb(final=False)
train_data1 = (x_train_1, y_train_1)
val_data1   = (x_val_1, y_val_1)

(x_train_2, y_train_2), (x_val_2, y_val_2), (i2w_2, w2i_2), numcls_2 = load_imdb_synth()
train_data2 = (x_train_2, y_train_2)
val_data2   = (x_val_2, y_val_2)

(x_train_3, y_train_3), (x_val_3, y_val_3), (i2w_3, w2i_3), numcls_3 = load_xor()
train_data3 = (x_train_3, y_train_3)
val_data3   = (x_val_3, y_val_3)

pad_idx1 = w2i_1['.pad']
pad_idx2 = w2i_2['.pad']
pad_idx3 = w2i_3['.pad']

# --- Run experiments with MultiHeadAttentionClassifier ---
results1 = grid_search_attention(train_data1, val_data1,
                                 vocab_size=len(i2w_1),
                                 num_classes=numcls_1,
                                 pad_idx=pad_idx1,
                                 num_epochs=20)

results2 = grid_search_attention(train_data2, val_data2,
                                 vocab_size=len(i2w_2),
                                 num_classes=numcls_2,
                                 pad_idx=pad_idx2,
                                 num_epochs=100)

results3 = grid_search_attention(train_data3, val_data3,
                                 vocab_size=len(i2w_3),
                                 num_classes=numcls_3,
                                 pad_idx=pad_idx3,
                                 num_epochs=100)