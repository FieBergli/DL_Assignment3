import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from models import BaselineClassifier
from q4 import train_epochs, iterate_batches, evaluate
from data import load_imdb, load_imdb_synth, load_xor

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
        super().__init__(vocab_size=vocab_size, emb_dim=emb, num_classes=num_classes, pool='first')

        assert emb % num_heads == 0, "Embedding dim must be divisible by num_heads"

        self.emb = emb
        self.num_heads = num_heads
        self.head_dim = emb // num_heads 

        self.to_q = nn.Linear(emb, emb)
        self.to_k = nn.Linear(emb, emb)
        self.to_v = nn.Linear(emb, emb)

        self.out_proj = nn.Linear(emb, emb)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        x: (B, T) with token indices
        Returns: logits of shape (B, num_classes)
        """
        B, T = x.size()          
        E = self.emb             
        H = self.num_heads       
        D = self.head_dim        
       
        x_emb = self.embedding(x)   

        #Linear projections to Q, K, V
        q = self.to_q(x_emb)         
        k = self.to_k(x_emb)         
        v = self.to_v(x_emb)         

        # 3) Multi-head: split E into (H, D) and move H out as its own dimension
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
        scores = torch.matmul(q, k.transpose(-2, -1))  

        #scale by sqrt(D) ---
        scores = scores / math.sqrt(D)

        attention = F.softmax(scores, dim=-1)              

        # Use attention weights to mix values:
        # attn: (B, H, T, T)
        # v:    (B, H, T, D)
        # Result: (B, H, T, D)
        z = torch.matmul(attention, v)                     

        # 5) Merge heads back: (B, H, T, D) -> (B, T, H, D) -> (B, T, E)
        z = z.permute(0, 2, 1, 3).contiguous()       
        z = z.view(B, T, E)                          

        # Final linear layer to mix information across heads
        attended = self.out_proj(z)                  

        # 6) Select pooling
        out = attended[:, 0, :]                     

        # 7) Classification layer from BaselineClassifier
        output = self.fc(out)                         
        return output
    
def grid_search_attention(train_data, val_data, vocab_size, num_classes, pad_idx, num_epochs, dataset_name:str, device= "cuda" if torch.cuda.is_available() else "cpu"):
    lrs = [1e-3, 5e-3, 1e-4]
    batch_sizes = [32, 64, 128]
    with open("results_q7.txt", "w") as f:
        f.write(f"\n Dataset: {dataset_name} \n")

    for lr, batch_size in itertools.product(lrs, batch_sizes):
        print(f"\n=== Training with lr={lr}, batch={batch_size} ===")
        model = MultiHeadSelfAttentionClassifier(vocab_size, emb=300, num_classes=num_classes, num_heads=6)
        model = model.to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        _, train_acc = train_epochs(model, train_data, batch_size, pad_idx, optimizer, num_epochs=num_epochs, device=device)
        _, val_acc = evaluate(model, val_data, batch_size, pad_idx, device=device)

        with open("results_q7.txt", "a") as f:
            f.write("\n New model:  \n")
            f.write(f"\n lr={lr}, batch={batch_size} | train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
        print(f'lr={lr}, batch={batch_size} | train_acc={train_acc:.3f}, val_acc={val_acc:.3f}')


if __name__ == "__main__":

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

    results1 = grid_search_attention(train_data1, val_data1,
                                    vocab_size=len(i2w_1),
                                    num_classes=numcls_1,
                                    pad_idx=pad_idx1,
                                    num_epochs=100, 
                                    dataset_name="IMDb")

"""
    results2 = grid_search_attention(train_data2, val_data2,
                                    vocab_size=len(i2w_2),
                                    num_classes=numcls_2,
                                    pad_idx=pad_idx2,
                                    num_epochs=100, 
                                    dataset_name="Synthetic IMDb")

    results3 = grid_search_attention(train_data3, val_data3,
                                    vocab_size=len(i2w_3),
                                    num_classes=numcls_3,
                                    pad_idx=pad_idx3,
                                    num_epochs=100, 
                                    dataset_name="XOR")
                                    """