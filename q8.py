import itertools
from torch import nn
import torch
from data import load_xor
from models import BaselineClassifier
import math
import torch.nn.functional as F
from q4 import evaluate, train_epochs

class MultiHeadSelfAttentionClassifier(BaselineClassifier):
    """
    Baseline + full multi-head self-attention layer.
    """
    def __init__(self, vocab_size, emb=300, num_classes=2, num_heads=6, max_len=256):
        super().__init__(vocab_size=vocab_size, emb_dim=emb, num_classes=num_classes, pool='first')
        assert emb % num_heads == 0, "Embedding dim must be divisible by num_heads"
        self.emb = emb
        self.num_heads = num_heads
        self.head_dim = emb // num_heads
        # position embedding layer
        self.pos_embedding = nn.Embedding(max_len, emb)
      
        self.to_q = nn.Linear(emb, emb)
        self.to_k = nn.Linear(emb, emb)
        self.to_v = nn.Linear(emb, emb)
        
        self.out_proj = nn.Linear(emb, emb)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        B, T = x.size() 
        E = self.emb 
        H = self.num_heads 
        D = self.head_dim 
        
        x_emb = self.embedding(x)
        
        pos_idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embedding(pos_idx)
        x_emb = x_emb + pos_emb
        
        q = self.to_q(x_emb)
        k = self.to_k(x_emb) 
        v = self.to_v(x_emb) 
      
        q = q.view(B, T, H, D).permute(0, 2, 1, 3)
        k = k.view(B, T, H, D).permute(0, 2, 1, 3)
        v = v.view(B, T, H, D).permute(0, 2, 1, 3)
        
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(D)
        attention = F.softmax(scores, dim=-1)
        z = torch.matmul(attention, v)
        z = z.permute(0, 2, 1, 3).contiguous()
        z = z.view(B, T, E)
        attended = self.out_proj(z)
        out = attended[:, 0, :]
        output = self.fc(out)
        return output

def grid_search_attention(train_data, val_data, vocab_size, num_classes, pad_idx, num_epochs, dataset_name:str, device= "cuda" if torch.cuda.is_available() else "cpu"):
    lrs = [1e-3, 5e-3, 1e-4]
    batch_sizes = [32, 64, 128]

    with open("results_q8.txt", "w") as f:
        f.write(f"\n Dataset: {dataset_name} \n")

    for lr, batch_size in itertools.product(lrs, batch_sizes):
        print(f"\n=== Training with lr={lr}, batch={batch_size} ===")
        model = MultiHeadSelfAttentionClassifier(vocab_size, emb=300, num_classes=num_classes, num_heads=6)
        model = model.to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        _, train_acc = train_epochs(model, train_data, batch_size, pad_idx, optimizer, num_epochs=num_epochs, device=device)
        _, val_acc = evaluate(model, val_data, batch_size, pad_idx, device=device)

        with open("results_q8.txt", "a") as f:
            f.write("\n New model:  \n")
            f.write(f"\n lr={lr}, batch={batch_size} | train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")
        print(f'lr={lr}, batch={batch_size} | train_acc={train_acc:.3f}, val_acc={val_acc:.3f}')


if __name__ == "__main__":

    (x_train_3, y_train_3), (x_val_3, y_val_3), (i2w_3, w2i_3), numcls_3 = load_xor()
    train_data3 = (x_train_3, y_train_3)
    val_data3   = (x_val_3, y_val_3)

    pad_idx3 = w2i_3['.pad']

    results3 = grid_search_attention(train_data3, val_data3,
                                    vocab_size=len(i2w_3),
                                    num_classes=numcls_3,
                                    pad_idx=pad_idx3,
                                    num_epochs=100, 
                                    dataset_name="XOR")