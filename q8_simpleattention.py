import itertools
import torch
import torch.nn.functional as F
import torch.nn as nn
from data import load_xor
from models import BaselineClassifier
from q4 import evaluate, train_epochs


class SimpleSelfAttentionClassifier(BaselineClassifier):
    """Baseline with one simple self-attention layer and positional embedding"""

    def __init__(self, vocab_size, emb=300, num_classes=2, pool="first", max_len=256):
        super().__init__(vocab_size, emb, num_classes, pool)
        self.max_len = max_len
        self.pos_embedding = nn.Embedding(max_len, emb)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        x = x[:, : self.max_len]
        B, T = x.size()

        x_emb = self.embedding(x)
        pos_idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embedding(pos_idx)
        x_emb = x_emb + pos_emb

        attention_output = torch.matmul(x_emb, x_emb.transpose(1, 2))
        attention_weights = F.softmax(attention_output, dim=-1)
        attended = torch.matmul(attention_weights, x_emb)

        if self.pool == "mean":
            out = attended.mean(dim=1)
        elif self.pool == "max":
            out, _ = attended.max(dim=1)
        else:  # "first"
            out = attended[:, 0, :]

        output = self.fc(out)
        return output


def grid_search_attention(
    train_data,
    val_data,
    vocab_size,
    num_classes,
    pad_idx,
    dataset_name: str,
    num_epochs=20,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    lrs = [1e-3, 5e-3, 1e-4]
    batch_sizes = [32, 64, 128]
    with open("results_q8.txt", "w") as f:
        f.write(f"Dataset: {dataset_name}")

    for lr, batch_size in itertools.product(lrs, batch_sizes):
        print(f"\n=== Training with lr={lr}, batch={batch_size} ===")
        model = SimpleSelfAttentionClassifier(
            vocab_size=vocab_size, num_classes=num_classes, pool="first", max_len=256
        )
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        _, train_acc = train_epochs(
            model,
            train_data,
            batch_size,
            pad_idx,
            optimizer,
            num_epochs=num_epochs,
            device=device,
        )
        _, val_acc = evaluate(model, val_data, batch_size, pad_idx, device=device)

        with open("results_q8.txt", "a") as f:
            f.write("\n New model:  \n")
            f.write(
                f"\n lr={lr}, batch={batch_size} | train_acc={train_acc:.3f}, val_acc={val_acc:.3f}"
            )


if __name__ == "__main__":

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
        dataset_name="XOR",
    )
