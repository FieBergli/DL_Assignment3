from q1 import pad_batch
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
import itertools
from data import load_imdb, load_imdb_synth, load_xor


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
        super().__init__()
        self.emb = emb_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes)
        self.pool = pool

    def forward(self, x: torch.LongTensor) -> torch.Tensor:

        # embed -> (batch, time, emb)
        x_emb = self.embedding(x)

        # global pooling -> (batch, emb)
        if self.pool == "mean":
            out = x_emb.mean(dim=1)
        elif self.pool == "max":
            out, _ = x_emb.max(dim=1)
        else:  # first
            out = x_emb[:, 0, :]

        output = self.fc(out)
        return output


class SimpleSelfAttentionClassifier(BaselineClassifier):
    """Baseline with one simple self-attention layer"""

    def __init__(self, vocab_size, emb=300, num_classes=2, pool="max", max_len=256):
        super().__init__(vocab_size, emb, num_classes, pool)
        self.max_len = max_len

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        x = x[:, : self.max_len]
        x_emb = self.embedding(x)

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


MAX_LEN = 256


def iterate_batches(dataset, batch_size, pad_idx):
    """
    dataset: (x_list, y_list)
    returns a list of (x_batch, y_batch) tuples
    """
    x_data, y_data = dataset
    indices = list(range(len(x_data)))

    batches = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        x_seqs = [x_data[j][:MAX_LEN] for j in batch_idx]
        y_labels = [y_data[j] for j in batch_idx]

        x = pad_batch(x_seqs, pad_idx)
        y = torch.tensor(y_labels, dtype=torch.long)
        batches.append((x, y))
    return batches


def train_epochs(
    model,
    train_data,
    val_data,
    batch_size,
    pad_idx,
    optimizer,
    num_epochs,
    patience=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.train()
    model.to(device)
    best_val_acc = 0
    no_improvement = 0
    for epoch in range(1, num_epochs + 1):
        total_loss, total_correct, total_examples = 0.0, 0, 0
        print(f"\nEpoch {epoch}/{num_epochs}")

        for x, y in iterate_batches(train_data, batch_size, pad_idx):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

            # stats
            batch_size_actual = x.size(0)
            total_loss += loss.item() * batch_size_actual
            preds = output.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_examples += batch_size_actual

        avg_loss = total_loss / total_examples
        train_acc = total_correct / total_examples

        if epoch % 5 == 0:
            _, val_acc = evaluate(model, val_data, batch_size, pad_idx)
            model.train()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_train_acc = train_acc
                epochs_trained = epoch
                best_loss = avg_loss
                no_improvement = 0 
            else:
                no_improvement += 1
                if no_improvement == patience:
                    break

            print(
                f"Epoch: {epoch} | Train loss: {avg_loss:.4f}  |  Train accuracy: {train_acc:.4f} | Val accuracy {val_acc:.4f}"
            )

    return best_loss, epochs_trained, best_train_acc, best_val_acc


def evaluate(
    model,
    val_data,
    batch_size,
    pad_idx,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.eval()
    total_loss, total_correct, total_examples = 0.0, 0, 0
    with torch.no_grad():
        for x, y in iterate_batches(val_data, batch_size, pad_idx):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = F.cross_entropy(output, y)
            batch_size_actual = x.size(0)
            total_loss += loss.item() * batch_size_actual
            preds = output.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_examples += batch_size_actual

    avg_loss = total_loss / total_examples
    acc = total_correct / total_examples
    return avg_loss, acc


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
    with open("results_q5.txt", "w") as f:
        f.write(f"Dataset: {dataset_name}")

    for lr, batch_size in itertools.product(lrs, batch_sizes):
        print(f"\n=== Training with lr={lr}, batch={batch_size} ===")
        model = SimpleSelfAttentionClassifier(
            vocab_size=vocab_size, num_classes=num_classes, pool="first", max_len=256
        )
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        _, epochs_trained, train_acc, val_acc = train_epochs(
            model,
            train_data,
            val_data,
            batch_size,
            pad_idx,
            optimizer,
            num_epochs=num_epochs,
            device=device,
        )

        with open("results_q5.txt", "a") as f:
            f.write("\n New model:  \n")
            f.write(
                f"\n lr={lr}, batch={batch_size}  , epochs trained: {epochs_trained} | train_acc={train_acc:.3f}, val_acc={val_acc:.3f}"
            )


if __name__ == "__main__":

    (x_train_1, y_train_1), (x_val_1, y_val_1), (i2w_1, w2i_1), numcls_1 = load_imdb(
        final=False
    )
    train_data1 = (x_train_1, y_train_1)
    val_data1 = (x_val_1, y_val_1)

    (x_train_2, y_train_2), (x_val_2, y_val_2), (i2w_2, w2i_2), numcls_2 = (
        load_imdb_synth()
    )
    train_data2 = (x_train_2, y_train_2)
    val_data2 = (x_val_2, y_val_2)

    (x_train_3, y_train_3), (x_val_3, y_val_3), (i2w_3, w2i_3), numcls_3 = load_xor()
    train_data3 = (x_train_3, y_train_3)
    val_data3 = (x_val_3, y_val_3)

    pad_idx1 = w2i_1[".pad"]
    pad_idx2 = w2i_2[".pad"]
    pad_idx3 = w2i_3[".pad"]

    results1 = grid_search_attention(
        train_data1,
        val_data1,
        vocab_size=len(i2w_1),
        num_classes=numcls_1,
        pad_idx=pad_idx1,
        num_epochs=20,
        dataset_name="IMDb",
    )

    results2 = grid_search_attention(
        train_data2,
        val_data2,
        vocab_size=len(i2w_2),
        num_classes=numcls_2,
        pad_idx=pad_idx2,
        num_epochs=100,
        dataset_name="IMBd synthetic",
    )

    results3 = grid_search_attention(
        train_data3,
        val_data3,
        vocab_size=len(i2w_3),
        num_classes=numcls_3,
        pad_idx=pad_idx3,
        num_epochs=100,
        dataset_name="XOR",
    )
