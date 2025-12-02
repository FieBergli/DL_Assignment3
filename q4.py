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
    
# --- Training and evaluation helpers ---
MAX_LEN = 256

def iterate_batches(dataset, batch_size, pad_idx, shuffle=True):
    """
    dataset: (x_list, y_list)
    returns a list of (x_batch, y_batch) tuples
    """
    x_data, y_data = dataset
    indices = list(range(len(x_data)))

    batches = []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        
        # truncate sequences here before calling pad_batch
        x_seqs = [x_data[j][:MAX_LEN] for j in batch_idx]
        y_labels = [y_data[j] for j in batch_idx]

        x = pad_batch(x_seqs, pad_idx)              # (B, T)
        y = torch.tensor(y_labels, dtype=torch.long)  # (B,)
        batches.append((x, y))
    return batches

def train_epochs(model, train_data, batch_size, pad_idx, optimizer, num_epochs=5):
    for epoch in range(1, num_epochs + 1):
        total_loss, total_correct, total_examples = 0.0, 0, 0
        print(f"\nEpoch {epoch}/{num_epochs}")

        for x, y in iterate_batches(train_data, batch_size, pad_idx, shuffle=True):
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
        acc = total_correct / total_examples
        print(f"Training loss: {avg_loss:.4f}  |  accuracy: {acc:.4f}")

    return avg_loss, acc

def evaluate(model, val_data, batch_size, pad_idx):
    total_loss, total_correct, total_examples = 0.0, 0, 0
    with torch.no_grad():
        for x, y in iterate_batches(val_data, batch_size, pad_idx, shuffle=False):
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

# --- Grid search for SimpleSelfAttentionClassifier ---
def grid_search_attention(train_data, val_data, vocab_size, num_classes, pad_idx):
    lrs = [1e-3, 5e-3]
    batch_sizes = [32, 64, 128]
    results = []

    for lr, batch_size in itertools.product(lrs, batch_sizes):
        print(f"\n=== Training with lr={lr}, batch={batch_size} ===")
        model = SimpleSelfAttentionClassifier(vocab_size=vocab_size, num_classes=num_classes, pool='max')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_loss, train_acc = train_epochs(model, train_data, batch_size, pad_idx, optimizer, num_epochs=20)
        val_loss, val_acc = evaluate(model, val_data, batch_size, pad_idx)

        print(f'lr={lr}, batch={batch_size} | train_acc={train_acc:.3f}, val_acc={val_acc:.3f}')
        results.append((lr, batch_size, train_acc, val_acc))

    return results

# --- Load datasets ---
(x_train_1, y_train_1), (x_val_1, y_val_1), (i2w_1, w2i_1), numcls_1 = load_imdb(final=False)
train_data1 = (x_train_1, y_train_1)
val_data1   = (x_val_1, y_val_1)

(x_train_2, y_train_2), (x_val_2, y_val_2), (i2w_2, w2i_2), numcls_2 = load_imdb_synth()
train_data2 = (x_train_2, y_train_2)
val_data2   = (x_val_2, y_val_2)

(x_train_3, y_train_3), (x_val_3, y_val_3), (i2w_3, w2i_3), numcls_3 = load_xor()
train_data3 = (x_train_3, y_train_3)
val_data3   = (x_val_3, y_val_3)

# --- Pad indices ---
pad_idx1 = w2i_1['.pad']
pad_idx2 = w2i_2['.pad']
pad_idx3 = w2i_3['.pad']

# --- Run experiments with SimpleSelfAttentionClassifier ---
results1 = grid_search_attention(train_data1, val_data1,
                                 vocab_size=len(i2w_1),
                                 num_classes=numcls_1,
                                 pad_idx=pad_idx1)

results2 = grid_search_attention(train_data2, val_data2,
                                 vocab_size=len(i2w_2),
                                 num_classes=numcls_2,
                                 pad_idx=pad_idx2)

results3 = grid_search_attention(train_data3, val_data3,
                                 vocab_size=len(i2w_3),
                                 num_classes=numcls_3,
                                 pad_idx=pad_idx3)
