from q11 import AutoRegressiveTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import load_toy
import math
from tqdm import tqdm


# Question 10
# Treat this as a data loader -> call it outside anything where gradients are computed, and no required_grad=True
def batch_dataset(dataset, batch_size: int, seq_len: int):
    "return a 2D tensor of shape (Batch_size, seq_len) --> Batch rows, sequence length columns"
    N = len(dataset)
    assert seq_len <= N, "seq_len cannot be larger than dataset length"
    max_start = N - seq_len
    start_pos = torch.randint(0, max_start + 1, size=(batch_size,), dtype=torch.long)
    instances = torch.arange(seq_len, dtype=torch.long)
    batch = dataset[start_pos.unsqueeze(1) + instances.unsqueeze(0)]
    return batch


def evaluate_val_bits_and_accuracy(
    model: AutoRegressiveTransformer,
    val_data: torch.LongTensor,
    batch_size: int,
    context_len: int,
    num_batches: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        total_bits = 0.0
        total_correct = 0
        total_tokens = 0

        for _ in tqdm(range(num_batches)):
            # Sample a batch of length L+1 from the *validation* data
            batch = batch_dataset(val_data, batch_size, context_len + 1)  # (B, L+1)
            batch = batch.to(device)

            x = batch[:, :-1]  # (B, L)  context
            y = batch[:, -1]  # (B,)    target: last token only

            # Forward pass
            output = model(x)  # (B, L, vocab_size)
            last_output = output[:, -1, :]  # (B, vocab_size)

            # Cross-entropy in nats, mean over the batch
            loss_nats = F.cross_entropy(last_output, y, reduction="mean")

            # Convert nats -> bits
            loss_bits = loss_nats / math.log(2.0)
            total_bits += loss_bits.item()

            # Accuracy at last position
            preds = last_output.argmax(dim=-1)  # (B,)
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()

        avg_bits_per_char = total_bits / num_batches
        accuracy = total_correct / total_tokens

    return avg_bits_per_char, accuracy


def train_autoregressive(
    train_data: torch.LongTensor,
    val_data: torch.LongTensor,
    context_len: int = 256,
    batch_size: int = 64,
    num_steps: int = 2000,  # one step means one gradient update using one batch of data
    eval_every: int = 1000,
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
):
    print(f"STARTING TRAINING ON {device}")
    lrs = [1e-3, 1e-4, 3e-4]
    result_dict = {}
    for lr in lrs:
        print(f"lr: {lr}")
        model = AutoRegressiveTransformer(
            vocab_size=len(i2c), emb=300, num_heads=6, max_len=context_len, num_layers=6
        )
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        model_dict = {}
        train_losses = []
        val_bits_history = []
        val_acc_history = []
        steps_eval = []

        for step in tqdm(range(1, num_steps + 1)):

            # --- sample batch from train data ---
            batch = batch_dataset(train_data, batch_size, context_len + 1)  # (B, L+1)
            batch = batch.to(device)

            x = batch[:, :-1]  # (B, L)
            y = batch[:, 1:]  # (B, L)  predict next token at every position

            # --- forward ---
            logits = model(x)  # (B, L, vocab_size)

            # reshape for cross_entropy: (B*L, vocab_size) vs (B*L)
            B, L, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * L, V),
                y.reshape(B * L),
                reduction="mean",
            )

            # --- backward + update ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

            # --- periodic evaluation on validation set (Q12) ---
            if step % eval_every == 0 or step == 1:
                val_bits, val_acc = evaluate_val_bits_and_accuracy(
                    model=model,
                    val_data=val_data,
                    batch_size=batch_size,
                    context_len=context_len,
                    num_batches=1000,  # as requested
                    device=device,
                )
                model.train()

                val_bits_history.append(val_bits)
                val_acc_history.append(val_acc)
                steps_eval.append(step)

                print(
                    f"learning rate: {lr},"
                    f"[step {step:6}],"
                    f"train_loss={loss.item():.4f}, "
                    f"val_bits={val_bits:.3f}, "
                    f"val_acc={val_acc:.3f}"
                )
        print(f"Model with lr: {lr}")
        print(f"train_losses: {train_losses}")
        print(f"Val bits: {val_bits_history}")
        print(f"Val accuracy: {val_acc_history}")
        print(f"steps at eval: {steps_eval}")

        model_dict["train_losses"] = train_losses
        model_dict["val_bits"] = val_bits_history
        model_dict["val_acc"] = val_acc_history
        model_dict["steps_eval"] = steps_eval
        result_dict[lr] = model_dict

    return result_dict


if __name__ == "__main__":
    (train, val), (i2c, c2i) = load_toy(final=False)
    train = train.long()
    val = val.long()

    results = train_autoregressive(
        train_data=train, val_data=val, context_len=256, batch_size=64
    )
