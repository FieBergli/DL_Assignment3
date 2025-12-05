import math
import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from q11 import AutoRegressiveTransformer
from q12 import batch_dataset
from data import load_toy
from q13 import generate_from_seed


def evaluate_val_bits_and_accuracy_per_token(
    model: AutoRegressiveTransformer,
    val_data: torch.LongTensor,
    batch_size: int,
    context_len: int,
    num_batches: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_nats = 0.0
    total_tokens = 0

    total_correct_last = 0
    total_last = 0

    with torch.no_grad():
        for _ in tqdm(range(num_batches)):
            batch = batch_dataset(val_data, batch_size, context_len + 1)
            batch = batch.to(device)

            x = batch[:, :-1]        # (B, L)
            y = batch[:, 1:]         # (B, L)

            logits = model(x)        # (B, L, V)
            B, L, V = logits.shape

            # per-token loss over all positions
            loss_nats = F.cross_entropy(
                logits.reshape(B * L, V),
                y.reshape(B * L),
                reduction="sum",
            )

            total_nats += loss_nats.item()
            total_tokens += B * L

            # last-token accuracy
            last_logits = logits[:, -1, :]   # (B, V)
            last_targets = y[:, -1]          # (B,)
            preds = last_logits.argmax(dim=-1)

            total_correct_last += (preds == last_targets).sum().item()
            total_last += B

    avg_nats_per_token = total_nats / total_tokens
    avg_bits_per_token = avg_nats_per_token / math.log(2.0)
    accuracy_last = total_correct_last / total_last

    return avg_bits_per_token, accuracy_last


def train_and_sample_q14(
    train_data: torch.LongTensor,
    val_data: torch.LongTensor,
    i2c: dict,
    c2i: dict,
    context_len: int = 256,
    batch_size: int = 64,
    total_steps: int = 50_000,
    eval_every: int = 2_000,          # more frequent eval for early stopping
    validate_num_batches: int = 1000,
    S_seed: int = 16,
    gen_length: int = 200,
    temperature_for_samples: float = 1.0,
    early_stopping_patience: int = 3, # stop if no improvement for N evals
    min_delta: float = 0.0,           # require at least this improvement in bits
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
):
    print("q14 TRAINING")
    with open("q14_results.txt", "a") as f:
        f.write("Autoregressive_training for q14 with early stopping:\n")

    print(
        f"Training on {device} for total_steps={total_steps}, eval_every={eval_every}"
    )

    model = AutoRegressiveTransformer(
        vocab_size=len(i2c), emb=300, num_heads=6, max_len=context_len, num_layers=6
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    train_loss_history = []
    grad_norm_history = []
    val_bits_history = []
    val_acc_history = []
    eval_steps = []
    samples_at_checkpoints = {}

    best_val_bits = float("inf")
    best_step = None
    best_state_dict = None
    epochs_without_improvement = 0

    step = 0
    pbar = tqdm(total=total_steps, desc="Training steps")

    while step < total_steps:
        step += 1

        # ----- TRAIN STEP -----
        batch = batch_dataset(train_data, batch_size, context_len + 1).to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        logits = model(x)  # (B, L, V)
        B, L, V = logits.shape

        loss_nats = F.cross_entropy(
            logits.reshape(B * L, V),
            y.reshape(B * L),
            reduction="sum",
        )
        loss = loss_nats / (B * L)   # nats per token

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss_history.append(loss.item())
        grad_norm_history.append(float(grad_norm))
        pbar.update(1)

        # ----- PERIODIC EVAL + EARLY STOPPING -----
        if step % eval_every == 0 or step == 1:
            model.eval()

            val_bits, val_acc = evaluate_val_bits_and_accuracy_per_token(
                model=model,
                val_data=val_data,
                batch_size=batch_size,
                context_len=context_len,
                num_batches=validate_num_batches,
                device=device,
            )
            val_bits_history.append(val_bits)
            val_acc_history.append(val_acc)
            eval_steps.append(step)

            # sampling for qualitative inspection
            seed_batch = batch_dataset(val_data, 1, S_seed).to(device)
            seed_ids = seed_batch[0].detach().cpu().tolist()

            generated_ids = generate_from_seed(
                model,
                seed_ids,
                gen_len=gen_length,
                temperature=temperature_for_samples,
                device=device,
            )

            def decode(ids):
                return "".join(i2c[int(i)] for i in ids)

            sample_text = decode(generated_ids)

            samples_at_checkpoints[step] = {
                "seed_ids": seed_ids,
                "generated_ids": generated_ids,
                "text": sample_text,
                "val_bits": val_bits,
                "val_acc": val_acc,
            }

            with open("q14_results.txt", "a") as f:
                f.write(
                    f"STEP {step}\n"
                    f"Seed_id: {seed_ids}\n"
                    f"Val bits: {val_bits:.4f}\n"
                    f"Val acc: {val_acc:.4f}\n"
                    f"Sample: {sample_text[:300]}\n"
                    "----\n"
                )

            print("\n=== EVAL @ step", step, "===")
            print(
                f"Validation loss: {val_bits:.4f} bits/char "
                f"(avg over {validate_num_batches} batches)"
            )
            print(f"Validation accuracy (last-token): {val_acc:.4f}")
            print("Sample (truncated):")
            print(sample_text[:400])
            print("=== end sample ===\n")

            # ---- early stopping logic ----
            if val_bits < best_val_bits - min_delta:
                best_val_bits = val_bits
                best_step = step
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                print(f"New best model at step {step}: {best_val_bits:.4f} bits/char")
            else:
                epochs_without_improvement += 1
                print(
                    f"No improvement for {epochs_without_improvement} evals "
                    f"(best={best_val_bits:.4f} bits/char at step {best_step})"
                )

            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"Early stopping triggered at step {step}. "
                    f"Best val bits={best_val_bits:.4f} at step {best_step}."
                )
                break

            model.train()

    pbar.close()

    # restore best model weights (for sampling/plots)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        torch.save(model.state_dict(), "best_model_q14.pt")
        print(f"Best model saved to best_model_q14.pt (step {best_step}).")

    # ---- plots ----
    plt.figure(figsize=(14, 4))

    # train loss (nats/token)
    plt.subplot(1, 3, 1)
    plt.plot(train_loss_history, alpha=0.4, label="train_loss (nats/token)")

    def moving_average(x, w=200):
        if len(x) < w:
            return x
        return [sum(x[i : i + w]) / w for i in range(len(x) - w + 1)]

    ma = moving_average(train_loss_history, w=200)
    plt.plot(range(len(ma)), ma, label="MA (w=200)")
    plt.title("Training loss (nats/token)")
    plt.xlabel("Step")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(grad_norm_history, label="grad_norm")
    plt.title("Gradient norm over steps")
    plt.xlabel("Step")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(eval_steps, val_bits_history, marker="o", label="val_bits (bits/char)")
    plt.title("Validation loss (bits/char) at eval checkpoints")
    plt.xlabel("Step")
    plt.legend()

    plt.tight_layout()
    plt.savefig("q13_training_plots.png", dpi=300)
    plt.show()

    results = {
        "train_loss_history": train_loss_history,
        "grad_norm_history": grad_norm_history,
        "val_bits_history": val_bits_history,
        "val_acc_history": val_acc_history,
        "eval_steps": eval_steps,
        "samples": samples_at_checkpoints,
        "best_val_bits": best_val_bits,
        "best_step": best_step,
    }
    return results
