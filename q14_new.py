import math
import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from torch.utils.tensorboard import SummaryWriter
from q11 import AutoRegressiveTransformer
from q12 import batch_dataset, evaluate_val_bits_and_accuracy
from data import load_toy
from q13 import generate_from_seed
from q13_fie import sample_context


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

            x = batch[:, :-1]
            y = batch[:, 1:]

            logits = model(x)
            B, L, V = logits.shape

            # per-token loss over ALL positions
            loss_nats = F.cross_entropy(
                logits.reshape(B * L, V),
                y.reshape(B * L),
                reduction="sum",
            )

            total_nats += loss_nats.item()
            total_tokens += B * L

            # last-token accuracy
            last_logits = logits[:, -1, :]
            last_targets = y[:, -1]
            preds = last_logits.argmax(dim=-1)

            total_correct_last += (preds == last_targets).sum().item()
            total_last += B

    avg_nats_per_token = total_nats / total_tokens
    avg_bits_per_token = avg_nats_per_token / math.log(2.0)
    accuracy_last = total_correct_last / total_last

    return avg_bits_per_token, accuracy_last


def train_ar_model_14(
    train_data,
    val_data,
    ind_char_mapping,
    vocab_size: int,
    num_steps: int = 50_000,
    eval_every: int = 100,
    batch_size: int = 32,
    device="cuda" if torch.cuda.is_available() else "cpu",
    emb: int = 300,
    num_heads: int = 6,
    context_len: int = 256,
    num_layers: int = 6,
    rot_emb: bool = False,
    lr: float = 1e-4,
    num_batches: int = math.ceil(10_000 / 32),
    seed_len: int = 16,
    job_id: str = "",
    patience: int = 5,
    delta: float = 0.0,
    l2_lambda: float = 0.0,
):
    print("q14 TRAINING")
    with open("q14_results_" + job_id + ".txt", "a") as f:
        f.write(f"Autoregressive_training for q14\n")
    writer = SummaryWriter(log_dir="logs/q14/" + job_id)

    model = AutoRegressiveTransformer(
        vocab_size=vocab_size,
        emb=emb,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=context_len,
        rot_emb=rot_emb,
    )
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)
    train_loss = []
    grad_norm_history = []
    val_loss = []
    val_accuracy = []
    generated_text = {}
    eval_steps = []

    best_val_bits = float("inf")
    best_step = None
    best_state_dict = None
    epochs_without_improvement = 0

    for step in range(1, num_steps + 1):
        batch = batch_dataset(
            train_data, batch_size=batch_size, seq_len=context_len + 1
        ).to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        output = model(x)
        B, L, V = output.shape
        loss = F.cross_entropy(
            output.reshape(B * L, V), y.reshape(B * L), reduction="sum"
        )
        loss_per_token = loss / (B * L)
        optimizer.zero_grad()
        loss_per_token.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss.append(loss_per_token.item())
        grad_norm_history.append(grad_norm.item())
        writer.add_scalar("train/loss_nats_per_token", loss_per_token.item(), step)
        writer.add_scalar("train/grad_norm", float(grad_norm), step)

        if step % eval_every == 0 or step == 1:
            val_bits, val_acc = evaluate_val_bits_and_accuracy_per_token(
                model=model,
                val_data=val_data,
                batch_size=batch_size,
                context_len=context_len,
                num_batches=num_batches,
                device=device,
            )
            val_accuracy.append(val_acc)
            val_loss.append(val_bits)
            eval_steps.append(step)
            print(
                f"Evaluation at step {step}, with validation log-loss in bits: {val_bits}"
            )
            text = sample_context(
                val_data,
                model,
                temperature=1.0,
                ind_char_mapping=ind_char_mapping,
                gen_txtlen=200,
                seed_len=seed_len,
                device=device,
            )
            generated_text[step] = text

            with open("q14_results_" + job_id + ".txt", "a") as f:
                f.write(
                    f"STEP {step}\n"
                    f"Val log-loss bits: {val_bits:.4f}\n"
                    f"Val acc: {val_acc:.4f}\n"
                    f"Text generated: {text[:300]}\n"
                    "----\n"
                )
            writer.add_scalar("val/bits_per_token", val_bits, step)
            writer.add_scalar("val/last_token_accuracy", val_acc, step)

            if val_bits < best_val_bits - delta:
                best_val_bits = val_bits
                best_step = step
                epochs_without_improvement = 0

                with open("q14_results_" + job_id + ".txt", "a") as f:
                    f.write(
                        f"NEW BEST @ step {step}: "
                        f"val_bits={best_val_bits:.4f}, val_acc={val_acc:.4f}\n"
                    )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping triggered at step {step}. "
                    f"Best val bits={best_val_bits:.4f} at step {best_step}."
                )
                break

    with open("q14_results_" + job_id + ".txt", "a") as f:
        f.write(
            f"TRAINING FINISHED.\n"
            f"Total steps run: {step}\n"
            f"Best val bits: {best_val_bits:.4f} at step {best_step}\n"
            "============================\n"
        )

    writer.close()

    return (
        train_loss,
        grad_norm_history,
        val_accuracy,
        val_loss,
        generated_text,
        eval_steps,
    )


def plot_train_and_grad(train_loss, grad_norm_history, id_):
    """
    Plots:
      - Training loss over all steps
      - Gradient norm over all steps
    side by side.
    """
    steps = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # training loss
    axes[0].plot(steps, train_loss)
    axes[0].set_title("Training loss over steps")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")

    # gradient norm
    axes[1].plot(steps, grad_norm_history)
    axes[1].set_title("Gradient norm over steps")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Gradient norm")
    fig.tight_layout()
    plt.savefig(f"q14_training_plots_" + id_ + ".png", dpi=300)
    plt.show()


def plot_val_metrics(val_loss, val_accuracy, id_, eval_steps):
    """
    Plots validation metrics:
      - bits per char vs step
      - accuracy vs step
      - bits per char vs accuracy
    in a single figure with three subplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(eval_steps, val_loss, marker="o")
    axes[0].set_title("Validation loss (bits/char)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Bits per char")

    axes[1].plot(eval_steps, val_accuracy, marker="o")
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy")

    axes[2].plot(val_loss, val_accuracy, marker="o")
    axes[2].set_title("Val accuracy vs bits/char")
    axes[2].set_xlabel("Bits per char")
    axes[2].set_ylabel("Accuracy")

    fig.tight_layout()
    plt.savefig("q14_eval_plots_" + id_ + ".png", dpi=300)
    plt.show()


if __name__ == "__main__":
    (train, val), (i2c, c2i) = load_toy(final=False)
    train = train.long()
    val = val.long()

    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("--id", type=str, default="0", help="id")
    args = parser.parse_args()

    (
        train_loss,
        grad_norm_history,
        val_accuracy,
        val_loss,
        generated_text,
        eval_steps,
    ) = train_ar_model_14(
        train_data=train,
        val_data=val,
        ind_char_mapping=i2c,
        vocab_size=len(i2c),
        num_steps=50_000,
        eval_every=50,
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        emb=300,
        num_heads=6,
        context_len=256,
        num_layers=6,
        rot_emb=True,
        lr=1e-4,
        num_batches=math.ceil(10_000 / 32),
        seed_len=16,
        job_id=args.id,
        patience=15,
        delta=0.0,
        l2_lambda=1e-4,
    )
    plot_train_and_grad(train_loss, grad_norm_history, id_=args.id)
    plot_val_metrics(val_loss, val_accuracy, id_=args.id, eval_steps=eval_steps)
