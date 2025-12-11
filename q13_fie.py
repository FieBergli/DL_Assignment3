import argparse
import torch.distributions as dist
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import load_toy
from q12 import batch_dataset, evaluate_val_bits_and_accuracy
from q11 import AutoRegressiveTransformer
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def sample(lnprobs, temperature=1.0):
    """Sample an element from a categorical distribution
    :param lnprobs: Outcome logits
    :param temperature: Sampling temperature. 1.0 follows the given distribution, 0.0 returns the maximum probability element.
    :return: The index of the sampled element."""
    if temperature == 0.0:
        return lnprobs.argmax()
    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(probs=p)
    return cd.sample()


"""
- Seed -> starting snippet of the text for the model to continue. We can call it context or prime text.
- When S=16, we say that the context string is of 16 characters. 
- Sample from validation data, as it is not the training set that the model trained on. 
"""


def sample_context(
    val,
    model,
    temperature,
    ind_char_mapping,
    gen_txtlen: int = 200,
    seed_len: int = 16,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    N = len(val)
    max_start = N - seed_len
    start = random.randint(0, max_start)
    context = val[start : start + seed_len]

    context_tensor = context.unsqueeze(0)  # add batch dim (1, S)
    context_tensor = context_tensor.to(device)

    model.eval()
    with torch.no_grad():

        for _ in range(gen_txtlen):
            output = model(context_tensor)
            last_outputs = output[:, -1, :]
            out_probs = F.log_softmax(last_outputs, dim=-1)
            out_probs = out_probs.squeeze(0)
            idx = sample(lnprobs=out_probs, temperature=temperature)
            next_token = idx.view(1, 1).to(device).long()
            context_tensor = torch.cat([context_tensor, next_token], dim=1)

    model.train()
    token_ids = context_tensor.squeeze(0).tolist()
    text = "".join(ind_char_mapping[i] for i in token_ids)
    return text


def train_ar_model(
    train_data,
    val_data,
    ind_char_mapping,
    vocab_size: int,
    num_steps: int = 50_000,
    eval_every: int = 10_000,
    batch_size: int = 32,
    device="cuda" if torch.cuda.is_available() else "cpu",
    emb: int = 300,
    num_heads: int = 6,
    context_len: int = 256,
    num_layers: int = 6,
    rot_emb: bool = False,
    lr: float = 1e-4,
    num_batches: int = 10_000,
    seed_len: int = 16,
    job_id: str = "",
):
    print("q13 TRAINING")
    with open("q13_results_" + job_id + ".txt", "a") as f:
        f.write(f"Autoregressive_training for q13\n")
    writer = SummaryWriter(log_dir="logs/q13/" + job_id)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    grad_norm_history = []
    val_loss = []
    val_accuracy = []
    generated_text = {}

    for step in range(1, num_steps + 1):
        batch = batch_dataset(
            train_data, batch_size=batch_size, seq_len=context_len + 1
        ).to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        output = model(x)
        B, L, V = output.shape
        loss = F.cross_entropy(
            output.reshape(B * L, V), y.reshape(B * L), reduction="mean"
        )
        optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss.append(loss.item())
        grad_norm_history.append(grad_norm.item())

        if step % eval_every == 0 or step == 1:
            val_bits, val_acc = evaluate_val_bits_and_accuracy(
                model=model,
                val_data=val_data,
                batch_size=batch_size,
                context_len=context_len,
                num_batches=num_batches,
                device=device,
            )
            val_accuracy.append(val_acc)
            val_loss.append(val_bits)
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

            with open("q13_results_" + job_id + ".txt", "a") as f:
                f.write(
                    f"STEP {step}\n"
                    f"Val log-loss bits: {val_bits:.4f}\n"
                    f"Val acc: {val_acc:.4f}\n"
                    f"Text generated: {text[:300]}\n"
                    "----\n"
                )

    writer.close()

    return train_loss, grad_norm_history, val_accuracy, val_loss, generated_text


def plot_train_and_grad(train_loss, grad_norm_history, id):
    """
    Plots:
      - Training loss over all steps
      - Gradient norm over all steps
    side by side.
    """
    steps = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: training loss
    axes[0].plot(steps, train_loss)
    axes[0].set_title("Training loss over steps")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")

    # Right: gradient norm
    axes[1].plot(steps, grad_norm_history)
    axes[1].set_title("Gradient norm over steps")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Gradient norm")
    fig.tight_layout()
    plt.savefig(f"q13_training_plots_{id}.png", dpi=300)
    plt.show()


def plot_val_metrics(val_loss, val_accuracy, id):
    """
    Plots validation metrics:
      - bits per char vs step
      - accuracy vs step
      - bits per char vs accuracy
    in a single figure with three subplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    eval_steps = len(val_loss)

    # 1) val bits vs step
    axes[0].plot(eval_steps, val_loss, marker="o")
    axes[0].set_title("Validation loss (bits/char)")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Bits per char")

    # 2) val accuracy vs step
    axes[1].plot(eval_steps, val_accuracy, marker="o")
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy")

    # 3) bits vs accuracy scatter/curve
    axes[2].plot(val_loss, val_accuracy, marker="o")
    axes[2].set_title("Val accuracy vs bits/char")
    axes[2].set_xlabel("Bits per char")
    axes[2].set_ylabel("Accuracy")

    fig.tight_layout()
    plt.savefig(f"q13_{id}_eval_plots.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    (train, val), (i2c, c2i) = load_toy(final=False)
    train = train.long()
    val = val.long()

    parser = argparse.ArgumentParser(description="Argument parser")
    parser.add_argument("--id", type=str, default="0", help="id")
    args = parser.parse_args()

    train_loss, grad_norm_history, val_accuracy, val_loss, generated_text = (
        train_ar_model(
            train_data=train,
            val_data=val,
            ind_char_mapping=i2c,
            vocab_size=len(i2c),
            num_steps=50_000,
            eval_every=10_000,
            batch_size=32,
            device="cuda" if torch.cuda.is_available() else "cpu",
            emb=300,
            num_heads=6,
            context_len=256,
            num_layers=6,
            rot_emb=False,
            lr=1e-4,
            num_batches=10_000,
            seed_len=16,
            job_id=args.id,
        )
    )
    plot_train_and_grad(train_loss, grad_norm_history, args.id)
    plot_val_metrics(val_loss, val_accuracy, args.id)
