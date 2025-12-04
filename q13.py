# q13_sample.py
# Full implementation for Question 13: sampling + training loop + periodic evaluation + plotting.
# This file assumes your project has:
#   - AutoRegressiveTransformer (from q11)
#   - batch_dataset and evaluate_val_bits_and_accuracy from your Question 12 code
#   - load_toy() that returns (train, val), (i2c, c2i)
# The script trains for 50k steps, evaluates every 10k steps (val bits over 1000 batches),
# and samples from a random seed of S=16 from the validation set at each evaluation.

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List


# import model and data functions: adjust import paths if your files are named differently
from q11 import AutoRegressiveTransformer
from q12 import batch_dataset, evaluate_val_bits_and_accuracy  # your Q12 utilities
from data import load_toy  # your toy data loader (as used earlier)

# ---------------------------
# EXACT sample function from the assignment (use this)
# ---------------------------
import torch.distributions as dist


def sample(lnprobs: torch.Tensor, temperature: float = 1.0) -> int:
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome logits (1D tensor, V,)
    :param temperature: Sampling temperature. 1.0 follows the given distribution, 0.0 returns argmax
    :return: The index of the sampled element (as int)
    """
    if temperature == 0.0:
        return int(lnprobs.argmax().item())
    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)
    return int(cd.sample().item())


# ---------------------------
# Helper: autoregressive generation given a seed of S tokens
# Uses the assignment sample() above.
# ---------------------------
def generate_from_seed(
    model: AutoRegressiveTransformer,
    seed_ids: List[int],
    gen_len: int,
    temperature: float = 1.0,
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
) -> List[int]:
    """
    Autoregressively generate `gen_len` additional tokens given an initial seed (list of token ids).
    Returns: list of token ids = seed + generated
    """
    model.eval()
    model.to(device)

    # Copy seed so we can append
    generated = list(seed_ids)

    with torch.no_grad():
        for _ in range(gen_len):
            # maybe this doesn't have to be an if/else because or model defintilty has a max_len / context_len of 256 right???????????!!!!!!!!!!!!!!!
            # Respect context window: if model has max_len, keep only the last max_len tokens
            context = generated[-model.max_len :]

            # Build input tensor shape (1, T)
            x = torch.tensor([context], dtype=torch.long, device=device)  # (1, T)
            logits = model(x)  # (1, T, V)
            # take logits for last time step -> shape (V,)
            last_logits = logits[0, -1, :]  # (V,)

            # sample next token using assignment's sample function
            next_id = sample(last_logits, temperature=temperature)
            generated.append(next_id)

    return generated


# ---------------------------
# Main training + sampling routine for Question 13
# ---------------------------
def train_and_sample_q13(
    train_data: torch.LongTensor,
    val_data: torch.LongTensor,
    i2c: dict,
    c2i: dict,
    context_len: int = 256,
    batch_size: int = 64,
    total_steps: int = 50_000,
    eval_every: int = 10_000,
    validate_num_batches: int = 1000,
    S_seed: int = 16,  # S = 16 as requested
    gen_length: int = 200,  # how many tokens to generate for inspection
    temperature_for_samples: float = 1.0,
    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ),
):
    """
    Train autoregressive transformer and at every eval_every steps:
      - compute validation bits (over validate_num_batches)
      - take a random seed of S characters from validation set, generate gen_length tokens,
        and save/print the generated sample
    Returns:
      results dict containing histories and samples
    """

    print(
        f"Training on {device} for total_steps={total_steps}, eval_every={eval_every}"
    )
    # Create model (same architecture choices as in your Q12 code)
    model = AutoRegressiveTransformer(
        vocab_size=len(i2c), emb=300, num_heads=6, max_len=context_len, num_layers=6
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4
    )  # lr chosen; you can tune

    # Storage for histories
    train_loss_history = []
    grad_norm_history = []
    val_bits_history = []
    val_acc_history = []
    eval_steps = []
    samples_at_checkpoints = {}  # step -> generated string

    # We'll do a progress bar over steps
    step = 0
    pbar = tqdm(total=total_steps, desc="Training steps")

    while step < total_steps:
        step += 1

        # --- 1) sample training batch of length L+1 ---
        batch = batch_dataset(train_data, batch_size, context_len + 1)  # (B, L+1)
        batch = batch.to(device)

        x = batch[:, :-1]  # (B, L)
        y = batch[:, 1:]  # (B, L) -> next-token targets at every position

        # --- 2) forward pass ---
        logits = model(x)  # (B, L, V)
        B, L, V = logits.shape

        # flatten time and batch dims for cross-entropy as usual
        loss = F.cross_entropy(
            logits.reshape(B * L, V), y.reshape(B * L), reduction="mean"
        )

        # --- 3) backward + gradient norm + clipping + step ---
        optimizer.zero_grad()
        loss.backward()

        # compute gradient norm and clip (clip_grad_norm_ returns total_norm)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # record training statistics
        train_loss_history.append(loss.item())
        grad_norm_history.append(float(grad_norm))

        pbar.update(1)

        # --- 4) periodic evaluation + sampling ---
        if step % eval_every == 0 or step == 1:
            model.eval()
            # compute validation bits & accuracy using your Q12 function
            val_bits, val_acc = evaluate_val_bits_and_accuracy(
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

            # --- sample: slice a random seed S characters from validation data ---
            # use batch_dataset(val_data, 1, S_seed) to get one random seed sequence of length S
            seed_batch = batch_dataset(val_data, 1, S_seed).to(device)  # shape (1, S)
            seed_ids = seed_batch[0].detach().cpu().tolist()  # list[int] length S

            # generate additional tokens autoregressively using the assignment sample()
            generated_ids = generate_from_seed(
                model,
                seed_ids,
                gen_len=gen_length,
                temperature=temperature_for_samples,
                device=device,
            )

            # decode to characters using i2c mapping
            def decode(ids):
                return "".join(
                    i2c[int(i)] for i in ids
                )  # mapping index->char; i2c uses indices

            sample_text = decode(generated_ids)

            # save/print sample and statistics
            samples_at_checkpoints[step] = {
                "seed_ids": seed_ids,
                "generated_ids": generated_ids,
                "text": sample_text,
                "val_bits": val_bits,
                "val_acc": val_acc,
            }

            print("\n=== EVAL @ step", step, "===")
            print(
                f"Validation bits (avg over {validate_num_batches} batches): {val_bits:.4f} bits/char"
            )
            print(f"Validation accuracy (last-token): {val_acc:.4f}")
            print(f"Sample (seed length S={S_seed} + generated {gen_length} chars):")
            # To avoid flooding terminal, print first ~400 chars
            print(sample_text[:400])
            print("=== end sample ===\n")

            model.train()  # set back to train mode

    pbar.close()

    # --- 5) plotting results ---
    # Plot training loss (moving average for readability), gradient norm, and val bits at eval points
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 4))

    # training loss (raw and smoothed)
    plt.subplot(1, 3, 1)
    plt.plot(train_loss_history, alpha=0.4, label="train_loss (raw)")

    # simple moving average smoothing
    def moving_average(x, w=100):
        if len(x) < w:
            return x
        return [sum(x[i : i + w]) / w for i in range(len(x) - w + 1)]

    ma = moving_average(train_loss_history, w=200)
    plt.plot(range(len(ma)), ma, color="red", label=f"MA (w=200)")
    plt.title("Training loss")
    plt.xlabel("Step")
    plt.legend()

    # gradient norm
    plt.subplot(1, 3, 2)
    plt.plot(grad_norm_history, label="grad_norm")
    plt.title("Gradient norm over steps")
    plt.xlabel("Step")
    plt.legend()

    # validation bits at eval steps
    plt.subplot(1, 3, 3)
    plt.plot(eval_steps, val_bits_history, marker="o", label="val_bits (bits/char)")
    plt.title("Validation bits at eval checkpoints")
    plt.xlabel("Step")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # return collected results plus samples
    results = {
        "train_loss_history": train_loss_history,
        "grad_norm_history": grad_norm_history,
        "val_bits_history": val_bits_history,
        "val_acc_history": val_acc_history,
        "eval_steps": eval_steps,
        "samples": samples_at_checkpoints,
    }
    return results


# ---------------------------
# Run the training+sampling experiment if this file is the main script
# ---------------------------
if __name__ == "__main__":
    # Load toy data (must return (train, val), (i2c, c2i) as in your code)
    (train, val), (i2c, c2i) = load_toy(final=False)
    # convert to long tensors
    train = train.long()
    val = val.long()

    # call training+sampling routine with assignment-specified parameters:
    results = train_and_sample_q13(
        train_data=train,
        val_data=val,
        i2c=i2c,
        c2i=c2i,
        context_len=256,
        batch_size=64,
        total_steps=50_000,
        eval_every=10_000,
        validate_num_batches=1000,
        S_seed=16,
        gen_length=200,
        temperature_for_samples=1.0,
    )

    # After this finishes you will have 'results' with samples at each eval checkpoint.
    # You can inspect results['samples'] for text samples and corresponding val_bits.
