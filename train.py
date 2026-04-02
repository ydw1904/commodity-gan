"""
Entry point for training QuantGAN on commodity futures data.

Usage:
    python train.py --data "quant_gan/preprocessing/C01_Comdty(1).xlsx"
    python train.py --data path/to/data.xlsx --epochs 400 --batch-size 64
"""
import argparse
import torch

from quant_gan.preprocessing.pipeline import preprocess
from quant_gan.training.gan import train
from quant_gan.evaluation.metrics import stylized_facts, print_summary


def parse_args():
    p = argparse.ArgumentParser(description="Train QuantGAN on commodity futures log-returns.")
    p.add_argument("--data", required=True, help="Path to the Excel data file.")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=3, help="Noise channels for generator.")
    p.add_argument("--n-hidden", type=int, default=80, help="TCN hidden channels.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-critic", type=int, default=5, help="Critic steps per generator step.")
    p.add_argument("--lambda-gp", type=float, default=10.0)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--device", default=None, help="e.g. 'cpu', 'cuda', 'mps'")
    p.add_argument("--generator", default="tcn", choices=["tcn", "svnn"],
                   help="Generator architecture: 'tcn' (pure TCN) or 'svnn' (C-SVNN, Wiese et al. Definition 5.1).")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading and preprocessing data from: {args.data}")
    windows, _ = preprocess(args.data)
    print(f"Dataset: {windows.shape[0]} windows of shape {windows.shape[1]}×{windows.shape[2]}")

    device = torch.device(args.device) if args.device else None

    G, _, _ = train(
        windows=windows,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        n_hidden=args.n_hidden,
        lr=args.lr,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp,
        device=device,
        checkpoint_every=args.checkpoint_every,
        generator_type=args.generator,
    )

    # ── Post-training evaluation ──────────────────────────────────────────────
    print("\nEvaluating stylized facts …")
    resolved_device = next(G.parameters()).device
    seq_len = windows.shape[2]
    n_samples = min(1000, windows.shape[0])

    with torch.no_grad():
        z = torch.randn(n_samples, args.latent_dim, seq_len, device=resolved_device)
        fake = G(z).cpu().numpy().flatten()

    real_flat = windows[:n_samples].numpy().flatten()
    metrics = stylized_facts(real_flat, fake)
    print_summary(metrics)


if __name__ == "__main__":
    main()
