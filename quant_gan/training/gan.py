import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from quant_gan.models import Generator, SVNNGenerator, Discriminator
from quant_gan.training.losses import discriminator_loss, generator_loss, gradient_penalty


def train(
    windows: torch.Tensor,
    n_epochs: int = 400,
    batch_size: int = 64,
    latent_dim: int = 3,
    n_hidden: int = 80,
    lr: float = 1e-4,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    clip_grad_norm: float = 5.0,
    device: torch.device | None = None,
    checkpoint_every: int = 50,
    checkpoint_path: str = "checkpoints/ckpt_{epoch:04d}.pt",
    generator_type: str = "tcn",
) -> tuple[Generator | SVNNGenerator, Discriminator, list[dict]]:
    """
    WGAN-GP training loop for QuantGAN.

    Args:
        windows:          Rolling-window dataset, shape (N, n_assets, seq_len).
        n_epochs:         Number of full passes over the dataset.
        batch_size:       Mini-batch size.
        latent_dim:       Noise channels fed to the generator.
        n_hidden:         Hidden channels in both G and D TCNs.
        lr:               Adam learning rate (paper uses 1e-4).
        n_critic:         Critic update steps per generator step (paper uses 5).
        lambda_gp:        Gradient-penalty coefficient.
        clip_grad_norm:   Max L2 norm for generator gradient clipping (0 = disabled).
        device:           Torch device (auto-detected if None).
        checkpoint_every: Save a checkpoint every N epochs (0 = never).
        checkpoint_path:  Format string for checkpoint filenames.
        generator_type:   "tcn" for pure TCN generator, "svnn" for C-SVNN generator
                          (Wiese et al. 2020, Definition 5.1).

    Returns:
        (generator, discriminator, history)
        history is a list of dicts with keys epoch, d_loss, g_loss.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    n_assets = windows.shape[1]
    seq_len = windows.shape[2]

    dataset = TensorDataset(windows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if generator_type == "svnn":
        G = SVNNGenerator(latent_dim=latent_dim, n_assets=n_assets, n_hidden=n_hidden).to(device)
    else:
        G = Generator(latent_dim=latent_dim, n_assets=n_assets, n_hidden=n_hidden).to(device)
    D = Discriminator(n_assets=n_assets, n_hidden=n_hidden).to(device)

    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=lr * 0.5, betas=(0.0, 0.9))

    history: list[dict] = []
    global_step = 0

    for epoch in range(1, n_epochs + 1):
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0
        n_batches = 0

        for (real_batch,) in loader:
            real_batch = real_batch.to(device)  # (B, n_assets, seq_len)
            current_bs = real_batch.size(0)

            # ── Critic update ─────────────────────────────────────────────
            for _ in range(n_critic):
                z = torch.randn(current_bs, latent_dim, seq_len, device=device)
                fake_batch = G(z).detach()

                real_scores = D(real_batch)
                fake_scores = D(fake_batch)

                gp = gradient_penalty(D, real_batch, fake_batch, device, lambda_gp)
                d_loss = discriminator_loss(real_scores, fake_scores) + gp

                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()

            # ── Generator update ──────────────────────────────────────────
            z = torch.randn(current_bs, latent_dim, seq_len, device=device)
            fake_batch = G(z)
            fake_scores = D(fake_batch)
            g_loss = generator_loss(fake_scores)

            opt_G.zero_grad()
            g_loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(G.parameters(), clip_grad_norm)
            opt_G.step()

            d_loss_epoch += d_loss.item()
            g_loss_epoch += g_loss.item()
            n_batches += 1
            global_step += 1

        d_loss_avg = d_loss_epoch / n_batches
        g_loss_avg = g_loss_epoch / n_batches
        history.append({"epoch": epoch, "d_loss": d_loss_avg, "g_loss": g_loss_avg})

        print(f"Epoch {epoch:4d}/{n_epochs}  D: {d_loss_avg:+.4f}  G: {g_loss_avg:+.4f}")

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            import pathlib
            path = checkpoint_path.format(epoch=epoch)
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "G_state": G.state_dict(),
                    "D_state": D.state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_D": opt_D.state_dict(),
                },
                path,
            )
            print(f"  → checkpoint saved to {path}")

    return G, D, history
