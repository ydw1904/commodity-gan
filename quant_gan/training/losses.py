import torch
import torch.nn as nn


def discriminator_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """WGAN critic loss: maximize E[D(real)] - E[D(fake)], i.e. minimise its negation."""
    return fake_scores.mean() - real_scores.mean()


def generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """WGAN generator loss: minimise -E[D(fake)]."""
    return -fake_scores.mean()


def gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    WGAN-GP gradient penalty (Gulrajani et al. 2017).

    Penalises the gradient norm of the critic at interpolations between
    real and generated samples to enforce the 1-Lipschitz constraint.

    Args:
        discriminator: The critic network.
        real:          Real samples  (batch, channels, seq_len).
        fake:          Fake samples  (batch, channels, seq_len).
        device:        Target device.
        lambda_gp:     Penalty coefficient (default 10).

    Returns:
        Scalar gradient penalty term.
    """
    batch_size = real.size(0)
    # random interpolation coefficient per sample
    alpha = torch.rand(batch_size, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    d_interp = discriminator(interpolated)

    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * gp
