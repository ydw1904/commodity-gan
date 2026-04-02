import torch
import torch.nn as nn

from .tcn import TCN


class Discriminator(nn.Module):
    """
    TCN-based discriminator (Wasserstein critic) for QuantGAN (Wiese et al. 2020).

    Scores a full sequence with a single scalar — no sigmoid, compatible with WGAN-GP loss.

    Args:
        n_assets: Number of input channels (assets).
        n_hidden: Number of hidden channels in each TCN layer (default 80).

    Input:  x  of shape (batch, n_assets, seq_len)
    Output:    of shape (batch, 1),        unbounded scalar score
    """

    def __init__(
        self,
        n_assets: int = 1,
        n_hidden: int = 80,
    ):
        super().__init__()

        self.tcn = TCN(n_inputs=n_assets, n_outputs=n_hidden, n_hidden=n_hidden)
        self.linear = nn.Linear(n_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_assets, seq_len)
        features = self.tcn(x)      # (batch, n_hidden, seq_len)
        last = features[:, :, -1]   # (batch, n_hidden) — use final time step
        return self.linear(last)    # (batch, 1)
