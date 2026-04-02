import torch
import torch.nn as nn

from .tcn import TCN


class Generator(nn.Module):
    """
    Pure TCN generator for QuantGAN (Wiese et al. 2020, Section 7.2 "Pure TCN").

    Maps i.i.d. Gaussian noise to synthetic log-return sequences.
    No output non-linearity: returns are unbounded, matching the paper.

    Args:
        latent_dim: Noise channels per time step (NZ = 3 in paper).
        n_assets:   Output channels / assets (NO = 1 in paper).
        n_hidden:   Hidden channels in each TCN layer (NH = 80 in paper).

    Input:  z  of shape (batch, latent_dim, seq_len)
    Output:    of shape (batch, n_assets,   seq_len)
    """

    def __init__(
        self,
        latent_dim: int = 3,
        n_assets: int = 1,
        n_hidden: int = 80,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_assets = n_assets

        self.tcn = TCN(n_inputs=latent_dim, n_outputs=n_assets, n_hidden=n_hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, latent_dim, seq_len)
        # No tanh — log-returns are not bounded to (-1, 1)
        return self.tcn(z)


class SVNNGenerator(nn.Module):
    """
    Stochastic Volatility Neural Network (SVNN) generator — constrained variant.

    Implements Definition 5.1 of Wiese et al. (2020):
        h_t        = TCN(Z_{t-T(g) : t-1})          # lagged noise through TCN
        σ_{t}      = |h_t[: n_assets]|               # volatility (positive)
        μ_{t}      = h_t[n_assets :]                 # drift
        ε_t        = Z_t[0]                          # constrained N(0,1) innovation
        R_t        = σ_t ⊙ ε_t + μ_t

    The innovation is pinned to Z_t[:,0], the first noise channel, which is
    i.i.d. N(0,1) — i.e. the "constrained log-return NP" (Section 5.5).

    Args:
        latent_dim: Total noise channels (NZ = 3; first channel = innovation).
        n_assets:   Number of assets / output channels (1 in paper).
        n_hidden:   TCN hidden dimension (NH = 50 for C-SVNN in paper).
    """

    def __init__(
        self,
        latent_dim: int = 3,
        n_assets: int = 1,
        n_hidden: int = 50,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_assets = n_assets

        # TCN outputs 2 * n_assets channels: first half = volatility, second = drift
        self.tcn = TCN(n_inputs=latent_dim, n_outputs=2 * n_assets, n_hidden=n_hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim, seq_len) — i.i.d. Gaussian noise

        Returns:
            R: (batch, n_assets, seq_len) — synthetic log-returns
        """
        # TCN sees lagged noise: shift by 1 so t-th output uses Z_{t-T:t-1}
        # In practice we feed the full sequence and the causal convolutions
        # ensure h_t only depends on z[:, :, :t].
        h = self.tcn(z)                                  # (B, 2*n_assets, T)

        sigma = torch.abs(h[:, : self.n_assets, :])      # (B, n_assets, T) — volatility ≥ 0
        mu    = h[:, self.n_assets :, :]                 # (B, n_assets, T) — drift

        # Constrained innovation: ε_t = Z_t[:,0], shape (B, 1, T)
        epsilon = z[:, 0:1, :]                           # (B, 1, T)  ~ N(0, 1)

        return sigma * epsilon + mu                       # (B, n_assets, T)
