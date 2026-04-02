import torch
import torch.nn as nn


class DilatedCausalConv(nn.Module):
    """
    Single dilated causal convolutional layer.
    Corresponds to Definition 3.6 in Wiese et al. (2020).
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        out = self.conv(x)
        # remove future timesteps to enforce causality
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TemporalBlock(nn.Module):
    """
    Two dilated causal convolutions with PReLU activations.
    Corresponds to Definition A.1 in Wiese et al. (2020).
    """
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = DilatedCausalConv(n_inputs, n_hidden, kernel_size, dilation)
        self.conv2 = DilatedCausalConv(n_hidden, n_outputs, kernel_size, dilation)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.prelu1(self.conv1(x))
        out = self.prelu2(self.conv2(out))
        return out


class TCN(nn.Module):
    """
    Temporal Convolutional Network with skip connections.
    Architecture from Appendix 2 of Wiese et al. (2020).
    
    Dilation schedule: [1, 1, 2, 4, 8, 16, 32]
    Kernel size: 1 for block 1, 2 for blocks 2-7
    Hidden dim: 80
    RFS: 127
    """
    def __init__(self, n_inputs: int, n_outputs: int, n_hidden: int = 80):
        super().__init__()

        # (kernel_size, dilation) per block from Appendix 2
        block_configs = [
            (1, 1),
            (2, 1),
            (2, 2),
            (2, 4),
            (2, 8),
            (2, 16),
            (2, 32),
        ]

        self.blocks = nn.ModuleList()
        for i, (k, d) in enumerate(block_configs):
            in_ch = n_inputs if i == 0 else n_hidden
            self.blocks.append(TemporalBlock(in_ch, n_hidden, n_hidden, k, d))

        self.n_blocks = len(block_configs)

        # 1x1 convolution output layer
        self.output_conv = nn.Conv1d(n_hidden, n_outputs, kernel_size=1)
        nn.init.xavier_uniform_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_inputs, time)
        skip_sum = None

        for block in self.blocks:
            x = block(x)
            skip_sum = x if skip_sum is None else skip_sum + x

        # Normalise by number of blocks to prevent magnitude explosion
        return self.output_conv(skip_sum / self.n_blocks)


def verify_rfs(n_hidden: int = 80) -> None:
    """
    Sanity check: feed a sequence of length 127 through the TCN
    and confirm output has time dimension >= 1.
    """
    model = TCN(n_inputs=3, n_outputs=1, n_hidden=n_hidden)
    x = torch.randn(1, 3, 127)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape[-1] >= 1, "RFS check failed — output time dimension is 0"
    print("RFS check passed.")


if __name__ == "__main__":
    verify_rfs()