import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseTCNLayer(nn.Module):
    """
    Stacked TCN Blocks.\n
    Input: (B, gcn_features, window_length)\n
    Output: (B, gtcn_features)
    """

    def __init__(self, channels, kernel_size, dilations, dropout):
        super().__init__()

        assert len(dilations) == len(channels) - 1

        tcn_blocks = []
        for i in range(len(channels) - 1):
            tcn_blocks.append(
                TCNBlock(
                    in_ch=channels[i],
                    out_ch=channels[i + 1],
                    kernel_size=kernel_size,
                    dilation=dilations[i],
                    dropout=dropout,
                )
            )
        self.net = nn.Sequential(*tcn_blocks)

        self.outdim = channels[-1]

    def get_outdim(self) -> int:
        return self.outdim


class TCNLayerLastStep(BaseTCNLayer):
    """
    Stacked TCN Blocks and take last time step.
    """

    def forward(self, x):
        out = self.net(x)
        out = out[:, :, 0]  # only take last time step
        return out


class TCNLayerMeanPool(BaseTCNLayer):
    """
    Stacked TCN Blocks and mean pool over time.
    """

    def forward(self, x):
        out = self.net(x)
        out = out.mean(dim=2)  # mean pool over time
        return out


class TCNLayerWeightPool(BaseTCNLayer):
    """
    Stacked TCN Blocks and linear weight pool over time.
    """

    def forward(self, x):
        T = x.shape[2]
        out = self.net(x)
        weights = torch.linspace(1, 0, steps=T).to(x.device)
        out = (out * weights.unsqueeze(0).unsqueeze(1)).sum(dim=2)
        return out


class TCNBlock(nn.Module):
    """
    One-dimensional Temporal Convolutional Network with residual, dropout\n
    Input: (B, in_ch, window_length)\n
    Output: (B, out_ch, window_length)
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.causal_padding = dilation * (kernel_size - 1)

        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

        self.residual = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        res = self.residual(x)

        x = F.pad(x, (self.causal_padding, 0))  # pad before sequence for causality
        out = self.conv(x)

        out = out + res
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)

        return out
