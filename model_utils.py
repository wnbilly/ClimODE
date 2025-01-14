import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


class BoundaryPad(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # Pad according to Appendix B.1 : Spherical geometry
        return F.pad(F.pad(input, (0, 0, 1, 1), 'reflect'), (1, 1, 0, 0), 'circular')


class ResidualBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            norm: bool = False,
            n_groups: int = 1,
    ):
        super().__init__()
        self.activation = nn.LeakyReLU(0.3)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=0)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=0.1)
        self.boundary_pad = BoundaryPad()

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # See GroupNorm here : https://arxiv.org/pdf/1803.08494
        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x: torch.Tensor):
        # First convolution layer
        h = self.boundary_pad(x)
        h = self.norm1(h)
        h = self.conv1(h)
        h = self.batch_norm1(h)
        h = self.activation(h)

        # Second convolution layer
        h = self.boundary_pad(h)
        h = self.norm2(h)
        h = self.conv2(h)
        h = self.batch_norm2(h)
        h = self.activation(h)

        h = self.dropout(h)

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class SelfAttnConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SelfAttnConv, self).__init__()
        self.query = self._conv(in_channels, in_channels // 8, stride=1)
        self.key = self.key_conv(in_channels, in_channels // 8, stride=2)
        self.value = self.key_conv(in_channels, out_channels, stride=2)
        self.post_map = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0))
        self.out_channels = out_channels

    def _conv(self, n_in, n_out, stride):
        return nn.Sequential(
            BoundaryPad(),
            nn.Conv2d(n_in, n_in // 2, kernel_size=(3, 3), stride=stride, padding=0),
            nn.LeakyReLU(0.3),
            BoundaryPad(),
            nn.Conv2d(n_in // 2, n_out, kernel_size=(3, 3), stride=stride, padding=0),
            nn.LeakyReLU(0.3),
            BoundaryPad(),
            nn.Conv2d(n_out, n_out, kernel_size=(3, 3), stride=stride, padding=0)
            )

    def key_conv(self, n_in, n_out, stride):
        return nn.Sequential(
            nn.Conv2d(n_in, n_in // 2, kernel_size=(3, 3), stride=stride, padding=0),
            nn.LeakyReLU(0.3),
            nn.Conv2d(n_in // 2, n_out, kernel_size=(3, 3), stride=stride, padding=0),
            nn.LeakyReLU(0.3),
            nn.Conv2d(n_out, n_out, kernel_size=(3, 3), stride=1, padding=0)
            )

    def forward(self, x):
        size = x.size()
        x = x.float()
        q, k, v = self.query(x).flatten(-2, -1), self.key(x).flatten(-2, -1), self.value(x).flatten(-2, -1)
        beta = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=1)
        o = torch.bmm(v, beta.transpose(1, 2))
        o = self.post_map(o.view(-1, self.out_channels, size[-2], size[-1]).contiguous())
        return o


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
