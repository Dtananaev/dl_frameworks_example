"""The cnn architecture."""
__copyright__ = """
Copyright (c) 2022 Tananaev Denis
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this permission
notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""


from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Module):
    """Two convolution block.

    Args:
        in_channels: input number of channels
        out_channels: number of convolutional fiters
        kernel: the size of convolutional kernel
        weight_decay: the weight decay regularization
    """

    def __init__(self, in_channels: int, out_channels: int, kernel: Union[int, Tuple[int, int]] = 3) -> None:
        """Initialization."""
        super().__init__()
        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=1, bias=True)
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, padding=1, bias=True)
        torch.nn.init.kaiming_normal_(self.conv2.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The call function."""
        features = self.activation(self.conv1(input))
        features = self.activation(self.conv2(features))
        return features


class EncoderBlock(nn.Module):
    """U-net encoder block.

    Args:
        in_channels: number of convolutional fiters input
        out_channels: number of convolutional fiters output
        kernel: the size of convolutional kernel
        pool_size: the size of pooling kernel
        dropout: the dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Union[int, Tuple[int, int]],
        pool_size: Union[int, Tuple[int, int]],
        dropout: float = 0.3,
    ) -> None:
        """Initialization."""
        super().__init__()
        self.double_conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, kernel=kernel)
        self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The call function."""
        skip_connection = features = self.double_conv(input)
        features = self.pool(features)
        # Note dropout works different than in tensorflow, no training argument should be included
        features = self.dropout(features)
        return skip_connection, features


class DecoderBlock(nn.Module):
    """U-net decoder block.

    Args:
        filters: number of convolutional filters
        kernel: convolutional kernel size
        strides: strides for transpose convolution
        weight_decay: weight decay regularization
        dropout: dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]],
        dropout: float = 0.3,
    ):
        """Initialization."""
        super().__init__()
        self.transp_conv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 2), stride=strides
        )
        self.dropout = nn.Dropout(p=dropout)
        self.double_conv = DoubleConv(in_channels=2 * out_channels, out_channels=out_channels, kernel=kernel)

    def forward(
        self,
        input: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """The call function."""
        skip_connection, features = input
        features = self.transp_conv(features)
        features = torch.cat([skip_connection, features], dim=1)
        features = self.dropout(features)
        features = self.double_conv(features)
        return features


class UnetEncoder(nn.Module):
    """U-net encoder.

    Args:
        dropout: dropout
    """

    def __init__(self, dropout=0.3) -> None:
        """Initialization."""
        super().__init__()
        self.block1 = EncoderBlock(in_channels=3, out_channels=64, kernel=(3, 3), pool_size=(2, 2), dropout=dropout)
        self.block2 = EncoderBlock(in_channels=64, out_channels=128, kernel=(3, 3), pool_size=(2, 2), dropout=dropout)
        self.block3 = EncoderBlock(in_channels=128, out_channels=256, kernel=(3, 3), pool_size=(2, 2), dropout=dropout)
        self.block4 = EncoderBlock(in_channels=256, out_channels=512, kernel=(3, 3), pool_size=(2, 2), dropout=dropout)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """The call function."""
        skip1, features = self.block1(input)
        skip2, features = self.block2(features)
        skip3, features = self.block3(features)
        skip4, features = self.block4(features)

        return features, (skip1, skip2, skip3, skip4)


class UnetDecoder(nn.Module):
    """U-net decoder.

    Args:
        weight_decay: weight decay
        dropout: dropout
    """

    def __init__(self, dropout=0.3) -> None:
        """Initialization."""
        super().__init__()

        self.block1 = DecoderBlock(in_channels=1024, out_channels=512, kernel=(3, 3), strides=(2, 2), dropout=dropout)
        self.block2 = DecoderBlock(in_channels=512, out_channels=256, kernel=(3, 3), strides=(2, 2), dropout=dropout)
        self.block3 = DecoderBlock(in_channels=256, out_channels=128, kernel=(3, 3), strides=(2, 2), dropout=dropout)
        self.block4 = DecoderBlock(in_channels=128, out_channels=64, kernel=(3, 3), strides=(2, 2), dropout=dropout)
        self.final_layer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1), padding=0, bias=True)
        torch.nn.init.kaiming_normal_(self.final_layer.weight)

    def forward(self, input: Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """The call function."""
        features, (skip1, skip2, skip3, skip4) = input

        features = self.block1((skip4, features))
        features = self.block2((skip3, features))
        features = self.block3((skip2, features))
        features = self.block4((skip1, features))
        output = self.final_layer(features)
        return output


class Unet(nn.Module):
    """U-net CNN.

    Note: see https://arxiv.org/abs/1505.04597 for details.
    We changed final layer to output depth instead of semseg.

    Args:
        weight_decay: weight decay
        dropout: dropout
    """

    def __init__(self, dropout: float = 0.3, name="Unet") -> None:
        """Initialization."""
        super().__init__()
        self.name = name
        self.encoder = UnetEncoder(dropout=dropout)
        self.bottleneck = DoubleConv(in_channels=512, out_channels=1024, kernel=(3, 3))
        self.decoder = UnetDecoder(dropout=dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The call function."""
        # Normalize image
        input /= 255.0
        features, skip_connections = self.encoder(input)
        features = self.bottleneck(features)
        output = self.decoder((features, skip_connections))

        return output
