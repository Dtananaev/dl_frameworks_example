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

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2


class DoubleConv(Model):
    """Two convolution block.

    Args:
        filters: number of convolutional fiters
        kernel: the size of convolutional kernel
        weight_decay: the weight decay regularization
    """

    def __init__(self, filters: int, kernel: Union[int, Tuple[int, int]], weight_decay: float = 0.0) -> None:
        """Initialization."""
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            kernel_initializer="he_normal",
            padding="same",
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            data_format="channels_first",
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel,
            kernel_initializer="he_normal",
            padding="same",
            activation="relu",
            kernel_regularizer=l2(weight_decay),
            data_format="channels_first",
        )

    def call(self, input: tf.Tensor) -> tf.Tensor:
        """The call function."""
        features = self.conv1(input)
        features = self.conv2(features)
        return features


class EncoderBlock(Model):
    """U-net encoder block.

    Args:
        filters: number of convolutional fiters
        kernel: the size of convolutional kernel
        pool_size: the size of pooling kernel
        weight_decay: the weight decay regularization
        dropout: the dropout probability
    """

    def __init__(
        self,
        filters: int,
        kernel: Union[int, Tuple[int, int]],
        pool_size: Union[int, Tuple[int, int]],
        weight_decay: float = 0.0,
        dropout: float = 0.3,
    ) -> None:
        """Initialization."""
        super().__init__()
        self.double_conv = DoubleConv(filters=filters, kernel=kernel, weight_decay=weight_decay)
        self.pool = tf.keras.layers.MaxPool2D(pool_size=pool_size, data_format="channels_first")
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, input: tf.Tensor, training: bool = False) -> tf.Tensor:
        """The call function."""
        skip_connection = features = self.double_conv(input)
        features = self.pool(features)
        # Note dropout works different in train and test time therefore training argument should be included
        features = self.dropout(features, training=training)
        return skip_connection, features


class DecoderBlock(Model):
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
        filters: int,
        kernel: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]],
        weight_decay: float = 0.0,
        dropout: float = 0.3,
    ):
        """Initialization."""
        super().__init__()

        self.transp_conv = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel,
            strides=strides,
            padding="same",
            kernel_regularizer=l2(weight_decay),
            data_format="channels_first",
        )
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)
        self.double_conv = DoubleConv(filters=filters, kernel=kernel, weight_decay=weight_decay)

    def call(self, input: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        """The call function."""
        skip_connection, features = input
        features = self.transp_conv(features)
        features = self.concat([skip_connection, features])
        features = self.dropout(features, training=training)
        features = self.double_conv(features)
        return features


class UnetEncoder(Model):
    """U-net encoder.

    Args:
        weight_decay: weight decay
        dropout: dropout
    """

    def __init__(self, weight_decay: float = 0.0, dropout=0.3) -> None:
        """Initialization."""
        super().__init__()
        self.block1 = EncoderBlock(
            filters=64, kernel=(3, 3), pool_size=(2, 2), weight_decay=weight_decay, dropout=dropout
        )
        self.block2 = EncoderBlock(
            filters=128, kernel=(3, 3), pool_size=(2, 2), weight_decay=weight_decay, dropout=dropout
        )
        self.block3 = EncoderBlock(
            filters=256, kernel=(3, 3), pool_size=(2, 2), weight_decay=weight_decay, dropout=dropout
        )
        self.block4 = EncoderBlock(
            filters=512, kernel=(3, 3), pool_size=(2, 2), weight_decay=weight_decay, dropout=dropout
        )

    def call(self, input: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, Tuple[tf.Tensor, ...]]:
        """The call function."""
        skip1, features = self.block1(input, training=training)
        skip2, features = self.block2(features, training=training)
        skip3, features = self.block3(features, training=training)
        skip4, features = self.block4(features, training=training)

        return features, (skip1, skip2, skip3, skip4)


class UnetDecoder(Model):
    """U-net decoder.

    Args:
        weight_decay: weight decay
        dropout: dropout
    """

    def __init__(self, weight_decay: float = 0.0, dropout=0.3) -> None:
        """Initialization."""
        super().__init__()
        self.block1 = DecoderBlock(
            filters=512, kernel=(3, 3), strides=(2, 2), weight_decay=weight_decay, dropout=dropout
        )
        self.block2 = DecoderBlock(
            filters=256, kernel=(3, 3), strides=(2, 2), weight_decay=weight_decay, dropout=dropout
        )
        self.block3 = DecoderBlock(
            filters=128, kernel=(3, 3), strides=(2, 2), weight_decay=weight_decay, dropout=dropout
        )
        self.block4 = DecoderBlock(
            filters=64, kernel=(3, 3), strides=(2, 2), weight_decay=weight_decay, dropout=dropout
        )
        self.final_layer = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            kernel_initializer="he_normal",
            padding="same",
            activation=None,
            kernel_regularizer=None,
            data_format="channels_first",
        )

    def call(self, input: Tuple[tf.Tensor, Tuple[tf.Tensor, ...]], training: bool = False) -> tf.Tensor:
        """The call function."""
        features, (skip1, skip2, skip3, skip4) = input

        features = self.block1((skip4, features), training=training)
        features = self.block2((skip3, features), training=training)
        features = self.block3((skip2, features), training=training)
        features = self.block4((skip1, features), training=training)
        output = self.final_layer(features)
        return output


class Unet(Model):
    """U-net CNN.

    Note: see https://arxiv.org/abs/1505.04597 for details.
    We changed final layer to output depth instead of semseg.

    Args:
        weight_decay: weight decay
        dropout: dropout
    """

    def __init__(self, weight_decay: float = 0.0, dropout: float = 0.3) -> None:
        """Initialization."""
        super().__init__()
        self.encoder = UnetEncoder(weight_decay=weight_decay, dropout=dropout)
        self.bottleneck = DoubleConv(filters=1024, kernel=(3, 3), weight_decay=weight_decay)
        self.decoder = UnetDecoder(weight_decay=weight_decay, dropout=dropout)

    def call(self, input: tf.Tensor, training: bool = False) -> tf.Tensor:
        """The call function."""
        # Normalize image
        input /= 255.0
        features, skip_connections = self.encoder(input, training=training)
        features = self.bottleneck(features)
        output = self.decoder((features, skip_connections), training=training)

        return output
