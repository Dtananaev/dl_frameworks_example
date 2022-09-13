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
import tensorflow as tf
from tensorflow.keras import Model
from typing import Union, Tuple
from tensorflow.keras.regularizers import l2


class EncoderBlock(Model):
    """U-net encoder block.
    
    Note: see https://arxiv.org/abs/1505.04597

    Args:
        filters: number of convolutional fiters
        kernel: the size of convolutional kernel
        pool_size: the size of pooling kernel
        weight_decay: the weight decay regularization
        dropout: the dropout probability
    """

    def __init__(self, filters: int, kernel:Union[int, Tuple[int, int]], pool_size:Union[int, Tuple[int, int]], weight_decay:float=0.0, dropout:float=0.3)->None:
        """Initialization."""
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size= kernel, kernel_initializer="he_normal", padding="same", activation="relu", kernel_regularizer=l2(weight_decay), data_format="channels_first")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size= kernel, kernel_initializer="he_normal", padding="same", activation="relu", kernel_regularizer=l2(weight_decay), data_format="channels_first")
        self.pool = tf.keras.layers.MaxPool2D(pool_size=pool_size, data_format="channels_first")
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, input: tf.Tensor, training:bool=False)-> tf.Tensor:
        """The call function."""
        features = self.conv1(input)
        features = self.conv2(features)
        features = self.pool(features)
        # Note dropout works different in train and test time therefore training argument should be included
        features = self.dropout(features, training=training)
        return features


class Bottleneck(Model):
    """U-net bottleneck block.
    
    Note: see https://arxiv.org/abs/1505.04597

    Args:
        filters: number of convolutional fiters
        kernel: the size of convolutional kernel
        weight_decay: the weight decay regularization
    """
    def __init__(self, filters: int, kernel:Union[int, Tuple[int, int]],  weight_decay:float=0.0)->None:
        """Initialization."""
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size= kernel, kernel_initializer="he_normal", padding="same", activation="relu", kernel_regularizer=l2(weight_decay), data_format="channels_first")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size= kernel, kernel_initializer="he_normal", padding="same", activation="relu", kernel_regularizer=l2(weight_decay), data_format="channels_first")
        
    def call(self, input: tf.Tensor) -> tf.Tensor:
        """The call function."""
        features = self.conv1(input)
        features = self.conv2(features)
        return features


class DecoderBlock(Model):
    