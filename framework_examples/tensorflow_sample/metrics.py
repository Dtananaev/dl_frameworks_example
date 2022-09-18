"""The metrics."""
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

from pickle import NONE
from typing import Any, Optional

import tensorflow as tf


class AbsoluteRelativeError(tf.keras.metrics.Metric):
    """Computes absolute relatie error.

    Note: see https://www.tensorflow.org/guide/keras/train_and_evaluate#custom_metrics

    Args:
        name: The metric name
    """

    def __init__(self, name: str = "absolute_relative_error", **kwargs: Optional[Any]) -> None:
        """Initialize."""
        super().__init__(name=name, **kwargs)

        self.abs_rel_error = self.add_weight(name="abs_rel_err", initializer="zeros")
        self.batch_count = self.add_weight(name="batch_count", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: tf.Tensor = None) -> None:
        """Updates sate for abs_rel_error."""

        values = tf.math.divide_no_nan(tf.abs(y_true - y_pred), y_true)
        num_samples = tf.reduce_sum(tf.cast(tf.math.not_equal(values, 0.0), self.dtype))

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)

        self.abs_rel_error.assign_add(tf.reduce_sum(values) / num_samples)
        self.batch_count.assign_add(1.0)

    def result(self) -> tf.Tensor:
        """Returns result."""
        return self.abs_rel_error / self.batch_count

    def reset_state(self) -> None:
        """Reset states."""
        self.abs_rel_error.assign(0.0)
        self.batch_count.assign(0.0)
