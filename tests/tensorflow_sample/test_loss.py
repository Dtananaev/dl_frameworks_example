"""The test for loss function."""
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

import os
import pytest 
import numpy as np
import tensorflow as tf
from framework_examples.tensorflow_sample.loss import HuberLoss
from framework_examples.file_io import load_image


@pytest.fixture(name="fake_groundtruth_data")
def fixture_fake_groundtruth_data()->np.ndarray:
    """Returns fake groundtruth depth."""
    depth_groundtruth = np.asarray([[1.0, 2.2, 3.0, np.nan],[2.0, 1.0, 3.0, 4.0], [5.0, 6.2, 7., 89.]])
    return depth_groundtruth


@pytest.fixture(name="real_groundtruth_data")
def fixture_real_groundtruth_data(data_dir: str)-> np.ndarray:
    """Returns real gt data."""
    path = os.path.join(data_dir, "depth", "test_label_frame.png")
    depth_groundtruth = load_image(path)
    return depth_groundtruth


def test_loss_fake_data(fake_groundtruth_data: np.ndarray)-> None:
    """Test loss with fake data."""

    # GIVEN huber loss function and prediction
    huber_loss = HuberLoss(treshold=1)
    prediction = np.ones((fake_groundtruth_data.shape))

    # WHEN computing result
    result = huber_loss(y_true=fake_groundtruth_data, y_pred=prediction)

    # THEN the output should be tensorflow tensor
    assert isinstance(result, tf.Tensor)
    # THEN result should be nan
    assert np.isnan(result)


def test_loss_real_data(real_groundtruth_data: np.ndarray)-> None:
    """Test loss with real data."""

    # GIVEN huber loss function and prediction
    huber_loss = HuberLoss(treshold=1)
    prediction = np.ones((real_groundtruth_data.shape))

    # WHEN computing result
    result = huber_loss(y_true=real_groundtruth_data, y_pred=prediction)

    # THEN the output should be tensorflow tensor
    assert isinstance(result, tf.Tensor)
    # THEN result should not be nan
    assert not np.isnan(result)


