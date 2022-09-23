"""The parameters file."""
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
import dataclasses
from collections import namedtuple
from typing import NamedTuple


@dataclasses.dataclass
class Parameters:
    """The parameters class."""

    # Path to the dataset folder
    dataset_basepath: str = "/home/deeplearning_workspace/dl_frameworks_example/dataset"

    # Summary dir
    summary_dir: str = "outputs/summary"
    # Checkpoint dir
    checkpoint_dir: str = "outputs/checkpoint"
    # batch size
    batchsize: int = 5

    # learning rate
    learning_rate = 1e-4
    # weight decay
    weight_decay: float = 1e-6

    # max number of epochs
    epochs: int = 10

    # input_resolution
    resolution: NamedTuple("input_resolution", (("width", int), ("height", int))) = namedtuple(
        "input_resolution", ["height", "width"]
    )(128, 256)
