"""The loss function."""
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

import torch
import torch.nn as nn
from typing import Optional

class HuberLoss(nn.Module):
    """This is implementation of the Huber loss.

    Note: see https://en.wikipedia.org/wiki/Huber_loss

    Args:
        treshold: treshold regualte ratio between L1 and L2 losses
    """
    def __init__(self, treshold: float = 1) -> None:
        """Initialization."""
        super().__init__()
        self.treshold = treshold

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """The call fuction."""
        error = inputs - targets
        is_small_error = torch.abs(error) <= self.treshold
        small_error_loss = torch.square(error) / 2.0
        big_error_loss = self.treshold * (torch.abs(error) - (0.5 * self.treshold))
        values =  torch.where(is_small_error, small_error_loss, big_error_loss)

        return values.sum()

