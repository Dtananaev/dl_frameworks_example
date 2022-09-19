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

from torchmetrics import Metric
from typing import Optional, Any
import torch

class AbsoluteRelativeError(Metric):
    """Computes absolute relatie error.

    Note: see https://torchmetrics.readthedocs.io/en/stable/pages/implement.html#implement
    """

    def __init__(self, **kwargs: Optional[Any]) -> None:
        """Initialize."""
        super().__init__(**kwargs)
        self.add_state("abs_rel_err", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("batch_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Updates sate for abs_rel_error."""
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        values = torch.div(torch.abs(preds - target), target)
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        num_samples = torch.sum((values != 0.0).type(torch.FloatTensor))
        self.abs_rel_err += torch.sum(values) / num_samples
        self.batch_count += 1.0

    def compute(self):
        """Result."""
        return self.abs_rel_err.float() / self.batch_count