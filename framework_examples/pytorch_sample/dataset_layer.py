"""The data loading layer."""
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
import torch
from torch.utils import data
import cv2
import numpy as np
from torch.utils.data import Dataset
from typing import NamedTuple, Tuple
from framework_examples.file_io import load_dataset_list, load_image
from framework_examples.parameters import Parameters


class DepthDatasetPytorch(Dataset):
    """This is dataset layer for pytorch.

    Args:
        dataset_basepath: the full path to the dataset folder
        dataset_name: the name of the dataset list (e.g. train.datatxt, val.datatxt, test.datatxt)
        resolution: resolution of the data
    """
    def __init__(
        self,
        dataset_basepath: str,
        dataset_name: str,
        resolution: NamedTuple("input_resolution", (("width", int), ("height", int))),
    ) -> None:
        """Initialization."""
        super().__init__()
        self.dataset_basepath = dataset_basepath
        self.dataset_name = dataset_name

        self.input_height = resolution.height
        self.input_width = resolution.width

        self.dataset_list = load_dataset_list(self.dataset_basepath, self.dataset_name)


    def __len__(self) -> float:
        """Returns lens of the dataset."""
        return len(self.dataset_list)
    
    def __getitem__(self, idx: int)->Tuple[torch.Tensor, torch.Tensor]:
        """Returns item given index."""
        img_path, label_path = self.dataset_list[idx]
        return self.load_data(img_path, label_path)

    def load_data(self, image_path:str, depth_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads data from string."""

        image_path = os.path.join(self.dataset_basepath, image_path)
        depth_path = os.path.join(self.dataset_basepath, depth_path)
        image, depth = load_image(image_path), load_image(depth_path)

        # resize
        dims = (self.input_width, self.input_height)
        image = cv2.resize(image, dims, interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, dims, interpolation=cv2.INTER_NEAREST)

        # Postprocess depth
        depth /= 100.0  # Make depth in meters
        disparity = 1.0 / depth

        # By default we train channels first
        image = np.transpose(image, (2, 0, 1))
        disparity = np.expand_dims(disparity, axis=0)
        sample = (torch.from_numpy(image), torch.from_numpy(disparity))
        return sample

if __name__=="__main__":
    """Example of dataset run."""
    parameters = Parameters()

    demo_dataset = DepthDatasetPytorch(
        dataset_basepath=parameters.dataset_basepath,
        dataset_name="train.datatxt",
        resolution=parameters.resolution,
    )
    demo_data_loader = data.DataLoader(
        demo_dataset, batch_size=parameters.batchsize, shuffle=False, drop_last=True, num_workers=12, prefetch_factor=32)

    for image, disparity in demo_data_loader:
        print(f"image {image.shape}, disparity {disparity.shape}")

