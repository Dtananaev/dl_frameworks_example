"""Creates dataset list."""
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
import argparse
import glob
import os
from typing import List

import numpy as np


class CreateDataList:
    """The class creates list of pairs image and label.

    Args:
        dataset_dir: the directory with data
    """

    def __init__(self, dataset_dir: str) -> None:
        """Initialize."""
        self.dataset_dir = dataset_dir

    def _remove_global_path(self, data: str) -> str:
        """The function removes dataset path from string."""
        return data[len(self.dataset_dir) + 1 :]

    def _get_data(self) -> List[str]:
        """Gets data list from given dataset directory."""

        search_string = os.path.join(self.dataset_dir, "images", "*.png")
        images_list = np.asarray(sorted(glob.glob(search_string)))

        images_list = np.asarray([self._remove_global_path(x) for x in images_list])
        search_string = os.path.join(self.dataset_dir, "depth", "*.png")
        depth_list = np.asarray(sorted(glob.glob(search_string)))
        depth_list = np.asarray([self._remove_global_path(x) for x in depth_list])

        data = np.concatenate(
            (
                images_list[:, None],
                depth_list[:, None],
            ),
            axis=1,
        )
        data = [";".join(x) for x in data]
        return data

    def create_datasets_file(self, split: str = "train") -> None:
        """Creates dataset list file.

        Args:
            split: train, val, test split
        """
        data_list = self._get_data()

        # Save dataset
        filename = os.path.join(self.dataset_dir, f"{split}.datatxt")

        with open(filename, "w") as f:
            for item in data_list:
                f.write("%s\n" % item)
        print(f"The dataset of the size {len(data_list)} saved in {filename}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset file.")
    parser.add_argument("--dataset_dir", default="../dataset")
    args = parser.parse_args()
    dataset_creator = CreateDataList(args.dataset_dir)

    splits = ["train", "val", "test"]
    for split in splits:
        # Note for demo purposes all splits contain the same data
        dataset_creator.create_datasets_file(split=split)
