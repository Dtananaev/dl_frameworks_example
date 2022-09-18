"""The helper functions for data loading and saving."""
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

from typing import List
import numpy as np
import cv2
import os

def load_dataset_list(dataset_basepath: str, dataset_filename: str, delimiter:str=";")-> List[str]:
    """Loads list of data from dataset file.

    Args:
     dataset_basepath: global path to the .datatxt file
     dataset_filename: the name of data file list

    Returns:
        dataset_list: the list with pairs of images and labels
    """

    file_path = os.path.join(dataset_basepath, dataset_filename)
    dataset_list = []
    with open(file_path) as f:
        dataset_list = f.readlines()
    dataset_list = [x.strip().split(delimiter) for x in dataset_list]
    return dataset_list

def load_image(filename: str)-> np.ndarray:
    """Load image as numpy array.
    
    Args:
        filename: filepath to the image

    Returns:
        image: numpy array float32 
    """
    image = np.asarray(cv2.imread(filename, -1), dtype=np.float32)
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


