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

import tensorflow as tf
import numpy as np
import os
import cv2
from framework_examples.parameters import Parameters
from framework_examples.file_io import load_dataset_list, load_image
from typing import List, Tuple, NamedTuple



class DepthDatasetTensorflow:
    """This is data loading layer for tensorflow 2.
    
    Args:
        dataset_basepath: the full path to the dataset folder
        dataset_name: the name of the dataset list (e.g. train.datatxt, val.datatxt, test.datatxt)
        batchsize: the size of batch
        shuffle: shuffle data if true
    """

    def __init__(self, dataset_basepath: str, dataset_name: str, batchsize: int, resolution: NamedTuple("input_resolution", (("width", int), ("height", int))), shuffle: bool=False)-> None:
        """Initialize."""
        self.dataset_basepath = dataset_basepath
        self.dataset_name = dataset_name
        self.batchsize = batchsize
        self.shuffle = shuffle

        self.input_height = resolution.height
        self.input_width = resolution.width


        # Get the data
        self.dataset_list = load_dataset_list(self.dataset_basepath, self.dataset_name)
        self.num_samples = len(self.dataset_list)
        self.num_it_per_epoch = int(self.num_samples /  self.batchsize)




        # Get data layer
        ds = tf.data.Dataset.from_tensor_slices(self.dataset_list)

        # Shuffle data if necessary
        if self.shuffle:
            ds = ds.shuffle(self.num_samples)

        # Load data
        ds = ds.map(
            map_func=lambda x: tf.py_function(
                self.load_data, [x], Tout=[tf.float32, tf.float32]
            ),
            num_parallel_calls=2,
        )
        # Make batch and prefetch in RAM samples
        ds = ds.batch(self.batchsize).prefetch(buffer_size=2)
        self.dataset = ds


    def load_data(self, data_input: List[str])-> Tuple[np.ndarray, np.ndarray]:
        """The function loads data.
        
        Args:
            data_input: the list with strings to image and label files

        Returns:
            image: image array
            disparity: disparity array
        """
        image_path, depth_path = data_input
        image_path = os.path.join(self.dataset_basepath, image_path.numpy().decode("utf-8"))
        depth_path = os.path.join(self.dataset_basepath, depth_path.numpy().decode("utf8"))
        image, depth = load_image(image_path), load_image(depth_path)

        # resize
        dims = (self.input_width, self.input_height)
        image = cv2.resize(image, dims, interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, dims, interpolation=cv2.INTER_NEAREST)
       
        # Postprocess depth
        depth /= 100.0 # Make depth in meters
        disparity = 1.0 / depth

        # By default we train channels first
        image = np.transpose(image, (2, 0, 1))
        disparity = np.expand_dims(disparity, axis=0)

        return image, disparity

if __name__ =="__main__":
    """Demo how to run the data layer."""
    parameters = Parameters()

    demo_dataset = DepthDatasetTensorflow(dataset_basepath=parameters.dataset_basepath,dataset_name="train.datatxt", batchsize=parameters.batchsize, resolution=parameters.resolution,shuffle=True)

    for image, disparity in demo_dataset.dataset:
        print(f"image {image.shape}, disparity {disparity.shape}")