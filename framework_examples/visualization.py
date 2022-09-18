"""Visualization script."""
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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def depth_to_image(images:np.ndarray, depth:np.ndarray, alpha:float=0.4)->np.ndarray:
    """Creates blended image with depth map.

    Args:
       images: input images batch of shape [batch, 3, height, width]
       depth: depth batch of shape [batch, 1, height, width]

    Returns:
        tinted_image: blended depth image [batch, height, width, channels]
    """
    tinted_images = []
    cmap = plt.cm.plasma

    for img, d_img in zip(images, depth):
        depth_relative = d_img / (np.percentile(d_img, 95) + 1e-8)
        d_image = 255.0 * cmap(np.squeeze(np.clip(depth_relative, 0., 1.0)))[..., :3]
        img = np.transpose(img, (1, 2, 0))
        tinted_img = alpha * img + (1.0 - alpha) * d_image
        tinted_images.append(tinted_img)
    return np.asarray(tinted_images)