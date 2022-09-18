"""The training script."""
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
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from framework_examples.parameters import Parameters
from framework_examples.tensorflow_sample.dataset_layer import \
    DepthDatasetTensorflow
from framework_examples.tensorflow_sample.loss import HuberLoss
from framework_examples.tensorflow_sample.metrics import AbsoluteRelativeError
from framework_examples.tensorflow_sample.model import Unet
from framework_examples.visualization import depth_to_image
from tqdm import tqdm


def setup_gpu():
    """The function sets up gpu to not allocate all memory."""
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        # Will not allocate all memory but only necessary amount
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


@tf.function
def train_step(
    optimizer: tf.keras.optimizers.Optimizer,
    model: tf.keras.Model,
    images: tf.Tensor,
    labels: tf.Tensor,
    loss_object: tf.keras.losses.Loss,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """This is single training step."""

    with tf.GradientTape() as tape:
        prediction = model(images, training=True)
        loss_value = loss_object(y_true=labels, y_pred=prediction)

    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return prediction, loss_value


@tf.function
def val_step(
    model: tf.keras.Model, images: tf.Tensor, labels: tf.Tensor, loss_object: tf.keras.losses.Loss
) -> Tuple[tf.Tensor, tf.Tensor]:
    """The validation step."""
    val_prediction = model(images, training=False)
    val_loss = loss_object(y_true=labels, y_pred=val_prediction)
    return val_prediction, val_loss


def train_for_one_epoch(
    train_dataset: tf.data.Dataset,
    optimizer: tf.keras.optimizers.Optimizer,
    model: tf.keras.Model,
    loss_object: tf.keras.losses.Loss,
    metric: tf.keras.metrics.Metric,
    summary_dir: str,
) -> List[float]:
    """This is training loop for one epoch."""
    losses, data = [], {}
    writer = tf.summary.create_file_writer(os.path.join(summary_dir, "train"))

    for (images, labels) in tqdm(train_dataset.dataset, total=train_dataset.num_it_per_epoch, desc="training"):

        prediction, loss_value = train_step(optimizer, model, images, labels, loss_object)
        # Here summaries on the batch end
        with writer.as_default():
            tf.summary.scalar("Loss", loss_value, step=optimizer.iterations)

        losses.append(loss_value)
        data = {"images": images.numpy(), "y_true": labels.numpy(), "y_pred": prediction.numpy()}
        # Update metrics
        metric.update_state(y_true=labels, y_pred=prediction)

    # Here summaries on the epoch end
    with writer.as_default():
        groundtruth = depth_to_image(data["images"], data["y_true"]) / 255.0
        prediction = depth_to_image(data["images"], data["y_pred"]) / 255.0
        tf.summary.image("Groundtruth", groundtruth, step=optimizer.iterations)
        tf.summary.image("Prediction", prediction, step=optimizer.iterations)

    return losses


def val_for_one_epoch(
    val_dataset: tf.data.Dataset,
    model: tf.keras.Model,
    loss_object: tf.keras.losses.Loss,
    metric: tf.keras.metrics.Metric,
) -> List[float]:
    """This is validation step"""
    losses = []
    for images, labels in tqdm(val_dataset.dataset, total=val_dataset.num_it_per_epoch, desc="validation"):
        val_prediction, val_loss = val_step(model, images, labels, loss_object)
        losses.append(val_loss)
        metric.update_state(y_true=labels, y_pred=val_prediction)
    return losses


def train():
    # Setup gpu
    setup_gpu()
    # Set global seed to make reproducable experiments
    tf.random.set_seed(2022)

    # get parameters
    param = Parameters()

    # Load dataset
    train_dataset = DepthDatasetTensorflow(
        dataset_basepath=param.dataset_basepath,
        dataset_name="train.datatxt",
        batchsize=param.batchsize,
        resolution=param.resolution,
        shuffle=True,
    )
    val_dataset = DepthDatasetTensorflow(
        dataset_basepath=param.dataset_basepath,
        dataset_name="val.datatxt",
        batchsize=param.batchsize,
        resolution=param.resolution,
        shuffle=False,
    )

    # Initialize model
    model = Unet(weight_decay=param.weight_decay)
    model.build((1, 3, param.resolution.height, param.resolution.width))
    model.summary()

    model_path = os.path.join(param.checkpoint_dir, "{model}-{epoch:04d}")

    # Set up scheduler and optimizer
    scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=param.learning_rate, first_decay_steps=train_dataset.num_it_per_epoch
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)

    # Set up the metric for evaluation
    train_abs_rel_metric = AbsoluteRelativeError()
    val_abs_rel_metric = AbsoluteRelativeError()
    # Set up loss
    loss_object = HuberLoss(treshold=1.0)

    # Main training and evaluation loop
    for epoch in range(param.epochs):
        print(f"Start of epoch {epoch}")
        save_dir = model_path.format(model=model.name, epoch=epoch)
        losses_train = train_for_one_epoch(
            train_dataset, optimizer, model, loss_object, train_abs_rel_metric, param.summary_dir
        )
        train_error = train_abs_rel_metric.result()

        losses_val = val_for_one_epoch(val_dataset, model, loss_object, val_abs_rel_metric)
        val_err = val_abs_rel_metric.result()

        losses_train_mean, losses_val_mean = np.mean(losses_train), np.mean(losses_val)
        print(
            f"\n Epoch {epoch}: Train loss: {losses_train_mean:.4f}  Validation Loss: {losses_val_mean:.4f}, Train abs rel err: {train_error:.4f}, Validation abs rel err {val_err:.4f}"
        )
        model.save(save_dir)
        train_abs_rel_metric.reset_state()
        val_abs_rel_metric.reset_state()


if __name__ == "__main__":
    train()
