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
import random
from typing import List

import numpy as np
import torch
import torchvision
from framework_examples.parameters import Parameters
from framework_examples.pytorch_sample.dataset_layer import DepthDatasetPytorch
from framework_examples.pytorch_sample.loss import HuberLoss
from framework_examples.pytorch_sample.metrics import AbsoluteRelativeError
from framework_examples.pytorch_sample.model import Unet
from framework_examples.visualization import depth_to_image
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric
from tqdm import tqdm


def set_random_seed(seed: int, deterministic: bool = True, benchmark: bool = True) -> None:
    """Set random seed.

    Note: Deterministic operations are often slower than nondeterministic operations,
        so single-run performance may decrease for your model.
        For further details see:
        https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        seed: random seed
        deterministic: if true pushes to get deterministic algorithms
        benchmark: if True fast and non-deterministics (if deterministic=False) otherwise False
        (Note: benchmark makes autotune of the trainig process thus first epoch can be slower,
         it should be false when using multiscale input or stochastic depth networks)
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Disabling the benchmarking feature causes cuDNN
    # to deterministically select an algorithm,
    # possibly at the cost of reduced performance
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id: int) -> None:
    """Seed workers.

    Note: Use worker_init_fn() to preserve same shuffling in dataloader
        For further details see:
        https://pytorch.org/docs/stable/notes/randomness.html
    Args:
        worker_id: the id of worker
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_for_one_epoch(
    train_dataloader: data.DataLoader,
    model: nn.Module,
    loss_object: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    device: str,
    metrics: Metric,
    summary_dir: str,
) -> List[float]:
    """This is training for one epoch."""
    model.train()
    losses, data = [], {}
    log_dir = os.path.join(summary_dir, "train")
    steps_per_epoch = len(train_dataloader)
    global_step = epoch * steps_per_epoch
    writer = SummaryWriter(log_dir=log_dir)
    for step, sample in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch}", total=steps_per_epoch):
        images, labels = sample[0].to(device), sample[1].to(device)

        # Compute prediction error
        prediction = model(images)
        loss_value = loss_object(prediction, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        scheduler.step(epoch + step / steps_per_epoch)

        metrics.update(preds=prediction, target=labels)
        losses.append(loss_value)

        data = {"images": images, "y_true": labels, "y_pred": prediction}

        global_step = epoch * steps_per_epoch + step

        # Here summaries on the batch end
        writer.add_scalar("loss", loss_value, global_step)

    # Here summaries on the epoch end
    images, y_true, y_pred = (
        data["images"].cpu().detach().numpy(),
        data["y_true"].cpu().detach().numpy(),
        data["y_pred"].cpu().detach().numpy(),
    )
    # Transpose to channels first
    groundtruth = np.transpose(depth_to_image(images, y_true), (0, 3, 1, 2)) / 255.0
    prediction = np.transpose(depth_to_image(images, y_pred), (0, 3, 1, 2)) / 255.0
    grid_gt = torchvision.utils.make_grid(torch.from_numpy(groundtruth))
    grid_pred = torchvision.utils.make_grid(torch.from_numpy(prediction))

    writer.add_image("Ground truth", grid_gt, global_step)
    writer.add_image("Prediction", grid_pred, global_step)
    writer.close()
    return losses


def val_for_one_epoch(
    val_dataloader: data.DataLoader, model: nn.Module, loss_object: nn.Module, device: str, metrics: Metric
) -> List[float]:
    """This is validation step."""
    model.eval()
    losses = []
    with torch.no_grad():
        for batch, sample in tqdm(enumerate(val_dataloader), desc=f"Validation", total=len(val_dataloader)):
            images, labels = sample[0].to(device), sample[1].to(device)
            prediction = model(images)
            loss = loss_object(prediction, labels)
            losses.append(loss)
            metrics.update(preds=prediction, target=labels)
    return losses


def train() -> None:
    # Set up gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")

    # Set random seed
    set_random_seed(seed=2022)

    # get parameters
    param = Parameters()

    # Load datasets
    train_dataset = DepthDatasetPytorch(
        dataset_basepath=param.dataset_basepath,
        dataset_name="train.datatxt",
        resolution=param.resolution,
    )
    train_data_loader = data.DataLoader(
        train_dataset,
        batch_size=param.batchsize,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        prefetch_factor=2,
        worker_init_fn=seed_worker,
    )

    val_dataset = DepthDatasetPytorch(
        dataset_basepath=param.dataset_basepath,
        dataset_name="val.datatxt",
        resolution=param.resolution,
    )
    val_data_loader = data.DataLoader(
        val_dataset, batch_size=param.batchsize, shuffle=False, drop_last=True, num_workers=2, prefetch_factor=2
    )

    # Initialize model
    model = Unet().to(device)
    model_path = os.path.join(param.checkpoint_dir, "{model}-{epoch:04d}")

    optimizer = torch.optim.Adam(model.parameters(), lr=param.learning_rate, weight_decay=param.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1, T_mult=1, eta_min=0, last_epoch=-1, verbose=False
    )
    # Set up the metric for evaluation
    train_abs_rel_metric = AbsoluteRelativeError().to(device)
    val_abs_rel_metric = AbsoluteRelativeError().to(device)
    # Set up loss
    loss_object = HuberLoss(treshold=1.0)

    # Main training and evaluation loop
    for epoch in range(param.epochs):
        print(f"Start of epoch {epoch}")
        save_dir = model_path.format(model=model.name, epoch=epoch)
        os.makedirs(save_dir, exist_ok=True)
        losses_train = train_for_one_epoch(
            train_dataloader=train_data_loader,
            model=model,
            loss_object=loss_object,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            device=device,
            metrics=train_abs_rel_metric,
            summary_dir=param.summary_dir,
        )
        train_error = train_abs_rel_metric.compute()

        losses_val = val_for_one_epoch(
            val_dataloader=val_data_loader,
            model=model,
            loss_object=loss_object,
            device=device,
            metrics=val_abs_rel_metric,
        )
        val_err = val_abs_rel_metric.compute()

        losses_train_mean, losses_val_mean = torch.mean(torch.stack(losses_train)), torch.mean(torch.stack(losses_val))
        print(
            f"\n Epoch {epoch}: Train loss: {losses_train_mean:.4f}  Validation Loss: {losses_val_mean:.4f}, Train abs rel err: {train_error:.4f}, Validation abs rel err {val_err:.4f}"
        )

        torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
        train_abs_rel_metric.reset()
        val_abs_rel_metric.reset()


if __name__ == "__main__":
    train()
