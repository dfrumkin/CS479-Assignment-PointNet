import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
import wandb
from dataloaders.modelnet import get_data_loaders
from model import PointNetCls, get_orthogonal_loss
from tqdm import tqdm
from utils.metrics import Accuracy
from utils.model_checkpoint import CheckpointManager


def step(points, labels, model):
    """
    Input :
        - points [B, N, 3]
        - ground truth labels [B]
    Output : loss
        - loss []
        - preds [B]
    """

    # DONE_TODO : Implement step function for classification.
    logits, t3, t64 = model(points)
    loss = F.cross_entropy(logits, labels)
    if t3 is not None:
        loss += get_orthogonal_loss(t3)
    if t64 is not None:
        loss += get_orthogonal_loss(t64)
    preds = logits.argmax(dim=1)
    return loss, preds


def train_step(points, labels, model, optimizer, train_acc_metric):
    loss, preds = step(points, labels, model)
    train_batch_acc = train_acc_metric(preds, labels.to(device))

    # DONE_TODO : Implement backpropagation using optimizer and loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss, train_batch_acc


def validation_step(points, labels, model, val_acc_metric):
    loss, preds = step(points, labels, model)
    val_batch_acc = val_acc_metric(preds, labels)

    return loss, val_batch_acc


def main(args):
    global device
    # DF: Modified to train on a Macbook
    # device = "cpu" if args.gpu == -1 else f"cuda:{args.gpu}"
    device = torch.device(
        f"cuda:{args.gpu}"
        if torch.cuda.is_available() and args.gpu >= 0
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # DF: Added for W&B tracking
    # Log in to WnB using WANDB_API_KEY or ~/.netrc
    wandb.login()

    wandb.init(
        project="cs479-assignment1-pointnet",
        config=args,
    )

    model = PointNetCls(num_classes=40, input_transform=True, feature_transform=True)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.5)

    # automatically save only topk checkpoints.
    if args.save:
        checkpoint_manager = CheckpointManager(
            # Specify the directory to save checkpoints.
            dirpath=datetime.now().strftime("checkpoints/classification/%m-%d_%H-%M-%S"),
            metric_name="val_acc",
            mode="max",
            # the number of checkpoints to save.
            topk=2,
            # Whether to maximize or minimize metric.
            verbose=True,
        )

    # It will download ModelNet dataset at the first time.
    (train_ds, val_ds, test_ds), (train_dl, val_dl, test_dl) = get_data_loaders(
        data_dir="./data", batch_size=args.batch_size, phases=["train", "val", "test"]
    )

    train_acc_metric = Accuracy()
    val_acc_metric = Accuracy()
    test_acc_metric = Accuracy()

    for epoch in range(args.epochs):
        # training step
        model.train()
        pbar = tqdm(train_dl)
        train_epoch_loss = []
        for points, labels in pbar:
            # DF: move to the right device - was missing
            points, labels = points.to(device), labels.to(device)

            train_batch_loss, train_batch_acc = train_step(points, labels, model, optimizer, train_acc_metric)
            train_epoch_loss.append(train_batch_loss)
            pbar.set_description(
                f"{epoch + 1}/{args.epochs} epoch | loss: {train_batch_loss:.4f} | accuracy: {train_batch_acc * 100:.1f}%"
            )

        train_epoch_loss = sum(train_epoch_loss) / len(train_epoch_loss)
        train_epoch_acc = train_acc_metric.compute_epoch()

        # validataion step
        model.eval()
        with torch.no_grad():
            val_epoch_loss = []
            for points, labels in val_dl:
                points, labels = points.to(device), labels.to(device)
                val_batch_loss, val_batch_acc = validation_step(points, labels, model, val_acc_metric)
                val_epoch_loss.append(val_batch_loss)

            val_epoch_loss = sum(val_epoch_loss) / len(val_epoch_loss)
            val_epoch_acc = val_acc_metric.compute_epoch()
            print(
                f"train loss: {train_epoch_loss:.4f} train acc: {train_epoch_acc * 100:.1f}% | val loss: {val_epoch_loss:.4f} val acc: {val_epoch_acc * 100:.1f}%"
            )

        if args.save:
            """
            Compare the current metric with history, and
            save ckpt only if the current metric is in topk.
            """
            checkpoint_manager.update(model, epoch, round(val_epoch_acc.item() * 100, 2), "Classification_ckpt")

        scheduler.step()

        # DF: Added logging to W&B
        wandb.log(
            {
                "loss/train": train_epoch_loss,
                "acc/train": train_epoch_acc,
                "loss/val": val_epoch_loss,
                "acc/val": val_epoch_acc,
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

    if args.save:
        checkpoint_manager.load_best_ckpt(model, device)
    model.eval()
    with torch.no_grad():
        for points, labels in test_dl:
            points, labels = points.to(device), labels.to(device)
            test_batch_loss, test_batch_acc = validation_step(points, labels, model, test_acc_metric)
        test_acc = test_acc_metric.compute_epoch()

        print(f"test acc: {test_acc * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointNet ModelNet40 Classification")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    args.gpu = 0
    args.save = True

    main(args)
