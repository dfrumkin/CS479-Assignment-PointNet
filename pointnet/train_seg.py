import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
import wandb
from dataloaders.shapenet_partseg import get_data_loaders
from model import PointNetPartSeg, get_orthogonal_loss
from tqdm import tqdm
from utils.metrics import Accuracy, mIoU
from utils.misc import save_samples
from utils.model_checkpoint import CheckpointManager


def step(points, pc_labels, class_labels, model):
    """
    Input :
        - points [B, N, 3]
        - ground truth pc_labels [B, N]
        - ground truth class_labels [B]
    Output : loss
        - loss []
        - logits [B, C, N] (C: num_class)
        - preds [B, N]
    """

    # DONE_TODO : Implement step function for segmentation.
    logits, t3, t64 = model(points)
    loss = F.cross_entropy(logits, pc_labels)
    preds = logits.argmax(dim=1)

    if t3 is not None:
        loss += get_orthogonal_loss(t3)
    if t64 is not None:
        loss += get_orthogonal_loss(t64)

    return loss, logits, preds


def train_step(points, pc_labels, class_labels, model, optimizer, train_acc_metric):
    loss, logits, preds = step(points, pc_labels, class_labels, model)
    train_batch_acc = train_acc_metric(preds, pc_labels.to(device))

    # DONE_TODO : Implement backpropagation using optimizer and loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss, train_batch_acc


def validation_step(points, pc_labels, class_labels, model, val_acc_metric, val_iou_metric):
    loss, logits, preds = step(points, pc_labels, class_labels, model)
    val_batch_acc = val_acc_metric(preds, pc_labels)
    val_batch_iou, masked_preds = val_iou_metric(logits, pc_labels, class_labels)

    return loss, masked_preds, val_batch_acc, val_batch_iou


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

    model = PointNetPartSeg()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.5)
    if args.save:
        checkpoint_manager = CheckpointManager(
            dirpath=datetime.now().strftime("checkpoints/segmentation/%m-%d_%H-%M-%S"),
            metric_name="val_iou",
            mode="max",
            topk=2,
            verbose=True,
        )

    # It will download Shapenet Dataset at the first time.
    (train_ds, val_ds, test_ds), (train_dl, val_dl, test_dl) = get_data_loaders(
        data_dir="./data", batch_size=args.batch_size, phases=["train", "val", "test"]
    )

    train_acc_metric = Accuracy()
    val_acc_metric = Accuracy()
    val_iou_metric = mIoU()

    for epoch in range(args.epochs):
        # training step
        model.train()
        pbar = tqdm(train_dl)
        train_epoch_loss = []
        for points, pc_labels, class_labels in pbar:
            # DF: move to the right device - was missing
            points, pc_labels, class_labels = points.to(device), pc_labels.to(device), class_labels.to(device)

            train_batch_loss, train_batch_acc = train_step(
                points, pc_labels, class_labels, model, optimizer, train_acc_metric
            )
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
            for points, pc_labels, class_labels in val_dl:
                points, pc_labels, class_labels = points.to(device), pc_labels.to(device), class_labels.to(device)
                val_batch_loss, val_batch_masked_preds, val_batch_acc, val_batch_iou = validation_step(
                    points,
                    pc_labels,
                    class_labels,
                    model,
                    val_acc_metric,
                    val_iou_metric,
                )
                val_epoch_loss.append(val_batch_loss)

            val_epoch_loss = sum(val_epoch_loss) / len(val_epoch_loss)
            val_epoch_acc = val_acc_metric.compute_epoch()
            val_epoch_iou = val_iou_metric.compute_epoch()
            print(
                f"train loss: {train_epoch_loss:.4f} | train acc: {train_epoch_acc * 100:.1f}% | val loss: {val_epoch_loss:.4f} | val acc: {val_epoch_acc * 100:.1f}% | val mIoU: {val_epoch_iou * 100:.1f}%"
            )

            if args.save:
                checkpoint_manager.update(model, epoch, round(val_epoch_iou.item() * 100, 2), "Segmentation_ckpt")
        scheduler.step()

        # DF: Added logging to W&B
        wandb.log(
            {
                "loss/train": train_epoch_loss,
                "acc/train": train_epoch_acc,
                "loss/val": val_epoch_loss,
                "acc/val": val_epoch_acc,
                "iou/val": val_epoch_iou,
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

    # After training, test on testset
    if args.save:
        checkpoint_manager.load_best_ckpt(model, device)
    model.eval()
    with torch.no_grad():
        test_acc_metric = Accuracy()
        test_iou_metric = mIoU()
        for points, pc_labels, class_labels in test_dl:
            points, pc_labels, class_labels = points.to(device), pc_labels.to(device), class_labels.to(device)
            test_batch_loss, test_batch_masked_preds, test_batch_acc, test_batch_iou = validation_step(
                points,
                pc_labels,
                class_labels,
                model,
                test_acc_metric,
                test_iou_metric,
            )
        test_acc = test_acc_metric.compute_epoch()
        test_iou = test_iou_metric.compute_epoch()

        print(f"test acc: {test_acc * 100:.1f}% | test mIoU: {test_iou * 100:.1f}%")
        save_samples(points[4:8], pc_labels[4:8], test_batch_masked_preds[4:8], "segmentation_samples.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointNet ShapeNet Part Segmentation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    args.gpu = 0
    args.save = True

    main(args)
