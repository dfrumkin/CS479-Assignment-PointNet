import argparse
from datetime import datetime

# pytorch3d is non-trivial to install, so let us implement Chamfer distance by ourselves!
# from pytorch3d.loss.chamfer import chamfer_distance
import einx
import torch
import wandb
from beartype import beartype
from dataloaders.modelnet import get_data_loaders
from jaxtyping import Float, jaxtyped
from model import PointNetAutoEncoder
from torch import Tensor
from tqdm import tqdm
from utils.model_checkpoint import CheckpointManager


@jaxtyped(typechecker=beartype)
def chamfer_distance(
    x: Float[Tensor, "b n d"],
    y: Float[Tensor, "b m d"],
) -> Float[Tensor, ""]:
    """
    Symmetric Chamfer distance between two point clouds using squared L2 distance.

    Args:
        x: (B, N, D) point cloud
        y: (B, M, D) point cloud

    Returns:
        Scalar Chamfer distance (mean over batch).
    """

    # Compute ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y without materializing (b, n, m, d).
    x2 = einx.sum("b n d -> b n 1", x * x)
    y2 = einx.sum("b m d -> b 1 m", y * y)
    xy = einx.dot("b n d, b m d -> b n m", x, y)

    dist = (x2 + y2 - 2.0 * xy).clamp_min(0.0)

    # (B, N): for each point in x, nearest neighbor in y
    min_xy, _ = einx.min("b n m -> b n", dist)

    # (B, M): for each point in y, nearest neighbor in x
    min_yx, _ = einx.min("b n m -> b m", dist)

    # mean over points, then mean over batch
    loss_per_batch = einx.mean("b n -> b", min_xy) + einx.mean("b m -> b", min_yx)
    return loss_per_batch.mean()


def step(points, model):
    """
    Input :
        - points [B, N, 3]
    Output : loss
        - loss []
        - preds [B, N, 3]
    """

    # DONE_TODO : Implement step function for AutoEncoder.
    # Hint : Use chamferDist defined in above
    # Hint : You can compute chamfer distance between two point cloud pc1 and pc2 by chamfer_distance(pc1, pc2)

    points = points.to(device)
    _, preds = model(points)
    loss = chamfer_distance(preds, points)

    return loss, preds


def train_step(points, model, optimizer):
    loss, preds = step(points, model)

    # DONE_TODO : Implement backpropagation using optimizer and loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss, preds


def validation_step(points, model):
    loss, preds = step(points, model)

    return loss, preds


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

    model = PointNetAutoEncoder(num_points=2048)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.5)

    # automatically save only topk checkpoints.
    if args.save:
        checkpoint_manager = CheckpointManager(
            dirpath=datetime.now().strftime("checkpoints/auto_encoding/%m-%d_%H-%M-%S"),
            metric_name="val_loss",
            mode="min",
            topk=2,
            verbose=True,
        )

    (train_ds, val_ds, test_ds), (train_dl, val_dl, test_dl) = get_data_loaders(
        data_dir="./data", batch_size=args.batch_size, phases=["train", "val", "test"]
    )

    for epoch in range(args.epochs):
        # training step
        model.train()
        pbar = tqdm(train_dl)
        train_epoch_loss = []
        for points, _ in pbar:
            train_batch_loss, train_batch_preds = train_step(points, model, optimizer)
            train_epoch_loss.append(train_batch_loss)
            pbar.set_description(f"{epoch + 1}/{args.epochs} epoch | loss: {train_batch_loss:.4f}")

        train_epoch_loss = sum(train_epoch_loss) / len(train_epoch_loss)

        # validataion step
        model.eval()
        with torch.no_grad():
            val_epoch_loss = []
            for points, _ in val_dl:
                val_batch_loss, val_batch_preds = validation_step(points, model)
                val_epoch_loss.append(val_batch_loss)

            val_epoch_loss = sum(val_epoch_loss) / len(val_epoch_loss)
            print(f"train loss: {train_epoch_loss:.4f} | val loss: {val_epoch_loss:.4f}")

        if args.save:
            checkpoint_manager.update(model, epoch, round(val_epoch_loss.item(), 4), "AutoEncoding_ckpt")

        scheduler.step()

        # DF: Added logging to W&B
        wandb.log(
            {
                "loss/train": train_epoch_loss,
                "loss/val": val_epoch_loss,
                "lr": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

    if args.save:
        checkpoint_manager.load_best_ckpt(model, device)
    model.eval()
    with torch.no_grad():
        test_epoch_loss = []
        for points, _ in test_dl:
            test_batch_loss, test_batch_preds = validation_step(points, model)
            test_epoch_loss.append(test_batch_loss)

        test_epoch_loss = sum(test_epoch_loss) / len(test_epoch_loss)
        print(f"test loss: {test_epoch_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointNet ModelNet40 AutoEncoder")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    args.gpu = 0
    args.save = True

    main(args)
