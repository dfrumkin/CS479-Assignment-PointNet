import einx
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)

        # Followed the original implementation to initialize a matrix as I.
        identity = Variable(torch.eye(self.k, dtype=torch.float)).reshape(1, self.k * self.k).expand(B, -1).to(device)
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """

    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # DONE_TODO : Implement point-wise mlp model based on PointNet Architecture.
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())

    @jaxtyped(typechecker=beartype)
    def forward(
        self, pointcloud: Float[Tensor, "b n 3"]
    ) -> tuple[Float[Tensor, "b 1024"], Float[Tensor, "b 3 3"] | None, Float[Tensor, "b 64 64"] | None]:
        """Compute the global feature vector for the point cloud.
        Args:
            pointcloud (Float[Tensor, "b n 3"]): Point cloud with n points.
        Returns:
            Float[Tensor, "b 1024"]: Global feature.
            Float[Tensor, "b 3 3"] | None: Input transform if used.
            Float[Tensor, "b 64 64"] | None: Feature transform if used.
        """
        # DONE_TODO : Implement forward function.
        x = einx.rearrange("b n c -> b c n", pointcloud, c=3)

        if self.input_transform:
            t3 = self.stn3(x)
            x = einx.dot("b i j, b j n -> b i n", t3, x, i=3, j=3)
        else:
            t3 = None

        x = self.conv1(x)

        if self.feature_transform:
            t64 = self.stn64(x)
            x = einx.dot("b i j, b j n -> b i n", t64, x, i=64, j=64)
        else:
            t64 = None

        x = self.conv3(self.conv2(x))

        feature = einx.max("b c n -> b c", x, c=1024)[0]

        return feature, t3, t64


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes

        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)

        # returns the final logits from the max-pooled features.
        # DONE_TODO : Implement MLP that takes global feature as an input and return logits.
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Should be configurable
            nn.Linear(256, num_classes),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, pointcloud: Float[Tensor, "b n 3"]
    ) -> tuple[Float[Tensor, "b num_classes"], Float[Tensor, "b 3 3"] | None, Float[Tensor, "b 64 64"] | None]:
        """Compute the logits for the point cloud.
        Args:
            pointcloud (Float[Tensor, "b n 3"]): Point cloud with n points.
        Returns:
            Float[Tensor, "b num_classes"]: Logits.
            Float[Tensor, "b 3 3"] | None: Input transform if used.
            Float[Tensor, "b 64 64"] | None: Feature transform if used.
        """
        # DONE_TODO : Implement forward function.
        feature, t3, t64 = self.pointnet_feat(pointcloud)
        logits = self.fc(feature)
        return logits, t3, t64


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        pass

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        pass


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat()

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.
        pass


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
