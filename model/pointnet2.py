"""
model/pointnet2.py
------------------
PointNet++ for point-wise semantic segmentation.

Supported input features:
    use_normals=False, use_rgb=False  ->  in_feat=0  -> input (B,N,3)
    use_normals=True,  use_rgb=False  ->  in_feat=3  -> input (B,N,6)
    use_normals=False, use_rgb=True   ->  in_feat=3  -> input (B,N,6)
    use_normals=True,  use_rgb=True   ->  in_feat=6  -> input (B,N,9)

The only change from the previous version is that in_feat
can now be 0, 3, or 6 depending on the active feature combination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Geometric utilities
# -----------------------------------------------------------------------------

def square_distance(src, dst):
    dist = -2 * torch.bmm(src, dst.permute(0, 2, 1))
    dist += (src ** 2).sum(-1, keepdim=True)
    dist += (dst ** 2).sum(-1, keepdim=True).permute(0, 2, 1)
    return dist.clamp(min=0)


def farthest_point_sample(xyz, npoint):
    B, N, _ = xyz.shape
    device   = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance  = torch.full((B, N), float("inf"), device=device)
    farthest  = torch.randint(0, N, (B,), device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest].unsqueeze(1)
        dist     = ((xyz - centroid) ** 2).sum(-1)
        distance = torch.min(distance, dist)
        farthest = distance.argmax(-1)
    return centroids


def index_points(points, idx):
    B      = points.shape[0]
    device = points.device
    vs = list(idx.shape); vs[1:] = [1] * (len(vs) - 1)
    rs = list(idx.shape); rs[0]  = 1
    bi = torch.arange(B, dtype=torch.long, device=device).view(vs).repeat(rs)
    return points[bi, idx, :]


def ball_query(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device  = xyz.device
    group_idx = torch.arange(N, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists  = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0:1].expand_as(group_idx)
    group_idx[group_idx == N] = group_first[group_idx == N]
    return group_idx


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

class SharedMLP(nn.Module):
    def __init__(self, in_ch, out_chs, dim=1):
        super().__init__()
        layers = []
        for o in out_chs:
            C  = nn.Conv1d  if dim == 1 else nn.Conv2d
            BN = nn.BatchNorm1d if dim == 1 else nn.BatchNorm2d
            layers += [C(in_ch, o, 1, bias=False), BN(o), nn.ReLU(inplace=True)]
            in_ch = o
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint  = npoint
        self.radius  = radius
        self.nsample = nsample
        self.mlp     = SharedMLP(in_channel, mlp, dim=2)

    def forward(self, xyz, points):
        new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))
        idx     = ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped = index_points(xyz, idx) - new_xyz.unsqueeze(2)
        if points is not None:
            grouped = torch.cat([grouped, index_points(points, idx)], dim=-1)
        new_pts = self.mlp(grouped.permute(0, 3, 2, 1)).max(2)[0].permute(0, 2, 1)
        return new_xyz, new_pts


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp = SharedMLP(in_channel, mlp, dim=1)

    def forward(self, xyz1, xyz2, points1, points2):
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interp = points2.repeat(1, N, 1)
        else:
            dists, idx = square_distance(xyz1, xyz2).sort(-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            w     = 1.0 / (dists + 1e-8)
            w     = w / w.sum(-1, keepdim=True)
            interp = (index_points(points2, idx) * w.unsqueeze(-1)).sum(2)
        new_pts = torch.cat([points1, interp], -1) if points1 is not None else interp
        return self.mlp(new_pts.permute(0, 2, 1)).permute(0, 2, 1)


# -----------------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------------

class PointNet2Segmentation(nn.Module):
    """
    PointNet++ for binary bark/wood segmentation.

    Args:
        num_classes:  number of classes (2: wood and bark)
        use_normals:  use normals as additional features (+3)
        use_rgb:      use scanner RGB as additional features (+3)

    in_feat combinations:
        use_normals=True,  use_rgb=False  ->  in_feat=3  (previous, compatible)
        use_normals=True,  use_rgb=True   ->  in_feat=6  (new with RGB)
        use_normals=False, use_rgb=True   ->  in_feat=3
        use_normals=False, use_rgb=False  ->  in_feat=0
    """

    def __init__(self, num_classes=2, use_normals=True, use_rgb=False):
        super().__init__()
        self.num_classes = num_classes
        self.use_normals = use_normals
        self.use_rgb     = use_rgb

        # Compute additional features (everything except XYZ)
        in_feat = 0
        if use_normals: in_feat += 3
        if use_rgb:     in_feat += 3

        # ── Encoder: 3 SetAbstraction layers ──────────────────────────────
        self.sa1 = SetAbstraction(1024, 0.1, 32,  3 + in_feat, [32, 32, 64])
        self.sa2 = SetAbstraction(256,  0.2, 64,  3 + 64,      [64, 64, 128])
        self.sa3 = SetAbstraction(64,   0.4, 128, 3 + 128,     [128, 128, 256])

        # ── Decoder: 3 FeaturePropagation layers ────────────────────────
        self.fp3 = FeaturePropagation(256 + 128, [256, 256])
        self.fp2 = FeaturePropagation(256 + 64,  [256, 128])
        self.fp1 = FeaturePropagation(128 + in_feat, [128, 128])

        # ── Classification head ──────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, xyz_feat):
        """
        Args:
            xyz_feat: (B, N, 3)  xyz only
                      (B, N, 6)  xyz + normals  OR  xyz + rgb
                      (B, N, 9)  xyz + normals + rgb

        Returns:
            logits: (B, N, num_classes)
        """
        xyz = xyz_feat[:, :, :3].contiguous()

        # Extract additional features based on configuration
        if self.use_normals and self.use_rgb:
            # (B, N, 9) -> normals=cols 3-5, rgb=cols 6-8
            f0 = xyz_feat[:, :, 3:].contiguous()    # (B, N, 6)
        elif self.use_normals:
            # (B, N, 6) -> normals=cols 3-5
            f0 = xyz_feat[:, :, 3:6].contiguous()   # (B, N, 3)
        elif self.use_rgb:
            # (B, N, 6) -> rgb=cols 3-5
            f0 = xyz_feat[:, :, 3:6].contiguous()   # (B, N, 3)
        else:
            f0 = None

        # Encoder
        x1, f1 = self.sa1(xyz, f0)
        x2, f2 = self.sa2(x1,  f1)
        x3, f3 = self.sa3(x2,  f2)

        # Decoder
        f2 = self.fp3(x2, x3, f2, f3)
        f1 = self.fp2(x1, x2, f1, f2)
        f0 = self.fp1(xyz, x1, f0, f1)

        # Head
        return self.head(f0.permute(0, 2, 1)).permute(0, 2, 1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Loss function
# -----------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for bark/wood class imbalance.
    gamma=2.0 as per original paper (Lin et al. 2017).
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Ensure tensors are contiguous before processing
        logits = logits.contiguous()
        targets = targets.contiguous()
        
        B, N, C  = logits.shape
        # Use reshape instead of view to avoid stride errors
        lf       = logits.reshape(-1, C)
        tf       = targets.reshape(-1)
        
        lp       = F.log_softmax(lf, -1)
        pt       = torch.exp(lp).gather(1, tf.unsqueeze(-1)).squeeze(-1)
        ce       = F.nll_loss(lp, tf, weight=self.alpha, reduction="none")
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss(class_weights=None, use_focal=True):
    if use_focal:
        return FocalLoss(alpha=class_weights, gamma=2.0)
    return nn.CrossEntropyLoss(weight=class_weights)