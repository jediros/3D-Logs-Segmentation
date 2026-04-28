from pathlib import Path
import numpy as np


def normalize_pointcloud(cloud):
    cloud = cloud.copy()
    centroid = cloud[:, :3].mean(axis=0)
    cloud[:, :3] -= centroid
    max_r = np.sqrt((cloud[:, :3] ** 2).sum(axis=1)).max()
    if max_r > 0:
        cloud[:, :3] /= max_r
    return cloud


def voxel_downsample(cloud, labels, voxel_size):
    voxel_idx = np.floor(cloud[:, :3] / voxel_size).astype(np.int32)
    seen, keep = {}, []
    for i, vi in enumerate(map(tuple, voxel_idx)):
        if vi not in seen:
            seen[vi] = i
            keep.append(i)
    keep = np.array(keep)
    return cloud[keep], labels[keep]
