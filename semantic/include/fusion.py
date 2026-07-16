"""Voxel fusion and map clean-up shared by the offline stitcher
(testandtraining/stitch_semantic_map.py) and the live map publisher
(rosnode/semantic_map_finisher.py)."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from config import LABEL_TO_NAME

NUM_CLASSES = len(LABEL_TO_NAME)
GROUND = 2


def load_poses(path):
    data = np.genfromtxt(path, delimiter=',', names=True)
    data = np.atleast_1d(data)
    order = np.argsort(data['timestamp'])
    stamps = data['timestamp'][order]
    trans = np.stack([data['x'], data['y'], data['z']], axis=1)[order]
    quats = np.stack([data['qx'], data['qy'], data['qz'], data['qw']], axis=1)[order]
    return stamps, trans, Rotation.from_quat(quats)


class PoseLookup:
    def __init__(self, stamps, trans, rots, max_dt=0.05, max_gap=5.0):
        self.stamps, self.trans, self.rots = stamps, trans, rots
        self.max_dt, self.max_gap = max_dt, max_gap
        self.slerp = Slerp(stamps, rots) if len(stamps) > 1 else None

    def get(self, t):
        """Return (R, t) at time t, or None if no trustworthy pose exists."""
        i = int(np.argmin(np.abs(self.stamps - t)))
        if abs(self.stamps[i] - t) <= self.max_dt:
            return self.rots[i], self.trans[i]
        if self.slerp is None or t < self.stamps[0] or t > self.stamps[-1]:
            return None
        j = int(np.searchsorted(self.stamps, t))
        if self.stamps[j] - self.stamps[j - 1] > self.max_gap:
            return None
        w = (t - self.stamps[j - 1]) / (self.stamps[j] - self.stamps[j - 1])
        trans = (1 - w) * self.trans[j - 1] + w * self.trans[j]
        return self.slerp([t])[0], trans


def remove_robot(xyz, labels, radius):
    """Drop sensor-frame points within a horizontal radius of the sensor
    (robot self-reflections). Returns filtered arrays + removed count."""
    if radius <= 0:
        return xyz, labels, 0
    far = np.einsum('ij,ij->i', xyz[:, :2], xyz[:, :2]) > radius ** 2
    return xyz[far], labels[far], int(len(xyz) - far.sum())


def _hash_keys(pts, voxel):
    keys = np.floor(pts / voxel).astype(np.int64) + (1 << 20)
    if keys.min() < 0 or keys.max() >= (1 << 21):
        raise ValueError('map exceeds hashable range — increase voxel size')
    return (keys[:, 0] << 42) | (keys[:, 1] << 21) | keys[:, 2]


def fuse(pts, labels, voxel, min_votes=1):
    """Voxel fusion with per-class majority vote.
    pts (N,3) float, labels (N,) int -> voxel centroids (M,3), labels (M,)."""
    h = _hash_keys(pts, voxel)
    _, inverse, n = np.unique(h, return_inverse=True, return_counts=True)
    counts = np.zeros((len(n), NUM_CLASSES), dtype=np.int64)
    np.add.at(counts, (inverse, labels.astype(np.int64)), 1)
    sums = np.zeros((len(n), 3), dtype=np.float64)
    np.add.at(sums, inverse, pts.astype(np.float64))

    keep = n >= min_votes
    label = np.argmax(counts[keep], axis=1).astype(np.uint8)
    xyz = (sums[keep] / n[keep, None]).astype(np.float32)
    return xyz, label


def denoise(xyz, label, min_neighbors, radius):
    """Remove isolated voxels (fewer than min_neighbors within radius)."""
    if min_neighbors <= 0 or len(xyz) == 0:
        return xyz, label, 0
    from sklearn.neighbors import KDTree
    neigh = KDTree(xyz).query_radius(xyz, r=radius, count_only=True)
    ok = neigh > min_neighbors            # count includes the voxel itself
    return xyz[ok], label[ok], int((~ok).sum())


def downsample_ground(xyz, label, ground_voxel, base_voxel):
    """Re-grid the (dominant) ground class to a coarser voxel size."""
    if ground_voxel <= base_voxel or len(xyz) == 0:
        return xyz, label
    ground = label == GROUND
    g_xyz = xyz[ground]
    if len(g_xyz) == 0:
        return xyz, label
    h = _hash_keys(g_xyz, ground_voxel)
    _, inv, n = np.unique(h, return_inverse=True, return_counts=True)
    sums = np.zeros((len(n), 3), dtype=np.float64)
    np.add.at(sums, inv, g_xyz.astype(np.float64))
    g_centroids = (sums / n[:, None]).astype(np.float32)
    xyz = np.concatenate([xyz[~ground], g_centroids])
    label = np.concatenate([label[~ground],
                            np.full(len(g_centroids), GROUND, dtype=np.uint8)])
    return xyz, label
