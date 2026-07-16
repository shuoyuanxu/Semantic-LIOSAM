#!/usr/bin/env python3
"""Stitch labeled frames into a global semantic map using LIO-SAM poses.

Inputs:
  - poses.csv written by LIO-SAM's save_map service
    (id,x,y,z,qx,qy,qz,qw,timestamp — optimized keyframe poses, lidar frame)
  - labeled_frames/ written by segmentation_node.py
    (frame_*.npz with xyz [sensor frame], label, intensity, stamp)

For every labeled frame, the pose at its timestamp is taken from the nearest
keyframe (within --max-dt) or interpolated between the two bracketing
keyframes. Points are transformed to the map frame and fused into a voxel
grid where overlapping observations vote per class; each voxel keeps the
majority label and the centroid of its points.

Output: a colored PLY (x,y,z,red,green,blue,label) viewable in
CloudCompare / RViz, plus optional per-class PLYs.

Example:
  python3 stitch_semantic_map.py \
      --poses ~/lio-sam-maps/keyframes/poses.csv \
      --frames ../output/labeled_frames \
      --out ../output/semantic_map.ply --voxel 0.10
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEMANTIC_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(SEMANTIC_DIR, 'include'))

from config import LABEL_TO_NAME, LABEL_TO_COLOR
from helper_ply import write_ply

NUM_CLASSES = len(LABEL_TO_NAME)


def load_poses(path):
    data = np.genfromtxt(path, delimiter=',', names=True)
    order = np.argsort(data['timestamp'])
    stamps = data['timestamp'][order]
    trans = np.stack([data['x'], data['y'], data['z']], axis=1)[order]
    quats = np.stack([data['qx'], data['qy'], data['qz'], data['qw']], axis=1)[order]
    return stamps, trans, Rotation.from_quat(quats)


class PoseLookup:
    def __init__(self, stamps, trans, rots, max_dt, max_gap):
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--poses', required=True, help='poses.csv from LIO-SAM save_map')
    ap.add_argument('--frames', required=True, help='directory of frame_*.npz')
    ap.add_argument('--out', default=os.path.join(SEMANTIC_DIR, 'output', 'semantic_map.ply'))
    ap.add_argument('--voxel', type=float, default=0.10, help='fusion voxel size [m]')
    ap.add_argument('--max-dt', type=float, default=0.05,
                    help='max |t_frame - t_keyframe| for a direct pose match [s]')
    ap.add_argument('--max-gap', type=float, default=5.0,
                    help='max keyframe gap for pose interpolation [s]')
    ap.add_argument('--min-votes', type=int, default=1,
                    help='drop voxels observed fewer than this many points')
    ap.add_argument('--split-classes', action='store_true',
                    help='also write one PLY per class')
    ap.add_argument('--preview', action='store_true',
                    help='also render a top-down/elevation PNG next to the PLY')
    args = ap.parse_args()

    stamps, trans, rots = load_poses(args.poses)
    print('Loaded %d keyframe poses (%.1f s span)' % (len(stamps), stamps[-1] - stamps[0]))
    lookup = PoseLookup(stamps, trans, rots, args.max_dt, args.max_gap)

    files = sorted(glob.glob(os.path.join(args.frames, 'frame_*.npz')))
    if not files:
        sys.exit('no frame_*.npz found in %s' % args.frames)
    print('Found %d labeled frames' % len(files))

    # collect map-frame points, then fuse in one vectorised pass
    inv_voxel = 1.0 / args.voxel
    all_pts, all_labels = [], []
    used = skipped = 0

    for f in files:
        d = np.load(f, allow_pickle=True)
        pose = lookup.get(float(d['stamp']))
        if pose is None:
            skipped += 1
            continue
        R, t = pose
        all_pts.append((d['xyz'].astype(np.float64) @ R.as_matrix().T + t).astype(np.float32))
        all_labels.append(d['label'].astype(np.uint8))
        used += 1
    print('Frames used: %d, skipped (no pose): %d' % (used, skipped))
    if not all_pts:
        sys.exit('nothing fused — check timestamps / --max-dt / --max-gap')

    pts = np.concatenate(all_pts)
    labels = np.concatenate(all_labels).astype(np.int64)
    del all_pts, all_labels
    print('Fusing %d points into %.2f m voxels ...' % (len(pts), args.voxel))

    # hash voxel index (21 bits per axis, origin-offset) into one int64
    keys = np.floor(pts * inv_voxel).astype(np.int64) + (1 << 20)
    if keys.min() < 0 or keys.max() >= (1 << 21):
        sys.exit('map exceeds the hashable range — increase --voxel')
    h = (keys[:, 0] << 42) | (keys[:, 1] << 21) | keys[:, 2]

    uniq, inverse, n = np.unique(h, return_inverse=True, return_counts=True)
    nvox = len(uniq)

    counts = np.zeros((nvox, NUM_CLASSES), dtype=np.int64)     # class votes
    np.add.at(counts, (inverse, labels), 1)
    sums = np.zeros((nvox, 3), dtype=np.float64)               # centroids
    np.add.at(sums, inverse, pts)

    keep = n >= args.min_votes
    label = np.argmax(counts[keep], axis=1).astype(np.uint8)   # majority vote
    xyz = (sums[keep] / n[keep, None]).astype(np.float32)

    palette = np.array([LABEL_TO_COLOR[i] for i in range(NUM_CLASSES)], dtype=np.uint8)
    rgb = palette[label]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_ply(args.out, [xyz, rgb[:, 0], rgb[:, 1], rgb[:, 2], label],
              ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    print('Wrote %s (%d voxels @ %.2f m)' % (args.out, len(xyz), args.voxel))

    for i in range(NUM_CLASSES):
        print('  %-8s %8d voxels' % (LABEL_TO_NAME[i], int(np.sum(label == i))))
        if args.split_classes:
            m = label == i
            if m.any():
                out_i = args.out.replace('.ply', '_%s.ply' % LABEL_TO_NAME[i])
                write_ply(out_i, [xyz[m], rgb[m, 0], rgb[m, 1], rgb[m, 2]],
                          ['x', 'y', 'z', 'red', 'green', 'blue'])
                print('           -> %s' % out_i)

    if args.preview:
        render_preview(xyz, label, args.out.replace('.ply', '_preview.png'))


def render_preview(xyz, label, out_png):
    """Top-down + side-elevation scatter of the fused map, coloured by class."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rgb = np.array([LABEL_TO_COLOR[i] for i in range(NUM_CLASSES)]) / 255.0
    fig, axes = plt.subplots(2, 1, figsize=(16, 14),
                             gridspec_kw={'height_ratios': [3, 1]})
    ax = axes[0]
    order = np.argsort(z)                     # draw low first so canopy on top
    ax.scatter(x[order], y[order], s=0.3, c=rgb[label[order]], linewidths=0)
    ax.set_aspect('equal')
    ax.set_title('Semantic map (top-down)')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    handles = [plt.Line2D([0], [0], marker='o', ls='', color=rgb[i],
                          label=LABEL_TO_NAME[i]) for i in range(NUM_CLASSES)]
    ax.legend(handles=handles, loc='upper right')

    ax2 = axes[1]
    m = np.abs(y - np.median(y)) < 30
    ax2.scatter(x[m], z[m], s=0.3, c=rgb[label[m]], linewidths=0)
    ax2.set_aspect('equal')
    ax2.set_title('Side elevation (|y - median| < 30 m)')
    ax2.set_xlabel('x [m]'); ax2.set_ylabel('z [m]')

    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    print('Preview -> %s' % out_png)


if __name__ == '__main__':
    main()
