#!/usr/bin/env python3
"""Convert the labeled forestry dataset into RandLA-Net training format.

Input: a directory tree where each labeled frame is a folder containing
    trunk.txt  crown.txt  ground.txt  others.txt
with one point per line, 7 whitespace-separated columns:
    x y z intensity reflectivity ambient label
(label column is ignored — the class comes from the file name:
 trunk=0, crown=1, ground=2, others=3)

Any folder whose path contains "Test" (case-insensitive) becomes a
validation cloud; everything else is used for training. Override with
--val-pattern.

Output (under --out):
    train_files.txt / val_files.txt
    original_ply/<name>.ply                full resolution + class
    input_0.040/<name>.ply                 grid-subsampled + class
    input_0.040/<name>_KDTree.pkl          KDTree over subsampled xyz
    input_0.040/<name>_proj.pkl            [proj_idx, full_labels] for eval

Example:
    python3 prepare_dataset.py --raw ../data/Labeled_dataset --out ../data/prepared
"""
import argparse
import os
import pickle
import sys
from os.path import join

import numpy as np
from sklearn.neighbors import KDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, join(os.path.dirname(BASE_DIR), 'include'))

from config import Config as cfg, LABEL_TO_NAME
from helper_ply import write_ply
from helper_tool import DataProcessing as DP

CLASS_FILES = {name: label for label, name in LABEL_TO_NAME.items()}
PLY_FIELDS = ['x', 'y', 'z', 'intensity', 'reflectivity', 'ambient', 'class']


def find_frames(raw_root):
    frames = []
    for dirpath, _dirnames, filenames in os.walk(raw_root):
        if any(c + '.txt' in filenames for c in CLASS_FILES):
            frames.append(dirpath)
    return sorted(frames)


def load_frame(frame_dir):
    parts = []
    for cname, label in CLASS_FILES.items():
        path = join(frame_dir, cname + '.txt')
        if not os.path.isfile(path):
            continue                      # frame simply has no points of this class
        data = np.loadtxt(path, dtype=np.float64)
        if data.size == 0:
            continue
        data = np.atleast_2d(data)
        labels = np.full((len(data), 1), label, dtype=np.float64)
        parts.append(np.hstack([data[:, :6], labels]))
    pts = np.vstack(parts)
    # drop invalid (0,0,0) returns, common in others.txt
    valid = np.linalg.norm(pts[:, :3], axis=1) > 1e-6
    return pts[valid]


def process_frame(frame_dir, name, out_root, sub_dir):
    pts = load_frame(frame_dir)
    xyz = pts[:, :3].astype(np.float32)
    # match the deployed inference pipeline: intensity/reflectivity through
    # uint8, ambient kept raw
    feats = np.stack([pts[:, 3].astype(np.uint8).astype(np.float32),
                      pts[:, 4].astype(np.uint8).astype(np.float32),
                      pts[:, 5].astype(np.float32)], axis=1)
    labels = pts[:, 6].astype(np.uint8)

    write_ply(join(out_root, 'original_ply', name + '.ply'),
              [xyz, feats[:, 0], feats[:, 1], feats[:, 2], labels], PLY_FIELDS)

    sub_xyz, sub_feat, sub_labels = DP.grid_sub_sampling(
        xyz, features=feats, labels=labels, grid_size=cfg.sub_grid_size)
    write_ply(join(sub_dir, name + '.ply'),
              [sub_xyz, sub_feat[:, 0], sub_feat[:, 1], sub_feat[:, 2],
               sub_labels.squeeze()], PLY_FIELDS)

    tree = KDTree(sub_xyz)
    with open(join(sub_dir, name + '_KDTree.pkl'), 'wb') as f:
        pickle.dump(tree, f)
    proj_idx = np.squeeze(tree.query(xyz, return_distance=False)).astype(np.int32)
    with open(join(sub_dir, name + '_proj.pkl'), 'wb') as f:
        pickle.dump([proj_idx, labels], f)

    return len(xyz), len(sub_xyz), np.bincount(labels, minlength=cfg.num_classes)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', required=True, help='root of the labeled dataset')
    ap.add_argument('--out', required=True, help='output root for prepared data')
    ap.add_argument('--val-pattern', default='test',
                    help='frames whose path contains this (case-insensitive) '
                         'become validation clouds')
    ap.add_argument('--limit', type=int, default=0,
                    help='only process the first N frames (0 = all)')
    args = ap.parse_args()

    frames = find_frames(args.raw)
    if not frames:
        sys.exit('no labeled frames (folders with %s) found under %s'
                 % ('/'.join(c + '.txt' for c in CLASS_FILES), args.raw))
    if args.limit:
        frames = frames[:args.limit]
    print('Found %d labeled frames' % len(frames))

    sub_dir = join(args.out, 'input_{:.3f}'.format(cfg.sub_grid_size))
    os.makedirs(join(args.out, 'original_ply'), exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)

    train_names, val_names = [], []
    totals = np.zeros(cfg.num_classes, dtype=np.int64)
    for i, frame_dir in enumerate(frames):
        rel = os.path.relpath(frame_dir, args.raw)
        name = rel.replace(os.sep, '_')
        n_full, n_sub, counts = process_frame(frame_dir, name, args.out, sub_dir)
        totals += counts
        split = val_names if args.val_pattern.lower() in rel.lower() else train_names
        split.append(name)
        print('[%3d/%d] %-24s %7d pts -> %6d sub (%s)'
              % (i + 1, len(frames), name, n_full, n_sub,
                 'val' if split is val_names else 'train'))

    with open(join(args.out, 'train_files.txt'), 'w') as f:
        f.write('\n'.join(train_names) + '\n')
    with open(join(args.out, 'val_files.txt'), 'w') as f:
        f.write('\n'.join(val_names) + '\n')

    print('\n%d training / %d validation clouds' % (len(train_names), len(val_names)))
    for i in range(cfg.num_classes):
        print('  %-8s %10d pts (%.1f%%)'
              % (LABEL_TO_NAME[i], totals[i], 100.0 * totals[i] / totals.sum()))
    print('Prepared data written to %s' % args.out)


if __name__ == '__main__':
    main()
