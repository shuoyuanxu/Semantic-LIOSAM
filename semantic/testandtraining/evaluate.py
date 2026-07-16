#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the validation clouds at full resolution.

Runs the same voting scheme as deployment: patches are sampled until every
subsampled point is covered, probabilities are accumulated with smoothing,
then projected back to the full-resolution points and compared to the labels.

Example:
    python3 evaluate.py --data ../data/prepared --checkpoint ../model/run1/checkpoint_best.tar
"""
import argparse
import os
import sys
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, join(os.path.dirname(BASE_DIR), 'include'))

from config import Config as cfg, LABEL_TO_NAME
from dataset import PreparedClouds, PatchSampler, to_device
from RandLANet import Network


def evaluate_cloud(net, device, tree, feats, min_coverage, test_smooth=0.95):
    points = np.array(tree.data, copy=False).astype(np.float32)
    n = len(points)
    possibility = np.random.rand(n) * 1e-3
    probs = np.zeros((n, cfg.num_classes), dtype=np.float32)

    while possibility.min() < min_coverage:
        point_ind = int(np.argmin(possibility))
        center = points[point_ind].reshape(1, -1)
        k = min(cfg.num_points, n)
        q_idx = tree.query(center, k=k)[1][0].astype(np.int64)
        np.random.shuffle(q_idx)

        q_xyz = points[q_idx] - center
        dists = np.sum(np.square(q_xyz), axis=1)
        possibility[q_idx] += np.square(1 - dists / np.max(dists))

        pad = np.random.choice(k, cfg.num_points - k) if k < cfg.num_points else []
        idx_full = np.concatenate([q_idx, q_idx[pad]]) if k < cfg.num_points else q_idx
        xyz_full = points[idx_full] - center
        feat_full = feats[idx_full]

        batch = PatchSampler.build_pyramid(xyz_full[None].astype(np.float32))
        batch['features'] = torch.from_numpy(
            np.concatenate([xyz_full[None], feat_full[None]], axis=-1)).float()
        batch = to_device(batch, device)
        with torch.no_grad():
            logits = net(batch)['logits']
        p = F.softmax(logits.transpose(1, 2), dim=2).reshape(-1, cfg.num_classes)
        probs[idx_full] = test_smooth * probs[idx_full] + \
            (1 - test_smooth) * p.cpu().numpy()

    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--split', default='validation', choices=['validation', 'training'])
    ap.add_argument('--min-coverage', type=float, default=0.5,
                    help='keep sampling patches until every point exceeds this')
    args = ap.parse_args()

    device = torch.device('cuda:%d' % args.gpu
                          if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    clouds = PreparedClouds(args.data, args.split)
    if not clouds.proj:
        sys.exit('no *_proj.pkl files — rerun prepare_dataset.py')

    net = Network(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    print('loaded %s (epoch %s)' % (args.checkpoint, ckpt.get('epoch', '?')))

    conf = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
    for i, name in enumerate(clouds.names):
        probs = evaluate_cloud(net, device, clouds.trees[i], clouds.features[i],
                               args.min_coverage)
        full_pred = probs[clouds.proj[i]].argmax(axis=1)
        gt = clouds.full_labels[i]
        np.add.at(conf, (gt.astype(np.int64), full_pred.astype(np.int64)), 1)
        print('[%2d/%d] %s: %d pts' % (i + 1, len(clouds.names), name, len(gt)))

    tp = np.diag(conf)
    iou = tp / np.maximum(conf.sum(0) + conf.sum(1) - tp, 1)
    print('\nOverall accuracy: %.4f' % (tp.sum() / conf.sum()))
    for i in range(cfg.num_classes):
        print('  IoU %-8s %.4f' % (LABEL_TO_NAME[i], iou[i]))
    print('mIoU: %.4f' % iou.mean())


if __name__ == '__main__':
    main()
