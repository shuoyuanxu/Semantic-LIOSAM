"""In-memory RandLA-Net inference for single LiDAR scans.

Replaces the legacy per-frame pipeline (write PLY + KDTree pickle to disk,
rebuild Dataset/DataLoader, reload checkpoint) with a resident model and a
pure in-memory patch sampler. The sampling / probability-accumulation logic
mirrors the legacy S3DISSampler_muti_apply + ModelTester behaviour.

Usage:
    engine = SegmentationEngine('semantic/model/checkpoint.tar')
    labels, probs = engine.segment(xyz, features)
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import Config as cfg
from helper_tool import DataProcessing as DP
from RandLANet import Network


class SegmentationEngine:
    """Loads the checkpoint once and segments scans in memory."""

    def __init__(self, checkpoint_path, device=None,
                 max_patches=8, min_coverage=0.2, test_smooth=0.95):
        """
        :param checkpoint_path: path to checkpoint.tar
        :param device: 'cuda', 'cuda:0', 'cpu' or None (auto)
        :param max_patches: hard cap on inference patches per scan
        :param min_coverage: stop early once every point's "possibility"
               exceeds this value (i.e. the whole scan has been covered)
        :param test_smooth: exponential smoothing of accumulated probabilities
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.max_patches = max_patches
        self.min_coverage = min_coverage
        self.test_smooth = test_smooth

        cfg.ignored_label_inds = []
        cfg.class_weights = DP.get_class_weights(cfg.name)

        self.net = Network(cfg)
        self.net.to(self.device)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError('checkpoint not found: %s' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()

    @torch.no_grad()
    def segment(self, xyz, features):
        """Segment one scan.

        :param xyz: (N,3) float32 sensor-frame coordinates
        :param features: (N,3) float32 [intensity, reflectivity, ambient],
               value ranges matching training (intensity/reflectivity cast
               to uint8 before conversion, as the legacy pipeline did)
        :return: labels (N,) uint8, probs (N,num_classes) float32
        """
        xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        features = np.ascontiguousarray(features, dtype=np.float32)
        n = xyz.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.uint8), np.zeros((0, cfg.num_classes), np.float32)

        tree = KDTree(xyz)
        # same "possibility" coverage mechanism as the legacy sampler
        possibility = np.random.rand(n) * 1e-3
        probs = np.zeros((n, cfg.num_classes), dtype=np.float32)

        for _ in range(self.max_patches):
            if possibility.min() > self.min_coverage:
                break

            center_ind = int(np.argmin(possibility))
            center = xyz[center_ind].reshape(1, -1)
            center = center + np.random.normal(scale=cfg.noise_init / 10,
                                               size=center.shape).astype(np.float32)

            k = min(cfg.num_points, n)
            q_idx = tree.query(center, k=k, return_distance=False)[0]
            q_idx = DP.shuffle_idx(q_idx)

            q_xyz = xyz[q_idx] - center.astype(np.float32)
            q_feat = features[q_idx]

            # update coverage before padding (padding duplicates points)
            dists = np.sum(np.square(q_xyz.astype(np.float32)), axis=1)
            delta = np.square(1 - dists / np.max(dists))
            possibility[q_idx] += delta

            if k < cfg.num_points:
                q_xyz, q_feat, q_idx = self._pad(q_xyz, q_feat, q_idx, cfg.num_points)

            batch = self._build_inputs(q_xyz, q_feat)
            end_points = self.net(batch)
            logits = end_points['logits']                     # (1, C, P)
            patch_probs = F.softmax(logits.transpose(1, 2), dim=2)
            patch_probs = patch_probs.reshape(-1, cfg.num_classes).cpu().numpy()

            probs[q_idx] = self.test_smooth * probs[q_idx] + \
                (1 - self.test_smooth) * patch_probs

        labels = np.argmax(probs, axis=1).astype(np.uint8)
        return labels, probs

    @staticmethod
    def _pad(xyz, feat, idx, num_out):
        """Duplicate random points so a small cloud fills a full patch
        (legacy code called a missing DP.data_aug_without_label here)."""
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in)
        return (np.concatenate([xyz, xyz[dup]], 0),
                np.concatenate([feat, feat[dup]], 0),
                np.concatenate([idx, idx[dup]], 0))

    def _build_inputs(self, q_xyz, q_feat):
        """Build the multi-scale input pyramid for one patch
        (batch size 1). Mirrors legacy tf_map + collate_fn."""
        batch_xyz = q_xyz[None, ...].astype(np.float32)          # (1, P, 3)
        batch_feat = np.concatenate([batch_xyz, q_feat[None, ...]], axis=-1)

        inputs = {'xyz': [], 'neigh_idx': [], 'sub_idx': [], 'interp_idx': []}
        for i in range(cfg.num_layers):
            neigh_idx = DP.knn_search(batch_xyz, batch_xyz, 2 * cfg.k_n)[:, :, 1::2]
            n_sub = batch_xyz.shape[1] // cfg.sub_sampling_ratio[i]
            sub_points = batch_xyz[:, :n_sub, :]
            pool_i = neigh_idx[:, :n_sub, :]
            up_i = DP.knn_search(sub_points, batch_xyz, 1)

            inputs['xyz'].append(torch.from_numpy(batch_xyz).float().to(self.device))
            inputs['neigh_idx'].append(torch.from_numpy(neigh_idx).long().to(self.device))
            inputs['sub_idx'].append(torch.from_numpy(pool_i).long().to(self.device))
            inputs['interp_idx'].append(torch.from_numpy(up_i).long().to(self.device))
            batch_xyz = sub_points

        inputs['features'] = torch.from_numpy(batch_feat).float().to(self.device)
        return inputs
