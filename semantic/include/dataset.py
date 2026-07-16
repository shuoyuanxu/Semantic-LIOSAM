"""Dataset classes for training/evaluating the forestry RandLA-Net on
clouds prepared by testandtraining/prepare_dataset.py.

Layout expected under the prepared-data root:
    train_files.txt / val_files.txt     one cloud name per line
    original_ply/<name>.ply             full-resolution cloud (+ class)
    input_<grid>/<name>.ply             grid-subsampled cloud (+ class)
    input_<grid>/<name>_KDTree.pkl      KDTree over the subsampled xyz
    input_<grid>/<name>_proj.pkl        [proj_idx, full_labels] for evaluation
"""
import os
import pickle
from os.path import join

import numpy as np
import torch
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import Config as cfg
from helper_ply import read_ply
from helper_tool import DataProcessing as DP


class PreparedClouds:
    """Loads all prepared clouds of one split into memory."""

    def __init__(self, data_root, split):
        list_file = join(data_root, 'train_files.txt' if split == 'training'
                         else 'val_files.txt')
        with open(list_file) as f:
            self.names = [l.strip() for l in f if l.strip()]
        if not self.names:
            raise RuntimeError('no clouds listed in %s' % list_file)

        sub_dir = join(data_root, 'input_{:.3f}'.format(cfg.sub_grid_size))
        self.trees, self.features, self.labels = [], [], []
        self.proj, self.full_labels = [], []
        for name in self.names:
            with open(join(sub_dir, name + '_KDTree.pkl'), 'rb') as f:
                self.trees.append(pickle.load(f))
            data = read_ply(join(sub_dir, name + '.ply'))
            self.features.append(np.vstack((data['intensity'],
                                            data['reflectivity'],
                                            data['ambient'])).T.astype(np.float32))
            self.labels.append(data['class'].astype(np.int64))
            proj_file = join(sub_dir, name + '_proj.pkl')
            if os.path.isfile(proj_file):
                with open(proj_file, 'rb') as f:
                    proj_idx, full_labels = pickle.load(f)
                self.proj.append(proj_idx)
                self.full_labels.append(full_labels)

        self.num_classes = cfg.num_classes

    def class_counts(self):
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for l in self.labels:
            counts += np.bincount(l, minlength=self.num_classes)
        return counts


class PatchSampler(Dataset):
    """Spatially-regular patch sampler over a PreparedClouds split
    (same 'possibility' mechanism as the original RandLA-Net pipeline)."""

    def __init__(self, clouds, split):
        self.clouds = clouds
        steps = cfg.train_steps if split == 'training' else cfg.val_steps
        batch = cfg.batch_size if split == 'training' else cfg.val_batch_size
        self.num_per_epoch = steps * batch
        self.possibility = [np.random.rand(t.data.shape[0]) * 1e-3
                            for t in clouds.trees]
        self.min_possibility = [float(np.min(p)) for p in self.possibility]

    def __len__(self):
        return self.num_per_epoch

    def __getitem__(self, _item):
        cloud_idx = int(np.argmin(self.min_possibility))
        tree = self.clouds.trees[cloud_idx]
        points = np.array(tree.data, copy=False)

        point_ind = np.argmin(self.possibility[cloud_idx])
        center = points[point_ind, :].reshape(1, -1)
        noise = np.random.normal(scale=cfg.noise_init / 10, size=center.shape)
        pick = center + noise.astype(center.dtype)

        k = min(cfg.num_points, len(points))
        q_idx = tree.query(pick, k=k)[1][0]
        q_idx = DP.shuffle_idx(q_idx)

        q_xyz = (points[q_idx] - pick).astype(np.float32)
        q_feat = self.clouds.features[cloud_idx][q_idx]
        q_label = self.clouds.labels[cloud_idx][q_idx]

        dists = np.sum(np.square(q_xyz), axis=1)
        delta = np.square(1 - dists / np.max(dists))
        self.possibility[cloud_idx][q_idx] += delta
        self.min_possibility[cloud_idx] = float(np.min(self.possibility[cloud_idx]))

        if k < cfg.num_points:
            q_xyz, q_feat, q_idx, q_label = DP.data_aug(
                q_xyz, q_feat, q_label, q_idx, cfg.num_points)

        return (q_xyz.astype(np.float32), q_feat.astype(np.float32),
                q_label.astype(np.int64), q_idx.astype(np.int64),
                np.array([cloud_idx], dtype=np.int64))

    @staticmethod
    def build_pyramid(batch_xyz):
        """Multi-scale neighbour / pooling / upsampling indices (tf_map)."""
        inputs = {'xyz': [], 'neigh_idx': [], 'sub_idx': [], 'interp_idx': []}
        for i in range(cfg.num_layers):
            neigh_idx = DP.knn_search(batch_xyz, batch_xyz, 2 * cfg.k_n)[:, :, 1::2]
            n_sub = batch_xyz.shape[1] // cfg.sub_sampling_ratio[i]
            sub_points = batch_xyz[:, :n_sub, :]
            inputs['xyz'].append(torch.from_numpy(batch_xyz).float())
            inputs['neigh_idx'].append(torch.from_numpy(neigh_idx).long())
            inputs['sub_idx'].append(torch.from_numpy(neigh_idx[:, :n_sub, :]).long())
            inputs['interp_idx'].append(
                torch.from_numpy(DP.knn_search(sub_points, batch_xyz, 1)).long())
            batch_xyz = sub_points
        return inputs

    @staticmethod
    def collate_fn(batch):
        xyz = np.stack([b[0] for b in batch])
        feat = np.stack([b[1] for b in batch])
        labels = np.stack([b[2] for b in batch])
        p_idx = np.stack([b[3] for b in batch])
        c_idx = np.stack([b[4] for b in batch])

        inputs = PatchSampler.build_pyramid(xyz)
        inputs['features'] = torch.from_numpy(
            np.concatenate([xyz, feat], axis=-1)).float()
        inputs['labels'] = torch.from_numpy(labels).long()
        inputs['input_inds'] = torch.from_numpy(p_idx).long()
        inputs['cloud_inds'] = torch.from_numpy(c_idx).long()
        return inputs


def to_device(batch, device):
    for key in batch:
        if isinstance(batch[key], list):
            batch[key] = [t.to(device) for t in batch[key]]
        else:
            batch[key] = batch[key].to(device)
    return batch
