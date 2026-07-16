# from open3d import linux as open3d
from os.path import join
import numpy as np
import colorsys, random, os, sys
try:
    import pandas as pd  # only needed by the semantic3d loaders
except ImportError:
    pd = None
try:
    import open3d  # only needed by the Plot visualisation helpers
except ImportError:
    open3d = None

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import utils.cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import utils.nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors


class DataProcessing:
    @staticmethod
    def load_pc_semantic3d(filename):
        pc_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float16)
        pc = pc_pd.values
        return pc

    @staticmethod
    def load_label_semantic3d(filename):
        label_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.uint8)
        cloud_labels = label_pd.values
        return cloud_labels

    @staticmethod
    def load_pc_kitti(pc_path):
        scan = np.fromfile(pc_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, 0:3]  # get xyz
        return points

    @staticmethod
    def load_label_kitti(label_path, remap_lut):
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = remap_lut[sem_label]
        return sem_label.astype(np.int32)

    @staticmethod
    def get_file_list(dataset_path, test_scan_num):
        seq_list = np.sort(os.listdir(dataset_path))

        train_file_list = []
        test_file_list = []
        val_file_list = []
        for seq_id in seq_list:
            seq_path = join(dataset_path, seq_id)
            pc_path = join(seq_path, 'velodyne')
            if seq_id == '08':
                val_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
                if seq_id == test_scan_num:
                    test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif int(seq_id) >= 11 and seq_id == test_scan_num:
                test_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])
            elif seq_id in ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']:
                train_file_list.append([join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

        train_file_list = np.concatenate(train_file_list, axis=0)
        val_file_list = np.concatenate(val_file_list, axis=0)
        if test_scan_num != 'None':
            test_file_list = np.concatenate(test_file_list, axis=0)
        else:
            test_file_list = None
        return train_file_list, val_file_list, test_file_list

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
        return neighbor_idx.astype(np.int32)

    @staticmethod
    def data_aug(xyz, color, labels, idx, num_out):
        num_in = len(xyz)
        dup = np.random.choice(num_in, num_out - num_in) #从num_in中随机抽取数字，并组成指定大小(size)的数组
        xyz_dup = xyz[dup, ...]
        xyz_aug = np.concatenate([xyz, xyz_dup], 0)
        color_dup = color[dup, ...]
        color_aug = np.concatenate([color, color_dup], 0)
        idx_dup = list(range(num_in)) + list(dup)
        idx_aug = idx[idx_dup]
        label_aug = labels[idx_dup]
        return xyz_aug, color_aug, idx_aug, label_aug

    @staticmethod
    def shuffle_idx(x):
        # random shuffle the index
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        return x[idx]

    @staticmethod
    def shuffle_list(data_list):
        indices = np.arange(np.shape(data_list)[0])
        np.random.shuffle(indices)
        data_list = data_list[indices]
        return data_list

    @staticmethod
    def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
        if (features is None) and (labels is None):
            return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
        elif labels is None:
            return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
        elif features is None:
            return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
        else:
            return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)

    @staticmethod
    def IoU_from_confusions(confusions):
        TP = np.diagonal(confusions, axis1=-2, axis2=-1)
        TP_plus_FN = np.sum(confusions, axis=-1)
        TP_plus_FP = np.sum(confusions, axis=-2)

        IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

        mask = TP_plus_FN < 1e-3
        counts = np.sum(1 - mask, axis=-1, keepdims=True)
        mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

        IoU += mask * mIoU
        return IoU

    @staticmethod
    def get_class_weights(dataset_name):
        num_per_class = []
        if dataset_name == 'Cunjia_liu':
            num_per_class = np.array([38232, 1665125, 1324780, 807374],
                                     dtype=np.int32)

        weight = num_per_class / float(sum(num_per_class))
        ce_label_weight = 1 / (weight + 0.02)
        return np.expand_dims(ce_label_weight, axis=0)

class Plot:
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb):
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            open3d.visualization.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
        open3d.visualization.draw_geometries([pc])
        return 0

    @staticmethod
    def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None):
        """
        pc_xyz: 3D coordinates of point clouds
        pc_sem_ins: semantic or instance labels
        plot_colors: custom color list
        """
        if plot_colors is not None:
            ins_colors = plot_colors
        else:
            ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=2)

        ##############################
        sem_ins_labels = np.unique(pc_sem_ins)
        sem_ins_bbox = []
        Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
        for id, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
            else:
                if plot_colors is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            ### bbox
            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0]);
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1]);
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2]);
            zmax = np.max(valid_xyz[:, 2])
            sem_ins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
        Plot.draw_pc(Y_semins)
        return Y_semins




