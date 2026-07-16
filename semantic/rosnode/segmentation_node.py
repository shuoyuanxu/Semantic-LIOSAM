#!/usr/bin/env python3
"""Keyframe-rate semantic segmentation node.

Runs alongside LIO-SAM (never in its critical path):
  - subscribes to the raw Ouster cloud (full fields incl. reflectivity/ambient)
  - keeps only the newest scan (drop-old, never builds a backlog)
  - segments it with the resident RandLA-Net model
  - publishes a labeled cloud for RViz
  - saves each labeled frame (sensor frame + timestamp) to disk, so the
    offline stitcher can compose the final semantic map with the
    loop-closure-corrected poses from poses.csv

Run inside the PyTorch conda env:
    rosrun lio_sam segmentation_node.py   (or python3 segmentation_node.py)
"""
import os
import sys
import threading

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEMANTIC_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, os.path.join(SEMANTIC_DIR, 'include'))

from config import TRAIN_CROP, LABEL_TO_NAME, LABEL_TO_COLOR
from inference import SegmentationEngine

# packed 0x00RRGGBB per class, for rviz RGB8 coloring
_PALETTE = np.array([(LABEL_TO_COLOR[i][0] << 16) | (LABEL_TO_COLOR[i][1] << 8)
                     | LABEL_TO_COLOR[i][2] for i in range(len(LABEL_TO_COLOR))],
                    dtype=np.uint32)

# PointField datatype -> numpy dtype
_PF_DTYPE = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}


def cloud_to_arrays(msg, wanted_fields):
    """Vectorised PointCloud2 -> dict of 1-D numpy arrays (no python loop)."""
    dtype_list = []
    offset = 0
    for f in msg.fields:
        if f.offset > offset:
            dtype_list.append(('_pad%d' % offset, 'V%d' % (f.offset - offset)))
            offset = f.offset
        np_dtype = _PF_DTYPE[f.datatype]
        dtype_list.append((f.name, np_dtype))
        offset = f.offset + np.dtype(np_dtype).itemsize
    if msg.point_step > offset:
        dtype_list.append(('_tail', 'V%d' % (msg.point_step - offset)))

    raw = np.frombuffer(msg.data, dtype=np.dtype(dtype_list),
                        count=msg.width * msg.height)
    return {name: np.array(raw[name]) for name in wanted_fields}


def param(name, default):
    """Config lookup: private ~name overrides the `semantic:` section of
    config/params.yml, which overrides the built-in default."""
    return rospy.get_param('~' + name, rospy.get_param('semantic/' + name, default))


class SegmentationNode:
    def __init__(self):
        rospy.init_node('semantic_segmentation')

        input_topic = param('input_topic', '/ouster/points')
        output_topic = param('output_topic', '/semantic/labeled_cloud')
        checkpoint = param('checkpoint',
                           os.path.join(SEMANTIC_DIR, 'model', 'checkpoint.tar'))
        device = param('device', '') or None
        # live alongside the rest of the LIO-SAM output (default ~/Downloads/LOAM/)
        loam_dir = os.path.expanduser('~') + rospy.get_param('lio_sam/savePCDDirectory',
                                                             '/Downloads/LOAM/')
        self.save_dir = param('save_dir',
                              os.path.join(loam_dir, 'semantic', 'labeled_frames'))
        self.min_interval = param('min_interval', 0.5)  # s between scans
        self.use_crop = param('use_crop', True)
        self.crop = {k: param('crop_' + k, v) for k, v in TRAIN_CROP.items()}
        # drop robot self-reflections before labeling (0 disables)
        self.robot_radius = param('robot_radius', 0.55)
        max_patches = param('max_patches', 8)
        min_coverage = param('min_coverage', 0.2)

        os.makedirs(self.save_dir, exist_ok=True)
        # fresh run overwrites the previous one, like the rest of LIO-SAM's output
        stale = [f for f in os.listdir(self.save_dir) if f.startswith('frame_')]
        for f in stale:
            os.remove(os.path.join(self.save_dir, f))
        if stale:
            rospy.loginfo('cleared %d labeled frames from previous run', len(stale))

        rospy.loginfo('Loading RandLA-Net checkpoint %s ...', checkpoint)
        self.engine = SegmentationEngine(checkpoint, device=device,
                                         max_patches=max_patches,
                                         min_coverage=min_coverage)
        rospy.loginfo('Model resident on %s', self.engine.device)

        self.lock = threading.Lock()
        self.latest_msg = None
        self.last_processed_stamp = -1e18
        self.last_seen_stamp = None

        self.pub = rospy.Publisher(output_topic, PointCloud2, queue_size=2)
        rospy.Subscriber(input_topic, PointCloud2, self.callback,
                         queue_size=1, buff_size=2 ** 24)
        rospy.loginfo('Subscribed to %s, publishing %s, saving frames to %s',
                      input_topic, output_topic, self.save_dir)

    def callback(self, msg):
        # a backward jump in sensor time means a bag was restarted:
        # clear the previous run's frames so maps don't mix
        stamp = msg.header.stamp.to_sec()
        if self.last_seen_stamp is not None and stamp < self.last_seen_stamp - 5.0:
            stale = [f for f in os.listdir(self.save_dir) if f.startswith('frame_')]
            for f in stale:
                os.remove(os.path.join(self.save_dir, f))
            self.last_processed_stamp = -1e18
            rospy.logwarn('bag restart detected (time jumped back %.0fs) — '
                          'cleared %d labeled frames from the previous run',
                          self.last_seen_stamp - stamp, len(stale))
        self.last_seen_stamp = stamp

        # keep only the newest scan; the worker picks it up when free
        with self.lock:
            self.latest_msg = msg

    def spin(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            with self.lock:
                msg, self.latest_msg = self.latest_msg, None
            if msg is not None:
                stamp = msg.header.stamp.to_sec()
                if stamp - self.last_processed_stamp >= self.min_interval:
                    try:
                        self.process(msg)
                        self.last_processed_stamp = stamp
                    except Exception as e:
                        rospy.logerr('segmentation failed: %s', e)
            rate.sleep()

    def process(self, msg):
        available = {f.name for f in msg.fields}
        ambient_field = 'ambient' if 'ambient' in available else 'near_ir'
        wanted = ['x', 'y', 'z', 'intensity', 'reflectivity', ambient_field]
        missing = [f for f in wanted if f not in available]
        if missing:
            rospy.logerr_throttle(10, 'missing fields %s in %s' % (missing, sorted(available)))
            return

        t0 = rospy.get_time()
        arr = cloud_to_arrays(msg, wanted)
        xyz = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float32)
        # feature dtypes must match the training pipeline:
        # intensity/reflectivity went through uint8, ambient stayed raw
        feats = np.stack([arr['intensity'].astype(np.uint8),
                          arr['reflectivity'].astype(np.uint8),
                          arr[ambient_field]], axis=1).astype(np.float32)

        valid = np.isfinite(xyz).all(axis=1) & (np.abs(xyz) > 1e-6).any(axis=1)
        xyz, feats = xyz[valid], feats[valid]

        if self.use_crop:
            m = ((xyz[:, 0] >= self.crop['x_min']) & (xyz[:, 0] <= self.crop['x_max']) &
                 (xyz[:, 1] >= self.crop['y_min']) & (xyz[:, 1] <= self.crop['y_max']))
            xyz, feats = xyz[m], feats[m]

        if self.robot_radius > 0:
            far = np.einsum('ij,ij->i', xyz[:, :2], xyz[:, :2]) > self.robot_radius ** 2
            xyz, feats = xyz[far], feats[far]

        # drop duplicate coordinates (degenerate for the KDTree), as legacy did
        _, uniq = np.unique(xyz, axis=0, return_index=True)
        xyz, feats = xyz[uniq], feats[uniq]

        if len(xyz) == 0:
            rospy.logwarn_throttle(10, 'no points left after filtering')
            return

        labels, _ = self.engine.segment(xyz, feats)

        stamp = msg.header.stamp
        fname = os.path.join(self.save_dir, 'frame_%010d_%09d.npz' % (stamp.secs, stamp.nsecs))
        np.savez(fname, xyz=xyz, intensity=feats[:, 0], label=labels,
                 stamp=np.float64(stamp.to_sec()), frame_id=msg.header.frame_id)

        self.publish(msg, xyz, feats[:, 0], labels)

        counts = np.bincount(labels, minlength=len(LABEL_TO_NAME))
        rospy.loginfo('scan %.3f: %d pts in %.2fs  [%s]', stamp.to_sec(), len(xyz),
                      rospy.get_time() - t0,
                      ', '.join('%s:%d' % (LABEL_TO_NAME[i], c) for i, c in enumerate(counts)))

    def publish(self, msg, xyz, intensity, labels):
        if self.pub.get_num_connections() == 0:
            return
        n = len(xyz)
        data = np.zeros(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                  ('intensity', 'f4'), ('rgb', 'u4'), ('label', 'u4')])
        data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        data['intensity'] = intensity
        data['rgb'] = _PALETTE[labels]        # same colors as /semantic/map
        data['label'] = labels

        out = PointCloud2()
        out.header = msg.header
        out.height, out.width = 1, n
        out.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=16, datatype=PointField.UINT32, count=1),
            PointField(name='label', offset=20, datatype=PointField.UINT32, count=1),
        ]
        out.is_bigendian = False
        out.point_step = 24
        out.row_step = 24 * n
        out.is_dense = True
        out.data = data.tobytes()
        self.pub.publish(out)


if __name__ == '__main__':
    try:
        SegmentationNode().spin()
    except rospy.ROSInterruptException:
        pass
