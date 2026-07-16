#!/usr/bin/env python3
"""Auto-generates the semantic map when a run ends.

Watches the raw lidar topic; when data has been flowing and then goes quiet
for ~idle_timeout seconds (bag finished / sensors stopped), it runs
stitch_semantic_map.py on the continuously-updated keyframes/poses.csv and
the labeled frames, writing the fused map + preview into the LOAM output
folder. Then it re-arms: play another bag in the same session and it will
stitch again when that one ends.

While the run is going it also publishes a CLEAN fused map (voxelized,
denoised, ground-thinned — same processing as the final map) on
~live_map_topic every ~live_interval seconds, rebuilt from the labeled
frames and the current optimized poses. This is what the RViz "Semantic
map" display shows.

If the launch is Ctrl-C'd mid-run instead, a detached stitch is spawned on
shutdown so the map still gets built from whatever was recorded.
"""
import glob
import os
import subprocess
import sys
import threading
import time

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STITCHER = os.path.join(os.path.dirname(BASE_DIR), 'testandtraining',
                        'stitch_semantic_map.py')
sys.path.insert(0, os.path.join(os.path.dirname(BASE_DIR), 'include'))

import fusion
from config import LABEL_TO_COLOR


def param(name, default):
    """Config lookup: private ~name overrides the `semantic:` section of
    config/params.yml, which overrides the built-in default."""
    return rospy.get_param('~' + name, rospy.get_param('semantic/' + name, default))


class Finisher:
    def __init__(self):
        rospy.init_node('semantic_map_finisher')

        loam_dir = os.path.expanduser('~') + rospy.get_param('lio_sam/savePCDDirectory',
                                                             '/Downloads/LOAM/')
        input_topic = param('input_topic', '/ouster/points')
        self.auto_stitch = param('auto_stitch', True)
        self.idle_timeout = param('idle_timeout', 15.0)   # wall seconds
        self.min_frames = param('min_frames', 10)
        # final map options (see the `semantic:` section of config/params.yml)
        self.voxel = param('map_voxel', 0.10)
        self.ground_voxel = param('ground_voxel', 0.3)
        self.denoise_neighbors = param('denoise_neighbors', 3)
        self.denoise_radius = param('denoise_radius', 0.35)
        self.min_votes = param('min_votes', 1)
        self.robot_radius = param('robot_radius', 0.55)
        self.raw_stitch = param('raw_stitch', False)
        self.split_classes = param('split_classes', True)
        self.preview = param('preview', True)
        self.poses = param('poses', os.path.join(loam_dir, 'keyframes', 'poses.csv'))
        self.frames_dir = param('frames_dir',
                                os.path.join(loam_dir, 'semantic', 'labeled_frames'))
        self.out = param('out', os.path.join(loam_dir, 'semantic', 'semantic_map.ply'))
        self.map_frame = param('map_frame', rospy.get_param('lio_sam/mapFrame', 'map'))
        self.live_interval = param('live_interval', 15.0)  # 0 disables
        live_topic = param('live_map_topic', '/semantic/map')
        self.live_voxel = param('live_voxel', max(self.voxel, 0.15))

        self.lock = threading.Lock()
        self.last_msg_wall = None      # wall time of last raw scan
        self.last_msg_stamp = None     # sensor time of last raw scan
        self.dirty = False             # data arrived since the last stitch
        self.frame_cache = {}          # npz path -> (stamp, xyz, label)
        self.last_live = 0.0

        self.live_pub = rospy.Publisher(live_topic, PointCloud2,
                                        queue_size=1, latch=True)
        rospy.Subscriber(input_topic, PointCloud2, self.callback, queue_size=1)
        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo('semantic_map_finisher: watching %s (idle timeout %.0fs); '
                      'live map on %s every %.0fs; final map -> %s',
                      input_topic, self.idle_timeout, live_topic,
                      self.live_interval, self.out)

    def callback(self, msg):
        stamp = msg.header.stamp.to_sec()
        with self.lock:
            # bag restarted (sensor time jumped back): forget the old run
            if self.last_msg_stamp is not None and stamp < self.last_msg_stamp - 5.0:
                self.frame_cache.clear()
                rospy.logwarn('bag restart detected — cleared cached frames '
                              'from the previous run')
            self.last_msg_stamp = stamp
            self.last_msg_wall = time.time()
            self.dirty = True

    # ------------------------- live clean map -------------------------

    def load_new_frames(self):
        for f in glob.glob(os.path.join(self.frames_dir, 'frame_*.npz')):
            if f in self.frame_cache:
                continue
            try:
                d = np.load(f, allow_pickle=True)
                self.frame_cache[f] = (float(d['stamp']),
                                       d['xyz'].astype(np.float32),
                                       d['label'].astype(np.uint8))
            except Exception:
                pass          # probably still being written — retry next tick

    def publish_live_map(self):
        if not os.path.isfile(self.poses):
            return
        self.load_new_frames()
        if len(self.frame_cache) < self.min_frames:
            return
        lookup = fusion.PoseLookup(*fusion.load_poses(self.poses))

        pts_list, lab_list = [], []
        for stamp, xyz, lab in self.frame_cache.values():
            pose = lookup.get(stamp)
            if pose is None:
                continue
            R, t = pose
            pts_list.append((xyz.astype(np.float64) @ R.as_matrix().T + t)
                            .astype(np.float32))
            lab_list.append(lab)
        if not pts_list:
            return

        if self.raw_stitch:
            # raw mode: publish every labeled point untouched (capped so the
            # message doesn't grow unbounded on very long runs)
            xyz = np.concatenate(pts_list)
            label = np.concatenate(lab_list)
            max_pts = 5_000_000
            if len(xyz) > max_pts:
                rospy.logwarn_throttle(
                    60, 'raw live map: %d points, showing a random %d — the '
                    'final semantic_map.ply still contains everything',
                    len(xyz), max_pts)
                pick = np.random.choice(len(xyz), max_pts, replace=False)
                xyz, label = xyz[pick], label[pick]
        else:
            # same clean-up chain as the final map, at live resolution
            xyz, label = fusion.fuse(np.concatenate(pts_list),
                                     np.concatenate(lab_list), self.live_voxel)
            xyz, label, _ = fusion.denoise(xyz, label, self.denoise_neighbors,
                                           max(self.denoise_radius, 2.5 * self.live_voxel))
            xyz, label = fusion.downsample_ground(
                xyz, label, max(self.ground_voxel, 2 * self.live_voxel), self.live_voxel)

        palette = np.array([LABEL_TO_COLOR[i] for i in range(4)], dtype=np.uint32)
        rgb = (palette[label, 0] << 16) | (palette[label, 1] << 8) | palette[label, 2]

        n = len(xyz)
        data = np.zeros(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                  ('rgb', 'u4'), ('label', 'u4')])
        data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        data['rgb'], data['label'] = rgb, label

        msg = PointCloud2()
        msg.header = Header(stamp=rospy.Time.now(), frame_id=self.map_frame)
        msg.height, msg.width = 1, n
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
            PointField(name='label', offset=16, datatype=PointField.UINT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 20
        msg.row_step = 20 * n
        msg.is_dense = True
        msg.data = data.tobytes()
        self.live_pub.publish(msg)
        rospy.loginfo_throttle(60, 'live semantic map: %d %s from %d frames',
                               n, 'raw points' if self.raw_stitch else 'clean voxels',
                               len(pts_list))

    def stitch_args(self):
        args = [sys.executable, STITCHER,
                '--poses', self.poses,
                '--frames', self.frames_dir,
                '--out', self.out,
                '--voxel', str(self.voxel),
                '--ground-voxel', str(self.ground_voxel),
                '--denoise-neighbors', str(self.denoise_neighbors),
                '--denoise-radius', str(self.denoise_radius),
                '--min-votes', str(self.min_votes),
                '--robot-radius', str(self.robot_radius)]
        if self.raw_stitch:
            args.append('--raw')
        if self.split_classes:
            args.append('--split-classes')
        if self.preview:
            args.append('--preview')
        return args

    def ready(self):
        if not os.path.isfile(self.poses):
            rospy.logwarn('no %s — was mapOptmization running?', self.poses)
            return False
        try:
            n = len([f for f in os.listdir(self.frames_dir) if f.startswith('frame_')])
        except OSError:
            n = 0
        if n < self.min_frames:
            rospy.logwarn('only %d labeled frames in %s — not stitching',
                          n, self.frames_dir)
            return False
        return True

    def stitch(self):
        if not self.ready():
            return
        rospy.loginfo('run finished — generating semantic map ...')
        result = subprocess.run(self.stitch_args(), capture_output=True, text=True)
        for line in result.stdout.strip().splitlines()[-8:]:
            rospy.loginfo('  %s', line)
        if result.returncode == 0:
            rospy.loginfo('semantic map written to %s', self.out)
        else:
            rospy.logerr('stitching failed:\n%s', result.stderr.strip()[-2000:])

    def on_shutdown(self):
        # Ctrl-C mid-run: stitch what we have, detached so roslaunch's
        # escalation timeout can't kill it half-way through
        with self.lock:
            pending = self.dirty
        if pending and self.auto_stitch and self.ready():
            print('[semantic_map_finisher] shutdown — stitching in background, '
                  'map will appear at %s' % self.out)
            subprocess.Popen(self.stitch_args(), start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def spin(self):
        # wall-clock sleep, NOT rospy.Rate: with use_sim_time the /clock stops
        # when the bag ends, which would freeze a sim-time loop right before
        # the one moment this node has to act
        while not rospy.is_shutdown():
            with self.lock:
                last, pending = self.last_msg_wall, self.dirty
            if pending and last is not None and time.time() - last > self.idle_timeout:
                if self.auto_stitch:
                    self.stitch()
                with self.lock:
                    self.dirty = False   # re-arm for a possible next bag
            elif (self.live_interval > 0 and pending
                    and time.time() - self.last_live > self.live_interval):
                try:
                    self.publish_live_map()
                except Exception as e:
                    rospy.logwarn_throttle(60, 'live map failed: %s', e)
                self.last_live = time.time()
            time.sleep(0.5)


if __name__ == '__main__':
    Finisher().spin()
