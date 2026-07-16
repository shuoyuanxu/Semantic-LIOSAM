#!/usr/bin/env python3
"""Auto-generates the semantic map when a run ends.

Watches the raw lidar topic; when data has been flowing and then goes quiet
for ~idle_timeout seconds (bag finished / sensors stopped), it runs
stitch_semantic_map.py on the continuously-updated keyframes/poses.csv and
the labeled frames, writing the fused map + preview into the LOAM output
folder. Then it re-arms: play another bag in the same session and it will
stitch again when that one ends.

If the launch is Ctrl-C'd mid-run instead, a detached stitch is spawned on
shutdown so the map still gets built from whatever was recorded.
"""
import os
import subprocess
import sys
import threading
import time

import rospy
from sensor_msgs.msg import PointCloud2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STITCHER = os.path.join(os.path.dirname(BASE_DIR), 'testandtraining',
                        'stitch_semantic_map.py')


class Finisher:
    def __init__(self):
        rospy.init_node('semantic_map_finisher')

        loam_dir = os.path.expanduser('~') + rospy.get_param('lio_sam/savePCDDirectory',
                                                             '/Downloads/LOAM/')
        input_topic = rospy.get_param('~input_topic', '/ouster/points')
        self.idle_timeout = rospy.get_param('~idle_timeout', 15.0)   # wall seconds
        self.min_frames = rospy.get_param('~min_frames', 10)
        self.voxel = rospy.get_param('~voxel', 0.10)
        self.split_classes = rospy.get_param('~split_classes', True)
        self.preview = rospy.get_param('~preview', True)
        self.poses = rospy.get_param('~poses', os.path.join(loam_dir, 'keyframes', 'poses.csv'))
        self.frames_dir = rospy.get_param('~frames_dir',
                                          os.path.join(loam_dir, 'semantic', 'labeled_frames'))
        self.out = rospy.get_param('~out', os.path.join(loam_dir, 'semantic', 'semantic_map.ply'))

        self.lock = threading.Lock()
        self.last_msg_wall = None      # wall time of last raw scan
        self.dirty = False             # data arrived since the last stitch

        rospy.Subscriber(input_topic, PointCloud2, self.callback, queue_size=1)
        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo('semantic_map_finisher: watching %s (idle timeout %.0fs); '
                      'map -> %s', input_topic, self.idle_timeout, self.out)

    def callback(self, _msg):
        with self.lock:
            self.last_msg_wall = time.time()
            self.dirty = True

    def stitch_args(self):
        args = [sys.executable, STITCHER,
                '--poses', self.poses,
                '--frames', self.frames_dir,
                '--out', self.out,
                '--voxel', str(self.voxel)]
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
        if pending and self.ready():
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
                self.stitch()
                with self.lock:
                    self.dirty = False   # re-arm for a possible next bag
            time.sleep(0.5)


if __name__ == '__main__':
    Finisher().spin()
