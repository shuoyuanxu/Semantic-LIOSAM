#!/usr/bin/env python3
"""
Generate clean crossseason bags in one pass:
  - Keep only LiDAR, IMU, GPS, tag detection topics
  - Strip /tf and /tf_static (pre-existing 'map' frame conflicts with LIO-SAM)
  - Fix /antobot_gps header.stamp: receiver clock was frozen, replace with bag
    recording time (t) so stamps match the LiDAR/IMU timeline
"""
import os
import sys
import rosbag
from sensor_msgs.msg import NavSatFix

BAGS_DIR = "/media/shuoyuan/CrucialX9/crossseason"

KEEP_TOPICS = [
    "/ouster/points", "/velodyne_points", "/ms/imu/data",
    "/antobot_gps", "/l/tag_detections", "/m/tag_detections", "/r/tag_detections",
]

BAGS = {
    "Cal1.bag":              "Nov2024_Cal1",
    "GNSSLidarMiddle.bag":   "Jun2024_GNSSLidarMiddle",
    "Loc1.bag":              "Nov2024_Loc1",
    "Loc2.bag":              "Nov2024_Loc2",
    "Loc3.bag":              "Nov2024_Loc3",
    "Middle.bag":            "Jun2024_Middle",
    "Normal.bag":            "Jun2024_Normal",
}


def free_gb(path):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize / 1e9


def process_bag(orig_name, out_stem):
    in_path = os.path.join(BAGS_DIR, orig_name)
    out_path = os.path.join(BAGS_DIR, out_stem + ".bag")

    if not os.path.exists(in_path):
        print(f"  SKIP — not found: {in_path}")
        return False

    if os.path.exists(out_path):
        size_gb = os.path.getsize(out_path) / 1e9
        print(f"  SKIP — already exists: {out_path} ({size_gb:.1f} GB)")
        return True

    avail = free_gb(BAGS_DIR)
    print(f"  Disk free: {avail:.0f} GB")
    if avail < 25:
        print(f"  ERROR: less than 25 GB free — aborting to avoid filling drive")
        return False

    print(f"  {in_path} ({os.path.getsize(in_path)/1e9:.0f} GB)")
    print(f"  -> {out_path}")

    gps_fixed = 0
    count = 0

    with rosbag.Bag(in_path, 'r') as inbag, rosbag.Bag(out_path, 'w') as outbag:
        for topic, msg, t in inbag.read_messages(topics=KEEP_TOPICS):
            if topic == "/antobot_gps":
                msg.header.stamp = t   # fix frozen GPS receiver clock
                gps_fixed += 1
            outbag.write(topic, msg, t)
            count += 1
            if count % 5000 == 0:
                done_gb = os.path.getsize(out_path) / 1e9
                print(f"    {count} msgs, {done_gb:.1f} GB written ...", end='\r', flush=True)

    out_gb = os.path.getsize(out_path) / 1e9
    print(f"\n  Done: {count} messages, {gps_fixed} GPS fixed, {out_gb:.1f} GB")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        targets = sys.argv[1:]  # e.g.  Cal1.bag Loc1.bag
    else:
        targets = list(BAGS.keys())

    for orig in targets:
        if orig not in BAGS:
            print(f"Unknown bag '{orig}'. Known: {list(BAGS.keys())}")
            continue
        print(f"\n=== {orig} -> {BAGS[orig]}.bag ===")
        ok = process_bag(orig, BAGS[orig])
        if not ok:
            print("Stopping.")
            break

    print("\nDone.")
