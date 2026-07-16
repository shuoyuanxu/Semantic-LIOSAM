#!/usr/bin/env python3
"""
Generate clean LIO-SAM bags from AereHFeb2026 recordings.
  - Keep: /ouster/points, /velodyne_points, /ms/imu/data, /antobot_gps
  - Strip: /tf, /tf_static, /antobot_robot/*, cameras, etc.
  - Fix GPS header.stamp: frozen receiver clock -> bag recording time
  - Fix Ouster ring field: all-zero ring in organized (64x1024) cloud
    -> recompute from point index: ring = i // cloud.width
"""
import os, sys, struct, array, rosbag
from sensor_msgs.msg import NavSatFix

BAGS_DIR = "/media/shuoyuan/CrucialX9/crossseason"
SRC_DIR  = os.path.join(BAGS_DIR, "AereHFeb2026")

KEEP = ["/ouster/points", "/velodyne_points", "/ms/imu/data", "/antobot_gps"]

BAGS = {
    "easy_AprilAdd.bag":   "Feb2026_easy",
    "easy2_AprilAdd.bag":  "Feb2026_easy2",
    "medium_AprilAdd.bag": "Feb2026_medium",
    "hard1_AprilAdd.bag":  "Feb2026_hard1",
    "hard2_AprilAdd.bag":  "Feb2026_hard2",
}


def free_gb():
    s = os.statvfs(BAGS_DIR)
    return s.f_bavail * s.f_frsize / 1e9


def fix_ouster_ring(msg):
    """Set ring field for each point to its row index in the organized cloud."""
    rf = next((f for f in msg.fields if f.name == "ring"), None)
    if rf is None or msg.height <= 1:
        return msg   # unorganized or no ring field — leave as-is

    # Check if ring is already populated (not all-zero)
    step = msg.point_step
    sample = struct.unpack_from("H", msg.data, rf.offset)[0]
    sample2 = struct.unpack_from("H", msg.data, rf.offset + step * (msg.width // 2))[0]
    if sample != 0 or sample2 != 0:
        return msg   # already correct

    # Rewrite ring field for every point
    data = bytearray(msg.data)
    for i in range(msg.width * msg.height):
        ring_val = i // msg.width   # row index = ring index
        struct.pack_into("H", data, rf.offset + i * step, ring_val)
    msg.data = bytes(data)
    return msg


def process(orig, out_stem):
    in_path  = os.path.join(SRC_DIR, orig)
    out_path = os.path.join(BAGS_DIR, out_stem + ".bag")

    if not os.path.exists(in_path):
        print(f"  SKIP — not found: {in_path}")
        return False

    if os.path.exists(out_path):
        print(f"  SKIP — exists: {out_path} ({os.path.getsize(out_path)/1e9:.1f} GB)")
        return True

    avail = free_gb()
    print(f"  Disk free: {avail:.0f} GB")
    if avail < 25:
        print(f"  ERROR: <25 GB free — stopping")
        return False

    print(f"  {in_path} ({os.path.getsize(in_path)/1e9:.0f} GB) -> {out_path}")

    count = gps_fixed = ring_fixed = 0

    with rosbag.Bag(in_path, 'r') as inbag, rosbag.Bag(out_path, 'w') as outbag:
        for topic, msg, t in inbag.read_messages(topics=KEEP):
            if topic == "/antobot_gps":
                msg.header.stamp = t
                gps_fixed += 1
            elif topic == "/ouster/points":
                before = struct.unpack_from("H", msg.data,
                    next(f.offset for f in msg.fields if f.name=="ring"))[0]
                msg = fix_ouster_ring(msg)
                after  = struct.unpack_from("H", msg.data,
                    next(f.offset for f in msg.fields if f.name=="ring"))[0]
                if before == 0 and after != 0:
                    ring_fixed += 1
            outbag.write(topic, msg, t)
            count += 1
            if count % 2000 == 0:
                done_gb = os.path.getsize(out_path) / 1e9
                print(f"    {count} msgs, {done_gb:.1f} GB ...", end='\r', flush=True)

    out_gb = os.path.getsize(out_path) / 1e9
    print(f"\n  Done: {count} msgs | GPS fixed: {gps_fixed} | Ouster ring fixed: {ring_fixed} scans | {out_gb:.1f} GB")
    return True


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(BAGS.keys())
    for orig in targets:
        if orig not in BAGS:
            print(f"Unknown: {orig}. Known: {list(BAGS.keys())}")
            continue
        print(f"\n=== {orig} -> {BAGS[orig]} ===")
        if not process(orig, BAGS[orig]):
            print("Stopping.")
            break
    print("\nDone.")
