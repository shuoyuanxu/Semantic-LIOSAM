#!/usr/bin/env python3
"""
Extract /antobot_gps messages from a bag and write gnss.csv.
Format: timestamp_s, latitude, longitude, altitude, status
The bag recording time (t) is used as timestamp (not the frozen GPS receiver clock).
"""
import os, sys, csv, rosbag

BAGS_DIR = "/media/shuoyuan/CrucialX9/crossseason"
GPS_TOPIC = "/antobot_gps"

# All bags: clean_bag_path -> output_dir
ENTRIES = [
    # Nov2024
    ("Nov2024_Cal1.bag",            "Nov2024_Cal1"),
    ("Nov2024_Loc1.bag",            "Nov2024_Loc1"),
    ("Nov2024_Loc2.bag",            "Nov2024_Loc2"),
    ("Nov2024_Loc3.bag",            "Nov2024_Loc3"),
    # Jun2024
    ("Jun2024_GNSSLidarMiddle.bag", "Jun2024_GNSSLidarMiddle"),
    ("Jun2024_Middle.bag",          "Jun2024_Middle"),
    ("Jun2024_Normal.bag",          "Jun2024_Normal"),
    # Feb2026 AereH — use clean bags (GPS timestamps fixed)
    ("Feb2026_easy.bag",   "Feb2026_easy"),
    ("Feb2026_easy2.bag",  "Feb2026_easy2"),
    ("Feb2026_medium.bag", "Feb2026_medium"),
    ("Feb2026_hard1.bag",  "Feb2026_hard1"),
    ("Feb2026_hard2.bag",  "Feb2026_hard2"),
]


def extract(bag_path, out_dir):
    csv_path = os.path.join(out_dir, "gnss.csv")
    if not os.path.exists(bag_path):
        print(f"  SKIP — bag not found: {bag_path}")
        return
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    with rosbag.Bag(bag_path) as b, open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_s", "latitude", "longitude", "altitude", "status"])
        for topic, msg, t in b.read_messages(topics=[GPS_TOPIC]):
            writer.writerow([
                f"{t.to_sec():.6f}",
                f"{msg.latitude:.9f}",
                f"{msg.longitude:.9f}",
                f"{msg.altitude:.4f}",
                int(msg.status.status),
            ])
            count += 1
    if count == 0:
        os.remove(csv_path)
        print(f"  {os.path.basename(bag_path)}: no GPS data found")
    else:
        print(f"  {os.path.basename(bag_path)}: {count} fixes -> {csv_path}")


if __name__ == "__main__":
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    for bag_rel, out_stem in ENTRIES:
        bag_path = os.path.join(BAGS_DIR, bag_rel)
        out_dir = os.path.join(BAGS_DIR, out_stem)
        if targets and not any(t in out_stem for t in targets):
            continue
        print(f"[{out_stem}]")
        extract(bag_path, out_dir)
    print("Done.")
