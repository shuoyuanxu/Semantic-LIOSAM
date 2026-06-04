#!/usr/bin/env python3
"""
Merge LIO-SAM keyframe PCDs into a single map using saved poses.
Preserves all PCD fields (x, y, z, intensity, ring, time, etc).

Usage:
    python3 merge_keyframes.py [keyframes_dir] [output.pcd] [--voxel LEAF] [--min-obs N]

Defaults:
    keyframes_dir : ~/Downloads/LOAM/keyframes
    output.pcd    : ~/Downloads/LOAM/merged_map.pcd
    voxel         : 0.1 m   (set to 0 to skip downsampling)
    min-obs       : 0       (set to e.g. 3 to remove dynamic objects)

Dynamic object removal (--min-obs):
    Each 3-D voxel must be observed (have a point land in it) from at least
    N distinct keyframes to be kept. Static structures are seen from many frames;
    moving objects occupy a voxel for only a few frames and are discarded.
    Typical values: 3–5 for slow-moving scenes, 5–10 for busier scenes.
"""

import argparse
import os
import sys
import struct
import numpy as np


# ---------------------------------------------------------------------------
# PCD I/O (binary, preserves all fields)
# ---------------------------------------------------------------------------

TYPE_MAP = {'F': {4: np.float32, 8: np.float64},
            'I': {1: np.int8,  2: np.int16,  4: np.int32,  8: np.int64},
            'U': {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}}


def _parse_header(f):
    header = {}
    while True:
        line = f.readline().decode('ascii', errors='ignore').strip()
        if line.lower().startswith('data'):
            header['data'] = line.split()[1].lower()
            break
        if line.startswith('#') or not line:
            continue
        key, *vals = line.split()
        header[key.upper()] = vals
    return header


def read_pcd(path):
    """Return (header_dict, structured_numpy_array) for a binary PCD file."""
    with open(path, 'rb') as f:
        hdr = _parse_header(f)
        data_start = f.tell()
        raw = f.read()

    fields  = hdr['FIELDS']
    sizes   = [int(s) for s in hdr['SIZE']]
    types   = hdr['TYPE']
    counts  = [int(c) for c in hdr['COUNT']]
    n_pts   = int(hdr['POINTS'][0])

    # build numpy dtype from header
    dtype_list = []
    for fname, ftype, fsize, fcount in zip(fields, types, sizes, counts):
        np_type = TYPE_MAP[ftype][fsize]
        if fcount == 1:
            dtype_list.append((fname, np_type))
        else:
            dtype_list.append((fname, np_type, (fcount,)))
    dtype = np.dtype(dtype_list)

    if hdr['data'] == 'binary':
        points = np.frombuffer(raw[:n_pts * dtype.itemsize], dtype=dtype).copy()
    elif hdr['data'] == 'ascii':
        points = np.loadtxt(path, skiprows=data_start, max_rows=n_pts,
                            dtype=dtype)
    else:
        raise ValueError(f"Unsupported PCD data type: {hdr['data']}")

    return hdr, points


def write_pcd(path, points):
    """Write a structured numpy array as a binary PCD file."""
    n = len(points)
    fields, sizes, types, counts = [], [], [], []
    for name in points.dtype.names:
        dt = points.dtype[name]
        if dt.subdtype:
            base_dt, shape = dt.subdtype
            cnt = int(np.prod(shape))
        else:
            base_dt = dt
            cnt = 1
        fields.append(name)
        sizes.append(base_dt.itemsize)
        counts.append(cnt)
        # reverse-map numpy dtype → PCD type letter
        kind = base_dt.kind
        if kind == 'f':
            types.append('F')
        elif kind == 'i':
            types.append('I')
        else:
            types.append('U')

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(fields)}\n"
        f"SIZE {' '.join(str(s) for s in sizes)}\n"
        f"TYPE {' '.join(types)}\n"
        f"COUNT {' '.join(str(c) for c in counts)}\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(points.tobytes())


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def quat_to_rot(qx, qy, qz, qw):
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n < 1e-10:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = qw*qx*s, qw*qy*s, qw*qz*s
    xx, xy, xz = qx*qx*s, qx*qy*s, qx*qz*s
    yy, yz, zz = qy*qy*s, qy*qz*s, qz*qz*s
    return np.array([
        [1-(yy+zz),  xy-wz,     xz+wy],
        [xy+wz,      1-(xx+zz), yz-wx],
        [xz-wy,      yz+wx,     1-(xx+yy)]
    ])


def transform_points(points, x, y, z, qx, qy, qz, qw):
    """Rotate + translate the x,y,z fields in-place (copy first)."""
    pts = points.copy()
    R = quat_to_rot(qx, qy, qz, qw)
    t = np.array([x, y, z], dtype=np.float64)
    xyz = np.column_stack([pts['x'].astype(np.float64),
                           pts['y'].astype(np.float64),
                           pts['z'].astype(np.float64)])
    xyz_t = (R @ xyz.T).T + t
    pts['x'] = xyz_t[:, 0].astype(pts['x'].dtype)
    pts['y'] = xyz_t[:, 1].astype(pts['y'].dtype)
    pts['z'] = xyz_t[:, 2].astype(pts['z'].dtype)
    return pts


# ---------------------------------------------------------------------------
# Voxel filter (preserves all fields, keeps first point per voxel)
# ---------------------------------------------------------------------------

def _voxel_keys(points, leaf):
    return np.floor(np.column_stack(
        [points['x'], points['y'], points['z']]) / leaf).astype(np.int64)


def _pack_keys(keys):
    # pack three int64 voxel coords into one int64 key (safe up to ±2^20 voxels per axis)
    return keys[:, 0] * (2**42) + keys[:, 1] * (2**21) + keys[:, 2]


def voxel_downsample(points, leaf):
    if leaf <= 0:
        return points
    packed = _pack_keys(_voxel_keys(points, leaf))
    _, first_idx = np.unique(packed, return_index=True)
    return points[np.sort(first_idx)]


# ---------------------------------------------------------------------------
# Dynamic object removal via temporal consistency voting
# ---------------------------------------------------------------------------

def remove_dynamic(chunks, poses, leaf):
    """
    Keep only voxels observed from >= min_obs distinct keyframes.

    chunks : list of transformed structured arrays (one per keyframe)
    poses  : Nx9 array (id, x, y, z, qx, qy, qz, qw, ts) — used for count only
    leaf   : voxel size used for voting (should match or be coarser than output voxel)

    Returns a boolean mask array over np.concatenate(chunks).
    """
    from collections import defaultdict

    # count how many distinct frames each voxel was observed in
    obs_count = defaultdict(int)
    for pts in chunks:
        if len(pts) == 0:
            continue
        keys = _pack_keys(_voxel_keys(pts, leaf))
        for k in np.unique(keys):
            obs_count[k] += 1

    # build mask over the full merged array
    masks = []
    for pts in chunks:
        if len(pts) == 0:
            masks.append(np.zeros(0, dtype=bool))
            continue
        keys = _pack_keys(_voxel_keys(pts, leaf))
        masks.append(np.vectorize(obs_count.__getitem__)(keys))
    return np.concatenate(masks)  # per-point observation count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Merge keyframe PCDs preserving all fields.")
    parser.add_argument("kf_dir", nargs="?",
                        default=os.path.expanduser("~/Downloads/LOAM/keyframes"))
    parser.add_argument("output", nargs="?",
                        default=os.path.expanduser("~/Downloads/LOAM/merged_map.pcd"))
    parser.add_argument("--voxel", type=float, default=0.1,
                        help="Voxel leaf size in metres (0 = no downsampling)")
    parser.add_argument("--min-obs", type=int, default=0,
                        help="Min keyframes a voxel must be seen from to keep it (0 = disabled)")
    args = parser.parse_args()

    poses_file = os.path.join(args.kf_dir, "poses.csv")
    if not os.path.isfile(poses_file):
        sys.exit(f"poses.csv not found in {args.kf_dir}")

    poses = np.loadtxt(poses_file, delimiter=",", skiprows=1)
    if poses.ndim == 1:
        poses = poses[np.newaxis, :]
    print(f"Loaded {len(poses)} poses")

    chunks = []
    missing = 0

    for row in poses:
        idx = int(row[0])
        x, y, z  = row[1], row[2], row[3]
        qx, qy, qz, qw = row[4], row[5], row[6], row[7]

        pcd_path = os.path.join(args.kf_dir, f"kf_{idx:05d}.pcd")
        if not os.path.isfile(pcd_path):
            missing += 1
            continue

        _, pts = read_pcd(pcd_path)
        if len(pts) == 0:
            continue

        pts = transform_points(pts, x, y, z, qx, qy, qz, qw)
        chunks.append(pts)

        if (idx + 1) % 50 == 0 or idx == int(poses[-1, 0]):
            total = sum(len(c) for c in chunks)
            print(f"  merged {idx+1}/{len(poses)} frames  ({total} pts so far)")

    if missing:
        print(f"Warning: {missing} PCD files were missing")

    if not chunks:
        sys.exit("No points — check that kf_*.pcd files exist in the directory.")

    # dynamic removal before concatenation (needs per-frame data)
    if args.min_obs > 0:
        vote_leaf = max(args.voxel, 0.2)   # voting voxel >= output voxel
        print(f"Dynamic removal: voting voxel={vote_leaf}m, min_obs={args.min_obs} frames...")
        obs = remove_dynamic(chunks, poses, vote_leaf)
        merged = np.concatenate(chunks)
        before = len(merged)
        merged = merged[obs >= args.min_obs]
        print(f"Dynamic removal: {before} → {len(merged)} pts "
              f"({before - len(merged)} removed)")
    else:
        merged = np.concatenate(chunks)

    print(f"Total before downsampling: {len(merged)} pts")

    if args.voxel > 0:
        merged = voxel_downsample(merged, args.voxel)
        print(f"After voxel ({args.voxel}m): {len(merged)} pts")

    write_pcd(args.output, merged)
    fields = list(merged.dtype.names)
    print(f"Saved → {args.output}  fields={fields}  points={len(merged)}")


if __name__ == "__main__":
    main()
