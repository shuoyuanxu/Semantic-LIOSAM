#!/usr/bin/env python3
"""
Merge LIO-SAM keyframe PCDs into a single map using saved poses.
Preserves all PCD fields (x, y, z, intensity, ring, time, etc).

Usage:
    python3 merge_keyframes.py [keyframes_dir] [output.pcd]
        [--voxel LEAF]
        [--voxel-ground LEAF] [--voxel-mid LEAF] [--voxel-ceil LEAF]
        [--cell-size M] [--ground-margin M] [--ceil-z M]
        [--min-obs N]
        [--sor] [--sor-k N] [--sor-std F]

Per-frame downsampling (in sensor frame, before transform):
    Each scan is classified into three height bands and downsampled
    independently before being transformed to world frame.
      ground : z < local_cell_zmin + ground-margin  (adaptive per cell)
      mid    : ground < z <= ceil-z                  (poles, vegetation)
      ceiling: z > ceil-z                            (fixed cutoff)

Dynamic object removal (--min-obs):
    Keeps only voxels observed from >= N distinct keyframes.
    Typical: 3-5 for slow scenes, 5-10 for busy ones.

Statistical outlier removal (--sor):
    Removes points whose neighbourhood distance is an outlier.
    --sor-std lower = more aggressive removal.
"""

import argparse
import os
import sys
import numpy as np


# ---------------------------------------------------------------------------
# PCD I/O
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
    with open(path, 'rb') as f:
        hdr = _parse_header(f)
        data_start = f.tell()
        raw = f.read()

    fields = hdr['FIELDS']
    sizes  = [int(s) for s in hdr['SIZE']]
    types  = hdr['TYPE']
    counts = [int(c) for c in hdr['COUNT']]
    n_pts  = int(hdr['POINTS'][0])

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
        points = np.loadtxt(path, skiprows=data_start, max_rows=n_pts, dtype=dtype)
    else:
        raise ValueError(f"Unsupported PCD data type: {hdr['data']}")
    return hdr, points


def write_pcd(path, points):
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
        types.append('F' if base_dt.kind == 'f' else
                     'I' if base_dt.kind == 'i' else 'U')

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(fields)}\n"
        f"SIZE {' '.join(str(s) for s in sizes)}\n"
        f"TYPE {' '.join(types)}\n"
        f"COUNT {' '.join(str(c) for c in counts)}\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\nDATA binary\n"
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
    return np.array([[1-(yy+zz), xy-wz,     xz+wy],
                     [xy+wz,     1-(xx+zz), yz-wx],
                     [xz-wy,     yz+wx,     1-(xx+yy)]])


def transform_points(points, x, y, z, qx, qy, qz, qw):
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
# Voxel filter — centroid averaging
# ---------------------------------------------------------------------------

def _voxel_keys(points, leaf):
    return np.floor(np.column_stack(
        [points['x'], points['y'], points['z']]) / leaf).astype(np.int64)


def _pack_keys(keys):
    return keys[:, 0] * (2**42) + keys[:, 1] * (2**21) + keys[:, 2]


def voxel_downsample(points, leaf):
    """Centroid-averaging voxel filter — averages float fields per voxel."""
    if leaf <= 0:
        return points

    packed = _pack_keys(_voxel_keys(points, leaf))
    sort_idx = np.argsort(packed, kind='stable')
    sorted_packed = packed[sort_idx]

    _, first_occ = np.unique(sorted_packed, return_index=True)
    counts = np.diff(np.concatenate([first_occ, [len(points)]]))

    out = points[sort_idx[first_occ]].copy()   # integer fields: keep first
    for field in points.dtype.names:            # float fields: centroid average
        if points.dtype[field].kind == 'f':
            vals = points[field][sort_idx].astype(np.float64)
            out[field] = (np.add.reduceat(vals, first_occ) / counts).astype(
                          points.dtype[field])
    return out


# ---------------------------------------------------------------------------
# Local height-band voxel filter (per-scan, sensor frame)
# ---------------------------------------------------------------------------

def _local_zmin(points, cell_size):
    """Smoothed per-cell z_min — removes hard seams at cell boundaries."""
    x = points['x'].astype(np.float64)
    y = points['y'].astype(np.float64)
    z = points['z'].astype(np.float64)

    cx = np.floor(x / cell_size).astype(np.int64)
    cy = np.floor(y / cell_size).astype(np.int64)

    cells = np.column_stack([cx, cy])
    unique_cells, cell_idx = np.unique(cells, axis=0, return_inverse=True)
    n_cells = len(unique_cells)

    z_min_raw = np.full(n_cells, np.inf)
    np.minimum.at(z_min_raw, cell_idx, z)

    cell_lookup = {(unique_cells[i, 0], unique_cells[i, 1]): i
                   for i in range(n_cells)}
    z_min_smooth = z_min_raw.copy()
    for i in range(n_cells):
        cxi, cyi = int(unique_cells[i, 0]), int(unique_cells[i, 1])
        nbr = [cell_lookup[k] for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               if (k := (cxi+dx, cyi+dy)) in cell_lookup and (dx or dy)]
        if nbr:
            z_min_smooth[i] = min(z_min_raw[i], z_min_raw[nbr].min())

    return z_min_smooth[cell_idx]


def voxel_downsample_local_bands(points, cell_size, ground_margin, ceil_z,
                                  voxel_ground, voxel_mid, voxel_ceil,
                                  verbose=True):
    z_min = _local_zmin(points, cell_size)
    z = points['z'].astype(np.float64)

    is_ground = z <= z_min + ground_margin
    is_ceil   = z > ceil_z
    is_mid    = ~is_ground & ~is_ceil

    parts = []
    for mask, leaf, label in [(is_ground, voxel_ground, 'ground'),
                               (is_mid,    voxel_mid,    'mid'),
                               (is_ceil,   voxel_ceil,   'ceiling')]:
        subset = points[mask]
        if len(subset) == 0:
            continue
        if leaf > 0:
            subset = voxel_downsample(subset, leaf)
        parts.append(subset)
        if verbose:
            print(f"  [{label:7s}]  {mask.sum():>9,} -> {len(subset):>9,}  (voxel={leaf}m)")

    return np.concatenate(parts) if parts else points[:0]


# ---------------------------------------------------------------------------
# Normal-based adaptive voxel (post-merge, world frame)
# ---------------------------------------------------------------------------

def normal_adaptive_downsample(points, normal_radius,
                                flat_threshold, voxel_flat, voxel_edge):
    """
    Estimate surface normals on the merged cloud, then split by flatness:
      |normal_z| > flat_threshold → flat surface (ground, walls, ceiling)
                                    → downsample at voxel_flat
      |normal_z| <= flat_threshold → curved/edge structure (poles, wires)
                                    → downsample at voxel_edge

    flat_threshold: 0.8 is a good start — near-vertical normals = flat surface
    normal_radius:  search radius for normal estimation (e.g. 0.1–0.3m)
    """
    import open3d as o3d

    xyz = np.column_stack([points['x'].astype(np.float64),
                           points['y'].astype(np.float64),
                           points['z'].astype(np.float64)])
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)

    print(f"  Estimating normals (radius={normal_radius}m)...")
    pc.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))

    normals = np.asarray(pc.normals)
    nz_abs  = np.abs(normals[:, 2])

    is_flat = nz_abs > flat_threshold
    is_edge = ~is_flat

    parts = []
    for mask, leaf, label in [(is_flat, voxel_flat, 'flat '),
                               (is_edge, voxel_edge, 'edge ')]:
        subset = points[mask]
        if len(subset) == 0:
            continue
        if leaf > 0:
            subset = voxel_downsample(subset, leaf)
        parts.append(subset)
        print(f"  [{label}] {mask.sum():>9,} pts -> {len(subset):>9,} pts  (voxel={leaf}m)")

    return np.concatenate(parts) if parts else points[:0]


# ---------------------------------------------------------------------------
# Dynamic object removal
# ---------------------------------------------------------------------------

def remove_dynamic(chunks, leaf, min_obs):
    from collections import defaultdict
    obs_count = defaultdict(int)
    for pts in chunks:
        if len(pts):
            for k in np.unique(_pack_keys(_voxel_keys(pts, leaf))):
                obs_count[k] += 1
    masks = []
    for pts in chunks:
        if len(pts) == 0:
            masks.append(np.zeros(0, dtype=bool))
        else:
            masks.append(np.vectorize(obs_count.__getitem__)(
                _pack_keys(_voxel_keys(pts, leaf))))
    return np.concatenate(masks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Merge keyframe PCDs preserving all fields.")
    parser.add_argument("kf_dir",  nargs="?", default=os.path.expanduser("~/Downloads/LOAM/keyframes"))
    parser.add_argument("output",  nargs="?", default=os.path.expanduser("~/Downloads/LOAM/merged_map.pcd"))
    parser.add_argument("--voxel",        type=float, default=0.1,  help="Uniform voxel size for all bands")
    parser.add_argument("--voxel-ground", type=float, default=None, help="Voxel size for ground band")
    parser.add_argument("--voxel-mid",    type=float, default=None, help="Voxel size for mid band (poles etc)")
    parser.add_argument("--voxel-ceil",   type=float, default=None, help="Voxel size for ceiling band")
    parser.add_argument("--cell-size",    type=float, default=1.0,  help="XY cell size for ground detection (default 1.0m)")
    parser.add_argument("--ground-margin",type=float, default=0.1,  help="Height above local z_min = ground (default 0.1m)")
    parser.add_argument("--ceil-z",       type=float, default=2.0,  help="Sensor-frame z cutoff for ceiling (default 2.0m)")
    parser.add_argument("--min-obs",      type=int,   default=0,    help="Min keyframes a voxel must be seen from (0=off)")
    parser.add_argument("--sor",               action="store_true",        help="Statistical outlier removal")
    parser.add_argument("--sor-k",             type=int,   default=20,    help="SOR neighbours (default 20)")
    parser.add_argument("--sor-std",           type=float, default=2.0,   help="SOR std-dev multiplier (default 2.0)")
    parser.add_argument("--normal-filter",     action="store_true",        help="Normal-based adaptive voxel after merging")
    parser.add_argument("--normal-radius",     type=float, default=0.15,  help="Normal estimation radius in metres (default 0.15)")
    parser.add_argument("--flat-threshold",    type=float, default=0.8,   help="|normal_z| above this = flat surface (default 0.8)")
    parser.add_argument("--voxel-flat",        type=float, default=0.3,   help="Voxel size for flat surfaces in normal filter (default 0.3)")
    parser.add_argument("--voxel-edge",        type=float, default=0.03,  help="Voxel size for edges/poles in normal filter (default 0.03)")
    args = parser.parse_args()

    voxel_ground = args.voxel_ground if args.voxel_ground is not None else args.voxel
    voxel_mid    = args.voxel_mid    if args.voxel_mid    is not None else args.voxel
    voxel_ceil   = args.voxel_ceil   if args.voxel_ceil   is not None else args.voxel
    multiband    = (args.voxel_ground is not None or
                    args.voxel_mid    is not None or
                    args.voxel_ceil   is not None)

    poses_file = os.path.join(args.kf_dir, "poses.csv")
    if not os.path.isfile(poses_file):
        sys.exit(f"poses.csv not found in {args.kf_dir}")

    poses = np.loadtxt(poses_file, delimiter=",", skiprows=1)
    if poses.ndim == 1:
        poses = poses[np.newaxis, :]
    print(f"Loaded {len(poses)} poses")

    chunks    = []
    missing   = 0
    total_raw = 0

    for row in poses:
        idx = int(row[0])
        x, y, z = row[1], row[2], row[3]
        qx, qy, qz, qw = row[4], row[5], row[6], row[7]

        pcd_path = os.path.join(args.kf_dir, f"kf_{idx:05d}.pcd")
        if not os.path.isfile(pcd_path):
            missing += 1
            continue
        _, pts = read_pcd(pcd_path)
        if len(pts) == 0:
            continue

        total_raw += len(pts)

        # downsample in sensor frame — ground is always at scan bottom here
        if multiband:
            pts = voxel_downsample_local_bands(
                pts, args.cell_size, args.ground_margin, args.ceil_z,
                voxel_ground, voxel_mid, voxel_ceil, verbose=False)
        elif args.voxel > 0:
            pts = voxel_downsample(pts, args.voxel)

        chunks.append(transform_points(pts, x, y, z, qx, qy, qz, qw))

        if (idx + 1) % 100 == 0 or idx == int(poses[-1, 0]):
            print(f"  frame {idx+1}/{len(poses)}  running total {sum(len(c) for c in chunks):,}")

    if missing:
        print(f"Warning: {missing} PCD files missing")
    if not chunks:
        sys.exit("No points — check that kf_*.pcd files exist.")

    print(f"Raw: {total_raw:,}  after per-frame downsample: {sum(len(c) for c in chunks):,}")
    merged = np.concatenate(chunks)

    if args.min_obs > 0:
        vote_leaf = max(args.voxel, 0.2)
        print(f"Dynamic removal: voxel={vote_leaf}m, min_obs={args.min_obs}...")
        obs = remove_dynamic(chunks, vote_leaf, args.min_obs)
        before = len(merged)
        merged = merged[obs >= args.min_obs]
        print(f"  {before:,} -> {len(merged):,} ({before-len(merged):,} removed)")

    dedup_leaf = min(voxel_ground, voxel_mid, voxel_ceil) if multiband else args.voxel
    if dedup_leaf > 0:
        before = len(merged)
        merged = voxel_downsample(merged, dedup_leaf)
        print(f"Dedup pass ({dedup_leaf}m): {before:,} -> {len(merged):,}")

    if args.normal_filter:
        print(f"Normal-based adaptive voxel  flat>{args.flat_threshold} "
              f"voxel_flat={args.voxel_flat}m  voxel_edge={args.voxel_edge}m...")
        merged = normal_adaptive_downsample(
            merged, args.normal_radius, args.flat_threshold,
            args.voxel_flat, args.voxel_edge)
        print(f"After normal filter: {len(merged):,} pts")

    if args.sor:
        import open3d as o3d
        print(f"SOR (k={args.sor_k}, std={args.sor_std})...")
        xyz = np.column_stack([merged['x'].astype(np.float64),
                               merged['y'].astype(np.float64),
                               merged['z'].astype(np.float64)])
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(xyz)
        _, inliers = pc.remove_statistical_outlier(
            nb_neighbors=args.sor_k, std_ratio=args.sor_std)
        before = len(merged)
        merged = merged[np.array(inliers)]
        print(f"  {before:,} -> {len(merged):,} ({before-len(merged):,} removed)")

    write_pcd(args.output, merged)
    print(f"Saved -> {args.output}  fields={list(merged.dtype.names)}  points={len(merged):,}")


if __name__ == "__main__":
    main()
