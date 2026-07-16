# Semantic LIO-SAM — forestry semantic mapping
<img width="1760" height="1540" alt="image" src="https://github.com/user-attachments/assets/aa007832-f5da-4c12-a640-04b2523cca3b" />

LiDAR-inertial SLAM with an integrated semantic segmentation pipeline that
produces class-labeled 3D maps of orchard environments
(classes: `trunk`, `crown`, `ground`, `others`).

## What this fork adds

- **`semantic/` module** — a RandLA-Net segmentation node that labels raw
  Ouster scans alongside LIO-SAM and automatic final map generation when a run ends
- **Continuous pose export** — `mapOptmization` writes every deskewed
  keyframe cloud and keeps `keyframes/poses.csv` for future place recognition usage
- **Training pipeline** — dataset preparation, training and full-resolution
  evaluation for the orchard model, plus the labeled dataset itself
  (`semantic/testandtraining/`, `semantic/data/`)
- **Launch integration** — `use_semantic` / `use_gps` switches in
  `run.launch`, RViz displays for live labels and the clean semantic map

## Running 

```bash
# terminal 1
source ~/catkin_ws/devel/setup.bash
roslaunch lio_sam run.launch use_semantic:=true

# terminal 2
rosbag play --clock /path/to/your.bag
```

`--clock` is required (the stack runs on sim time). 

Every new run **overwrites** these outputs, like the rest of the LOAM
folder. To re-fuse by hand with different settings:

### Configuration

| Param | Default | Meaning |
|---|---|---|
| `raw_stitch` | `false` | `true` = raw stitching: no voxelisation, no clean-up — every labeled point kept (`--raw` on the CLI) |
| `map_voxel` | `0.10` m | fusion voxel size of the final map (bigger = more downsampled) |
| `ground_voxel` | `0.3` m | coarser re-grid of the ground class so trees stand out (0 = keep `map_voxel`) |
| `denoise_neighbors` / `denoise_radius` | `3` / `0.35` m | drop isolated noise voxels (0 = off) |
| `robot_radius` | `0.55` m | remove robot self-reflections around the sensor (0 = off) |
| `min_votes` | `1` | drop voxels observed by fewer points |
| `min_interval` | `0.5` s | time between labeled scans |
| `live_interval` / `live_voxel` | `15` s / `0.15` m | refresh rate / resolution of the live RViz map (`/semantic/map`) |
| `auto_stitch` / `idle_timeout` | `true` / `15` s | automatic final map when the lidar goes quiet |
| `split_classes` / `preview` | `true` | per-class PLYs / preview PNG |

## Training a model

### Data format

One folder per labeled frame, containing up to four files (a missing file
just means the frame has no points of that class):

```
<frame>/trunk.txt   crown.txt   ground.txt   others.txt
```

Each line is one point, 7 whitespace-separated columns:

```
x  y  z  intensity  reflectivity  ambient  label
```

- (0,0,0) points are dropped automatically

### Prepare → train → evaluate → deploy

```bash

# 1. convert to RandLA-Net format
python3 prepare_dataset.py --raw ../data/Labeled_dataset --out ../data/prepared

# 2. train (50 epochs; add --quick first as a smoke test)
python3 train.py --data ../data/prepared --out ../model/run1

# 3. full-resolution voting eval on the held-out clouds:
python3 evaluate.py --data ../data/prepared --checkpoint ../model/run1/checkpoint_best.tar

# 4. deploy — the ROS node picks it up on next launch
cp ../model/run1/checkpoint_best.tar ../model/checkpoint.tar
```

If you change classes or features, update `semantic/include/config.py`
(`num_classes`, `LABEL_TO_NAME`, `LABEL_TO_COLOR`) — the checkpoint and
config must agree.

## Based on LIO-SAM

This repository is built on **[LIO-SAM](https://github.com/TixiaoShan/LIO-SAM)**
by Tixiao Shan et al., which provides the underlying tightly-coupled
lidar-inertial odometry and factor-graph mapping. Please refer to the
upstream repository for the original documentation, dependencies (ROS,
GTSAM) and sensor setup guidance, and cite the original work:

The semantic segmentation network is
**[RandLA-Net](https://arxiv.org/abs/1911.11236)** (Hu et al., CVPR 2020),
PyTorch implementation lineage: [RandLA-Net-pytorch](https://github.com/qiqihaer/RandLA-Net-pytorch).
