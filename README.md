# Semantic LIO-SAM — forestry / orchard semantic mapping
<img width="1760" height="1540" alt="image" src="https://github.com/user-attachments/assets/aa007832-f5da-4c12-a640-04b2523cca3b" />

LiDAR-inertial SLAM with an integrated semantic segmentation pipeline that
produces class-labeled 3D maps of orchard / forestry environments
(classes: `trunk`, `crown`, `ground`, `others`). One launch + one bag
in, a clean fused semantic map out — no manual steps.

## Based on LIO-SAM

This repository is built on **[LIO-SAM](https://github.com/TixiaoShan/LIO-SAM)**
by Tixiao Shan et al., which provides the underlying tightly-coupled
lidar-inertial odometry and factor-graph mapping. Please refer to the
upstream repository for the original documentation, dependencies (ROS,
GTSAM) and sensor setup guidance, and cite the original work:

```bibtex
@inproceedings{liosam2020shan,
  title     = {LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping},
  author    = {Shan, Tixiao and Englot, Brendan and Meyers, Drew and Wang, Wei and Ratti, Carlo and Rus, Daniela},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages     = {5135-5142},
  year      = {2020},
  organization = {IEEE}
}
```

The semantic segmentation network is
**[RandLA-Net](https://arxiv.org/abs/1911.11236)** (Hu et al., CVPR 2020),
PyTorch implementation lineage: [RandLA-Net-pytorch](https://github.com/qiqihaer/RandLA-Net-pytorch).

## What this fork adds

- **`semantic/` module** — a RandLA-Net segmentation node that labels raw
  Ouster scans alongside LIO-SAM (never in its critical path), a live clean
  fused map in RViz, and automatic final map generation when a run ends
  (`semantic/rosnode/`, `semantic/include/`)
- **Continuous pose export** — `mapOptmization` writes every deskewed
  keyframe cloud and keeps `keyframes/poses.csv` up to date on every
  keyframe and loop closure, so the optimized trajectory is always on disk
  (no `save_map` service call needed)
- **Map post-processing** — voxel fusion with per-class majority voting,
  robot self-reflection removal, isolated-noise removal, ground
  downsampling (`semantic/include/fusion.py`)
- **Training pipeline** — dataset preparation, training and full-resolution
  evaluation for the forestry model, plus the labeled dataset itself
  (`semantic/testandtraining/`, `semantic/data/`)
- **Launch integration** — `use_semantic` / `use_gps` switches in
  `run.launch`, RViz displays for live labels and the clean semantic map

## Running semantic SLAM

```bash
# terminal 1
source ~/catkin_ws/devel/setup.bash
roslaunch lio_sam run.launch use_semantic:=true

# terminal 2
rosbag play --clock /path/to/your.bag
```

`--clock` is required (the stack runs on sim time). If LIO-SAM's mapping
can't keep up at full rate (smeared/doubled rows in the map), slow the
playback down with `-r 0.5`.

GPS is independent of the semantic pipeline: `use_gps` defaults to on, and
the stitched map automatically inherits whatever pose corrections GPS
factors provide. Add `use_gps:=false` only when the GNSS in a dataset is
unreliable (e.g. under dense canopy).

What happens automatically:

- `mapOptmization` keeps `~/Downloads/LOAM/keyframes/poses.csv` current on
  every keyframe and after every loop closure
- `segmentation_node` labels ~2 scans/s into
  `~/Downloads/LOAM/semantic/labeled_frames/` (cleared at each new launch)
- a clean fused map is published on `/semantic/map` every ~15 s and shown
  in RViz ("Semantic map" display, same color scheme as the per-scan labels)
- ~15 s after the lidar goes quiet (bag ended), `semantic_map_finisher`
  stitches and writes to `~/Downloads/LOAM/semantic/`:
  `semantic_map.ply` (view in CloudCompare — colored by class, `label`
  field), one PLY per class, and `semantic_map_preview.png`
- Ctrl-C mid-run also stitches whatever was recorded (in the background)

Every new run **overwrites** these outputs, like the rest of the LOAM
folder. To re-fuse by hand with different settings:

```bash
python3 semantic/testandtraining/stitch_semantic_map.py \
    --poses ~/Downloads/LOAM/keyframes/poses.csv \
    --frames ~/Downloads/LOAM/semantic/labeled_frames \
    --out ~/Downloads/LOAM/semantic/semantic_map.ply \
    --voxel 0.10 --split-classes --preview
```

### Configuration

All semantic knobs live in the **`semantic:` section of
[`config/params.yaml`](config/params.yaml)** — the same file as the LIO-SAM
parameters. The ones you'll touch most:

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

The manual stitcher accepts the same options as CLI flags
(`--voxel`, `--ground-voxel`, `--denoise-neighbors`, `--denoise-radius`,
`--robot-radius`, `--min-votes`, plus `--max-dt`/`--max-gap` for pose
matching), so you can re-fuse the same run at different resolutions without
touching the yaml.

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

- `x y z` in metres, **sensor frame**, within the working crop
  (x ∈ [-2, 3], y ∈ [-4, 4] — see `semantic/include/config.py: TRAIN_CROP`)
- `intensity reflectivity ambient` raw Ouster values
- the trailing `label` column is ignored — class comes from the file name
  (trunk=0, crown=1, ground=2, others=3)
- (0,0,0) points are dropped automatically

Frame folders can be nested any way you like (e.g. `Apr/1/`, `May/12/`).
**Frames whose path contains `Test` become the validation set** — keep a
representative held-out set there (change with `--val-pattern`).

The existing dataset (92 frames) is in `semantic/data/Labeled_dataset/`.

### Prepare → train → evaluate → deploy

```bash
cd semantic/testandtraining

# 1. convert to RandLA-Net format (0.04 m subsampled PLYs + KDTree/projection)
python3 prepare_dataset.py --raw ../data/Labeled_dataset --out ../data/prepared

# 2. train (50 epochs; add --quick first as a smoke test)
python3 train.py --data ../data/prepared --out ../model/run1

# 3. honest number: full-resolution voting eval on the held-out clouds
python3 evaluate.py --data ../data/prepared --checkpoint ../model/run1/checkpoint_best.tar

# 4. deploy — the ROS node picks it up on next launch
cp ../model/run1/checkpoint_best.tar ../model/checkpoint.tar
```

Training uses class-weighted cross-entropy (weights computed from your
data), logs per-epoch val IoU/mIoU to `<out>/train_log.txt`, and writes
`checkpoint.tar` (latest) + `checkpoint_best.tar` (best val mIoU). The
currently deployed model reaches **mIoU 0.86** (OA 0.99) on the held-out
validation clouds.

If you change classes or features, update `semantic/include/config.py`
(`num_classes`, `LABEL_TO_NAME`, `LABEL_TO_COLOR`) — the checkpoint and
config must agree.

## Repository layout (semantic parts)

| Path | Content |
|---|---|
| `semantic/data/` | datasets (raw labeled data in git; `prepared/` is regenerable and gitignored) |
| `semantic/model/` | deployed weights `checkpoint.tar` (in git); training runs gitignored |
| `semantic/include/` | library code: `RandLANet.py`, `inference.py`, `fusion.py`, `dataset.py`, `config.py`, helpers, compiled ops |
| `semantic/testandtraining/` | scripts run by hand: `prepare_dataset.py`, `train.py`, `evaluate.py`, `stitch_semantic_map.py` |
| `semantic/rosnode/` | ROS nodes: `segmentation_node.py`, `semantic_map_finisher.py` |

## Environment

Beyond the standard LIO-SAM dependencies (ROS1, GTSAM — see upstream), the
semantic pipeline needs python3 with PyTorch (CUDA recommended),
scikit-learn and scipy. The compiled ops (`semantic/include/utils/`) ship
prebuilt `.so` files for py3.7/3.8; to rebuild for another python:

```bash
cd semantic/include/utils && sh cpp_wrappers/compile_wrappers.sh
cd nearest_neighbors && python setup.py install --home="."
```

## Classes / colors

| id | class | RGB (PLY export) |
|---|---|---|
| 0 | trunk | 139,69,19 brown |
| 1 | crown | 34,139,34 green |
| 2 | ground | 128,128,96 olive |
| 3 | others | 170,170,170 grey |

In RViz both the per-scan labels and the fused map color by the `label`
channel (bright rainbow); switch the display's Color Transformer to RGB8
for the true class colors above.
