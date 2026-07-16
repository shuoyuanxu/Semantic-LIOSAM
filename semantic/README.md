# Semantic mapping for LIO-SAM (forestry RandLA-Net)

Semantic SLAM add-on: LIO-SAM provides geometry, a RandLA-Net node labels the
raw Ouster scans (classes `trunk / crown / ground / others`), and a stitcher
fuses everything into a global semantic map using the optimized keyframe
poses. Fully automatic — launch, play a bag, the map appears when it ends.

## Folder layout

| Folder | Content |
|---|---|
| `data/` | datasets (raw labeled data, prepared training data) — **gitignored** |
| `model/` | deployed weights `checkpoint.tar` + training runs — **gitignored** |
| `include/` | library code, not run directly: `RandLANet.py`, `inference.py`, `dataset.py`, `config.py`, helpers, compiled ops (`utils/`) |
| `testandtraining/` | scripts you run by hand: `prepare_dataset.py`, `train.py`, `evaluate.py`, `stitch_semantic_map.py` |
| `rosnode/` | ROS nodes: `segmentation_node.py`, `semantic_map_finisher.py` |

## 1. Running semantic SLAM

```bash
# terminal 1
source ~/catkin_ws/devel/setup.bash
roslaunch lio_sam run.launch use_gps:=false use_semantic:=true

# terminal 2
rosbag play --clock -r 0.5 /path/to/your.bag
```

That's everything. What happens automatically:

- `mapOptmization` keeps `~/Downloads/LOAM/keyframes/poses.csv` current on
  every keyframe and after every loop closure (no `save_map` call needed)
- `segmentation_node` labels ~2 scans/s into
  `~/Downloads/LOAM/semantic/labeled_frames/` (cleared at each new launch)
- ~15 s after the lidar goes quiet (bag ended), `semantic_map_finisher`
  stitches and writes to `~/Downloads/LOAM/semantic/`:
  `semantic_map.ply` (view in CloudCompare — colored by class, `label` field),
  one PLY per class, and `semantic_map_preview.png`
- Ctrl-C mid-run also stitches whatever was recorded (in the background)

Every new run **overwrites** these outputs, like the rest of the LOAM folder.
To re-fuse by hand with different settings:

```bash
python3 testandtraining/stitch_semantic_map.py \
    --poses ~/Downloads/LOAM/keyframes/poses.csv \
    --frames ~/Downloads/LOAM/semantic/labeled_frames \
    --out ~/Downloads/LOAM/semantic/semantic_map.ply \
    --voxel 0.10 --split-classes --preview
```

Launch args (set in `run.launch` → `module_semantic.launch`): `min_interval`
(0.5 s between labeled scans), `use_crop` (near-field training crop),
`auto_stitch`, `stitch_voxel` (0.10 m), `idle_timeout` (15 s), `device`.

## 2. Training a model

### 2.1 Data format

One folder per labeled frame, containing exactly these four files:

```
<frame>/trunk.txt   crown.txt   ground.txt   others.txt
```

Each line is one point, 7 whitespace-separated columns:

```
x  y  z  intensity  reflectivity  ambient  label
```

- `x y z` in metres, **sensor frame**, within the working crop
  (x ∈ [-2, 3], y ∈ [-4, 4] — see `include/config.py: TRAIN_CROP`)
- `intensity reflectivity ambient` raw Ouster values
- the trailing `label` column is ignored — class comes from the file name
  (trunk=0, crown=1, ground=2, others=3)
- (0,0,0) points are dropped automatically

Frame folders can be nested any way you like (e.g. `Apr/1/`, `May/12/`).
**Frames whose path contains `Test` become the validation set** — keep a
representative held-out set there (change with `--val-pattern`).

The existing dataset (90 frames) is in `data/Labeled_dataset/`.

### 2.2 Prepare

```bash
cd testandtraining
python3 prepare_dataset.py --raw ../data/Labeled_dataset --out ../data/prepared
```

Converts to RandLA-Net format: 0.04 m grid-subsampled PLYs + KDTree/projection
pickles + `train_files.txt` / `val_files.txt`. Rerun whenever the raw data
changes. Prints per-class point counts — check `trunk` isn't starved.

### 2.3 Train

```bash
python3 train.py --data ../data/prepared --out ../model/run1
# smoke test first if you changed anything:  add --quick
```

- 50 epochs by default (`--epochs`), Adam 1e-2 with 0.95/epoch decay,
  class-weighted cross-entropy (weights computed from your data)
- per-epoch log: train/val loss, OA, per-class IoU, mIoU
  (also appended to `<out>/train_log.txt`)
- writes `checkpoint.tar` (latest) and `checkpoint_best.tar` (best val mIoU)
- ~20k-point patches; a training epoch is 300 steps × batch 4

### 2.4 Evaluate (full-resolution, voting — the honest number)

```bash
python3 evaluate.py --data ../data/prepared --checkpoint ../model/run1/checkpoint_best.tar
```

Reports overall accuracy + per-class IoU + mIoU on the validation clouds at
full resolution (patch voting + projection, same scheme as deployment).

### 2.5 Deploy

```bash
cp ../model/run1/checkpoint_best.tar ../model/checkpoint.tar
```

The ROS node picks it up on next launch. If you change classes or features,
update `include/config.py` (`num_classes`, `LABEL_TO_NAME`, `LABEL_TO_COLOR`)
— the checkpoint and config must agree.

## Environment

System python3 on this machine works as-is (torch + CUDA + rospy + sklearn +
scipy). The compiled ops (`include/utils/`) ship prebuilt `.so` for
py3.7/3.8; to rebuild for another python:

```bash
cd include/utils && sh cpp_wrappers/compile_wrappers.sh
cd nearest_neighbors && python setup.py install --home="."
```

## Classes / colors

| id | class | RGB |
|---|---|---|
| 0 | trunk | 139,69,19 brown |
| 1 | crown | 34,139,34 green |
| 2 | ground | 128,128,96 olive |
| 3 | others | 170,170,170 grey |
