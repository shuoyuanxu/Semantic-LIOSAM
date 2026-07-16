#!/bin/bash
# run_all_bags.sh
# For each bag: run LIO-SAM with GNSS -> save outputs -> delete original -> repeat.
# Plays directly from Thunderbolt external drive (no local copy needed).
#
# Output per bag (in BAGS_DIR/<MonYYYY_Name>/):
#   trajectory.pcd, transformations.pcd, CornerMap.pcd, SurfMap.pcd, GlobalMap.pcd
#   poses.csv
#   keyframes/kf_*.pcd
#
# Usage:
#   ./run_all_bags.sh               # all bags
#   ./run_all_bags.sh Cal1          # single bag by stem

set -e

BAGS_DIR="/media/shuoyuan/CrucialX9/crossseason"
LOAM_DIR="$HOME/Downloads/LOAM"
LIO_SAM_WS="$HOME/catkin_slam_ws"

source "$LIO_SAM_WS/devel/setup.bash"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# clean_bag | original_bag (or "" if already deleted) | out_stem
BAGS=(
    "Nov2024_Cal1.bag|Cal1.bag|Nov2024_Cal1"
    "Jun2024_GNSSLidarMiddle.bag|GNSSLidarMiddle.bag|Jun2024_GNSSLidarMiddle"
    "Nov2024_Loc1.bag|Loc1.bag|Nov2024_Loc1"
    "Nov2024_Loc2.bag|Loc2.bag|Nov2024_Loc2"
    "Nov2024_Loc3.bag|Loc3.bag|Nov2024_Loc3"
    "Jun2024_Middle.bag|Middle.bag|Jun2024_Middle"
    "Jun2024_Normal.bag|Normal.bag|Jun2024_Normal"
)

kill_ros() {
    pkill -f "lio_sam_|roslaunch|rosbag play|roscore" 2>/dev/null || true
    sleep 4
}

start_liosam() {
    log "[ros] Starting roscore + LIO-SAM with GPS..."
    rm -rf "$LOAM_DIR" && mkdir -p "$LOAM_DIR/keyframes"
    roscore > /tmp/roscore.log 2>&1 &
    sleep 4
    rosparam set /use_sim_time true
    roslaunch lio_sam run.launch use_gps:=false > /tmp/liosam_run.log 2>&1 &
    LIOSAM_PID=$!
    log "[ros] Waiting 12s for nodes..."
    sleep 12
    echo "$LIOSAM_PID"
}

save_and_copy() {
    local out_dir="$1"
    local liosam_pid="$2"

    log "[save] Bag finished. Waiting 10s for last keyframes..."
    sleep 10

    log "[save] Calling save_map service..."
    rosservice call /lio_sam/save_map "{}" 2>/dev/null || {
        log "[save] Service call failed, falling back to SIGINT..."
        kill -INT "$liosam_pid" 2>/dev/null || pkill -INT -f "roslaunch" || true
    }
    sleep 30

    mkdir -p "$out_dir/keyframes"

    # 5 global map PCDs → root of output dir
    for pcd in trajectory.pcd transformations.pcd CornerMap.pcd SurfMap.pcd GlobalMap.pcd; do
        if [ -f "$LOAM_DIR/$pcd" ]; then
            cp "$LOAM_DIR/$pcd" "$out_dir/$pcd"
        else
            log "[save] WARNING: $pcd not found"
        fi
    done

    # poses.csv → root of output dir
    if [ -f "$LOAM_DIR/keyframes/poses.csv" ]; then
        cp "$LOAM_DIR/keyframes/poses.csv" "$out_dir/poses.csv"
    else
        log "[save] WARNING: poses.csv not found"
    fi

    # keyframe PCDs → keyframes/ subfolder
    if ls "$LOAM_DIR/keyframes"/kf_*.pcd 2>/dev/null | grep -q .; then
        cp "$LOAM_DIR/keyframes"/kf_*.pcd "$out_dir/keyframes/"
        local n
        n=$(ls "$out_dir/keyframes"/kf_*.pcd | wc -l)
        log "[save] Saved: $n keyframes + poses.csv + 5 map PCDs -> $out_dir"
    else
        log "[save] WARNING: no keyframe PCDs found"
    fi
}

run_bag() {
    local clean_bag="$1"
    local orig_bag="$2"
    local out_dir="$3"

    # Check clean bag exists
    if [ ! -f "$BAGS_DIR/$clean_bag" ]; then
        log "ERROR: clean bag not found: $BAGS_DIR/$clean_bag — skipping"
        return
    fi

    kill_ros
    LIOSAM_PID=$(start_liosam)

    log "[play] Playing $clean_bag directly from Thunderbolt drive..."
    rosbag play "$BAGS_DIR/$clean_bag" --clock -r 1.0 --start 1 \
        --topics /ouster/points /ms/imu/data \
        > /tmp/bag_play.log 2>&1
    log "[play] Bag done."

    save_and_copy "$out_dir" "$LIOSAM_PID"
    kill_ros

    # Delete original to free space (skip if already gone)
    local orig_path="$BAGS_DIR/$orig_bag"
    if [ -f "$orig_path" ]; then
        local sz
        sz=$(du -sh "$orig_path" | cut -f1)
        rm "$orig_path"
        log "[delete] Removed $orig_bag ($sz)"
    fi
}

# ── main ──────────────────────────────────────────────────────────────────
SINGLE="${1:-}"

for entry in "${BAGS[@]}"; do
    IFS='|' read -r clean_name orig_name out_stem <<< "$entry"
    out_dir="$BAGS_DIR/$out_stem"

    # Single-bag mode
    if [ -n "$SINGLE" ] && [[ "$out_stem" != *"$SINGLE"* ]]; then
        continue
    fi

    echo ""
    echo "================================================================="
    log "BAG: $clean_name  ->  $out_stem"
    echo "================================================================="

    # Skip if fully done (has keyframes AND all 5 PCDs)
    if [ -d "$out_dir/keyframes" ] && \
       ls "$out_dir/keyframes"/kf_*.pcd 2>/dev/null | grep -q . && \
       [ -f "$out_dir/GlobalMap.pcd" ]; then
        log "SKIP — already complete: $(ls "$out_dir/keyframes"/kf_*.pcd | wc -l) keyframes + GlobalMap.pcd"
        continue
    fi

    run_bag "$clean_name" "$orig_name" "$out_dir"
    log "DONE: $out_stem"
done

echo ""
log "ALL DONE."
ls -la "$BAGS_DIR"/
