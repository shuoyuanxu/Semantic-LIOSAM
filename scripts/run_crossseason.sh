#!/bin/bash
# run_crossseason.sh
# For each bag in BAGS_DIR:
#   1. Filter to keep only LiDAR, IMU, GPS, tag detection topics — strips /tf and /tf_static
#      (the bags contain a pre-existing "map" frame that conflicts with LIO-SAM's GPS map frame)
#   2. Run LIO-SAM with GNSS (rviz shown automatically via run.launch)
#   3. Copy keyframes + poses.csv to output folder named by recording timestamp
#
# Usage:
#   ./run_crossseason.sh                # smoke-test on GNSSLidarMiddle.bag only
#   ./run_crossseason.sh --all          # process all bags sequentially
#   ./run_crossseason.sh --filter-only  # only create filtered bags, skip LIO-SAM
#   ./run_crossseason.sh Cal1.bag       # run one specific bag by original name

set -e

BAGS_DIR="/media/shuoyuan/CrucialX9/crossseason"
LOAM_TMP="$HOME/Downloads/LOAM"
LIO_SAM_WS="$HOME/catkin_slam_ws"

# Keep only sensor topics — explicitly exclude /tf and /tf_static to avoid
# the pre-existing "map" frame in the bags conflicting with LIO-SAM's own map frame.
TOPIC_EXPR='topic in ["/ouster/points", "/velodyne_points", "/ms/imu/data", "/antobot_gps", "/l/tag_detections", "/m/tag_detections", "/r/tag_detections"]'

# Map original bag name -> MonYYYY prefix (derived from first-message timestamp)
declare -A PREFIX=(
    [Cal1.bag]=Nov2024
    [GNSSLidarMiddle.bag]=Jun2024
    [Loc1.bag]=Nov2024
    [Loc2.bag]=Nov2024
    [Loc3.bag]=Nov2024
    [Middle.bag]=Jun2024
    [Normal.bag]=Jun2024
)

# Keyframe output dir names (timestamp-based, unambiguous)
declare -A OUTDIR=(
    [Cal1.bag]=Nov2024_Cal1
    [GNSSLidarMiddle.bag]=Jun2024_GNSSLidarMiddle
    [Loc1.bag]=Nov2024_Loc1
    [Loc2.bag]=Nov2024_Loc2
    [Loc3.bag]=Nov2024_Loc3
    [Middle.bag]=Jun2024_Middle
    [Normal.bag]=Jun2024_Normal
)

source "$LIO_SAM_WS/devel/setup.bash"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# -----------------------------------------------------------------------
filtered_name() {
    local orig="$1"
    local stem="${orig%.bag}"
    echo "${BAGS_DIR}/${PREFIX[$orig]}_${stem}.bag"
}

filter_bag() {
    local orig="$1"
    local filtered
    filtered=$(filtered_name "$orig")

    if [ -f "$filtered" ]; then
        log "[filter] $(basename "$filtered") already exists ($(du -sh "$filtered" | cut -f1)), skipping."
        return
    fi

    log "[filter] $orig -> $(basename "$filtered")"
    log "[filter] Keeping: ouster, velodyne, ms/imu, antobot_gps, tag_detections"
    log "[filter] Stripping: /tf /tf_static (pre-existing map frame)"
    rosbag filter "$BAGS_DIR/$orig" "$filtered" "$TOPIC_EXPR"
    log "[filter] Done. Size: $(du -sh "$filtered" | cut -f1)"
}

# -----------------------------------------------------------------------
run_lio_sam() {
    local filtered_bag="$1"
    local out_dir="$2"

    log "[lio-sam] Cleaning LOAM tmp dir"
    ps aux | grep -E "lio_sam_|roscore|roslaunch" | grep -v grep | awk '{print $2}' | xargs kill 2>/dev/null || true
    sleep 2

    rm -rf "$LOAM_TMP"
    mkdir -p "$LOAM_TMP/keyframes"

    log "[lio-sam] Launching (RViz will open)..."
    # Start roscore separately so we can set use_sim_time before nodes init
    roscore &
    ROSCORE_PID=$!
    sleep 3
    bash -lc 'rosparam set /use_sim_time true'
    nohup roslaunch lio_sam run.launch > /tmp/liosam_run.log 2>&1 &
    ROS_PID=$!
    log "[lio-sam] Waiting 12s for nodes to init..."
    sleep 12

    log "[play] Playing $(basename "$filtered_bag") at 1× speed..."
    # --start 1: skip first second so LIO-SAM TF is up before first GPS fix arrives
    # Explicit topics: avoids --wait-for-subscribers hanging on velodyne (no LIO-SAM subscriber)
    rosbag play "$filtered_bag" --clock -r 1.0 --start 1 \
        --topics /ouster/points /ms/imu/data /antobot_gps \
                 /l/tag_detections /m/tag_detections /r/tag_detections
    log "[play] Bag finished."

    log "[save] Waiting 20s for LIO-SAM to flush remaining frames..."
    sleep 20

    log "[save] Sending SIGINT -> triggers poses.csv save..."
    kill -INT $ROS_PID 2>/dev/null || true
    sleep 25

    ps aux | grep -E "lio_sam_|roscore|roslaunch" | grep -v grep | awk '{print $2}' | xargs kill 2>/dev/null || true
    sleep 3

    log "[copy] Copying keyframes to $out_dir ..."
    mkdir -p "$out_dir"
    if [ -d "$LOAM_TMP/keyframes" ] && [ "$(ls -A "$LOAM_TMP/keyframes" 2>/dev/null)" ]; then
        cp -r "$LOAM_TMP/keyframes/." "$out_dir/"
        local n_kf
        n_kf=$(ls "$out_dir"/kf_*.pcd 2>/dev/null | wc -l)
        local csv_ok
        csv_ok=$([ -f "$out_dir/poses.csv" ] && echo "YES" || echo "MISSING!")
        log "[copy] Done: $n_kf keyframe PCDs, poses.csv $csv_ok"
    else
        log "[copy] WARNING: No keyframes found — did the bag play correctly?"
    fi
}

# -----------------------------------------------------------------------
process_bag() {
    local orig="$1"
    local filtered
    filtered=$(filtered_name "$orig")
    local out_dir="$BAGS_DIR/${OUTDIR[$orig]}"

    if [ -z "${PREFIX[$orig]}" ]; then
        log "ERROR: Unknown bag '$orig'. Known: ${!PREFIX[*]}"
        exit 1
    fi

    echo ""
    echo "========================================================="
    log "Processing: $orig  ->  $(basename "$filtered")"
    echo "========================================================="

    filter_bag "$orig"

    if [ "$FILTER_ONLY" = "1" ]; then
        log "Filter-only mode — skipping LIO-SAM."
        return
    fi

    run_lio_sam "$filtered" "$out_dir"
    log "COMPLETE: keyframes in $out_dir"
}

# -----------------------------------------------------------------------
FILTER_ONLY=0
MODE="test"

if   [ "$1" = "--all" ];         then MODE="all"
elif [ "$1" = "--filter-only" ]; then MODE="all"; FILTER_ONLY=1
elif [ -n "$1" ] && [[ "$1" != --* ]]; then MODE="single"; SINGLE_BAG="$1"
fi

case "$MODE" in
    test)
        echo "=== SMOKE-TEST: GNSSLidarMiddle.bag only ==="
        echo "=== Use --all to process all 7 bags       ==="
        process_bag "GNSSLidarMiddle.bag"
        ;;
    all)
        for bag in Cal1.bag GNSSLidarMiddle.bag Loc1.bag Loc2.bag Loc3.bag Middle.bag Normal.bag; do
            process_bag "$bag"
        done
        ;;
    single)
        process_bag "$SINGLE_BAG"
        ;;
esac

echo ""
log "All done. Results in $BAGS_DIR/"
ls -la "$BAGS_DIR/"
