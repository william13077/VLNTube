#!/bin/bash
# VLN Pipeline Runner
# Runs the three pipeline stages: sample walkable -> generate goals -> render video

# ==========================================================
# Shared configuration — edit these once, used by all stages
# ==========================================================
DATAROOT="/mnt/6t/dataset/vlnverse"
METAROOT="/data/lsh/scene_summary/metadata/"
USD_ROOT="/mnt/6t/dataset/vlnverse"
SCENE_GRAPH="/data/lsh/scene_summary/scene_summary/"
TASK_DIR="goalnav_discrete"
SEQ_DIR="sequence_discrete"
SPLITS_FILE="splits/scene_splits.json"

EXIT_CODE_ALL_DONE=10
EXIT_CODE_SKIP_SCENE=11

all_dirs=("$DATAROOT"/*/)
# NOTE: first is the start index, second is the length
selected_dirs=("${all_dirs[@]: 0: 200}")

# ==========================================================
# Stage 1: Sample walkable points
# ==========================================================
echo "========== STAGE 1: Sample Walkable Points =========="
python -m vistube.stage1_sample_walkable \
    --dataroot "$DATAROOT" \
    --metaroot "$METAROOT" \
    --sample-dir "$SAMPLE_DIR" \
    --splits-file "$SPLITS_FILE"
echo "Stage 1 complete."
echo "========================================================"

# ==========================================================
# Stage 2: Generate goals and discrete paths
# ==========================================================
echo "========== STAGE 2: Generate Goals =========="
for SUB_DIR in "${selected_dirs[@]}"; do
    if [ -d "$SUB_DIR" ]; then
        echo "Shell: Stage 2 processing -> $SUB_DIR"
        python -m vistube.stage2_generate_goals "$SUB_DIR" \
            --dataroot "$DATAROOT" \
            --metaroot "$METAROOT" \
            --usd-root "$USD_ROOT" \
            --scene-graph "$SCENE_GRAPH" \
            --task-dir "$TASK_DIR" \
            --sample-dir "$SAMPLE_DIR" \
            --splits-file "$SPLITS_FILE"
        echo "----------------------------------------"
    fi
done
echo "Stage 2 complete."
echo "========================================================"

# ==========================================================
# Stage 3: Render navigation videos
# ==========================================================
echo "========== STAGE 3: Render Videos =========="
for SUB_DIR in "${selected_dirs[@]}"; do
    if [ -d "$SUB_DIR" ]; then
        echo "Shell: Stage 3 processing -> $SUB_DIR"
        python -m vistube.stage3_render_video "$SUB_DIR" \
            --dataroot "$DATAROOT" \
            --metaroot "$METAROOT" \
            --usd-root "$USD_ROOT" \
            --task-dir "$TASK_DIR" \
            --seq-dir "$SEQ_DIR" \
            --splits-file "$SPLITS_FILE"
        echo "----------------------------------------"
    fi
done
echo "Stage 3 complete."
echo "========================================================"

echo "All specified scenes have been processed."
