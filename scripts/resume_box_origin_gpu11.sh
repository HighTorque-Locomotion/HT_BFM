#!/usr/bin/env bash
set -euo pipefail

# GPU11 is the user's A100 host; its historical physical CUDA index is 2.
# Refuse non-A100 endpoints so an SSH alias for another machine cannot launch here.
# Wait without touching unrelated jobs, then launch one persistent box lineage.
PROJECT_DIR="${HT_BFM_BOX_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${HT_BFM_BOX_RUN_DIR:-$PROJECT_DIR/results/bfmzero-piplus-h0w-box-origin-20260906}"
GPU_INDEX="${HT_BFM_BOX_GPU_INDEX:-2}"
LOG_DIR="$PROJECT_DIR/logs/remote_launch"
LAUNCH_LOG="$LOG_DIR/bfmzero_piplus_h0w_box_origin_20260906.log"

mkdir -p "$RUN_DIR" "$LOG_DIR" "$PROJECT_DIR/cache/tmp" "$PROJECT_DIR/cache/IsaacLab"

gpu_is_idle() {
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
        | awk -F, -v target="$GPU_INDEX" '
            {gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); gsub(/[[:space:]]/, "", $3)}
            $1 == target { found=1; ok=($2 < 8000 && $3 < 10) }
            END { exit !(found && ok) }
        '
}

gpu_is_a100() {
    nvidia-smi --query-gpu=index,name --format=csv,noheader \
        | awk -F, -v target="$GPU_INDEX" '
            {gsub(/[[:space:]]/, "", $1); name=$2}
            $1 == target { found=1; ok=(name ~ /A100/) }
            END { exit !(found && ok) }
        '
}

target_worker_alive() {
    for pid_dir in /proc/[0-9]*; do
        pid="${pid_dir##*/}"
        [[ "$pid" == "$$" ]] && continue
        exe="$(readlink "$pid_dir/exe" 2>/dev/null || true)"
        [[ "$exe" == */python* || "$exe" == */uv ]] || continue
        cwd="$(readlink "$pid_dir/cwd" 2>/dev/null || true)"
        [[ "$cwd" == "$PROJECT_DIR" || "$cwd" == "$PROJECT_DIR"/* ]] || continue
        cmd="$(tr '\0' ' ' < "$pid_dir/cmdline" 2>/dev/null || true)"
        [[ "$cmd" == *"humanoidverse.train"* && "$cmd" == *"$RUN_DIR"* ]] && return 0
    done
    return 1
}

{
    echo "started_at=$(date --iso-8601=seconds)"
    echo "project_dir=$PROJECT_DIR"
    echo "run_dir=$RUN_DIR"
    echo "gpu_index=$GPU_INDEX"
    echo "init_mode=terrain_origin"
    echo "disable_domain_randomization=true"
    echo "disable_obs_noise=true"
    if [[ -f "$RUN_DIR/checkpoint/train_status.json" ]]; then
        echo "resume_from=$RUN_DIR"
    else
        echo "warm_start=$PROJECT_DIR/bootstrap"
    fi
} >> "$LAUNCH_LOG"

if ! gpu_is_a100; then
    echo "$(date --iso-8601=seconds) refusing_non_a100_endpoint index=$GPU_INDEX" >> "$LAUNCH_LOG"
    exit 2
fi

while ! gpu_is_idle; do
    echo "$(date --iso-8601=seconds) waiting_for_gpu index=$GPU_INDEX" >> "$LAUNCH_LOG"
    sleep 300
done

if target_worker_alive; then
    echo "$(date --iso-8601=seconds) existing_target_worker_found; refusing_duplicate" >> "$LAUNCH_LOG"
    exit 0
fi

echo "$(date --iso-8601=seconds) gpu_ready; launching_training" >> "$LAUNCH_LOG"
cd "$PROJECT_DIR"
RESUME_ARGS=(--warm_start_model_from "$PROJECT_DIR/bootstrap")
if [[ -f "$RUN_DIR/checkpoint/train_status.json" ]]; then
    RESUME_ARGS=(--resume_from "$RUN_DIR")
fi
exec env \
    PYTHONPATH="$PROJECT_DIR" \
    HT_URDF_ROOT="$PROJECT_DIR/ht_urdf" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    TMPDIR="$PROJECT_DIR/cache/tmp" TEMP="$PROJECT_DIR/cache/tmp" TMP="$PROJECT_DIR/cache/tmp" \
    BFMZERO_ASSET_CACHE_DIR="$PROJECT_DIR/cache/IsaacLab" \
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    "$PROJECT_DIR/.venv/bin/python" -u -m humanoidverse.train \
    --robot PiPlus_S_12L8A0G2H0W \
    --box_climb \
    "${RESUME_ARGS[@]}" \
    --no_resume_replay_buffer \
    --no_wandb \
    --work_dir "$RUN_DIR" \
    >> "$LAUNCH_LOG" 2>&1
