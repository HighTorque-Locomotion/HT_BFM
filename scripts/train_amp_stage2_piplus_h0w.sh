#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

exec python -m humanoidverse.amp_stage2 \
  --bfm-checkpoint "huiying/bfmzero-piplus-h0w-isaac-20260730_102224(1)/checkpoint" \
  --expert-dataset dataset/piplus_nowaist_walk_run/piplus_nowaist_walk_run.pkl \
  --robot-config humanoidverse/config/robot/piplus/PiPlus_S_12L8A0G2H0W.yaml \
  --disable-lora \
  --work-dir runs/amp_stage2_piplus_h0w_nowaist \
  "$@"
