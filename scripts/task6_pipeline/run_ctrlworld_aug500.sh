#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
PYTHON=${PYTHON:-python}

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

INPUT_DIR=${INPUT_DIR:-$ROOT/outputs/lerobot_policy_data/task6_demo_plus_policy_rollout50}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs/lerobot_policy_data/task6_ctrlworld_aug500}

: "${CTRLWORLD_CKPT_PATH:?Set CTRLWORLD_CKPT_PATH}"
: "${CTRLWORLD_DATA_STAT_PATH:?Set CTRLWORLD_DATA_STAT_PATH}"
: "${DYN_CKPT_PATH:?Set DYN_CKPT_PATH}"
: "${DYN_STAT_PATH:?Set DYN_STAT_PATH}"
: "${CTRLWORLD_POLICY_CKPT:?Set CTRLWORLD_POLICY_CKPT}"
: "${CONFIG_NAME:?Set CONFIG_NAME to the policy config used with CTRLWORLD_POLICY_CKPT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/src:$ROOT/scripts:${PYTHONPATH:-}"

cd "$ROOT"
exec "$PYTHON" scripts/augment_lerobot_with_ctrl_world.py \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --sample_mode "${SAMPLE_MODE:-per_episode}" \
  --points_per_episode "${POINTS_PER_EPISODE:-10}" \
  --num_aug "${NUM_AUG:-500}" \
  --seed "${SEED:-6}" \
  --ctrl_interactions "${CTRL_INTERACTIONS:-5}" \
  --steps_per_ctrl_interaction "${STEPS_PER_CTRL_INTERACTION:-12}" \
  --target_suffix_steps "${TARGET_SUFFIX_STEPS:-60}" \
  --min_prefix_steps "${MIN_PREFIX_STEPS:-1}" \
  --synthetic_success "${SYNTHETIC_SUCCESS:-false}" \
  --config_name "$CONFIG_NAME" \
  --pi_ckpt "$CTRLWORLD_POLICY_CKPT" \
  --ckpt_path "$CTRLWORLD_CKPT_PATH" \
  --data_stat_path "$CTRLWORLD_DATA_STAT_PATH" \
  --dyn_ckpt_path "$DYN_CKPT_PATH" \
  --dyn_stat_path "$DYN_STAT_PATH" \
  --overwrite "${OVERWRITE:-false}" \
  "$@"
