#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
PYTHON=${PYTHON:-/public/home/chenyuyao1/venv/venv_pi/bin/python}

TASK_SUITE=${TASK_SUITE:-libero_10}
TASK_ID=${TASK_ID:-7}
INIT_STATE_INDICES=${INIT_STATE_INDICES:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49"}

POLICY1_CONFIG=${POLICY1_CONFIG:-pi05_libero_task7_sft}
POLICY1_EXP=${POLICY1_EXP:-policy1_task7}
POLICY1_CKPT_STEP=${POLICY1_CKPT_STEP:-2500}
POLICY1_ROLLOUT_REPO=${POLICY1_ROLLOUT_REPO:-policy1_2500_task7_rollout100}

DEMO_DATA_DIR=${DEMO_DATA_DIR:-$ROOT/outputs/lerobot_policy_data/task7_demo}
ROLLOUT_OUTPUT_DIR=${ROLLOUT_OUTPUT_DIR:-$ROOT/outputs/rollout_data}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-$ROLLOUT_OUTPUT_DIR/$POLICY1_ROLLOUT_REPO}
DEMO_PREPARED_DIR=${DEMO_PREPARED_DIR:-$ROOT/outputs/lerobot_policy_data/task7_demo_positive_prepared}
COMBINED_INPUT_DIR=${COMBINED_INPUT_DIR:-$ROOT/outputs/lerobot_policy_data/task7_demo_plus_policy1_2500_rollout100}
PROGRESS_DATA_DIR=${PROGRESS_DATA_DIR:-$ROOT/outputs/lerobot_policy_data/task7_demo_plus_policy1_2500_rollout100_progress}
ADV_DATA_DIR=${ADV_DATA_DIR:-$ROOT/outputs/lerobot_policy_data/task7_demo_plus_policy1_2500_rollout100_progress_adv_vlm10k}

VJEPA_MODEL_PATH=${VJEPA_MODEL_PATH:-/public/home/chenyuyao1/model/vjepa2/vitg-384.pt}
VJEPA_DEVICE_ID=${VJEPA_DEVICE_ID:-0}
PREFIX_STRIDE=${PREFIX_STRIDE:-1}
VJEPA_CACHE_DIR=${VJEPA_CACHE_DIR:-$ROOT/cache/task7_demo_plus_policy1_2500_rollout100_progress_stride1}

VLM_CKPT_DIR=${VLM_CKPT_DIR:-$ROOT/checkpoints/value_task7_demo_plus_policy1_2500_rollout100_progress_10k_b32}
VLM_CKPT_NAME=${VLM_CKPT_NAME:-step_00010000}
VLM_RUN_NAME=${VLM_RUN_NAME:-value_task7_demo_plus_policy1_2500_rollout100_progress_10k_b32}
TOKENIZER_PATH=${TOKENIZER_PATH:-/public/home/chenyuyao1/.cache/openpi/big_vision/paligemma_tokenizer.model}

PISTAR_CONFIG=${PISTAR_CONFIG:-pi05_star_libero}
PISTAR_EXP=${PISTAR_EXP:-policy2_srpo_task7_progress_adv_vlm10k}
PISTAR_ASSET_ID=${PISTAR_ASSET_ID:-ybpy/libero_pistar}
PISTAR_STEPS=${PISTAR_STEPS:-10000}
PISTAR_SAVE_INTERVAL=${PISTAR_SAVE_INTERVAL:-2500}
PISTAR_KEEP_PERIOD=${PISTAR_KEEP_PERIOD:-10000}
PISTAR_BATCH_SIZE=${PISTAR_BATCH_SIZE:-64}
PISTAR_NUM_WORKERS=${PISTAR_NUM_WORKERS:-4}
PISTAR_FSDP_DEVICES=${PISTAR_FSDP_DEVICES:-8}

LOG_DIR=${LOG_DIR:-$ROOT/outputs/training_logs}
ROLLOUT_LOG_DIR=${ROLLOUT_LOG_DIR:-$ROOT/outputs/rollout_logs}
mkdir -p "$LOG_DIR" "$ROLLOUT_LOG_DIR"

export http_proxy=${http_proxy:-http://127.0.0.1:17890}
export https_proxy=${https_proxy:-http://127.0.0.1:17890}
export HTTP_PROXY=${HTTP_PROXY:-http://127.0.0.1:17890}
export HTTPS_PROXY=${HTTPS_PROXY:-http://127.0.0.1:17890}
export WANDB_MODE=${WANDB_MODE:-online}
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export PYTHONPATH="$ROOT/src:$ROOT/third_party/libero:${PYTHONPATH:-}"


cd "$ROOT"
if [[ "${OVERWRITE_ADV_DATA:-true}" == "true" ]]; then
  rm -rf "$ADV_DATA_DIR"
fi
if [[ ! -d "$ADV_DATA_DIR" ]]; then
  cp -a "$PROGRESS_DATA_DIR" "$ADV_DATA_DIR"
fi

exec "$PYTHON" -u scripts/label_advantage_from_vlm.py \
  --data_dir "$ADV_DATA_DIR" \
  --checkpoint_dir "$VLM_CKPT_DIR" \
  --checkpoint_name "$VLM_CKPT_NAME" \
  --lookahead ${ADV_LOOKAHEAD:-15} \
  --top_percent ${ADV_TOP_PERCENT:-30} \
  --batch_size ${ADV_BATCH_SIZE:-8} \
  --num_workers ${ADV_NUM_WORKERS:-0} \
  --reward_col ${ADV_REWARD_COL:-reward_label} \
  --tokenizer_path "$TOKENIZER_PATH"
