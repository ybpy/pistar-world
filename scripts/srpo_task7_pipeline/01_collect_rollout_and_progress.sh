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

CKPT_DIR=${POLICY1_CKPT_DIR:-$ROOT/checkpoints/$POLICY1_CONFIG/$POLICY1_EXP/$POLICY1_CKPT_STEP}
PORT=${PORT:-8027}
TRIALS=${TRIALS:-100}
REPLAN_STEPS=${REPLAN_STEPS:-5}
DRIVER_LOG=${DRIVER_LOG:-$LOG_DIR/task7_srpo_rollout_progress.log}
SERVER_LOG=${SERVER_LOG:-$ROLLOUT_LOG_DIR/serve_policy1_${POLICY1_CKPT_STEP}_task${TASK_ID}.log}
ROLLOUT_LOG=${ROLLOUT_LOG:-$ROLLOUT_LOG_DIR/collect_${POLICY1_ROLLOUT_REPO}_task${TASK_ID}.log}
STOP_TRAIN_AT_CKPT=${STOP_TRAIN_AT_CKPT:-true}

log() { echo "[$(date -Is)] $*" | tee -a "$DRIVER_LOG"; }

checkpoint_ready() {
  [[ -d "$CKPT_DIR" ]] || return 1
  [[ -e "$CKPT_DIR/_CHECKPOINT_METADATA" || -e "$CKPT_DIR/checkpoint" || -d "$CKPT_DIR/params" ]] || return 1
}

log "Waiting for checkpoint: $CKPT_DIR"
until checkpoint_ready; do sleep 300; done

if [[ "$STOP_TRAIN_AT_CKPT" == "true" ]]; then
  mapfile -t pids < <(pgrep -f "scripts/train.py $POLICY1_CONFIG .*--exp_name $POLICY1_EXP" || true)
  if [[ "${#pids[@]}" -gt 0 ]]; then
    log "Stopping policy1 training after checkpoint $POLICY1_CKPT_STEP: ${pids[*]}"
    kill "${pids[@]}" || true
  fi
fi

if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  log "Port $PORT is already in use; aborting."
  exit 1
fi

log "Starting policy server from $CKPT_DIR on port $PORT"
"$PYTHON" scripts/serve_policy.py \
  --port "$PORT" \
  policy:checkpoint \
  --policy.config "$POLICY1_CONFIG" \
  --policy.dir "$CKPT_DIR" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" >/dev/null 2>&1 || true' EXIT

for _ in $(seq 1 180); do
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    tail -n 120 "$SERVER_LOG" | tee -a "$DRIVER_LOG" || true
    exit 1
  fi
  if "$PYTHON" -c "import socket; s=socket.create_connection(('127.0.0.1', $PORT), 1); s.close()" >/dev/null 2>&1; then
    break
  fi
  sleep 10
done

log "Collecting $TRIALS rollout episodes for $TASK_SUITE task $TASK_ID"
"$PYTHON" examples/libero/main.py \
  --args.task_suite_name "$TASK_SUITE" \
  --args.task_ids "$TASK_ID" \
  --args.init_state_indices $INIT_STATE_INDICES \
  --args.num_trials_per_task "$TRIALS" \
  --args.replan_steps "$REPLAN_STEPS" \
  --args.host 127.0.0.1 \
  --args.port "$PORT" \
  --args.save_lerobot_rollout true \
  --args.rollout_output_dir "$ROLLOUT_OUTPUT_DIR" \
  --args.rollout_repo_id "$POLICY1_ROLLOUT_REPO" \
  --args.rollout_overwrite true \
  >"$ROLLOUT_LOG" 2>&1

kill "$SERVER_PID" >/dev/null 2>&1 || true
trap - EXIT

log "Preparing successful demo copy with positive labels"
rm -rf "$DEMO_PREPARED_DIR"
cp -a "$DEMO_DATA_DIR" "$DEMO_PREPARED_DIR"
DEMO_PREPARED_DIR="$DEMO_PREPARED_DIR" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(os.environ["DEMO_PREPARED_DIR"])
for path in sorted((root / "data").rglob("*.parquet")):
    df = pd.read_parquet(path)
    n = len(df)
    df["success"] = True
    df["intervention"] = np.ones((n,), dtype=np.int64)
    df["reward_label"] = np.zeros((n,), dtype=np.float32)
    df["value_label"] = np.zeros((n,), dtype=np.float32)
    df["adv_ind"] = "positive"
    df.to_parquet(path, index=False)
info_path = root / "meta" / "info.json"
if info_path.exists():
    info = json.loads(info_path.read_text())
    features = info.setdefault("features", {})
    features["success"] = {"dtype": "bool", "shape": [1], "names": ["success"]}
    features["intervention"] = {"dtype": "int64", "shape": [1], "names": ["intervention_flag"]}
    features["reward_label"] = {"dtype": "float32", "shape": [1], "names": ["reward_label"]}
    features["value_label"] = {"dtype": "float32", "shape": [1], "names": ["value_label"]}
    features["adv_ind"] = {"dtype": "string", "shape": [1], "names": ["adv_ind"]}
    info_path.write_text(json.dumps(info, indent=2) + "\n")
PY

log "Merging demo + policy1 rollout"
"$PYTHON" scripts/merge_datasets.py \
  --sources "$DEMO_PREPARED_DIR" "$ROLLOUT_DATA_DIR" \
  --output "$COMBINED_INPUT_DIR" \
  --overwrite

log "Computing per-frame prefix progress labels: prefix_stride=$PREFIX_STRIDE"
"$PYTHON" -u scripts/label_lerobot_rollout_progress.py \
  --input_dir "$COMBINED_INPUT_DIR" \
  --output_dir "$PROGRESS_DATA_DIR" \
  --prefix_stride "$PREFIX_STRIDE" \
  --model_path "$VJEPA_MODEL_PATH" \
  --device_id "$VJEPA_DEVICE_ID" \
  --cache_dir "$VJEPA_CACHE_DIR" \
  --overwrite

log "Progress-labeled dataset: $PROGRESS_DATA_DIR"
