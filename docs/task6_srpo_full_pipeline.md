# Task6 SRPO Pipeline

This document describes the portable task6 pipeline that runs fully inside this repository.
It starts from `pi05_base`, creates SRPO-style labels, trains a value VLM, uses the VLM to
select positive/negative frames, and trains a PiStar policy.

## Repository Layout

- `scripts/label_lerobot_rollout_progress.py`: V-JEPA2 trajectory-progress labeling.
- `scripts/train_value.py`: value VLM training.
- `scripts/label_advantage_from_vlm.py`: VLM inference and `adv_ind` generation.
- `scripts/train.py`: PiStar policy training from `pi05_base`.
- `scripts/task6_pipeline/run_srpo_from_pi05base.py`: end-to-end task6 policy pipeline.
- `scripts/task6_pipeline/run_ctrlworld_aug500.sh`: CtrlWorld augmentation launcher.
- `third_party/vjepa2`: vendored V-JEPA2 code used by `src/progress/encoder.py`.

Generated datasets, checkpoints, logs, videos, and caches should stay under ignored runtime
directories such as `outputs/`, `checkpoints/`, `cache/`, and `wandb/`.

## External Inputs

Set these paths in your shell or in a local `.env` copied from
`scripts/task6_pipeline/env.example`. The example contains placeholders and should be edited
before sourcing.

- `PYTHON`: Python executable.
- `TOKENIZER_PATH`: PaliGemma tokenizer model.
- `VJEPA_MODEL_PATH`: V-JEPA2 `vitg-384.pt` weights.
- `PI05_BASE_PARAMS`: `pi05_base` parameter checkpoint. The default can use the OpenPI GCS path.
- `TASK6_SOURCE_DATA`: LeRobot dataset containing task6 demos plus rollout50.

For CtrlWorld augmentation also set:

- `CTRLWORLD_CKPT_PATH`: CtrlWorld checkpoint.
- `CTRLWORLD_DATA_STAT_PATH`: CtrlWorld world-model stats.
- `DYN_CKPT_PATH`: dynamics checkpoint.
- `DYN_STAT_PATH`: dynamics stats.
- `CTRLWORLD_POLICY_CKPT`: policy checkpoint used to drive CtrlWorld interactions.

Do not commit local machine paths or downloaded model weights.

## Main Policy Pipeline

Example:

```bash
cp scripts/task6_pipeline/env.example .env
$EDITOR .env
source .env

python scripts/task6_pipeline/run_srpo_from_pi05base.py \
  --source_data "$TASK6_SOURCE_DATA" \
  --run_name task6_srpo_s1_f01_a12 \
  --alpha 12 \
  --failure_scale 0.1 \
  --top_percent 30 \
  --vlm_steps 5000 \
  --policy_steps 5000 \
  --vlm_batch_size 32 \
  --policy_batch_size 64 \
  --fsdp_devices 4 \
  --gpus 0,1,2,3 \
  --label_gpu 0 \
  --infer_gpu 0 \
  --overwrite
```

The script runs these stages:

1. Copy the source LeRobot dataset and add SRPO-style dense `reward_label`/`value_label`.
2. Train a value VLM for `--vlm_steps`.
3. Run VLM inference and mark `adv_ind` with `--top_percent`.
4. Train policy with `pi05_star_libero_task6_srpo_baseinit` from `PI05_BASE_PARAMS`.

The current SRPO-style label formula is:

```text
d_norm = clip((d - q01(all_prefix_distances)) / (q99(all_prefix_distances) - q01(...)), 0, 1)
p_t = scale / (1 + exp(-alpha * (0.5 - d_norm)))
scale = 1.0 for success trajectories, failure_scale for failed trajectories
value_label_t = p_t - 1
reward_label_t = p_t - p_{t+1}, terminal uses p_T - 1 for failed trajectories
```

The current best completed task6 sweep used `failure_scale=0.1`; ongoing experiments compare
`alpha=12` and `alpha=8`.

## CtrlWorld Augmentation

CtrlWorld augmentation is optional and creates synthetic suffixes from existing rollout episodes.
The default launcher samples 10 cut points per source episode from 50 rollout episodes, producing
500 augmented trajectories.

Default temporal settings:

- `CTRL_INTERACTIONS=5`
- `STEPS_PER_CTRL_INTERACTION=12`
- `TARGET_SUFFIX_STEPS=60`

With the rollout video writer at 10 FPS and the policy downsample pattern used by CtrlWorld,
60 raw policy steps correspond to about 20 saved frames, roughly 2 seconds of suffix video.

Example:

```bash
source .env

INPUT_DIR=outputs/lerobot_policy_data/task6_demo_plus_policy_rollout50 \
OUTPUT_DIR=outputs/lerobot_policy_data/task6_ctrlworld_aug500 \
OVERWRITE=true \
scripts/task6_pipeline/run_ctrlworld_aug500.sh
```

The generated augmented dataset should be inspected before mixing it into policy training. The
recommended default for the main task6 value-VLM step is still to train labels on demo+rollout50,
then use the trained VLM to infer labels/advantages for any augmented data.
