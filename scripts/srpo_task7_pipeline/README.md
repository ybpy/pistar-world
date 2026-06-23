# Task-7 SRPO Pipeline

This directory contains the executable SRPO flow for LIBERO-10 task id 7. The pipeline starts from an existing `policy1` checkpoint and runs from real-environment rollout onward.

## Preconditions

Prepare these assets before running the scripts:

```text
DEMO_DATA_DIR      successful task-7 demo LeRobot dataset
POLICY1_CKPT_DIR  policy1 checkpoint used for rollout
VJEPA_MODEL_PATH   V-JEPA2 vitg-384.pt weights
TOKENIZER_PATH     Gemma tokenizer.model
VLM_SIGLIP_PATH    SigLIP checkpoint for VLM/value training
VLM_GEMMA_PATH     Gemma checkpoint for VLM/value training
```

The rollout step also needs LIBERO, MuJoCo/EGL, `third_party/libero`, and GPU access. Default outputs are under this repository's `outputs/` and `checkpoints/`; override paths with environment variables if assets live elsewhere.

## Steps

1. `01_collect_rollout_and_progress.sh`
   Starts a policy server from `POLICY1_CKPT_DIR`, collects task-7 rollout in real LIBERO, prepares successful demos as positive, merges demo + rollout, and writes progress-based `value_label` / `reward_label` using V-JEPA2.

2. `02_train_vlm_value_10k.sh`
   Trains the VLM/value model on `PROGRESS_DATA_DIR`.

3. `03_label_advantage_vlm10k.sh`
   Copies `PROGRESS_DATA_DIR` to `ADV_DATA_DIR`, loads the VLM/value checkpoint, computes TD advantage, and writes `adv_ind`. Demo frames remain positive.

4. `04_check_adv_labels.py`
   Prints demo/rollout positive-negative counts and fails if demo frames are not all positive.

5. `05_train_pistar_srpo_10k.sh`
   Trains `policy2_srpo` with PiStar on `ADV_DATA_DIR`.

## Minimal Command Sequence

```bash
DEMO_DATA_DIR=/path/to/task7_demo \
POLICY1_CKPT_DIR=/path/to/policy1/2500 \
VJEPA_MODEL_PATH=/path/to/vitg-384.pt \
bash scripts/srpo_task7_pipeline/01_collect_rollout_and_progress.sh

VLM_SIGLIP_PATH=/path/to/siglip2_so400m14_224.npz \
VLM_GEMMA_PATH=/path/to/gemma-3-270m \
TOKENIZER_PATH=/path/to/tokenizer.model \
bash scripts/srpo_task7_pipeline/02_train_vlm_value_10k.sh

bash scripts/srpo_task7_pipeline/03_label_advantage_vlm10k.sh
python scripts/srpo_task7_pipeline/04_check_adv_labels.py
bash scripts/srpo_task7_pipeline/05_train_pistar_srpo_10k.sh
```

## Default Stage Outputs

```text
PROGRESS_DATA_DIR = outputs/lerobot_policy_data/task7_demo_plus_policy1_2500_rollout100_progress
VLM_CKPT_DIR      = checkpoints/value_task7_demo_plus_policy1_2500_rollout100_progress_10k_b32
ADV_DATA_DIR      = outputs/lerobot_policy_data/task7_demo_plus_policy1_2500_rollout100_progress_adv_vlm10k
PISTAR_EXP        = policy2_srpo_task7_progress_adv_vlm10k
```

Important convention: the physical policy dataset remains demo episodes + rollout episodes. Prefix information is represented by per-frame `value_label` / `reward_label` and V-JEPA2 prefix encodings; frames are not expanded into separate LeRobot episodes.
