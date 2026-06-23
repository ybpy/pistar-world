# SRPO Method Chain

This file documents only our progress-based PiStar recap method. Baseline comparison files are intentionally not included in this repository cleanup.

## Method

The chain is:

```text
pi05_base
  -> successful LIBERO-10 task-7 demo SFT
  -> policy1
  -> policy1 real LIBERO rollout
  -> merge successful demos + policy1 rollout
  -> V-JEPA2 prefix progress labeling
  -> VLM/value training
  -> VLM TD-advantage adv_ind labeling
  -> PiStar policy2_srpo continued training
```

No Ctrl-World rollout, world-model rollout, baseline comparison, or multi-iteration recap is part of this method package.

## Core Definitions

For each trajectory prefix, V-JEPA2 encodes visual prefixes and the progress code maps distance-to-success-centers into `p_t in (0, 1]`.

Per-frame labels are:

```text
value_label[t] = p_t - 1
reward_label[t<T-1] = p_t - p_(t+1)
```

Terminal rule:

```text
success terminal: p_T = 1, reward_label[T-1] = 0
failure terminal: p_T < 1, reward_label[T-1] = p_T - 1
```

VLM advantage labeling uses:

```text
A_t = sum_{k=0}^{N-1} r_{t+k} + V_{t+N} - V_t
```

with the task-7 default `N = 15`; if `t + N >= T`, it uses `V_{T-1}` and sums rewards to the end. Rollout non-intervention frames in the top advantage percentile are labeled `positive`, remaining rollout frames are `negative`, and successful demo frames remain `positive`.

## Included Code

- `src/progress/`: progress scoring, clustering, V-JEPA2 encoder wrapper, and reward/value label computation.
- `third_party/vjepa2/`: vendored V-JEPA2 source used by `src/progress/encoder.py`; weights are not included.
- `scripts/label_lerobot_rollout_progress.py`: applies progress reward/value labels to LeRobot data.
- `scripts/train_value.py`: trains the VLM/value model on progress-labeled data.
- `scripts/label_advantage_from_vlm.py`: exports VLM values and writes `adv_ind` using TD advantage.
- `scripts/merge_datasets.py`: existing repository helper used to merge successful demos and policy rollout data.
- `scripts/srpo_task7_pipeline/`: executable task-7 SRPO pipeline from rollout collection onward.

## External Assets

The repository includes V-JEPA2 code but not model weights. Set:

```bash
export VJEPA_MODEL_PATH=/path/to/vitg-384.pt
```

The VLM/value model also needs local SigLIP/Gemma/tokenizer assets, configurable through the existing environment variables used in `src/openpi/training/weight_loaders.py`.

## Task-7 Protocol

Current task:

```text
suite       = libero_10
task id     = 7
task text   = put both the alphabet soup and the cream cheese box in the basket
init states = 0-49
```

Current results should be described as in-sample over init-state indices `0-49` unless a separate held-out init-state protocol is added.

## How To Run

This pipeline starts from an existing `policy1` checkpoint. It does not train `policy1`; train `policy1` separately from successful task-7 demonstrations with `pi05_libero_task7_sft`.

Required external assets:

```text
DEMO_DATA_DIR      successful task-7 demo LeRobot dataset
POLICY1_CKPT_DIR  policy1 checkpoint used for rollout
VJEPA_MODEL_PATH   V-JEPA2 vitg-384.pt weights
TOKENIZER_PATH     Gemma tokenizer.model for VLM/value inference
VLM_SIGLIP_PATH    SigLIP checkpoint for VLM/value training
VLM_GEMMA_PATH     Gemma checkpoint for VLM/value training
```

Default paths are under this repository's `outputs/` and `checkpoints/`. Override them with environment variables when data or checkpoints live elsewhere.

Run from the repository root:

```bash
# 1. Collect policy1 rollout, merge successful demos + rollout, and write progress labels.
DEMO_DATA_DIR=/path/to/task7_demo \
POLICY1_CKPT_DIR=/path/to/policy1/2500 \
VJEPA_MODEL_PATH=/path/to/vitg-384.pt \
bash scripts/srpo_task7_pipeline/01_collect_rollout_and_progress.sh

# 2. Train the VLM/value model on the progress-labeled dataset.
VLM_SIGLIP_PATH=/path/to/siglip2_so400m14_224.npz \
VLM_GEMMA_PATH=/path/to/gemma-3-270m \
TOKENIZER_PATH=/path/to/tokenizer.model \
bash scripts/srpo_task7_pipeline/02_train_vlm_value_10k.sh

# 3. Label rollout adv_ind from VLM TD advantage.
bash scripts/srpo_task7_pipeline/03_label_advantage_vlm10k.sh

# 4. Check demo/rollout positive-negative label counts.
python scripts/srpo_task7_pipeline/04_check_adv_labels.py

# 5. Train policy2_srpo with PiStar on demo + advantage-labeled rollout.
bash scripts/srpo_task7_pipeline/05_train_pistar_srpo_10k.sh
```

Stage outputs:

```text
01 -> PROGRESS_DATA_DIR  demo + rollout with progress reward/value labels
02 -> VLM_CKPT_DIR       VLM/value checkpoint
03 -> ADV_DATA_DIR       demo + rollout with adv_ind labels
05 -> checkpoints/pi05_star_libero/$PISTAR_EXP
```

The rollout step requires a working LIBERO installation, a policy server checkpoint, MuJoCo/EGL runtime, and GPU access. The VLM/value and policy training steps require GPU access and local VLM/OpenPI checkpoints.
