# PiStar World

This repository contains the single-repo LIBERO task6 SRPO pipeline:

1. Generate or provide task6 demo + rollout data in LeRobot format.
2. Label trajectory progress with the vendored V-JEPA2 encoder.
3. Train the value VLM.
4. Run VLM inference to write advantage-conditioned `adv_ind` labels.
5. Fine-tune the PiStar policy from `pi05_base`.
6. Optionally generate CtrlWorld augmentation data for later policy-data experiments.

The current maintained entry point is:

```bash
python scripts/task6_pipeline/run_srpo_from_pi05base.py --help
```

For setup, required external model paths, labeling semantics, and CtrlWorld augmentation details,
see:

```text
docs/task6_srpo_full_pipeline.md
scripts/task6_pipeline/env.example
```

Runtime outputs are intentionally ignored by git:

```text
outputs/
checkpoints/
cache/
wandb/
```

V-JEPA2 code is vendored in `third_party/vjepa2`. Model weights are not committed; provide them
with `VJEPA_MODEL_PATH`.
