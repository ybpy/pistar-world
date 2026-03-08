# Usage of PiStar

## Environment Setup

Using uv to manage virtual environment. 

```bash
git clone https://github.com/ybpy/pistar.git

git submodule update --init --recursive

uv venv --python 3.11.9 /path/to/create/pistar/venv

source /path/to/your/pistar/venv/bin/activate

cd /path/to/pistar

GIT_LFS_SKIP_SMUDGE=1 uv sync --active

GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

uv pip install -r pistar_requirements.txt
```

## Fine-Tuning

Before executing following command, make sure the pistar venv have been activated. 

### 1. Convert your data to a LeRobot dataset

```bash
python examples/libero/pistar_data_processing.py \
    --data_dir /path/to/modified_libero_rlds \
    --default_adv_ind positive
or 
python -u examples/libero/pistar_data_processing.py ...
```

### 2. Compute the normalization statistics for the training data

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/compute_norm_stats.py --config-name pi05_star_libero
```

### 3. Run training

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_star_libero --exp-name=my_experiment --overwrite
```

You can use `--resume` to replace `--overwrite` in above command to restore latest checkpoint for continuation training. 

## Evaluation and Rollout

### 1. Client

Create virtual environment for Simulation enviroment first.
```bash
# Create virtual environment
uv venv --python 3.10 /path/to/create/libero_rollout/venv
source /path/to/your/libero_rollout/venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
uv pip install --no-deps git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
# To insure installing these package in your libero venv, you can add --python /path/to/your/libero/venv/bin/python behind above command, like uv pip install -e third_party/libero --python /path/to/your/libero/venv/bin/python
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```

For evalution:
```bash
# Run the simulation
# Recommend using egl for rendering
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python examples/libero/main.py --args.adv_ind_input positive
# If you have egl errors, fix by running the following command
sudo -E apt-get update
sudo -E apt-get install -y libegl1 libgl1 libglvnd0 libgles2 libdrm2 libgbm1
export MUJOCO_GL=glx
python examples/libero/main.py --args.adv_ind_input positive
# If failed, try following command
export MUJOCO_GL=glx
xvfb-run -a python examples/libero/main.py --args.adv_ind_input positive
```

For rollout:
```bash
# Run the simulation
# Recommend using egl for rendering
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python examples/libero/main.py --args.adv_ind_input positive --args.num_trials_per_task 5 --args.save_lerobot_rollouts --args.rollout_overwrite
# If you have egl errors, fix by running the following command
sudo -E apt-get update
sudo -E apt-get install -y libegl1 libgl1 libglvnd0 libgles2 libdrm2 libgbm1
export MUJOCO_GL=glx
python examples/libero/main.py --args.adv_ind_input positive --args.num_trials_per_task 5 --args.save_lerobot_rollouts --args.rollout_overwrite
# If failed, try following command
export MUJOCO_GL=glx
xvfb-run -a python examples/libero/main.py --args.adv_ind_input positive --args.num_trials_per_task 5 --args.save_lerobot_rollouts --args.rollout_overwrite
```

### 2. Server

```bash
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_star_libero_infer --policy.dir=checkpoints/pi05_star_libero/my_experiment/10000
```

