'''
Usage:

1) 不指定 task_ids：评估 suite 全任务
python /public/home/chenyuyao1/code/pistar/examples/libero/main.py \
  --args.task_suite_name libero_spatial \
  --args.num_trials_per_task 50

2) 指定 task_ids：仅评估 task 0, 3, 7
python /public/home/chenyuyao1/code/pistar/examples/libero/main.py \
  --args.task_suite_name libero_spatial \
  --args.task_ids 0 3 7 \
  --args.num_trials_per_task 10

3) 指定 task_ids + 导出 LeRobot rollout
python /public/home/chenyuyao1/code/pistar/examples/libero/main.py \
  --args.task_suite_name libero_10 \
  --args.task_ids 2 5 \
  --args.save_lerobot_rollout true \
  --args.rollout_output_dir /public/home/chenyuyao1/dataset/lerobot \
  --args.rollout_repo_id ybpy/libero_rollouts_subset

More args can be found in the Args dataclass.
'''

import collections
import dataclasses
import logging
import math
import pathlib
import shutil
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    task_ids: Optional[list[int]] = None  # Specific task ids to evaluate and rollout (if None, evaluate all tasks in suite)
    init_state_indices: Optional[list[int]] = None  # Restrict collection/evaluation to selected per-task init states

    #################################################################################################################
    # PI* specific parameters
    #################################################################################################################
    adv_ind_input: Optional[str] = None  # Advantage indicator for inference ("positive" or "negative"). If None, no adv_ind is used.
    # Usage: python examples/libero/main.py --args.adv_ind_input positive

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    #################################################################################################################
    # Optional LeRobot rollout export
    #################################################################################################################
    save_lerobot_rollout: bool = False  # Whether to save rollout trajectories as a LeRobot dataset
    rollout_repo_id: str = "ybpy/libero_rollout"  # LeRobot repo id
    rollout_output_dir: Optional[str] = None  # Custom output dir; defaults to HF_LEROBOT_HOME
    rollout_overwrite: bool = False  # If True, remove existing dataset directory before writing
    rollout_robot_type: str = "panda"
    rollout_fps: int = 10
    rollout_penalty_value: float = -1.0  # Value label used for failed episodes

    seed: int = 7  # Random Seed (for reproducibility)

    #################################################################################################################
    # Debug raw LIBERO observations (no model inference)
    #################################################################################################################
    debug_raw_obs_only: bool = False  # If True, run a raw observation sanity test and exit
    debug_task_id: int = 0  # Task id used by raw observation sanity test
    debug_num_steps: int = 5  # Number of LIBERO_DUMMY_ACTION steps for raw observation sanity test
    debug_out_dir: str = "data/libero/debug_raw_obs"  # Directory to save raw / processed debug images and video


class LiberoRolloutLeRobotWriter:
    """Write LIBERO rollout trajectories to a LeRobot dataset with PI* value labels."""

    def __init__(self, args: Args):
        try:
            from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise ImportError(
                "save_lerobot_rollout=True requires lerobot. Install dependencies first."
            ) from exc
        
        self._LeRobotDataset = LeRobotDataset
        self.repo_id = args.rollout_repo_id
        self.penalty_value = args.rollout_penalty_value
        # Rollout dataset stores flipped raw camera images at original LIBERO render resolution.
        self.image_size = LIBERO_ENV_RESOLUTION

        if args.rollout_output_dir:
            self.dataset_path = pathlib.Path(args.rollout_output_dir) / self.repo_id
        else:
            self.dataset_path = pathlib.Path(HF_LEROBOT_HOME) / self.repo_id

        if args.rollout_overwrite and self.dataset_path.exists():
            logging.warning(f"Removing existing rollout dataset at {self.dataset_path}")
            shutil.rmtree(self.dataset_path)

        self.dataset = self._create_or_load_dataset(args)
        logging.info(f"Initialized LeRobot dataset at {self.dataset_path}")

    def _create_or_load_dataset(self, args: Args):
        if self.dataset_path.exists() and (self.dataset_path / "meta").exists():
            dataset = self._LeRobotDataset(repo_id=self.repo_id, root=self.dataset_path)
            if hasattr(dataset, "start_image_writer"):
                dataset.start_image_writer(num_processes=5, num_threads=10)
            logging.info(f"Appending rollouts to existing LeRobot dataset ({dataset.num_episodes} episodes).")
            return dataset

        if self.dataset_path.exists():
            raise ValueError(
                f"Rollout output path exists but is not a valid LeRobot dataset: {self.dataset_path}. "
                "Use --args.rollout_overwrite true to recreate it."
            )

        return self._LeRobotDataset.create(
            repo_id=self.repo_id,
            root=self.dataset_path,
            robot_type=args.rollout_robot_type,
            fps=args.rollout_fps,
            features={
                "image": {
                    "dtype": "image",
                    "shape": (self.image_size, self.image_size, 3),
                    "names": ["height", "width", "channel"],
                },
                "wrist_image": {
                    "dtype": "image",
                    "shape": (self.image_size, self.image_size, 3),
                    "names": ["height", "width", "channel"],
                },
                "state": {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": ["actions"],
                },
                "intervention": {
                    "dtype": "int64",
                    "shape": (1,),
                    "names": ["intervention_flag"],
                },
                "success": {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": ["success"],
                },
                "value_label": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["value_label"],
                },
                "reward": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["reward"],
                },
                "reward_label": {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": ["reward_label"],
                },
                "adv_ind": {
                    "dtype": "string",
                    "shape": (1,),
                    "names": ["adv_ind"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

    def save_episode(self, *, steps: list[dict], task: str, success: bool) -> None:
        if not steps:
            logging.warning("Empty rollout episode, skip LeRobot save.")
            return

        value_labels = self._compute_value_labels(len(steps), success)
        rewards = self._compute_rewards(len(steps), success)
        reward_labels = self._compute_reward_labels(len(steps), success)
        for idx, step in enumerate(steps):
            frame = {
                "image": step["image"],
                "wrist_image": step["wrist_image"],
                "state": step["state"],
                "actions": step["actions"],
                "intervention": np.asarray([0], dtype=np.int64),  # No manual intervention in LIBERO deployment
                "success": np.asarray([success], dtype=np.bool_),
                "value_label": np.asarray([value_labels[idx]], dtype=np.float32),
                "reward": np.asarray([rewards[idx]], dtype=np.float32),
                "reward_label": np.asarray([reward_labels[idx]], dtype=np.float32),
                "adv_ind": "none",  # Advantage indicator will be inferenced using vlm later
            }
            self._add_frame(frame, task=task)

        self.dataset.save_episode()

    def _add_frame(self, frame: dict, task: str):
        frame_with_task = dict(frame)
        frame_with_task["task"] = task

        try:
            self.dataset.add_frame(frame_with_task)
            return
        except TypeError:
            self.dataset.add_frame(frame)

    def _compute_value_labels(self, episode_length: int, success: bool) -> np.ndarray:
        if success:
            t = np.arange(episode_length, dtype=np.float32)
            value_labels = -(episode_length - 1 - t) / float(episode_length)
            return value_labels.astype(np.float32)
        return np.full((episode_length,), self.penalty_value, dtype=np.float32)

    def _compute_rewards(self, episode_length: int, success: bool) -> np.ndarray:
        # 01 reward
        if success:
            rewards = np.zeros((episode_length,), dtype=np.float32)
            rewards[-1] = 1.0
            return rewards
        return np.zeros((episode_length,), dtype=np.float32)
        
    def _compute_reward_labels(self, episode_length: int, success: bool) -> np.ndarray:
        reward_labels = np.full((episode_length,), -1.0 / float(episode_length), dtype=np.float32)
        if success:
            reward_labels[-1] = 0.0
        else:
            reward_labels[-1] = -1.0
        return reward_labels


def eval_libero(args: Args) -> None:
    if args.debug_raw_obs_only:
        _debug_libero_raw_obs(args)
        return

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    rollout_writer = LiberoRolloutLeRobotWriter(args) if args.save_lerobot_rollout else None

    # Start evaluation
    total_episodes, total_successes = 0, 0
    if args.task_ids is not None:
        filtered_task_ids = [task_id for task_id in args.task_ids if 0 <= task_id < num_tasks_in_suite]
        invalid_task_ids = [task_id for task_id in args.task_ids if task_id < 0 or task_id >= num_tasks_in_suite]
        if invalid_task_ids:
            logging.warning(
                "Ignoring out-of-range task ids: %s (valid range: [0, %d))",
                invalid_task_ids,
                num_tasks_in_suite,
            )
        if not filtered_task_ids:
            logging.error("No valid task ids left after filtering; nothing to evaluate.")
            return
        logging.info(f"Evaluating specified task ids: {filtered_task_ids}")
        task_ids = filtered_task_ids
    else:
        logging.info("Evaluating all tasks in suite.")
        task_ids = range(num_tasks_in_suite)
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)
        num_initial_states = len(initial_states)
        init_state_indices = args.init_state_indices or list(range(num_initial_states))
        invalid_init_indices = [idx for idx in init_state_indices if idx < 0 or idx >= num_initial_states]
        if invalid_init_indices:
            raise ValueError(f"Invalid init_state_indices for task {task_id}: {invalid_init_indices}; valid range is [0, {num_initial_states})")

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[init_state_indices[episode_idx % len(init_state_indices)]])

            # Setup
            t = 0
            done = False
            replay_images = []
            rollout_steps = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img_flipped = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img_flipped = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img_flipped, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img_flipped, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)
                    obs_state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ).astype(np.float32)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": obs_state,
                            "prompt": str(task_description),
                        }
                        
                        # Add adv_ind if specified
                        if args.adv_ind_input is not None:
                            element["adv_ind"] = args.adv_ind_input

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if rollout_writer is not None:
                        rollout_steps.append(
                            {
                                "image": np.ascontiguousarray(img_flipped),
                                "wrist_image": np.ascontiguousarray(wrist_img_flipped),
                                "state": np.asarray(obs_state, dtype=np.float32),
                                "actions": np.asarray(action, dtype=np.float32),
                                # "abs_pose": np.asarray(np.concatenate((obs_state[:6], action[-1:]), axis=-1), dtype=np.float32),
                            }
                        )
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            if replay_images:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
            if rollout_writer is not None:
                rollout_writer.save_episode(
                    steps=rollout_steps,
                    task=str(task_description),
                    success=done,
                )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Final results for task suite: {args.task_suite_name}")
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    if rollout_writer is not None:
        logging.info(f"LeRobot rollout export completed. Dataset saved at {rollout_writer.dataset_path}.")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _debug_libero_raw_obs(args: Args) -> None:
    """Run a small no-policy sanity test to inspect raw LIBERO camera observations."""
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    if args.debug_task_id < 0 or args.debug_task_id >= task_suite.n_tasks:
        raise ValueError(
            f"debug_task_id={args.debug_task_id} is out of range for {args.task_suite_name} (n_tasks={task_suite.n_tasks})."
        )

    task = task_suite.get_task(args.debug_task_id)
    initial_states = task_suite.get_task_init_states(args.debug_task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    out_dir = pathlib.Path(args.debug_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("[debug_raw_obs] task_suite=%s task_id=%d task=%s", args.task_suite_name, args.debug_task_id, task_description)
    logging.info("[debug_raw_obs] camera resolution configured in env: %dx%d", LIBERO_ENV_RESOLUTION, LIBERO_ENV_RESOLUTION)

    env.reset()
    obs = env.set_init_state(initial_states[0])

    # Let the scene settle a bit.
    for _ in range(max(0, args.num_steps_wait)):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    viz_frames = []

    for step_idx in range(max(1, args.debug_num_steps)):
        raw_agent = np.ascontiguousarray(obs["agentview_image"])
        raw_wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])

        proc_agent = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(raw_agent[::-1, ::-1], args.resize_size, args.resize_size)
        )
        proc_wrist = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(raw_wrist[::-1, ::-1], args.resize_size, args.resize_size)
        )

        logging.info(
            "[debug_raw_obs][step=%d] raw agent shape=%s dtype=%s min=%s max=%s | raw wrist shape=%s dtype=%s min=%s max=%s",
            step_idx,
            raw_agent.shape,
            raw_agent.dtype,
            raw_agent.min(),
            raw_agent.max(),
            raw_wrist.shape,
            raw_wrist.dtype,
            raw_wrist.min(),
            raw_wrist.max(),
        )

        row_top = np.concatenate([raw_agent, raw_wrist], axis=1)
        row_bottom = np.concatenate([proc_agent, proc_wrist], axis=1)
        if row_bottom.shape[1] != row_top.shape[1]:
            row_bottom = image_tools.resize_with_pad(row_bottom, row_top.shape[0], row_top.shape[1])
        panel = np.concatenate([row_top, row_bottom], axis=0)

        imageio.imwrite(out_dir / f"debug_step_{step_idx:03d}_raw_agent.png", raw_agent)
        imageio.imwrite(out_dir / f"debug_step_{step_idx:03d}_raw_wrist.png", raw_wrist)
        imageio.imwrite(out_dir / f"debug_step_{step_idx:03d}_proc_agent.png", proc_agent)
        imageio.imwrite(out_dir / f"debug_step_{step_idx:03d}_proc_wrist.png", proc_wrist)
        imageio.imwrite(out_dir / f"debug_step_{step_idx:03d}_panel.png", panel)
        viz_frames.append(panel)

        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    if viz_frames:
        imageio.mimwrite(out_dir / "debug_raw_vs_processed.mp4", [np.asarray(f) for f in viz_frames], fps=5)

    env.close()
    logging.info("[debug_raw_obs] Finished. Files saved to: %s", out_dir)


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
