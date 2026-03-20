# Repository Guidelines

## Project Structure & Module Organization
- `src/openpi/`: core library code (models, policies, training, serving, and shared utilities).
- `packages/openpi-client/src/openpi_client/`: client/runtime package used by external apps.
- `scripts/`: runnable entrypoints such as `train.py`, `train_pytorch.py`, `serve_policy.py`, and `compute_norm_stats.py`.
- `examples/`: platform-specific examples (`droid`, `libero`, `aloha_*`, `simple_client`, `ur5`).
- `docs/`: setup and usage docs; `third_party/` contains vendored upstream projects and large assets.

## Build, Test, and Development Commands
- `GIT_LFS_SKIP_SMUDGE=1 uv sync --all-extras --dev`: install project + dev dependencies.
- `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`: editable install for local development.
- `uv run pytest --strict-markers -m "not manual"`: run the CI-equivalent automated test set.
- `ruff check .` and `ruff format .`: lint and format code before opening a PR.
- Example train/serve flow:
  - `uv run scripts/train.py pi05_libero --exp-name=my_experiment`
  - `uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/...`

## Coding Style & Naming Conventions
- Python `>=3.11`, 4-space indentation, max line length `120` (configured in `pyproject.toml`).
- Follow Ruff import ordering (single-line imports are enforced by project config).
- Use `snake_case` for files/functions/variables, `PascalCase` for classes.
- Keep config names explicit and task-scoped (for example, `pi05_libero`, `pi05_droid`).

## Testing Guidelines
- Framework: `pytest`; test discovery paths are `src`, `scripts`, and `packages`.
- Name tests with `_test.py` and colocate them near the code they validate (for example, `src/openpi/models/pi0_test.py`).
- Use `@pytest.mark.manual` only for tests that cannot run in CI; automated PR checks exclude these.

## Commit & Pull Request Guidelines
- Commit messages in this repo are typically short, imperative, and lower-case (for example, `support specifying task_ids in libero eval and rollout`).
- Keep each commit focused on one logical change.
- PRs should include a clear description, related issue/discussion link when applicable, and evidence of validation (commands run + results).
- Before requesting review, run: `pre-commit` (after `pre-commit install`), `ruff check .`, `ruff format .`, and `pytest`.
