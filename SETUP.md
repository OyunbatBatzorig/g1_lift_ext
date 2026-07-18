# Setting up on the new machine

## 1. Environment

`environment/requirements.txt` (pip freeze, 273 packages) is the real source of
truth -- almost everything here was pip-installed, not conda-installed, so
`environment/environment_from_history.yml` only pins Python 3.11 and is not
useful on its own.

```bash
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
pip install -r environment/requirements.txt
```

Notes:
- This re-downloads Isaac Sim (large — the original env was ~21GB), so this
  step needs a good internet connection and will take a while. That's expected
  and is the reliable way to do this — Isaac Sim's Kit extension system can
  cache machine-specific paths/GPU shader data that doesn't survive a raw
  folder copy cleanly.
- If the new laptop has a different GPU/driver than this machine, some pinned
  versions in requirements.txt (CUDA-related packages especially) may need
  adjusting — pip freeze captures exactly what was installed here, not
  necessarily what's compatible elsewhere.
- `requirements.txt` includes `isaaclab`/`isaaclab_rl`/`isaaclab_tasks`/etc.
  as `-e git+https://github.com/isaac-sim/IsaacLab.git@d94504bc...` (pinned
  to the exact commit this project was built against) -- pip will clone and
  install these automatically, BUT into an internal build location, not a
  clean predictable directory. **Every training command in this project
  assumes IsaacLab lives at a known path with `isaaclab.sh` and
  `scripts/reinforcement_learning/rsl_rl/train.py` reachable in it.** Clone
  it explicitly to get that:
  ```bash
  git clone https://github.com/isaac-sim/IsaacLab.git ~/projects/IsaacLab
  cd ~/projects/IsaacLab
  git checkout d94504bcf91cb7ab7ff956a2d48ecd1bca82797a   # exact commit used here
  ```
  After this, training commands look like (run from `g1_lift_ext_transfer/`,
  with `env_isaaclab` activated):
  ```bash
  ~/projects/IsaacLab/isaaclab.sh -p ~/projects/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-G1-Lift-Ext-v0 --headless
  ```

## 2. Install the g1_lift_rl package

```bash
cd g1_lift_ext_transfer   # wherever this repo landed
pip install -e .
```

This registers the gym task IDs (`Isaac-G1-Lift-Ext-v0`, `Isaac-G1-Policy2/3/4-Ext-v0`, etc.)
the same way it worked on the original machine.

## 3. Where things are

- `g1_lift_rl/` — full source package (env configs, rewards, constants, agent configs).
- `*.py` at the root — the loose diagnostic scripts (`check_assets.py`, `arm_teleop.py`, etc.), `plot_training.py`.
- `checkpoints/policy{1,2,3,4}/` — one checkpoint per policy (see TRANSFER_NOTES.md for exactly which run and its status). Each has `exported/policy.pt` (jit) + `exported/policy.onnx`, plus the raw numbered checkpoint for resuming training if needed.
- `obsidian_vault/` — full copy of the Obsidian vault (Isaac Lab notes, including `Projects/g1_lift_ext.md`, the single source of truth for this project's status).

See `TRANSFER_NOTES.md` for the honest state of each policy — none of them are
in a fully "done" state, this is a snapshot of work in progress, not a
finished deliverable.
