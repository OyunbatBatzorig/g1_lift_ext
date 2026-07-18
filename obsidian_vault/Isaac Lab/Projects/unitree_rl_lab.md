# unitree_rl_lab — Unitree's Official RL Repo (vendored reference, locally patched)

`/home/virtual-acc/projects/unitree_rl_lab`. Vendored clone of `github.com/unitreerobotics/unitree_rl_lab`, `main` branch, up to date with `origin/main` (no local commits ahead) — but **not a pristine reference copy**: several files are locally modified/untracked to work around library version drift.

## What it is
Unitree's own more mature version of the "Isaac Lab extension + sim2sim + sim2real deploy" pattern the custom `g1_*` projects imitate: `tasks/locomotion` (velocity-tracking, Go2/H1/G1-29dof) and `tasks/mimic` (motion imitation, currently G1-29dof only — `dance_102` and `gangnanm_style` reference motions). G1 velocity task: `Unitree-G1-29dof-Velocity`, RSL-RL PPO (`hidden_dims=[512,256,128]`, `max_iterations=50000`, `learning_rate=1e-3`), reward mixes velocity tracking, posture (`base_height_l2` target 0.78m), and gait-shaping terms (`feet_gait`, `foot_clearance_reward`). Has its own `CLAUDE.md` documenting the `deploy/` C++ stack for sim2real.

## Local modifications (uncommitted)
`scripts/rsl_rl/{train,play}.py` and `assets/robots/unitree.py` are patched locally — import-path fixes for `rsl_rl`/`isaaclab_rl` API drift, broader actor-detection fallback in `play.py`'s policy export, and `UNITREE_MODEL_DIR`/`UNITREE_ROS_DIR` pointed at real local paths (`unitree_model`, `unitree_ros`).

## Status — training attempted, crashed before completion
Only two G1 runs exist under `logs/rsl_rl/unitree_g1_29dof_velocity/`, both 2026-04-30:
- First run died almost immediately (`KeyError: 'class_name'` — `rsl-rl` package/cfg-format mismatch, per `journal.txt`).
- Second run trained to iteration 31900/50000 (through 2026-05-02), then hit `RuntimeError: normal expects all elements of std >= 0.0` — a PPO divergence/NaN crash, confirmed in `journal.txt` under the header "It stopped the training". Its `exported/` directory exists but is **empty** — `play.py` was invoked but the export failed on a since-patched `ModuleNotFoundError` for `pretrained_checkpoint`.

**No locally-trained `policy.pt`/`policy.onnx` exists.** The `policy.onnx`/`deploy.yaml` files under `deploy/robots/g1_29dof/config/policy/` are git-tracked upstream demo assets (identical mtimes to clone time), not anything trained locally.

## Relation to other projects
Structural template all four custom `g1_*` packages imitate (task registration via `gym.register`, `agents/rsl_rl_ppo_cfg.py`, `mdp/` layout). Paired with [[unitree_mujoco]] for sim2sim (per its `CLAUDE.md` and the vault's sim-to-sim-transfer note) and [[unitree_sdk2]] for the C++ `deploy/` stack's real-hardware DDS I/O. See [[overview]].
