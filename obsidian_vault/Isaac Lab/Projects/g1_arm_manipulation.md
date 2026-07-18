# g1_arm_manipulation — G1 Arm Reach (prior experiment)

`/home/virtual-acc/projects/g1_arm_manipulation`. No own `.git` (part of the outer uncommitted repo). Prior experiment, kept for thesis comparison — not under active development.

## Task
Single skill: G1 right-arm reach, following the NVIDIA Isaac Lab tutorial pattern directly (per its own docstring, modeled on the "Getting Started with Isaac Lab" tutorial + a Kinova sim2real reference). Registers `G1-Reach-v0` / `G1-Reach-Play-v0` in `g1_arm_manipulation/tasks/reach/__init__.py`. Pure arm-only scene (no table), robot config `G1_SINGLE_ARM_CFG` imported from **`g1_pick_place.assets.robots.g1_cfg`** — i.e. this project depends on [[g1_pick_place]]'s asset definitions rather than defining its own.

Target EE pose sampled via `UniformPoseCommandCfg` in robot-body frame: `pos_x=(0.20,0.40)`, `pos_y=(-0.28,0.00)`, `pos_z=(0.05,0.20)`, top-down approach (`roll=-π/2` fixed). Episode = 4s (60Hz sim, 30Hz policy via decimation=2).

## Approach
RSL-RL PPO (`agents/rsl_rl_ppo_cfg.py`): `actor/critic_hidden_dims=[256,128,64]`, elu, `init_noise_std=1.0`, `num_steps_per_env=24`, `max_iterations=2000`, `learning_rate=3e-4`, `entropy_coef=0.005`, standard PPO clip/GAE settings (`clip_param=0.2`, `gamma=0.99`, `lam=0.95`).

Reward (all in `reach_env_cfg.py` / `mdp/rewards.py`): `end_effector_position_tracking` (L2, w=-0.2) + `..._fine_grained` (tanh std=0.05, w=0.5) + `success_bonus` (w=1.0, +10 bonus inside 5cm) + `combined_success_bonus` (w=3.0, +10 bonus when position AND orientation both succeed) + `end_effector_orientation_tracking` (L2, w=-0.1) + `..._fine_grained` (tanh std=1.0, w=0.15) + `action_rate` (w=-0.0001) + `joint_vel` (w=-0.0001). Curriculum ramps `action_rate`→-0.005 and `joint_vel`→-0.001 over 4500 steps. Termination requires holding within 5cm for 2 continuous seconds (`ee_held_at_command`), not just a single touch.

Also ships a `PreviewEnvCfg`/`preview_init.py` — single-env scene with a draggable red cube for visually checking workspace reachability before training.

## Status
14 training runs logged under `logs/g1_reach_v0/`, dated 2026-05-19 to 2026-05-20 — a short, contained experiment (about one day of iteration), earlier than [[g1_lift_ext]] and [[g1_redblock_ext]]'s work. No concrete success-rate metric was captured in the note beyond the reward design itself; would need to open the TensorBoard logs to quantify.

## Relation to other projects
Structurally the "tutorial-clean" starting point — imports its robot config from [[g1_pick_place]], and its reach-only scope (no grasp/lift) makes it the simplest of the four custom `g1_*` experiments. Distinct code lineage from [[g1_lift_ext]]/[[g1_redblock_ext]] (`scripts/{train,play}.py` + `tasks/` layout, vs. their `constants.py` + root-diagnostic-script pattern). See [[overview]].
