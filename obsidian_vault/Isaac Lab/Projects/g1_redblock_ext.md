# g1_redblock_ext — G1 Red-Block Task (prior experiment, direct predecessor of g1_lift_ext)

`/home/virtual-acc/projects/g1_redblock_ext`. No own `.git` (part of the outer uncommitted repo). Prior experiment, kept for thesis comparison — not under active development. Structurally and conceptually this is the **direct predecessor** [[g1_lift_ext]] was cloned from: identical file layout (`env_cfg.py`, `constants.py`, `mdp/{events,observations,rewards,terminations}.py`, `agents/rsl_rl_ppo_cfg.py`, root-level `check_assets.py`/`diag_grasp.py`/`plot_training.py`), and `g1_lift_ext`'s own code comments cite it directly ("verified in g1_redblock_ext", "ported from g1_redblock_ext's patch_finger_collision.py").

## Task
Package `g1_redblock_rl`, two env configs: `env_cfg.py` and `lift_env_cfg.py` (same reach→grasp→lift stage split later carried into `g1_lift_ext`'s policy1/2/3 chain). `mdp/goal.py` defines goal sampling/placement logic — the direct ancestor of the goal-marker mechanism `g1_lift_ext`'s Policy 3 uses.

## Approach
Same family of reward shaping later refined in `g1_lift_ext`: grasp/lift/inspect/place-style terms in `mdp/rewards.py`, RSL-RL PPO via `agents/rsl_rl_ppo_cfg.py`. `constants.py` holds the hardcoded pose/gripper constants that `g1_lift_ext` re-tuned and re-verified rather than invented from scratch.

## Status — where work stopped, and why it moved to g1_lift_ext
`patch_finger_collision.py` is a specific, targeted collision-geometry patch (the same fix `g1_lift_ext` explicitly ports forward) — its existence, plus the cluster of diagnostic scripts (`check_assets.py`, `check_inspect_target.py`, `check_zero_agent.py`, `diag_grasp.py`, `measure_hold_pos.py`), points to unresolved finger/hand collision and grasp-timing problems being actively chased when work on this project wound down. Checkpoints exist under `checkpoints/`/`logs/`/`outputs/`, but the geometry problems this project was debugging are exactly what `g1_lift_ext`'s `check_hand_base.py` → `check_hand_geometry_v2.py` → `hand_base_to_object` observation lineage went on to fix properly (v2 found the v1 bug undercounted the mounting-plate mesh's reach by ~9cm). Reasonable read: this project's unresolved hand-collision/grasp issues are the direct reason work forked into a fresh `g1_lift_ext` copy rather than continuing here.

## Relation to other projects
Direct code ancestor of [[g1_lift_ext]] (see that note for the full reward/status breakdown of the continuation). Distinct lineage from [[g1_pick_place]]/[[g1_arm_manipulation]]'s `scripts/`+`tasks/` layout. See [[overview]].
