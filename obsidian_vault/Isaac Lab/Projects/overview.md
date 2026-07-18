# Projects Overview

Workspace: `/home/virtual-acc/projects`. Master case study: sim-to-sim transfer, IsaacLab → MuJoCo (see [[../sim-to-sim-transfer]]).

## Current focus
- **[[g1_lift_ext]]** — G1 lift task, 3-policy chain (reach/hold → grasp/lift/carry → place/release). Active development, real training progress; Policy 3 (place/release) is the live bottleneck. This is where new work should go.

## Prior experiments (kept for thesis report comparison, not under active development)
- **[[g1_redblock_ext]]** — direct code ancestor of `g1_lift_ext`; work forked away from here after unresolved finger/hand-collision issues.
- **[[g1_pick_place]]** — most structurally ambitious (own git repo, 6-skill hierarchical breakdown, teleop data collection), but only `reach`/`grasp`/`reach_grasp` were ever actually trained; `lift`/`inspect`/`place`/`full_task` are scaffolded but unrun. Source of the `g1_cfg.py` robot config other projects reuse.
- **[[g1_arm_manipulation]]** — simplest, tutorial-clean single reach skill; imports its robot config from `g1_pick_place`. Shortest-lived (~1 day of runs).

## Vendored infrastructure / reference repos (read-only, not experiments)
- **[[unitree_rl_lab]]** — Unitree's own official RL repo; the structural template all four custom projects imitate. Locally patched (not a pristine clone) to work around library drift; one G1 locomotion training attempt was made and crashed with a PPO divergence before completing export.
- **[[unitree_mujoco]]** — MuJoCo sim2sim verifier, built and configured for G1 (domain 0, to match `unitree_rl_lab`), paired with its `deploy/` C++ stack.
- **[[unitree_sdk2]]** — low-level C++ DDS SDK for real hardware; built and installed to `/opt/unitree_robotics`; underlies both `unitree_mujoco` and `unitree_rl_lab`'s deploy stack.
- **[[unitree_ros]]** — ROS/Gazebo packages + URDFs; vendored only, never actually run; not used by any custom project (they use USD assets instead).

## How the pieces connect
```
g1_pick_place (robot cfg, hierarchical template)
     │
     ├── g1_arm_manipulation (reach-only, imports g1_cfg.py)
     │
     └── g1_redblock_ext ──clone──> g1_lift_ext  ← CURRENT FOCUS
                                          │
                          (assets from unitree_sim_isaaclab)

unitree_rl_lab (official reference pattern + sim2real deploy)
     │
     ├── unitree_mujoco (sim2sim, DDS domain 0)
     └── unitree_sdk2   (DDS transport, real hardware)
                │
          unitree_ros / unitree_model (URDF / USD asset sources — vendored, unused directly by g1_* training)
```

None of the custom `g1_*` Isaac Lab training projects have reached a sim2sim or sim2real stage yet — that pipeline exists only in `unitree_rl_lab`'s own (crashed, incomplete) attempt. The report's sim-to-sim case study is documented separately in [[../sim-to-sim-transfer]].
