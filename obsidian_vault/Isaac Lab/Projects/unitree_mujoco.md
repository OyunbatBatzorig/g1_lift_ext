# unitree_mujoco — Sim2Sim Verifier (vendored infrastructure)

`/home/virtual-acc/projects/unitree_mujoco`. Clean vendored clone of `github.com/unitreerobotics/unitree_mujoco`, `main`, up to date with origin (only local change: `simulate/config.yaml`, unstaged).

## What it is
MuJoCo-based simulator built on [[unitree_sdk2]] (C++, `simulate/`) / `unitree_sdk2_python` (`simulate_python/`), speaking the same DDS transport (LowCmd/LowState, plus G1-only IMUState) as real hardware. Purpose: sim-to-real verification of controllers, not RL training. Supports b2, b2w, g1 (`g1_23dof.xml`/`g1_29dof.xml`), go2, go2w, h1, h1_2.

## Usage evidence — actually built and configured for G1
`simulate/build/` has a real compiled binary (4.6MB, 2026-04-30). `config.yaml` was hand-edited to `robot: "g1"`, `robot_scene: "scene_29dof.xml"`, `domain_id: 0` with a comment noting "domain 0 to match unitree_rl_lab". `journal.txt`'s "Phase 4" describes it as one half of a DDS-domain-0 sim2sim pair with [[unitree_rl_lab]]'s `g1_ctrl` deploy binary.

## Relation to other projects
The sim2sim half of [[unitree_rl_lab]]'s workflow (see that repo's `CLAUDE.md` and the vault's `sim-to-sim-transfer` note — the actual glue script, `depoly/sim2sim.py`, lived on the `unitree_rl_lab` side and was since deleted there; nothing analogous lives in this repo, which is a passive DDS-speaking simulator with no sim2sim code of its own). Not referenced by any of the custom `g1_*` training projects — none of them have reached a sim2sim stage yet. See [[overview]].
