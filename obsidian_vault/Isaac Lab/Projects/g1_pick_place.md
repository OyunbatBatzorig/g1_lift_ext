# g1_pick_place — Hierarchical Pick/Inspect/Place (prior experiment)

`/home/virtual-acc/projects/g1_pick_place`. Has its own nested `.git` repo (only custom project that does). Prior experiment, kept for thesis comparison — not under active development.

## Task
Hierarchical RL breakdown of full pick-and-place into 6 skills, each a separate gym-registered task: `G1-Reach-v0`, `G1-Grasp-v0`, `G1-Lift-v0`, `G1-Inspect-v0`, `G1-Place-v0`, `G1-FullTask-v0` (`setup.py` entry points → `g1_pick_place.tasks.<skill>:<Skill>EnvCfg`). Idea: reach → grasp & lift off surface → inspect at eye level → place at goal and release, motivated explicitly (per README) by avoiding the degenerate "just push the object along the floor" shortcut a flat single reward would find.

Code layout: `g1_pick_place/tasks/{reach,grasp_lift,inspect,place,full_task}/`, `mdp/{observations,rewards,terminations,events}.py`, `assets/{robots,objects}/`. This is the project [[g1_arm_manipulation]] imports its robot config (`g1_cfg.py`) from.

## Approach
Per-skill reward shaping (see README's tuning table): `penalize_object_on_ground` weight discourages pushing instead of lifting; `no_dropping` weight discourages dropping during inspect. Full task composition (`tasks/full_task/`) is intended to combine the pretrained skill policies — worth re-checking at read time whether it's a learned high-level selector or a scripted sequencer, as the README doesn't make this explicit.

## Status
`trained_policies/` only has **`reach/`**, **`grasp/`**, and **`reach_grasp/`** timestamped run directories (reach: 6 runs 2026-05-11→05-16; grasp: 4 runs 2026-05-19; reach_grasp: 9 runs 2026-05-05→05-06) — **no `lift/`, `inspect/`, or `place/` runs exist**, meaning those skills and the full hierarchical task were scaffolded (env configs exist) but never actually trained. `scripts/teleop_collect.py` + `data/demos/` suggest a teleop demonstration-collection path was set up, likely to bootstrap the harder skills, but no output data was confirmed. This reads as the most ambitious but least-finished of the four custom experiments — good scaffolding, training stalled after the first two skills.

## Relation to other projects
The structural template the other three custom projects grew out of / diverged from: `g1_cfg.py` is reused directly by [[g1_arm_manipulation]]; the multi-skill/hierarchical framing is the conceptual ancestor of [[g1_lift_ext]]'s 3-policy chain, though `g1_lift_ext` is a direct code-clone of [[g1_redblock_ext]] rather than of this repo. See [[overview]].
