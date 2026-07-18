# Transfer snapshot -- 2026-07-18

Honest status of each policy's included checkpoint. None are fully finished;
this is a work-in-progress snapshot, not a final deliverable.

## Policy 1 (reach + hold at pre-grasp point)
- Checkpoint: `2026-07-16_09-18-57`, `match_pregrasp` weight=2.5, `ALIGN_XY`=0.05.
- **Solid**: the wrist hard-limit saturation bug is resolved and verified
  (`right_wrist_roll_joint` converges to +0.12 rad, not the former -1.972 rad
  limit). This is the best-validated Policy 1 checkpoint.
- **Not yet reflected in this checkpoint**: three scene-geometry changes made
  after this checkpoint was trained (`CUBE_ROT` = cube yawed +12° so a face
  faces the approach direction, `BLOCK_INIT_POS` repositioned under the
  gripper's natural convergence point + pushed 1cm further from the gripper,
  and the reset jitter changed from independent world x/y to a properly
  rotated cube-local frame, ±1cm perpendicular / ±2cm along the approach).
  Code has these changes; this checkpoint was trained *before* them, so it
  doesn't exploit the improved geometry. A fresh training run against the
  current `g1_lift_rl/env_cfg.py` is the next step, not yet done.
- Known minor open issue even with the geometry fixes: ~9-15% of envs show
  shallow (2-4mm) finger-cube contact at the converged hover pose, tracked to
  EE-position imprecision (~3-4cm deviation from the ideal grasp-offset
  target). Two attempted fixes for this (weight 3.3, `ALIGN_XY` 0.03) both
  backfired and were reverted -- see `g1_lift_rl/env_cfg.py`'s comments and
  the Obsidian note for the full diagnosis.

## Policy 2 (grasp + carry to inspection)
- Checkpoint: `2026-07-09_12-30-06`, 1200 iterations.
- **Solid, no open items.** This is the most mature of the four policies --
  reward plateaus cleanly, `inspect_bonus` saturates near max, episode length
  steady at full length. Untouched this session.

## Policy 3 (move to goal + place -- narrowed scope)
- Checkpoint: `2026-07-16_12-23-10`, resumed run with `penalty_low_carry`
  (fixes the cube being dragged/skimmed low above the table instead of
  carried) active and verified working (peak carry height improved from
  ~1.2-1.8cm to ~4cm above resting).
- **NOT included**: the table-penetration fix (`penalty_table_clearance_
  near_goal_excluded`'s capped-instead-of-fully-excluded version, addressing
  ~0.5cm of sustained hand/table interpenetration during placing). This fix
  is in the code (`g1_lift_rl/mdp/rewards.py`), but the from-scratch retrain
  meant to pick it up **failed to converge** (`move_to_goal`/`settle` near
  zero, 42% drop rate, elevated action_std) -- possibly a bad seed, possibly
  a GPU-contention artifact from running 3 training jobs concurrently at the
  time, not yet re-isolated and re-tried. **This is the least finished of the
  four policies** -- the included checkpoint does NOT have the table-
  penetration fix trained in, only the low-carry fix.
- Also: this checkpoint predates the original Policy 3/4 architecture split
  being fully re-verified end to end.

## Policy 4 (release + return to ready -- new this session)
- Checkpoint: `2026-07-16_08-44-24`, 800 iterations, first and only run so far.
- **Release works well**: sustained (not a flicker -- verified via replay),
  ~770/800 steps per episode once achieved.
- **`return_to_ready` does not work**: the arm never moves back toward
  `READY_ARM_POSE` despite the reward window being open for most of the
  episode. Diagnosed as likely an exploration/budget problem (release is an
  easier behavior to discover than moving 7 joints ~2 rad), not a reward-
  structure bug. Fix applied: `max_iterations` raised 800->1500, and
  `reward_release` given a `RELEASE_HOLD_STEPS=15` sustain requirement (a
  safety net, validated as not rejecting the checkpoint's real behavior, but
  not the actual fix for `return_to_ready`). **A 1500-iteration retrain was
  queued but not completed** -- this checkpoint is the pre-fix, 800-iteration
  one.

## Recommended next steps on the new machine
1. Retrain Policy 1 fresh against the current scene geometry (cube rotation/position/jitter).
2. Re-attempt Policy 3's from-scratch retrain in isolation (no concurrent GPU jobs) to see if the earlier failure was a seed/contention artifact.
3. Retrain Policy 4 at 1500 iterations.
4. `plot_training.py --policy N` works for all four now (it was hardcoded to 1-3 before this session).
