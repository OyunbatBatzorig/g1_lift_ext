# Diagnostic scripts from the 2026-07-16/18 session

All run the same way: `isaaclab.sh -p diagnostics/<script>.py --policy_path <exported policy.pt> --headless [args]`.
Each writes a `_result.txt` file next to itself with the findings, and also
prints to stdout.

## Policy 1 (wrist-limit / finger-cube-contact investigation)

- `check_wrist_roll_limit.py`, `check_all_arm_limits.py` — query the robot's
  actual joint limits directly and compare against a checkpoint's converged
  values. Used to diagnose the wrist hard-limit saturation bug.
- `check_policy1_final_vs_pregrasp.py` — the core Policy 1 diagnostic: replays
  a checkpoint, reads converged per-joint values + EE position vs
  `PRE_GRASP_ARM_POSE`/the ideal grasp-offset target. Use this after any
  Policy 1 retrain to check real convergence, not just the training curve.
- `check_finger_cube_contact_v2.py` — full 7-body Dex1 finger/hand geometry
  penetration check against the cube. The v1 version (checking only fingertip
  origin points) undercounted real contact -- use v2.
- `measure_approach_angle.py` — measures the actual EE-to-cube approach angle
  from a converged checkpoint. Used to derive `CUBE_ROT`.
- `check_contact_correlation.py`, `check_tilt_correlation.py` — correlate
  finger-cube contact against cube reset jitter and straddle tilt
  respectively. Both came back negative (no correlation) -- the real driver is
  EE-position imprecision, not jitter or tilt.

## Policy 3 (dragging / table-penetration investigation)

- `policy3_cube_height_timeline.py` — per-step cube height above resting,
  checks whether the cube is being carried at a safe height or dragged/skimmed
  low. Used to diagnose the issue `penalty_low_carry` fixes.
- `policy3_table_timeline.py` — per-step table-clearance violation timeline,
  useful for checking exactly when/where contact starts relative to reset vs.
  approach vs. settling at the goal.
- `check_link2_table_contact.py` — per-body, per-phase (transit vs near-goal)
  table penetration breakdown. This is what found the ~0.5cm sustained
  `right_hand_base_link` penetration near the goal that the capped
  `penalty_table_clearance_near_goal_excluded` fix addresses.
- `verify_table_penalty_fix.py` — sanity-checks the capped near-goal penalty
  against a known-bad checkpoint before spending a full retrain on it.

## Policy 4

- `verify_release_fix.py` — sanity-checks the `RELEASE_HOLD_STEPS` sustained-
  release requirement against a checkpoint, without a full retrain.
