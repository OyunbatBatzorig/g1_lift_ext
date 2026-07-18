# g1_lift_ext — G1 Lift Task (current focus)

`/home/virtual-acc/projects/g1_lift_ext` — Python package `g1_lift_rl`. Direct structural clone of [[g1_redblock_ext]] (identical file layout: `env_cfg.py`, `constants.py`, `mdp/{events,observations,rewards,terminations}.py`, `agents/rsl_rl_ppo_cfg.py`, root-level diagnostic scripts) — in-code comments explicitly say "verified in g1_redblock_ext" and "ported from g1_redblock_ext's patch_finger_collision.py". No own `.git`; it's part of the outer uncommitted repo, so this timeline is built from file/log mtimes, not commits.

Started 2026-06-30. Most recent edits and training runs are **2026-07-16** — this is actively worked, not idle.

## Task: four chained single-purpose policies, not one monolithic skill

Original design (`~/Downloads/MDP_SPEC.md`, never copied into the repo) was a single "reach→grasp→lift→hold" policy. It was split into independently-trained policies, each starting from a hand-off pose approximating where the previous policy is expected to end up — no runtime dependency between them at train time.

**Architecture change (2026-07-16)**: what was Policy 3 (place + release) got split again, into Policy 3 (move to goal + place) and a new **Policy 4** (release + return to ready). Reason: a matched-checkpoint comparison showed clinging-at-goal was a stable, already-rewarded local optimum (`settle`+`place` alone paid 3.5 just for holding the cube still) that `release`'s sparse bonus could never reliably outcompete — across two independent training approaches (plain PPO and a self-imitation-learning variant, both tried, see below), release only ever fired in 1-3 of 16-64 envs. Splitting the reliable skill from the fragile one removes the reward conflict at its root: Policy 3 now has zero incentive to ever open the gripper, so nothing competes with settle/place; Policy 4 has zero incentive to keep clinging (no move_to_goal/settle/place reward at all), so releasing is the only path to any reward.

`g1_lift_rl/__init__.py` registers 8 gym IDs (4 policies × train/play):

| Task ID | Env cfg | PPO cfg | Job |
|---|---|---|---|
| `Isaac-G1-Lift-Ext-v0` / `-Play-v0` | `env_cfg.G1LiftEnvCfg` | `G1LiftPPORunnerCfg` | Policy 1: reach to pre-grasp point + hold, gripper open, cube undisturbed |
| `Isaac-G1-Policy2-Ext-v0` / `-Play-v0` | `env_cfg_policy2.G1Policy2EnvCfg` | `G1Policy2PPORunnerCfg` | Policy 2: grasp, lift, carry to inspect position |
| `Isaac-G1-Policy3-Ext-v0` / `-Play-v0` | `env_cfg_policy3.G1Policy3EnvCfg` | `G1Policy3PPORunnerCfg` | Policy 3: carry held cube to goal, place it (stays gripped) |
| `Isaac-G1-Policy4-Ext-v0` / `-Play-v0` | `env_cfg_policy4.G1Policy4EnvCfg` | `G1Policy4PPORunnerCfg` | Policy 4: release the placed cube, return arm to ready pose |

- **Policy 1** (`env_cfg.py`): robot G1+Dex1, no grasp/lift reward at all — only positioning + not knocking the cube. No success termination; must hold for the full 8s episode.
- **Policy 2** (`env_cfg_policy2.py`): starts near Policy 1's hand-off state (`PRE_GRASP_ARM_POSE ± 0.03 rad` noise, gripper open). Reuses `LiftSceneCfg`; custom reset couples the cube's spawn position to the robot's *actual* post-reset EE pose (not an independent jitter) to avoid reset-time overlap explosions.
- **Policy 3** (`env_cfg_policy3.py`): starts already holding the cube (`INSPECT_ARM_POSE`, gripper closed at `INSPECT_GRIP_VALUE`, cube at `INSPECT_POS`). Adds a visual yellow goal-marker disc. Must carry to `GOAL_POS` and place it — no release, ends the episode still gripping.
- **Policy 4** (`env_cfg_policy4.py`, new): starts already at the goal, holding the cube (`GOAL_ARM_POSE`/`GOAL_GRIP_VALUE` — measured empirically from Policy 3's own converged checkpoint, same discipline as `INSPECT_ARM_POSE`). Must open the gripper (sustained, not a flicker) and pull the arm back toward `READY_ARM_POSE`.

USD asset: `unitree_sim_isaaclab/assets/robots/g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1_fingerfilter.usd` — a finger-collision-filtered variant of the standard G1+Dex1 asset. `unitree_model` is not used directly.

## Approach

**PPO hyperparameters** (`g1_lift_rl/agents/rsl_rl_ppo_cfg.py`, same core algorithm settings across all runner cfgs):
- `actor_hidden_dims=critic_hidden_dims=[256,128,64]`, activation `elu`, `init_noise_std=0.6`
- `num_steps_per_env=24`, `num_learning_epochs=5`, `num_mini_batches=4`, `learning_rate=1e-3`, schedule `adaptive`, `desired_kl=0.01`, `clip_param=0.2`, `value_loss_coef=1.0`, `use_clipped_value_loss=True`, `max_grad_norm=1.0`, `gamma=0.99`, `lam=0.95`
- `entropy_coef=0.004` across the board (lowered from an earlier 0.008 after it caused growing `action_std` and worsening drop/launch/disturbance behavior on Policy 1)
- `max_iterations`: Policy1=1500, Policy2=800, Policy3=1500 (narrowed reward, ongoing retrains), Policy4=1500 (raised from an initial 800 — see Status)
- `experiment_name`: `g1_lift_policy1`, `g1_policy2_grasp_carry`, `g1_policy3_place_release`, `g1_policy4_release_return` (→ `logs/rsl_rl/<name>/`)
- **`SILPPO` (`agents/sil_ppo.py`) exists but is unused** — a self-imitation-learning PPO subclass tried for Policy 3's old release-consolidation problem, reverted (see Status). `class_name` defaults to plain `"PPO"` everywhere now.

### Reward structure (`g1_lift_rl/mdp/rewards.py`)

**Policy 1**: `hover`(w=1.0) · `descend`(w=1.5, dense pull to verified grasp offset, gated on xy-alignment) · `match_pregrasp`(**w=2.5**, raised from 1.0 — see Status; pulls joints toward `PRE_GRASP_ARM_POSE` directly) · `early_close`(w=-0.5) · `contact_disturbance`(w=-0.1) · `base_clearance`(w=-3.0) · `action_rate`(w=-0.01) · `joint_vel`(w=-1e-4). `ALIGN_XY` (the xy-alignment gate for `descend`/`match_pregrasp`) tightened `0.05→0.03` (see Status). `K_JOINT` (tanh steepness for `match_pregrasp`/`reward_return_to_ready`) fixed `1.5→0.5` after it was found to saturate the gradient near-zero.

**Policy 2**: `descend`(1.0) · `close_gradient`(1.0) · `grasp`(2.0) · `lift`(4.0) · `inspect`(2.0) · `inspect_bonus`(4.0) · `early_close`(-0.5) · `contact_disturbance`(-0.1) · `base_clearance`(-3.0) · `torso_clearance`(-3.0) · `action_rate`(-0.01) · `joint_vel`(-1e-4). No open items.

**Policy 3** (narrowed 2026-07-16): `move_to_goal`(3.0) · `settle`(1.5) · `place`(2.0) · `low_carry`(**new**, -2.0, penalizes carrying the cube too low/skimmed above the table instead of lifted — see Status) · `table_clearance` via `penalty_table_clearance_near_goal_excluded`(-3.0) · `contact_disturbance`(-0.1) · `action_rate`(-0.01) · `joint_vel`(-1e-4). `release`/`return_to_ready` **removed** (moved to Policy 4).

**Policy 4** (new): `release`(6.0, now requires `RELEASE_HOLD_STEPS=15` consecutive steps of the settled+open condition, not an instantaneous check — see Status) · `return_to_ready`(1.0) · `table_clearance` via `penalty_table_clearance_near_goal_excluded`(-3.0) · `contact_disturbance`(-0.1) · `action_rate`(-0.01) · `joint_vel`(-1e-4). No move_to_goal/settle/place — clinging pays literally zero here, by design.

**Constants** (`constants.py`): gripper actuator `stiffness=800/damping=3`; `GRIPPER_OPEN=-0.02`/`GRIPPER_CLOSE=+0.0245`; `GRASP_OFFSET=(-0.010,0.027,0.012)` teleop-verified; hand-off poses `PRE_GRASP_ARM_POSE`, `INSPECT_ARM_POSE`, `INSPECT_GRIP_VALUE=-0.0144`, `GOAL_POS=(0.070,-0.389,0.846)`, and **new**: `GOAL_ARM_POSE`/`GOAL_GRIP_VALUE=-0.0088` (Policy 3→4 handoff, measured from Policy 3's actual converged checkpoint the same way `INSPECT_ARM_POSE` was measured from Policy 2's).

## Status — Policy 1 & 2 solid, Policy 3/4 split mid-flight

### Policy 1 — wrist-limit issue resolved; new finger-cube-contact issue found and being worked

**Wrist-limit saturation (previously the open item) is RESOLVED.** Diagnosis: `right_wrist_roll_joint` was converging to exactly its hard limit (`-1.972` rad). Root cause found via a full-run TensorBoard trace: `match_pregrasp` peaked early (iter 375) then eroded as `descend` climbed — a genuine competing-objective signature, not undertraining. `reward_descend` only scores EE *position* (no orientation term), so for this 7-DOF arm it has zero preference among the many joint configs reaching the same EE point; `descend`'s larger weight (1.5) was winning against `match_pregrasp`'s (1.0). Fix: raised `match_pregrasp` weight `1.0→2.5`. **Confirmed via direct per-joint replay** (not just the reward curve) on the resulting checkpoint (`2026-07-16_09-18-57`): `right_wrist_roll_joint` now sits at `+0.12` rad, nowhere near the limit; total joint-space distance from `PRE_GRASP_ARM_POSE` dropped from ~2.67 rad to ~0.53 rad.

**Tried and reverted: weight 2.5→3.3.** The noisy training curve suggested this helped further (raw `match_pregrasp` 0.545→0.607), but a deterministic per-joint replay showed it actually regressed on every axis: total joint-space distance got *worse* (0.53→0.65 rad), EE-position deviation from the ideal grasp geometry got *worse* (3.0cm→5.5cm), and per-env consistency collapsed (e.g. `wrist_roll` std went 0.027→0.138 rad — envs stopped converging to one stable pose). User visually confirmed the 3.3 checkpoint's gripper wasn't grasp-ready. Reverted to 2.5, the measurably better value. **Lesson reinforced**: stochastic training-curve averages can mislead; deterministic final-checkpoint replay is the reliable read (same discipline as the rest of this project).

**New issue found: occasional finger-cube contact at the hover pose**, visually reported by user. Diagnosed with real per-body-geometry contact checks (not just the fingertip origin — an initial check using only fingertip points found zero contact, which was wrong; redone with the full 7-body Dex1 finger/hand bounding-box set found real, if shallow, contact: 12.5% of envs (2/16), up to 0.43cm penetration, concentrated on the `Link2` finger chain, not both sides).

Two hypotheses tested and **both ruled out with data**:
- **Cube reset jitter**: tried narrowing `reset_object`'s `pose_range` `±0.02→±0.012→±0.01`, then checked correlation directly — mean jitter was actually *slightly lower* in contact envs (0.699cm) than no-contact envs (0.785cm). No relationship. **Reverted to `±0.02`.**
- **Approach tilt** (user's visual hypothesis: gripper not perfectly vertical): checked using the same fingertip-height-difference metric `reward_straddle_orientation` already established — contact envs showed *less* tilt on average (1.385cm) than no-contact envs (1.539cm). Also ruled out.

**Actual driver: EE-position imprecision at the converged hover pose.** Every contact env had EE deviation ≥4.1cm from the ideal grasp-offset target (mean 4.28cm); every env with small deviation never contacted (5 envs at ~1.5cm deviation, 0 contacts). This is the same underlying problem as the wrist-limit fix — not a separate bug.

**Current mitigation, retrain pending**: tightened `ALIGN_XY` (the gate for `descend`/`match_pregrasp`) `0.05→0.03`, forcing the policy to get closer before either term pays out. Accepted risk: too tight a gate could turn into a sparse-reward problem instead of a precision fix — watch `hover`/`aligned_frac` for collapse if this backfires. Retrain not yet run (queued behind Policy 3/4).

### Policy 3 — narrowed scope, new low-carry issue found and fixed, retrain in progress

Split off `release`/`return_to_ready` into Policy 4 (see Task section). Retrained the narrowed (move_to_goal+place only) version from the pre-split checkpoint; final numbers reasonable (`move_to_goal`=2.04, `settle`=0.41, `place`=0.50).

**New issue found: cube dragged along the table instead of carried.** User visually observed what looked like dragging during transit. Diagnosed with a per-step timeline (not just phase-bucketed averages): the cube starts ~13cm above resting height at reset (matching Policy 2's own handoff height, `INSPECT_POS` is 15.7cm above `_CUBE_REST_Z`) but collapses to ~1.2-1.8cm clearance within ~10 steps and stays there for the rest of the episode. Root cause: nothing in Policy 3's reward set rewards carry *height* — `move_to_goal` only scores xyz distance to `GOAL_POS`, so skimming the cube just above the table scores identically to carrying it safely.

Fix: added `penalty_low_carry` (gated on `_is_grasping` and excluded near the goal, same pattern as `penalty_table_clearance_near_goal_excluded`, reusing `MIN_CARRY_HEIGHT = LIFT_CAP = 0.12m` — Policy 1/2's already-validated lift threshold, not a fresh guess). **Design was corrected once already**: the first version copied `table_clearance`'s raw-meters-shortfall pattern and weight (-3.0), but that penalty's typical violation (3-5cm) is a much smaller natural scale than this term's (up to the full 12cm), so the same weight number wasn't actually comparable — it was closer to fighting `move_to_goal` outright than intended. Reformulated as a normalized `[0,1]` ratio (the same pattern `reward_lift` already uses, fitting since `MIN_CARRY_HEIGHT` *is* `LIFT_CAP`) with weight `-2.0`, comparable to `place`. Confirmed via `reward_settle_near_goal`'s own formula (full 3D distance to goal, not just XY) that this doesn't create a new competing-objective conflict with `settle`/`place` — the handoff between "stay elevated during transit" and "come down near the goal" is a clean spatial handoff at the same `SETTLE_NEAR_RADIUS` boundary, not a simultaneous fight.

Retrain launched 2026-07-16 (resumed from `2026-07-13_17-29-24`, model_2298, +800 iterations) — in progress as of this note.

### Policy 4 — new, first run diagnosed a real bug (not the one it first looked like), fix applied, retrain pending

Built from scratch 2026-07-16: `env_cfg_policy4.py` (reset couples robot+cube into the settled-at-goal state via `GOAL_ARM_POSE`/`GOAL_GRIP_VALUE`, measured from Policy 3's own converged checkpoint), `G1Policy4PPORunnerCfg`, gym registration.

First training run (`2026-07-16_08-44-24`, 800 iterations): `release`=3.71/6.0 in training-curve terms — looked promising. Post-hoc replay initially seemed to show a **flicker exploit**: the release condition was satisfied momentarily then immediately reversed (grip back to closed value, arm at exactly the reset distance from `READY_ARM_POSE`, std=0.0000 across envs — the arm never moved at all). Added `RELEASE_HOLD_STEPS=15` to `reward_release`, requiring the settled+open condition to hold for 15 consecutive steps (not an instant), to close what looked like a single-frame reward exploit.

**Re-diagnosis found the flicker read was wrong** (a bug in the diagnostic script — it called `reward_release()` a second time explicitly on top of the reward manager's own internal call, double-incrementing a hold-counter and inflating the numbers). Corrected replay showed the gripper genuinely opens and **stays open for ~770 of 800 steps** — a real, sustained release, not a flicker. The `RELEASE_HOLD_STEPS` fix is still valid (doesn't reject this legitimate behavior, 16/16 envs clear the 15-step bar easily) and is being kept as a safety net, but it was not the actual bug.

**The real problem**: `return_to_ready` is gated on `NOT _is_grasping`, and since the gripper is open ~770/800 steps, that gate *is* open almost the whole episode — confirmed the AND-condition in `_is_grasping` opens correctly the instant the gripper crosses threshold, regardless of hand-to-object distance. Yet the arm's joint positions never move from `GOAL_ARM_POSE` the entire episode. This reads as an exploration/learning problem, not a reward-structure bug: `release` (crossing one threshold) is much easier to discover than `return_to_ready` (moving 7 joints ~2 rad), so with only 800 iterations the easier half of this composite skill got learned and the harder half simply hasn't been discovered yet.

Fix: raised `max_iterations` `800→1500` (matching Policy 3's budget, since this is a two-phase composite skill). Retrain queued, not yet run as of this note.

### SIL (self-imitation learning) — tried for the old Policy 3 release problem, reverted

Before the Policy 3/4 split, a `SILPPO` variant (`agents/sil_ppo.py`) was built and tried as a fix for Policy 3's old release-consolidation problem (rare discovery, lost on resume with plain PPO). Implementation: a demo buffer capturing (obs, action) pairs from release events, seeded from an offline replay of the one checkpoint that ever produced release, plus an auxiliary BC loss via a separate optimizer. It worked mechanically (buffer filled, BC step ran every iteration) but a matched-checkpoint replay comparison against the pre-SIL checkpoint showed **no reliable improvement on release** (3/64 vs 1/64 envs ever releasing — arguably still just noise) **and real regressions elsewhere**: `place` -38%, `action_rate` +78%, `joint_vel` +50%. Net harmful, not neutral. Reverted to plain PPO. The Policy 3/4 architectural split (above) is the approach that actually worked — removing the reward conflict at its source instead of trying to force consolidation through an auxiliary loss.

## Planned next stage: scripted (not learned) policy hand-off — now a 4-policy chain, still not built

Once all 4 policies are individually solid, the plan is a **scripted orchestrator** running Policy 1 → 2 → 3 → 4 end-to-end (sim demo, then real robot), explicitly **not** a learned high-level/hierarchical policy. A learned switcher would be a second black-box decision layer, harder to verify and with its own failure modes; a scripted readiness check is fully interpretable and testable in isolation, which matters more for real-robot safety.

Agreed readiness-check logic (not yet implemented, and now needs a 3→4 transition added to the original 3-policy version):
- Each transition requires the readiness condition to hold for a **sustained ~100-step (~1s) window**, not a single instant.
- **Policy 1→2**: EE aligned with cube (`xy_err < ALIGN_XY`) and arm joint velocity below a small threshold, sustained.
- **Policy 2→3**: reuse `reward_inspect_bonus`'s condition — `_is_grasping() & _lifted() & (dist to INSPECT_POS < INSPECT_RADIUS)` — sustained.
- **Policy 3→4** (new, not yet designed in detail): analogous condition using `_at_goal_settled()` (already exists, used by `reward_place`/`reward_release`) sustained for the same ~100-step window.
- **Tolerance**: accept up to **1cm** deviation from the ideal target as a good-enough handoff; anything worse should flag/abort rather than continue silently.
- Fallback: a max-time limit per phase so the pipeline can't hang indefinitely.
- Explicitly gated on all 4 policies being individually verified first — not started.

### Diagnostic tooling

Project-root scripts (permanent, reusable): `check_assets.py`, `check_geometry.py`, `check_reach.py`, `check_hand_base.py`/`check_hand_geometry_v2.py`, `diag_grasp.py`, `pose_finder.py`, `arm_teleop.py`, `zero_agent.py`, `measure_gripper.py` (fingertip-separation-vs-joint-value measurement), and **`plot_training.py`** (TensorBoard curve plotter — updated 2026-07-16 to add Policy 4 support, was hardcoded to policies 1-3 only).

Most of the 2026-07-16 diagnostics (finger-cube contact geometry checks, jitter/tilt/EE-deviation correlation checks, cube-height timelines, table-contact phase checks) were written as one-off scratchpad scripts, not added to the project root — they answered specific questions and aren't intended as reusable tooling the way the list above is.

## Relation to other projects

- Direct structural/conceptual clone of [[g1_redblock_ext]] — not [[g1_pick_place]] or [[g1_arm_manipulation]], which use a different `scripts/{train,play}.py` + `tasks/` layout with no `constants.py`/root-diagnostic-script pattern.
- Robot/object USD assets sourced from `unitree_sim_isaaclab`, not `unitree_model`.
- See [[overview]] for how this fits against the other three prior-experiment projects.
