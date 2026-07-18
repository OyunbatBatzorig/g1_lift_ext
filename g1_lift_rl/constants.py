# g1_lift_rl/constants.py
"""Verified physical facts about the robot and scene (Phase 0/2 territory).

Everything in this file is a measured/verified fact, not an RL design choice --
joint names, limits, the finger-collision fix, the left-arm hang pose. MDP design
(reward thresholds, observation terms) lives in mdp/* and is decided in Phase 1,
not here.

Geometry is expressed in the per-environment local frame (relative to env_origin,
which sits on the ground plane at z = 0).
"""

# ---------------------------------------------------------------------------
# Robot asset
# ---------------------------------------------------------------------------
# Patched copy (see patch_finger_collision.py, ported from g1_redblock_ext): filters
# collision between the two right-hand finger chains + hand base via
# UsdPhysics.FilteredPairsAPI, so the gripper can fully close while
# enabled_self_collisions stays True everywhere else. Verified empirically: closure
# reaches +0.0245 (vs. ~+0.0007 on the unpatched USD under self-collision). Original
# (unpatched) asset under unitree_sim_isaaclab is untouched.
ROBOT_USD = (
    "/home/virtual-acc/projects/unitree_sim_isaaclab/assets/robots/"
    "g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1_fingerfilter.usd"
)

# Base pose. rot is (w, x, y, z); this quaternion faces the robot toward -Y.
ROBOT_POS = (0.0, 0.0, 0.76)
ROBOT_ROT = (0.7071, 0.0, 0.0, -0.7071)

# ---------------------------------------------------------------------------
# Joints / bodies (verified via introspect_g1_dex1.py / check_assets.py)
# ---------------------------------------------------------------------------
ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
GRIPPER_JOINTS = ["right_hand_Joint1_1", "right_hand_Joint2_1"]

# Measured via measure_gripper.py / check_assets.py gripper sweep:
# -0.02 -> ~9.0cm fingertip separation (open), +0.0245 -> ~0.1cm (closed).
# Closes toward POSITIVE.
GRIPPER_OPEN = -0.02
GRIPPER_CLOSE = 0.0245

# Fingertip links; their midpoint is the grasp centre.
EE_LINKS = ["right_hand_Link1_3", "right_hand_Link2_3"]

# Verified EE-to-cube offset at a genuine, physical grasp (arm_teleop.py):
# ee = cube + GRASP_OFFSET, i.e. cube = ee - GRASP_OFFSET. Two independent
# teleop readings agreed (grasp point and held/inspection point both gave
# magnitude ~0.0312m) -- see mdp/rewards.py:reward_descend's docstring history
# for the full verification. Was inline inside reward_descend only; pulled out
# here so other code (Policy 2's coupled cube reset) can reuse the exact same
# value instead of duplicating the literal and risking drift.
GRASP_OFFSET = (-0.010, 0.027, 0.012)

# Head camera link (for the future inspection/perception phase).
HEAD_CAMERA_LINK = "d435_link"

# Solid mounting plate between the two fingers -- visually confirmed (check_hand_
# base.py + direct observation) striking the cube's TOP surface during descent.
# Neither EE_LINKS-based rewards/observations nor penalty_contact_disturbance
# track this body at all, so the policy had no visibility into this risk.
# Measured empirically: trails the EE_LINKS midpoint by a fixed ~9.8cm.
HAND_BASE_LINK = "right_hand_base_link"

# Finger-chain links whose real mesh extends measurably beyond their tracked
# ORIGIN point (probe_base_mesh_extent.py: right_hand_Link1_3's mesh reaches
# ~4.75cm forward / 2.84cm down beyond its own origin -- the same origin-vs-mesh
# gap that caused the HAND_BASE_LINK bug, just smaller). Confirmed via
# check_finger_geometry.py (32 envs x 300 steps against the corrected Policy 1
# checkpoint): of 59 real disturbance events, 100% implicated ONLY these two
# bodies (31 + 29 events) and 100% happened with the gripper still OPEN / not
# xy-aligned (an unwanted knock during approach, not a legitimate grasp) -- the
# mirror-image chain (Link2_1/2/3) and Link1_1 showed zero events in that run.
FINGER_CLEARANCE_LINKS = ["right_hand_Link1_2", "right_hand_Link1_3"]

# All 7 hand bodies with a measured mesh bbox (probe_base_mesh_extent.py), tracked
# for TABLE clearance specifically -- unlike FINGER_CLEARANCE_LINKS (only 2 bodies
# implicated for the CUBE specifically), check_table_contact.py found events
# against the TABLE's much larger flat surface spread across ALL of them (base +
# every finger-chain link), since any part of the hand sweeping low over any part
# of the table counts, not just the small area right around the cube.
TABLE_CLEARANCE_LINKS = [
    "right_hand_base_link",
    "right_hand_Link1_1", "right_hand_Link1_2", "right_hand_Link1_3",
    "right_hand_Link2_1", "right_hand_Link2_2", "right_hand_Link2_3",
]

# "Ready" posture for the right arm at reset. Carried over from g1_redblock_ext as a
# starting point -- PLACEHOLDER, re-check visually/numerically in Phase 2 for this
# package rather than assuming it's still appropriate.
#
# FIXED: right_shoulder_pitch_joint 0.30 -> 0.60. Measured directly
# (check_ready_pose_table_overlap.py) that at the original value,
# right_hand_base_link started ~0.232m INSIDE the table's volume at every
# single reset -- resolved by the physics engine's depenetration response,
# producing a violent velocity spike (18-33 rad/s on the arm, confirmed via
# diagnose_policy1_time.py) at t=0 of every episode, which was also
# disturbing the cube (7/16 envs showed >5cm displacement from this event
# alone). +0.60 pulls the hand back out of the table's xy footprint entirely
# (verified: worst_clearance goes from -0.232m to +inf -- no longer even
# positioned over the table) -- a swept comparison (fix_ready_pose.py) of
# several shoulder_pitch/elbow combinations found this the simplest, smallest
# change that fully clears the table.
READY_ARM_POSE = {
    "right_shoulder_pitch_joint": 0.60,
    "right_shoulder_roll_joint": -0.20,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.80,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.40,
    "right_wrist_yaw_joint": 0.0,
}

# Left arm: stow down by the side, out of the right arm's workspace. Held fixed via
# mdp/events.py:reset_robot_to_default, which writes a position TARGET for every
# joint at reset (not just actuated ones) -- otherwise passive joints drift to the
# USD default and the arm flops onto the table. Verified numerically in
# g1_redblock_ext's check_zero_agent.py: holds within ~2.4deg of this pose.
LEFT_ARM_STOW = {
    "left_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.25,   # small outward so it clears the torso
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
}

# ---------------------------------------------------------------------------
# Scene geometry (env-local frame)
# ---------------------------------------------------------------------------
TABLE_TOP_Z = 0.81
# -y is IN FRONT of the robot (it faces -Y), so the table/cube sit at negative y.
TABLE_POS = (0.04, -0.33, TABLE_TOP_Z - 0.74 / 2.0)  # box centre; top lands at TABLE_TOP_Z
TABLE_SIZE = (0.60, 0.50, 0.74)

BLOCK_SIZE = 0.06
# FIXED: (-0.05, -0.33) -> (-0.0620, -0.3247). User asked to center the cube's
# spawn point under where the gripper naturally converges, not just fix its
# rotation -- computed as EE_position - GRASP_OFFSET (same derivation pattern
# already used for INSPECT_POS/GOAL_POS elsewhere in this project), using the
# measured EE position from the known-good checkpoint (2026-07-16_09-18-57,
# match_pregrasp weight=2.5): EE=(-0.0720,-0.2977). Small shift (~1.2cm x,
# ~0.5cm y). Caveat: this EE measurement predates the CUBE_ROT/jitter-frame
# changes above (nothing has been retrained against the new geometry yet), so
# it's the best available estimate, not verified against post-rotation
# behavior -- may need re-deriving once a fresh checkpoint exists.
#
# FIXED: (-0.0620,-0.3247) -> (-0.0585,-0.3341). That point was the IDEAL,
# zero-error grasp position (EE - GRASP_OFFSET exactly) -- using it as the
# JITTER CENTER meant half the sampled resets landed even closer to the
# gripper than the ideal point, which is the riskier direction (closer than
# expected risks premature/deeper contact; farther than expected just risks
# an incomplete reach, not a collision). Pushed 1cm further away from the
# gripper along the same EE-to-cube line GRASP_OFFSET already defines, so the
# ideal point sits near the NEAR EDGE of the jitter range instead of its
# center.
BLOCK_INIT_POS = (-0.0585, -0.3341, TABLE_TOP_Z + BLOCK_SIZE / 2.0)  # resting on the table

# NEW: cube yaw rotation so a FACE, not an edge/corner, faces the gripper's actual
# approach direction -- user visually observed the gripper (side approach) meeting
# the cube near a corner rather than square against a face, which gives less
# clearance margin for a given position error than a face would. Measured directly
# (measure_approach_angle.py) on the converged checkpoint (2026-07-16_09-18-57):
# EE-to-cube XY offset averages -34.76 deg from world +Y across 16 envs -- but with
# real spread (std=15.35 deg, individual envs ranging -13 to -64 deg), so that mean
# was already an approximation, not a single "correct" angle.
#
# FIXED: -34.76 -> -12 deg. Visually inspecting the live scene, the measured mean
# looked over-rotated (user's estimate from the actual viewer: 10-14 deg) -- the
# quaternion math itself checked out against the raw offset vectors (no bug), but
# the mean is only a central estimate over a genuinely noisy per-env angle, so
# there's no reason to trust it over direct visual inspection of the real scene.
#
# FIXED: -12 -> +12 deg (sign flip). User confirmed the rotation direction itself
# was backwards -- turned the face toward the robot's LEFT hand side instead of
# the right hand side it approaches from. Same magnitude, opposite direction.
CUBE_ROT = (0.9945, 0.0, 0.0, 0.1045)  # (w,x,y,z), +12 deg around Z
CUBE_YAW_RAD = 0.2094  # same angle, radians -- used to jitter in the cube's
                         # own rotated frame (see env_cfg.py's reset_object),
                         # not naively in world x/y which no longer lines up
                         # with which axis actually threatens face/corner
                         # alignment once the cube itself is rotated.

# Inspection point: target for the held cube. UPDATED (2nd time) -- the previous
# value (-0.027, -0.225, 0.960) trained a policy whose carry-to-inspect pose
# brought the right elbow uncomfortably close to torso_link (visually confirmed
# by the user; geometric mesh-overlap diagnostics for this specific USD asset
# proved unreliable -- three attempts gave contradictory numbers -- so this was
# NOT re-verified by bbox overlap, only by joint torque/tracking-error, which
# showed no sign of a hard block). No reward term penalizes arm-vs-own-body
# clearance at all, so the policy had zero incentive to avoid crowding the torso;
# fixed at the source by moving the target, not by adding a new penalty (this
# project's clearance-penalty terms have repeatedly proven fragile when
# combined/added ad hoc). Re-teleoperated a new, farther-out 'inspection' pose
# (arm_teleop.py) and set INSPECT_POS to EE_mid - GRASP_OFFSET from that
# recording (-0.047,-0.195,+1.009) - (-0.010,0.027,0.012) = (-0.037,-0.222,0.997)
# -- using the offset-derived value (not the recording's own directly-measured
# cube position) for consistency with how the trained policy's own verified
# grasp geometry actually places the cube, since this recording's straddle was
# slightly off-centre (perp_dist 0.044m > the 0.030m clean-grasp guideline).
# Elbow-to-torso origin distance at the new arm pose measured 0.256m vs 0.174m
# at the current reset pose (farther, not closer); applied torque comparable
# magnitude at both, no joint showing a large sustained tracking error the way
# the gripper-vs-cube contact block does (~0.038 rad sustained gap there).
# Was (-0.027, -0.225, 0.960); that in turn replaced (-0.116, -0.139, 0.933), an
# earlier, less-verified estimate predating the teleop work. Replaced, not kept
# alongside, to avoid multiple disagreeing "inspection point" constants.
INSPECT_POS = (-0.037, -0.222, 0.997)

# ---------------------------------------------------------------------------
# 3-policy handoff poses -- each policy trains independently by starting from an
# approximation of where the PREVIOUS policy is expected to hand off, rather than
# needing that policy to actually run first. All teleop-verified via arm_teleop.py.
# ---------------------------------------------------------------------------

# Policy 2 (grasp + carry to inspection) reset pose: approximates where Policy 1
# is trained to arrive and hold (gripper OPEN, cube still freely on the table).
#
# FIXED (2nd time): the original value here (labeled "teleop-verified") actually
# put the EE 10.1cm from the cube -- see the first fix below for that whole
# story. That first fix re-solved this pose GEOMETRICALLY (CEM search targeting
# cube+grasp_offset exactly, converged to 0.16mm residual) -- correct in theory,
# but check_policy1_vs_policy2_handoff.py found it didn't match what Policy 1
# ACTUALLY converges to and holds: reward_hover (always-on, pulls to a waypoint
# above/behind the cube) and reward_descend (only pays once aligned, pulls to
# the exact grasp point) pull in different directions.
#
# FIXED (4th time): replaced with MEASURED mean joint positions from a real
# Policy 1 rollout against model_600.pt (2026-07-07_16-24-53 -- a fresh,
# CONFIRMED-healthy base_clearance-only run, not the noisy model_1499 that
# broke the 3rd attempt). 6400 samples (32 envs x steps 200-400). Per-joint std
# here is low (elbow std=0.18, wrist_yaw std=0.04 -- comparable to the OTHER
# confirmed-good run's model_600, nowhere near model_1499's 0.42-0.45), so this
# mean pose represents a real, physically consistent hold, not a blend of
# scattered configurations. Cross-checks against the FIRST (also model_600-
# based) empirical measurement: EE-to-cube height offset is 6.8cm above the
# cube here vs 7.8cm there -- consistent across two independent training runs,
# confirming this ~6-8cm-above-grasp_offset hold height is a real, reproducible
# feature of how Policy 1 converges (reward_hover vs reward_descend tension),
# not run-to-run noise.
#
# FIXED (5th time), SUPERSEDED: the pose above put the EE close enough that the
# OPEN gripper's real finger mesh overlapped the cube's box in most resets
# (checked via check_cube_overlap_reset.py) -- confirmed via
# check_reset_overlap_velocity.py to be a real, serious problem, not a benign
# artifact: PhysX's depenetration response launched the cube up to 26.9 m/s
# right at reset, before the policy ever acts, in 83% of sampled episodes.
# First re-solve (solve_pregrasp_no_overlap.py) only checked CUBE overlap and
# got it to exactly zero -- but check_reset_overlap_velocity.py STILL showed
# launches up to 15.2 m/s afterward. diagnose_reset_launch.py found why: ALL
# 48/48 still-launched envs had TABLE overlap (up to 7.2cm) instead -- pulling
# the pose back from the cube had pushed it down into the table.
#
# FIXED (6th time), SUPERSEDED: re-solved (solve_pregrasp_no_overlap_v2.py)
# checking BOTH cube overlap (5-point +-0.05m jitter grid) AND table overlap
# simultaneously. Converged to EXACTLY ZERO overlap on both -- but
# check_reset_overlap_velocity.py showed this made things WORSE (up to 71.9
# m/s launches). Root cause: this solve only checked the exact nominal pose --
# zero overlap means zero MARGIN, the tightest possible fit. Policy 2's real
# reset adds +-0.03 rad noise PER JOINT on top of this pose
# (env_cfg_policy2.py's _ARM_POSE_NOISE), which pushes a zero-margin solution
# back into overlap with high probability -- worse, this joint configuration
# has near-singular Jacobian directions (found earlier this session: singular
# values as low as 0.37 vs 472 elsewhere), so small joint noise can produce
# unpredictably large Cartesian swings.
#
# FIXED (7th time): re-solved (solve_pregrasp_no_overlap_v3.py) requiring a
# REAL 2cm margin (not just >=0) AND evaluating each candidate against actual
# noise-perturbed samples (+-0.03 rad per joint, matching the real reset noise
# exactly) crossed with the jitter grid, not just the exact nominal point --
# the solution has to survive what training will actually throw at it.
# Converged to a worst-case (under noise+jitter) clearance of 2.4cm from the
# cube and 6.3cm from the table, 0.311 rad total move from the original
# measured pose. Verified via check_reset_overlap_velocity.py before trusting
# this one (see that check's result, not just the solve's own numbers).
# FIXED (8th time), SUPERSEDED: v3 (above) turned out to have the SAME bug in a
# subtler form -- its cost minimized distance FROM THE ORIGINAL POSE, not
# distance TO THE ACTUAL GRASP TARGET, so once safety required moving away
# from the cube, there was no pull back toward it. check_policy2_reset.py
# caught it: nominal dist3d had ballooned to 15.6cm and 0% of resets landed in
# grasp range. Re-solved minimizing nominal EE-to-(cube+grasp_offset) 3D
# distance directly -- got to 10.2cm, but check_policy2_reset.py caught a
# SECOND subtlety: aligned_frac (xy_err < ALIGN_XY, the thing that actually
# gates reward_descend firing at all) dropped to 17.7% -- pure 3D distance let
# the optimizer trade away XY alignment for a slightly smaller blended number,
# even though XY specifically is a hard gate and Z is not.
#
# FIXED (9th time), SUPERSEDED: re-solved (solve_pregrasp_no_overlap_v4.py)
# weighting XY error far above Z error (50x vs 5x, on top of the 1000x safety
# weight) -- XY gates reward_descend firing at all; Z only scales the reward
# once aligned. Converged to XY error of 0.00006m with Z error of 13.9cm --
# safe (verified) but visually the hand sat noticeably above the cube, and the
# large Z gap wasn't wanted.
#
# FIXED (10th time): replaced with a FRESH teleop recording (provided directly
# by the user, not solved) -- EE=(-0.069,-0.308,+0.852), cube=(-0.059,-0.335,
# +0.840), EE-to-cube dist=0.032m (closely matches the independently-verified
# grasp_offset magnitude of ~0.029-0.031m from earlier teleop sessions),
# gripper OPEN. Confirmed via the SAME empirical checks used for every solved
# pose: 27/32 resets overlapped the cube under Policy 2's real +-0.05m jitter,
# and check_reset_overlap_velocity.py showed launches up to 63.1 m/s -- a real
# teleop recording still needs the same checks as a solved pose, since it
# wasn't recorded against the actual jitter/noise distribution Policy 2 uses.
#
# FIXED (11th time): replaced with a SECOND, more rigorously verified teleop
# recording -- includes the full straddle-geometry check (t=+0.453 in [0,1],
# perp_dist=0.023m < cube half-size 0.030m -- the recording tool's own "clean,
# centred, face-on straddle" criterion), EE-to-cube dist=0.023m, REL speed
# 0.027 m/s (low, confirms genuine coupling not noise). Being re-verified now
# the same way as the first recording -- caught a SEVERE problem this time:
# check_table_overlap_reset.py found the WHOLE hand (all 7 tracked bodies)
# embedded up to 13.7cm into the TABLE in 100% of sampled resets. My earlier
# "safe" verdict (from cube-velocity alone) was wrong -- cube velocity looking
# low doesn't mean the ARM isn't being violently ejected from the table, it
# just means that particular ejection didn't happen to sweep through the cube.
#
# FIXED (12th time): re-solved (solve_pregrasp_anchored.py) anchored to this
# teleop recording -- same method as the 9th fix (real margin under actual
# reset noise+jitter for BOTH cube and table, XY error weighted far above Z),
# but with closeness to THIS recording as the tie-breaker instead of the
# earlier measured-from-Policy-1 pose, so it stays as faithful as possible to
# what was actually recorded. Converged to 1.85cm cube clearance, 6.68cm table
# clearance (both above the 1.5cm margin), XY error 0.56cm (excellent
# alignment). Moved 0.64 rad total from the recorded pose -- larger than
# previous solves needed, because this recording started so deep inside the
# table. Verified via the full suite (cube overlap, table overlap, launch
# velocity, alignment) before trusting it.
# SUPERSEDED before verification: a 3rd teleop recording was applied here but
# never checked -- the user interrupted with a 4th, more accurate recording
# before the verification suite ran.
#
# FIXED (14th time): 4th teleop recording, applied EXACTLY as recorded (per
# explicit instruction) -- EE=(-0.043,-0.281,+0.869), cube=(-0.034,-0.301,
# +0.840), EE-to-cube dist=0.036m, straddle t=+0.517 in [0,1] with
# perp_dist=0.036m (slightly ABOVE the recording tool's own ~0.030m "clean
# straddle" guideline, by 0.006m -- worth noting, not necessarily
# disqualifying), REL speed 0.035 m/s. Being verified now via the full suite
# (cube overlap, table overlap, launch velocity, alignment) -- results not yet
# known as this comment is written.
PRE_GRASP_ARM_POSE = {
    "right_shoulder_pitch_joint": +0.200,
    "right_shoulder_roll_joint": -0.100,
    "right_shoulder_yaw_joint": +0.400,
    "right_elbow_joint": -0.250,
    "right_wrist_roll_joint": -0.000,
    "right_wrist_pitch_joint": +0.000,
    "right_wrist_yaw_joint": +0.000,
}

# Policy 3 (place + release) reset pose: approximates where Policy 2 is trained to
# arrive with the cube already held (gripper CLOSED).
#
# FIXED (3rd time): previous versions used a TELEOP-recorded reference pose,
# which turned out to only match Policy 2's actual convergence in cube position
# (8mm off -- fine), not arm configuration: measured directly
# (check_policy2_to_policy3_handoff.py against Policy 2's trained checkpoint,
# 2026-07-11_12-00-16) that Policy 2's real joint angles differed from the
# teleop pose by up to 0.86 rad (right_wrist_yaw) and 1.46 rad
# (right_wrist_roll) -- expected, since 7 DOF means many joint configs reach
# the same EE point, and PPO has no reason to find the same one a human
# teleop demo did. Since Policy 3's observations include raw joint angles
# (arm_joint_pos_rel), training it from a pose 30-80+ degrees off from what
# Policy 2 actually hands off meant Policy 3 never saw the real handoff
# configuration during training, only a matching cube position.
#
# Now set DIRECTLY from Policy 2's measured converged state (mean over 16 envs,
# std 0.001-0.003 rad -- Policy 2 converges to an almost identical pose every
# episode, so this is a stable, well-defined target, not noise):
INSPECT_ARM_POSE = {
    "right_shoulder_pitch_joint": +0.1272,
    "right_shoulder_roll_joint": -0.6230,
    "right_shoulder_yaw_joint": +0.8217,
    "right_elbow_joint": -0.5414,
    "right_wrist_roll_joint": -0.1649,
    "right_wrist_pitch_joint": -0.4380,
    "right_wrist_yaw_joint": -0.6681,
}

# Policy 2's converged gripper joint value (same measurement) -- the real,
# physically-blocked grasp (~-0.0144, see GRIP_CLOSED_THRESHOLD in rewards.py),
# NOT the unreachable full-closure GRIPPER_CLOSE (+0.0245) the reset used to
# hardcode. Policy 3's observations include gripper_joint_pos, so matching this
# keeps the reset distribution-consistent with what Policy 2 actually produces,
# not just geometrically coupled (which GRASP_OFFSET already guaranteed
# regardless of the exact gripper value used).
INSPECT_GRIP_VALUE = -0.0144

# Policy 3's goal: where the cube should be carried and released. Teleop-verified
# (estimated-cube-if-held at the recorded "target" arm pose) -- a different spot on
# the SAME table, not a separate platform (z is close to normal resting height).
#
# UPDATED (2nd recording): user flagged persistent visible table contact while
# reviewing a trained checkpoint and re-teleoperated a fresh reference "place"
# pose to replace the guessed/first-recording value. EE_mid=(0.060,-0.362,0.858),
# gripper CLOSED, REL speed 0.028 m/s confirmed a genuine held coupling (not a
# loose/incidental contact). Using EE_mid - GRASP_OFFSET (the canonical
# offset, not this recording's own directly-measured cube reading) for
# consistency with how the trained policy's own grasp geometry actually holds
# the cube, same reasoning as INSPECT_POS's derivation.
GOAL_POS = (0.070, -0.389, 0.846)

# Policy 4 (release + return-to-ready) reset pose: where Policy 3 (move to goal +
# place, narrowed after release/return_to_ready were split off -- see
# env_cfg_policy3.py's RewardsCfg docstring) actually converges to, same
# measurement discipline as INSPECT_ARM_POSE (replay the real trained checkpoint,
# read the converged joint angles directly, don't guess or reuse a teleop pose).
# Measured via measure_policy3_to_policy4_handoff.py against checkpoint
# 2026-07-13_17-29-24 (the narrowed-reward retrain), restricted to the 21/64 envs
# where _at_goal_settled() was genuinely true at episode end -- std 0.004-0.047
# rad per joint, a stable converged pose, not noise.
GOAL_ARM_POSE = {
    "right_shoulder_pitch_joint": -0.7747,
    "right_shoulder_roll_joint": +0.1841,
    "right_shoulder_yaw_joint": -0.1673,
    "right_elbow_joint": +1.4127,
    "right_wrist_roll_joint": -0.3798,
    "right_wrist_pitch_joint": -0.6768,
    "right_wrist_yaw_joint": +0.3613,
}

# Policy 3's converged gripper joint value at the goal (same measurement, same
# 21/64 settled envs) -- the real, physically-blocked grasp, not the unreachable
# full-closure GRIPPER_CLOSE. Policy 4's observations include gripper_joint_pos,
# so matching this keeps its reset distribution-consistent with what Policy 3
# actually hands off (same reasoning as INSPECT_GRIP_VALUE).
GOAL_GRIP_VALUE = -0.0088
