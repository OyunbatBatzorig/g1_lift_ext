# g1_lift_rl/mdp/rewards.py -- SIMPLE STAGED LADDER (full replacement)
# Stages: hover -> descend (only when centered) -> grasp -> lift -> inspect.
# Design rule: every stage strictly out-pays parking in the previous one, and the
# "down" gradient exists ONLY when the hand is centered above the cube, so the
# side-poke that knocks the cube off is never rewarded.
#
# RewardsCfg to use (env_cfg.py):
#   hover         = RewTerm(func=mdp.reward_hover,         weight=1.0)
#   descend       = RewTerm(func=mdp.reward_descend,       weight=1.0)
#   grasp         = RewTerm(func=mdp.reward_grasp,         weight=2.0)
#   lift          = RewTerm(func=mdp.reward_lift,          weight=4.0)
#   inspect       = RewTerm(func=mdp.reward_inspect,       weight=2.0)
#   inspect_bonus = RewTerm(func=mdp.reward_inspect_bonus, weight=4.0)
#   early_close   = RewTerm(func=mdp.penalty_early_close,  weight=-0.5)
#   action_rate   = RewTerm(func=mdp.penalty_action_rate,  weight=-0.01)
#   joint_vel     = RewTerm(func=mdp.penalty_joint_vel,    weight=-1.0e-4)
# (reach, holding, close_grip terms are GONE -- remove them from RewardsCfg and
#  their imports from mdp/__init__.py; export reward_descend instead.)
"""Simple staged reward ladder for Policy 1 (lift).

hover:   dense pull to a waypoint above the cube (always on).
descend: dense pull DOWN onto the cube, active only when xy-centered above it.
grasp:   bonus when enveloping AND gripper closing (the gate; binary gripper makes
         this discoverable once the hand is inside 4cm).
lift:    height above table, gated on grasp -- continuous, so it is also the
         bodyguard that keeps the grasp held.
inspect: dense pull of the held cube to INSPECT_POS + bonus inside the radius.
"""
from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

from ..constants import (
    ARM_JOINTS, EE_LINKS, GRIPPER_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSE,
    TABLE_TOP_Z, TABLE_POS, TABLE_SIZE, BLOCK_SIZE, INSPECT_POS, GOAL_POS,
    HAND_BASE_LINK, FINGER_CLEARANCE_LINKS, TABLE_CLEARANCE_LINKS, GRASP_OFFSET,
    READY_ARM_POSE, PRE_GRASP_ARM_POSE,
)

GRASP_DIST = 0.04        # m -- enveloped: EE (fingertip midpoint) within this of cube centre
# TRIED AND REVERTED: 0.05 -> 0.03. The idea (check_contact_correlation.py
# found finger-cube contact tracks EE-position imprecision, not jitter/tilt --
# so tightening the gate that turns on descend/match_pregrasp should force
# more precision before either pays out) backfired exactly as the accepted
# risk predicted: at 0.03 the gate became hard enough to satisfy that the
# policy stopped reliably triggering it at all. Direct per-joint replay of the
# resulting checkpoint (2026-07-16_13-35-55) showed EE-position deviation
# WORSENED to 8.75cm (was 3.0cm at the good 0.05 config) -- specifically
# +8.74cm too HIGH (visually confirmed by user: "too far or above the cube"),
# and total joint-space distance from PRE_GRASP_ARM_POSE got worse too
# (~0.80 rad vs ~0.53 rad). Back to 0.05, the measurably-working value.
ALIGN_XY = 0.05          # m -- xy error under which "descend" pays (fingers can straddle)
STRADDLE_Z_SCALE = 0.02  # m -- decay scale for reward_straddle_orientation.
                         # FIXED: reward_descend only checks fingertip-midpoint
                         # POSITION (ee - descend_target), never orientation --
                         # GRASP_OFFSET's geometry (large xy component, 2.88cm,
                         # vs small z, 1.2cm) implies a SIDE-STRADDLE approach
                         # (palm facing sideways, both fingertips near the same
                         # height), but nothing enforced that. Measured directly
                         # (check_gripper_orientation.py) that a trained Policy 1
                         # checkpoint converged to a ~29deg tilted approach
                         # instead -- the two fingertip chains (EE_LINKS)
                         # separated 7.9-8.0cm horizontally but ALSO 4.2-4.7cm
                         # vertically (should be ~0 for a clean side straddle),
                         # one chain sitting consistently lower -- very likely
                         # what was dragging that chain down toward the table.
                         # 0.02 chosen so the CURRENT bad state (z~0.045)
                         # scores low (exp(-0.045/0.02)=0.11) -- clear pressure
                         # to improve -- while z~0 scores near 1.0.
HOVER_OFFSET = 0.03      # m -- waypoint height above cube centre
HOVER_BACK_OFFSET = 0.06  # m -- waypoint pulled toward the robot's own body (+y;
                          # table/cube sit at y=-0.33, robot base at y=0, so +y
                          # moves back toward the torso). Purely horizontal, height
                          # unchanged -- HOVER_OFFSET was already tuned down once
                          # to fix a different problem (parking at hover instead of
                          # descending); this fixes the approach ANGLE instead of
                          # the approach HEIGHT, so it doesn't undo that. Without
                          # this, the hover waypoint sits directly above the cube in
                          # x/y, so the natural path drops straight down onto its
                          # top face -- exactly the "hitting the cube from above"
                          # behavior observed in training. Estimated, not
                          # teleop-verified like the grasp offset -- revisit after
                          # the next training run.
LIFT_CAP = 0.12          # m -- lift reward saturates here
INSPECT_RADIUS = 0.08    # m -- inspect bonus radius
K = 5.0                  # tanh steepness everywhere
GRIP_CLOSED_THRESHOLD = -0.018  # rad -- FIXED: _is_grasping() used to require
                         # grip_mean > 0.0, i.e. past the actuator's own midpoint.
                         # Measured directly (find_real_grasp_threshold.py: arm
                         # held still, gripper commanded all the way from
                         # GRIPPER_OPEN=-0.02 to GRIPPER_CLOSE=+0.0245 with a real
                         # cube coupled between the fingers) that a genuine,
                         # stable, load-bearing grasp on the cube physically BLOCKS
                         # the joint at just -0.0071 to -0.0174 across 32 samples --
                         # the fingers contact the cube itself (not self-collision,
                         # which is filtered) and the actuator can never reach
                         # anywhere near 0.0, let alone GRIPPER_CLOSE, while
                         # actually holding something. grip_mean > 0.0 was
                         # therefore unreachable during any real grasp, which is
                         # why grasp/lift/inspect never fired despite the policy
                         # visibly picking up and carrying the cube (confirmed via
                         # user's own diagnostic: cube height climbing, contact/
                         # disturbance event, all while grip_mean sat at -0.014).
                         # -0.018 sits with margin above true GRIPPER_OPEN (-0.02,
                         # measured with ~0.0001 noise) and below the worst-case
                         # (least-closed) measured real grasp (-0.0174).
EARLY_CLOSE_DIST = 0.04  # m -- equals GRASP_DIST: "safe to close" and "counts as
                         # grasping" are now the SAME zone. Was 0.032, almost exactly
                         # coincident with the verified ~31.2mm grasp offset -- that
                         # left only ~0.8mm margin, so normal settling jitter kept
                         # straddling the boundary and the gripper got penalized for
                         # closing right at the position it needed to close at.
FINGER_ALIGN_MARGIN = 0.01  # m -- extra slack added to ALIGN_XY when deciding
                         # whether penalty_finger_clearance should back off. The
                         # verified descend target (grasp_offset, xy=2.88cm) sits
                         # INSIDE ALIGN_XY (5cm) already, so a hard cutoff exactly
                         # at ALIGN_XY still leaves near-zero margin against normal
                         # approach noise. This widens the "safe to descend, don't
                         # penalize" zone to ALIGN_XY + this margin, without moving
                         # ALIGN_XY itself (reward_descend still requires the
                         # tighter 5cm to actually pay reward) -- gives the policy
                         # room to move near the boundary instead of a razor edge.

# Cube's resting CENTER height (it's a solid cube, not a point -- its centre sits
# BLOCK_SIZE/2 above the table surface even at rest). reward_lift/inspect/
# inspect_bonus all measure "lifted" relative to THIS, not TABLE_TOP_Z directly --
# the old TABLE_TOP_Z baseline handed out 25% of LIFT_CAP (1.0 of the 4.0-weight
# lift term) for free, just from grasping in place with zero actual lift.
_CUBE_REST_Z = TABLE_TOP_Z + BLOCK_SIZE / 2.0
LIFTED_MARGIN = 0.02    # m -- above _CUBE_REST_Z that counts as "genuinely lifted"
                        # (not just grasped in place); gates reward_inspect/_bonus.
DISTURBANCE_VEL_THRESHOLD = 0.05  # m/s -- cube linear velocity above this, while the
                        # gripper is still open, counts as a real knock (not just
                        # normal settling jitter). Measured contact explosions were
                        # ~8.9-9.6 m/s, ~180x this -- plenty of margin either side.
DISTURBANCE_VEL_CAP = 2.0  # m/s -- hard cap before computing the penalty; beyond
                        # this magnitude more penalty teaches nothing extra, it only
                        # risks feeding an extreme/NaN value into training (this is
                        # what crashed a run: NaN value loss -> invalid policy std).

# Mounting-plate (HAND_BASE_LINK) clearance thresholds.
#
# FIXED: the first version measured distance from the base link's ORIGIN POINT,
# which read ~7.8-18cm from the cube and never looked dangerous. But the plate is
# not a point -- its mesh (measured via USD bounding-box query, see
# probe_base_mesh_extent.py) is a real ~7x7x9cm block extending well beyond its
# own origin toward the fingers:
#   local bbox: x=[-0.035,+0.035]  y=[0,+0.0738]  z=[-0.035,+0.089]
# A corrected, corner-based diagnostic (check_hand_geometry_v2.py) transformed all
# 8 corners into world space and found the plate's ACTUAL edge sustaining ~1.4cm
# xy-distance and ~2-3mm clearance above the cube for multiple consecutive steps
# during a real disturbance event -- i.e. it genuinely does graze the cube, this
# was just invisible to the origin-only check. _hand_base_closest_corner_w below
# uses the same full-extent, corner-based approach for both the reward and the
# observation. Thresholds tightened accordingly (now measuring the real edge, not
# an origin point that was always ~9cm short of it).
BASE_DANGER_XY_RADIUS = 0.03    # m -- xy distance (of the CLOSEST corner) under which the base counts as "over" the cube
BASE_CLEARANCE_MARGIN = 0.015   # m -- required clearance (of the CLOSEST corner) above the cube's top surface

# Base link mesh bounding box in its OWN local frame (probe_base_mesh_extent.py).
_HAND_BASE_BBOX_MIN = (-0.035, 0.000, -0.035)
_HAND_BASE_BBOX_MAX = (0.035, 0.0738, 0.089)
_HAND_BASE_CORNERS_LOCAL = [
    (x, y, z)
    for x in (_HAND_BASE_BBOX_MIN[0], _HAND_BASE_BBOX_MAX[0])
    for y in (_HAND_BASE_BBOX_MIN[1], _HAND_BASE_BBOX_MAX[1])
    for z in (_HAND_BASE_BBOX_MIN[2], _HAND_BASE_BBOX_MAX[2])
]

# Finger-chain link bboxes (own local frame, probe_base_mesh_extent.py) for the
# two bodies check_finger_geometry.py implicated (see FINGER_CLEARANCE_LINKS).
_FINGER_BBOXES = {
    "right_hand_Link1_2": ((-0.006, -0.0068, -0.0304), (0.013, 0.075, 0.000)),
    "right_hand_Link1_3": ((-0.000, -0.002, -0.0284), (0.004, 0.0475, 0.000)),
}
_FINGER_CORNERS_LOCAL = [
    [
        (x, y, z)
        for x in (_FINGER_BBOXES[name][0][0], _FINGER_BBOXES[name][1][0])
        for y in (_FINGER_BBOXES[name][0][1], _FINGER_BBOXES[name][1][1])
        for z in (_FINGER_BBOXES[name][0][2], _FINGER_BBOXES[name][1][2])
    ]
    for name in FINGER_CLEARANCE_LINKS
]

# All 7 tracked hand-body bboxes (own local frame, probe_base_mesh_extent.py), for
# TABLE clearance specifically -- see TABLE_CLEARANCE_LINKS for why this is a
# wider set than FINGER_CLEARANCE_LINKS.
_TABLE_BBOXES = {
    "right_hand_base_link": ((-0.035, 0.000, -0.035), (0.035, 0.0738, 0.089)),
    "right_hand_Link1_1":   ((-0.080, -0.005, -0.008), (-0.000, 0.009, 0.0065)),
    "right_hand_Link1_2":   ((-0.006, -0.0068, -0.0304), (0.013, 0.075, 0.000)),
    "right_hand_Link1_3":   ((-0.000, -0.002, -0.0284), (0.004, 0.0475, 0.000)),
    "right_hand_Link2_1":   ((0.000, -0.005, -0.0065), (0.080, 0.009, 0.008)),
    "right_hand_Link2_2":   ((-0.013, -0.0068, 0.000), (0.006, 0.075, 0.0304)),
    "right_hand_Link2_3":   ((-0.004, -0.002, -0.0284), (0.000, 0.0475, 0.000)),
}
_TABLE_CORNERS_LOCAL = [
    [
        (x, y, z)
        for x in (_TABLE_BBOXES[name][0][0], _TABLE_BBOXES[name][1][0])
        for y in (_TABLE_BBOXES[name][0][1], _TABLE_BBOXES[name][1][1])
        for z in (_TABLE_BBOXES[name][0][2], _TABLE_BBOXES[name][1][2])
    ]
    for name in TABLE_CLEARANCE_LINKS
]

# Table's world-space (env-local) footprint, from the SAME constants the scene
# itself is built from -- not re-measured/guessed. Confirmed via
# check_table_contact.py: TABLE_TOP_Z matches the box's actual top face exactly.
_TABLE_X_RANGE = (TABLE_POS[0] - TABLE_SIZE[0] / 2, TABLE_POS[0] + TABLE_SIZE[0] / 2)
_TABLE_Y_RANGE = (TABLE_POS[1] - TABLE_SIZE[1] / 2, TABLE_POS[1] + TABLE_SIZE[1] / 2)
TABLE_CLEARANCE_MARGIN = BASE_CLEARANCE_MARGIN  # m -- reusing the same verified
                                                 # margin rather than a new guess

# Arm-vs-own-body (torso/head self-collision) clearance. UNLIKE base_clearance/
# table_clearance, this deliberately uses plain body-ORIGIN distance (Isaac
# Lab's own trusted body_pos_w), not a corner/mesh-extent check -- three
# separate attempts to calibrate a correct local bbox for torso_link/head_link
# against this USD asset's live simulation state gave three contradictory
# answers, while origin distance was measured consistently.
#
# CRITICAL, checked before picking a threshold (check_torso_clearance_
# calibration.py): at Policy 2's normal RESET/REACH pose (PRE_GRASP_ARM_POSE,
# gripper open, arm extending toward the table -- a legitimate, necessary
# configuration), every arm body already sits within 0.155-0.219m of
# torso_link. A threshold anywhere near that range, applied unconditionally,
# would penalize the reach phase itself -- the exact "verified pose sits
# inside its own penalty's danger zone" mistake that broke the first version of
# penalty_finger_clearance. Fix: gate this penalty on _is_grasping() (in
# penalty_torso_clearance below), so it only applies during the carry-to-
# inspect phase, where the actual risk is (the screenshot that motivated this
# showed the held cube/gripper pressed against torso_link during carry, not
# during reach).
#
# Threshold calibrated against the CARRY phase's own safe reference (the
# teleop-verified inspect arm pose): measured distances there ranged
# 0.215-0.256m across all 5 bodies (closest: right_wrist_pitch_link at
# 0.2154m). TORSO_CLEARANCE_MIN_DIST sits below that whole range with margin,
# so the verified-safe carry pose itself is not penalized, while configurations
# noticeably closer than that (e.g. the ~0.17m tension case) are.
TORSO_CLEARANCE_LINKS = [
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link", "right_hand_base_link",
]
TORSO_BODY_LINKS = ["torso_link", "head_link"]
TORSO_CLEARANCE_MIN_DIST = 0.19  # m -- origin-to-origin, see reasoning above

# --- Policy 3 (place + release) thresholds. PLACE_DIST/PLACE_HEIGHT_MARGIN/
# PLACE_SETTLE_VEL mirror the pattern already used (and proven) in the earlier
# g1_redblock_ext project for the same "is it actually placed" question, adapted
# to this project's shared helpers -- not new guesses.
PLACE_DIST = 0.05           # m -- xy radius around GOAL_POS that counts as "there"
PLACE_HEIGHT_MARGIN = 0.04  # m -- above _CUBE_REST_Z that still counts as "on the table"
PLACE_SETTLE_VEL = 0.05     # m/s -- cube linear speed below which it counts as settled

# reward_settle_near_goal thresholds. FIXED: place/release stayed at ~0.000 for
# 873/1500 iterations of a real training run despite move_to_goal climbing
# steadily toward its max -- diagnosed as a reward-shaping gap, not a threshold
# bug (unlike the GRIP_CLOSED_THRESHOLD case): move_to_goal is dense and
# continuous (rewards getting CLOSE), but place/release are sparse, all-or-
# nothing bonuses requiring simultaneous proximity + low height + LOW VELOCITY,
# with no gradient anywhere encouraging the policy to actually decelerate and
# stop once near the goal -- it can collect most of move_to_goal's reward by
# hovering near the goal while still moving, with no pressure to fully commit.
SETTLE_NEAR_RADIUS = 0.15   # m -- wider than PLACE_DIST so the reward gradient
                            # starts before the exact success zone, guiding the
                            # policy IN toward slowing down, not just rewarding
                            # it after the fact once already stopped there.
SETTLE_VEL_SCALE = 0.08     # m/s -- decay scale for the velocity term; chosen
                            # so that reaching PLACE_SETTLE_VEL (0.05) already
                            # gives a substantial reward (exp(-0.05/0.08)=0.53),
                            # not just an all-or-nothing cliff at the threshold.

# reward_return_to_ready: after release, nothing pulled the arm back to a
# known configuration -- the whole point of the 3-policy chain is that each
# policy's end state approximates the next one's start, so the robot is ready
# for another pick-and-place cycle, not left with the hand wherever it
# happened to be. Pulls the arm's joints (radian-space, not meters) back
# toward READY_ARM_POSE (Policy 1's own reset target) once released, closing
# the loop back to Policy 1's expected starting state. K_JOINT is separate
# from the shared K=5.0 (tuned for metre-scale position errors elsewhere) --
# joint-space L2 distances run much larger (up to a few radians), so K=5.0
# would saturate almost immediately and give no usable gradient across the
# real range.
#
# FIXED: 1.5 -> 0.5. The first value (1.5) was ITSELF still too steep --
# confirmed empirically: a full Policy 1 training run with reward_match_
# pregrasp_pose (same K_JOINT) converged to a REAL joint-space distance of
# ~2.67 rad from PRE_GRASP_ARM_POSE, at which tanh(1.5*2.67)=0.999, giving a
# reward of ~0.0007 -- matched the logged Episode_Reward/match_pregrasp value
# (0.0006) almost exactly, confirming the gradient had been saturated near
# zero for essentially the entire run. The policy had no usable signal to
# follow at all and just ignored the pose-matching objective. 0.5 keeps a
# real, non-saturated gradient across the actual observed problem range (e.g.
# tanh(0.5*2.67)=0.716 -> reward 0.284 at the previous failure point, vs
# tanh(0.5*0.5)=0.245 -> reward 0.755 once genuinely close).
K_JOINT = 0.5


def _ee_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_bodies(EE_LINKS)
    return robot.data.body_pos_w[:, ids, :].mean(dim=1)


def _hand_base_closest_corner_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """World position of whichever corner of the base plate's ACTUAL mesh extent
    (not just its origin -- confirmed to undercount the real risk by ~9cm) is
    closest, in xy, to the cube. (N, 3)"""
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_bodies([HAND_BASE_LINK])
    pos = robot.data.body_pos_w[:, ids[0], :]      # (N, 3)
    quat = robot.data.body_quat_w[:, ids[0], :]    # (N, 4)
    cube_pos = _cube_pos_w(env)

    corners_local = torch.tensor(_HAND_BASE_CORNERS_LOCAL, device=env.device)  # (8, 3)
    n = pos.shape[0]
    q = quat.unsqueeze(1).expand(n, 8, 4).reshape(n * 8, 4)
    c = corners_local.unsqueeze(0).expand(n, 8, 3).reshape(n * 8, 3)
    world_offsets = quat_apply(q, c).reshape(n, 8, 3)
    world_corners = pos.unsqueeze(1) + world_offsets  # (N, 8, 3)

    xy_dist = torch.norm(world_corners[..., :2] - cube_pos[:, None, :2], dim=-1)  # (N, 8)
    closest_idx = xy_dist.argmin(dim=-1, keepdim=True)  # (N, 1)
    idx_expanded = closest_idx.unsqueeze(-1).expand(-1, -1, 3)  # (N, 1, 3)
    return torch.gather(world_corners, 1, idx_expanded).squeeze(1)  # (N, 3)


def _cube_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj: RigidObject = env.scene["object"]
    return obj.data.root_pos_w


def _ee_obj_dist(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.norm(_ee_pos_w(env) - _cube_pos_w(env), dim=-1)


def _grip_mean(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    gids, _ = robot.find_joints(GRIPPER_JOINTS)
    return robot.data.joint_pos[:, gids].mean(dim=-1)


def _is_grasping(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Enveloped (dist < GRASP_DIST) AND gripper closed as far as it can get
    while physically blocked by the cube (see GRIP_CLOSED_THRESHOLD)."""
    return (_ee_obj_dist(env) < GRASP_DIST) & (_grip_mean(env) > GRIP_CLOSED_THRESHOLD)


def _closedness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gripper position normalized to 0 (GRIPPER_OPEN) .. 1 (GRIPPER_CLOSE)."""
    return torch.clamp((_grip_mean(env) - GRIPPER_OPEN) / (GRIPPER_CLOSE - GRIPPER_OPEN), 0.0, 1.0)


def _lifted(env: ManagerBasedRLEnv) -> torch.Tensor:
    """True once the cube is genuinely off the table (height measured from its own
    resting centre, _CUBE_REST_Z, not just grasped in place with zero real lift)."""
    return (_cube_pos_w(env)[:, 2] - _CUBE_REST_Z) > LIFTED_MARGIN


# ---------------- stages ----------------
def reward_hover(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 1: dense pull to the waypoint HOVER_OFFSET above AND HOVER_BACK_OFFSET
    behind (toward the robot body) the cube. Always on."""
    target = _cube_pos_w(env).clone()
    target[:, 1] += HOVER_BACK_OFFSET
    target[:, 2] += HOVER_OFFSET
    return 1.0 - torch.tanh(K * torch.norm(_ee_pos_w(env) - target, dim=-1))


def reward_descend(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 2: dense pull to the verified grasp approach position (offset from cube center)."""
    ee, cube = _ee_pos_w(env), _cube_pos_w(env)
    xy_err = torch.norm(ee[:, :2] - cube[:, :2], dim=-1)
    aligned = (xy_err < ALIGN_XY).float()

    # Verified via arm_teleop.py: pre-grasp approach pose measured EE-cube offset
    # (-0.010, +0.027, +0.012), dist=0.032. A second reading at the grasped/held/
    # inspection pose measured (-0.016, +0.027, -0.005), dist=0.032 -- the y-offset
    # (+0.027) agrees across both independently-posed readings, so it's the real
    # fingertip-midpoint-to-cube-centre offset of the Dex1 envelope grasp, not noise.
    # magnitude = 0.0312 m, safely inside GRASP_DIST (0.04 m) with ~0.9cm margin --
    # unlike the previous +0.045 guess (0.0476 m), which sat OUTSIDE the grasp gate
    # and made grasp/lift/inspect structurally unreachable (see diagnostic).
    grasp_offset = torch.tensor(GRASP_OFFSET, device=env.device)
    descend_target = cube + grasp_offset

    return aligned * (1.0 - torch.tanh(K * torch.norm(ee - descend_target, dim=-1)))


def reward_straddle_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Encourages the two-finger straddle to stay roughly HORIZONTAL (matching
    the verified teleop side-approach GRASP_OFFSET implies), not tilted/
    palm-down. See STRADDLE_Z_SCALE's comment for the diagnosis this fixes.
    Gated the same way as reward_descend (xy-aligned) -- orientation only
    matters once the hand is actually near the cube, not during transit."""
    robot: Articulation = env.scene["robot"]
    ee, cube = _ee_pos_w(env), _cube_pos_w(env)
    xy_err = torch.norm(ee[:, :2] - cube[:, :2], dim=-1)
    aligned = (xy_err < ALIGN_XY).float()

    ids, _ = robot.find_bodies(EE_LINKS)
    p1 = robot.data.body_pos_w[:, ids[0], :]
    p2 = robot.data.body_pos_w[:, ids[1], :]
    offset = p1 - p2
    z_mag = torch.abs(offset[:, 2])
    return aligned * torch.exp(-z_mag / STRADDLE_Z_SCALE)


def reward_match_pregrasp_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """SUPERSEDES reward_straddle_orientation for real-deployment safety:
    that reward only encouraged "some horizontal straddle," which fixed the
    palm-down tilt but converged to an orientation that still didn't match
    PRE_GRASP_ARM_POSE (visually confirmed "upside down" relative to Policy
    2's actual reset). For a real robot where control switches from Policy 1
    to Policy 2 using whatever state Policy 1 actually left the arm in (not a
    synthetic reset), EE-position matching alone isn't enough -- Policy 2's
    observations include raw joint angles (arm_joint_pos_rel/arm_joint_vel),
    so a joint configuration Policy 2 never saw during ITS OWN training
    (PRE_GRASP_ARM_POSE +- _ARM_POSE_NOISE=0.03 rad) risks out-of-distribution
    behavior even with correct EE/cube geometry. Pulls the arm's joints
    directly toward PRE_GRASP_ARM_POSE (which already IS a genuine, verified
    grasp orientation -- matching it exactly subsumes the straddle-alignment
    goal, no need for both). Same alignment gate as reward_descend -- pose
    matching only matters once actually near the cube, not during transit."""
    robot: Articulation = env.scene["robot"]
    ee, cube = _ee_pos_w(env), _cube_pos_w(env)
    xy_err = torch.norm(ee[:, :2] - cube[:, :2], dim=-1)
    aligned = (xy_err < ALIGN_XY).float()

    arm_ids = [robot.find_joints([n])[0][0] for n in ARM_JOINTS]
    current = robot.data.joint_pos[:, arm_ids]
    target = torch.tensor([PRE_GRASP_ARM_POSE[j] for j in ARM_JOINTS], device=env.device)
    dist = torch.norm(current - target, dim=-1)
    return aligned * (1.0 - torch.tanh(K_JOINT * dist))


def reward_close_gradient(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bridge between descend and grasp: dense reward for closing PROGRESSIVELY as
    the hand approaches, instead of only paying at the exact grasp instant.

    FIXED: the first version used the shared K=5 tanh falloff (same as hover/
    descend, tuned for a ~20cm approach). That decayed too slowly relative to the
    4cm GRASP_DIST/EARLY_CLOSE_DIST boundary -- closing was net-rewarding (this
    term's payoff beat penalty_early_close's) anywhere within ~11cm of the cube,
    reopening the close-too-early problem early_close exists to prevent (confirmed
    empirically: dropped-episode rate roughly doubled after adding it). Now the
    falloff is tied directly to GRASP_DIST (linear ramp to 0 by 2x GRASP_DIST,
    i.e. ~8cm) instead of an unrelated, slower shared constant, and it's gated on
    xy-alignment exactly like descend, so it can't reward closing from an
    off-centre angle either (the tight ~1.5cm finger-vs-cube clearance makes that
    a real physical knock-off risk, not just a reward-shaping one)."""
    ee, cube = _ee_pos_w(env), _cube_pos_w(env)
    xy_err = torch.norm(ee[:, :2] - cube[:, :2], dim=-1)
    aligned = (xy_err < ALIGN_XY).float()
    dist = _ee_obj_dist(env)
    proximity = torch.clamp(1.0 - dist / (2.0 * GRASP_DIST), 0.0, 1.0)
    return aligned * proximity * _closedness(env)


def reward_grasp(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 3: bonus while enveloping AND closing."""
    return _is_grasping(env).float()


def reward_lift(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 4: height above the cube's OWN resting height (capped), gated on
    grasp. Continuous, so it is also the bodyguard that keeps the grasp held.

    FIXED: baseline was TABLE_TOP_Z, but the cube's resting CENTRE sits
    BLOCK_SIZE/2 above that (it's a solid cube, not a point) -- the old baseline
    handed out 25% of LIFT_CAP (1.0 of this term's 4.0 weight) for free, just from
    grasping in place with zero actual lift. Baseline is now _CUBE_REST_Z, so
    height is genuinely 0 at rest."""
    height = torch.clamp(_cube_pos_w(env)[:, 2] - _CUBE_REST_Z, 0.0, LIFT_CAP)
    return (height / LIFT_CAP) * _is_grasping(env).float()


def reward_inspect(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 5: dense pull of the HELD AND LIFTED cube toward INSPECT_POS.

    FIXED: previously gated only on _is_grasping, so it paid partial credit
    (~0.39 at rest, dist(cube_at_rest, INSPECT_POS)~=0.22m) just for grasping in
    place, with zero actual lift -- same root cause as the reward_lift baseline
    bug. Now also requires _lifted()."""
    inspect_w = torch.tensor(INSPECT_POS, device=env.device) + env.scene.env_origins
    dist = torch.norm(_cube_pos_w(env) - inspect_w, dim=-1)
    return (_is_grasping(env) & _lifted(env)).float() * (1.0 - torch.tanh(K * dist))


def reward_inspect_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Stage 5 bonus: HELD AND LIFTED cube inside INSPECT_RADIUS of INSPECT_POS.
    (INSPECT_RADIUS alone already excluded the at-rest case here -- ~0.22m is far
    outside 0.08m -- but _lifted() is added too for consistency with reward_inspect
    and to stay safe if INSPECT_POS or the cube's rest position ever change.)"""
    inspect_w = torch.tensor(INSPECT_POS, device=env.device) + env.scene.env_origins
    dist = torch.norm(_cube_pos_w(env) - inspect_w, dim=-1)
    return (_is_grasping(env) & _lifted(env) & (dist < INSPECT_RADIUS)).float()


# ---------------- Policy 3: place + release ----------------
def _at_goal_settled(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cube resting near GOAL_POS (xy + table height) AND settled (low velocity) --
    regardless of gripper state. Shared by reward_place (doesn't care if still
    gripped) and reward_release (which additionally requires the gripper open)."""
    obj: RigidObject = env.scene["object"]
    goal_w = torch.tensor(GOAL_POS, device=env.device) + env.scene.env_origins
    d_xy = torch.norm(obj.data.root_pos_w[:, :2] - goal_w[:, :2], dim=-1)
    near_table = obj.data.root_pos_w[:, 2] < (_CUBE_REST_Z + PLACE_HEIGHT_MARGIN)
    settled = torch.norm(obj.data.root_lin_vel_w, dim=-1) < PLACE_SETTLE_VEL
    return (d_xy < PLACE_DIST) & near_table & settled


def reward_move_to_goal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense pull of the held cube toward GOAL_POS (xy + height together), gated on
    genuinely still holding it (not a height floor, which would fight the descent
    needed to reach the table-height goal)."""
    goal_w = torch.tensor(GOAL_POS, device=env.device) + env.scene.env_origins
    dist = torch.norm(_cube_pos_w(env) - goal_w, dim=-1)
    return _is_grasping(env).float() * (1.0 - torch.tanh(K * dist))


def reward_settle_near_goal(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense bridge between reward_move_to_goal (rewards getting close) and
    reward_place/reward_release (sparse, all-or-nothing 'settled' bonus) --
    see SETTLE_NEAR_RADIUS/SETTLE_VEL_SCALE's comment for why this was added.
    Rewards being close to GOAL_POS AND moving slowly, continuously, so the
    policy has a gradient actively pulling it toward decelerating and
    stopping as it nears the goal, not just toward proximity alone."""
    obj: RigidObject = env.scene["object"]
    goal_w = torch.tensor(GOAL_POS, device=env.device) + env.scene.env_origins
    dist = torch.norm(_cube_pos_w(env) - goal_w, dim=-1)
    near = torch.clamp(1.0 - dist / SETTLE_NEAR_RADIUS, 0.0, 1.0)
    speed = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    slow = torch.exp(-speed / SETTLE_VEL_SCALE)
    return _is_grasping(env).float() * near * slow


def reward_place(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bonus: cube resting at the goal, settled -- regardless of gripper state.
    reward_release adds the actual let-go requirement on top of this, and must
    out-pay this term alone so releasing beats clinging (see reward_release)."""
    return _at_goal_settled(env).float()


RELEASE_HOLD_STEPS = 15  # ~0.15s at decimation=2/sim.dt=0.005 -- see reward_release

def reward_release(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bonus: _at_goal_settled AND gripper genuinely, SUSTAINEDLY open -- the
    real 'let go' behaviour, not just parking a still-held cube at the goal.
    Combined with reward_place and reward_move_to_goal's weights, releasing at
    the goal nets strictly more total reward than holding there (the "release
    flip": a policy that learned to grip will cling forever unless letting go
    pays more).

    FIXED: was a pure instantaneous check (settled AND grip momentarily below
    threshold), no minimum duration. Policy 4's first trained checkpoint
    (2026-07-16_08-44-24) exploited this directly: post-hoc replay showed
    32/32 envs satisfying the condition at least once (ever_released), but the
    episode ended with grip_mean back at the closed/holding value and the arm
    at EXACTLY its reset distance from READY_ARM_POSE (std=0.0000 across all
    envs) -- the policy flickered the gripper open for a single instant to
    collect the reward, then immediately re-closed around the same cube and
    never moved, since return_to_ready's NOT-grasping gate never got a real
    window to activate. Requiring the condition to hold for RELEASE_HOLD_STEPS
    consecutive steps (same reasoning _at_goal_settled already applies to
    velocity -- a momentary dip shouldn't count) closes that exploit: a
    single-frame flicker can no longer pay out, only an actual sustained
    release can.

    Needs a persistent per-env counter, unlike every other reward term in this
    module (all pure functions of live state) -- lazily attached to `env`
    itself since RewTerm functions have no other place to keep state between
    steps. Cleared for envs starting a fresh episode (episode_length_buf<=1 is
    the first step a reward function ever observes post-reset, since
    _reset_idx runs after reward computation -- see manager_based_rl_env.py's
    step()) so a near-miss from a just-terminated episode can't carry over."""
    if not hasattr(env, "_release_hold_counter"):
        env._release_hold_counter = torch.zeros(env.num_envs, device=env.device)
    fresh = env.episode_length_buf <= 1
    env._release_hold_counter[fresh] = 0.0

    gripper_open = _grip_mean(env) <= GRIP_CLOSED_THRESHOLD  # see GRIP_CLOSED_THRESHOLD
    instantaneous = _at_goal_settled(env) & gripper_open
    env._release_hold_counter = torch.where(
        instantaneous, env._release_hold_counter + 1.0, torch.zeros_like(env._release_hold_counter)
    )
    return (env._release_hold_counter >= RELEASE_HOLD_STEPS).float()


def reward_return_to_ready(env: ManagerBasedRLEnv) -> torch.Tensor:
    """NEW: dense pull of the arm's joints back toward READY_ARM_POSE (Policy
    1's own reset target) once released -- see K_JOINT's comment for why this
    exists. Gated on NOT grasping so it can't compete with move_to_goal/
    settle/place/release, which are all only active while still holding the
    cube -- these gates are naturally sequential (grasping -> released), not
    overlapping, so this can't create the kind of accidental incentive
    conflict reward_settle_near_goal did with reward_release."""
    robot: Articulation = env.scene["robot"]
    arm_ids = [robot.find_joints([n])[0][0] for n in ARM_JOINTS]
    current = robot.data.joint_pos[:, arm_ids]
    target = torch.tensor([READY_ARM_POSE[j] for j in ARM_JOINTS], device=env.device)
    dist = torch.norm(current - target, dim=-1)
    not_grasping = (~_is_grasping(env)).float()
    return not_grasping * (1.0 - torch.tanh(K_JOINT * dist))


# ---------------- penalties ----------------
def penalty_contact_disturbance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalizes cube linear velocity above DISTURBANCE_VEL_THRESHOLD while the
    gripper is still open -- directly discourages knocking/hitting the cube during
    approach (measured directly: ~8.9-9.6 m/s launches happening before any close
    attempt). Complements, doesn't replace, the physics-level depenetration-velocity
    cap -- that limits how hard contact CAN resolve; this gives the policy an actual
    training signal to avoid causing the contact in the first place. Only linear
    velocity is used (not angular) for simplicity -- the two spike together in the
    measured events, so linear alone is a sufficient, simpler proxy."""
    obj: RigidObject = env.scene["object"]
    lin_vel = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    # Defensive: a genuinely pathological contact/penetration event could in
    # principle produce NaN or an extreme velocity from the physics engine itself.
    # torch.clamp alone does NOT sanitize NaN (it passes through unchanged) -- this
    # crashed a training run (NaN value loss -> invalid policy std -> sampler
    # error) once already, so both a NaN guard and a hard magnitude cap are used.
    lin_vel = torch.nan_to_num(lin_vel, nan=0.0, posinf=DISTURBANCE_VEL_CAP, neginf=0.0)
    lin_vel = torch.clamp(lin_vel, max=DISTURBANCE_VEL_CAP)
    excess = torch.clamp(lin_vel - DISTURBANCE_VEL_THRESHOLD, min=0.0)
    # FIXED: was `<= 0.0`, matching _is_grasping()'s old (unreachable-while-
    # grasping) threshold -- that meant this stayed "still open" even during a
    # real, held grasp, so it penalized the cube's own legitimate carrying
    # velocity as if it were an unwanted knock. Now consistent with
    # GRIP_CLOSED_THRESHOLD/_is_grasping().
    still_open = (_grip_mean(env) <= GRIP_CLOSED_THRESHOLD).float()
    return excess * still_open


def penalty_early_close(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Closed gripper while far = penalized; keeps the approach open-handed."""
    return _closedness(env) * (_ee_obj_dist(env) > EARLY_CLOSE_DIST).float()


def penalty_base_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalizes the solid mounting plate (HAND_BASE_LINK) sitting over AND too
    low relative to the cube -- visually confirmed AND empirically verified
    (check_hand_geometry_v2.py: sustained ~1.4cm xy / ~2-3mm clearance during a
    real disturbance event) striking the cube's TOP surface during descent.
    Neither the fingertip-based approach reward nor penalty_contact_disturbance
    (velocity-only, reacts after the fact) have any visibility into this body at
    all; this is a dedicated, POSITION-based, anticipatory signal specifically for
    it. Uses the plate's CLOSEST CORNER (full mesh extent), not its origin -- the
    origin-only version undercounted the real risk by ~9cm and never fired."""
    corner = _hand_base_closest_corner_w(env)
    cube_pos = _cube_pos_w(env)
    xy_dist = torch.norm(corner[:, :2] - cube_pos[:, :2], dim=-1)
    cube_top_z = cube_pos[:, 2] + BLOCK_SIZE / 2.0
    clearance = corner[:, 2] - (cube_top_z + BASE_CLEARANCE_MARGIN)
    over_cube = (xy_dist < BASE_DANGER_XY_RADIUS).float()
    return torch.clamp(-clearance, min=0.0) * over_cube


def _finger_closest_corners_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """World position of each FINGER_CLEARANCE_LINKS body's closest-in-xy corner
    (full mesh extent, not origin). (N, B, 3), B = len(FINGER_CLEARANCE_LINKS)."""
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_bodies(FINGER_CLEARANCE_LINKS, preserve_order=True)
    pos = robot.data.body_pos_w[:, ids, :]      # (N, B, 3)
    quat = robot.data.body_quat_w[:, ids, :]    # (N, B, 4)
    cube_pos = _cube_pos_w(env)

    corners_local = torch.tensor(_FINGER_CORNERS_LOCAL, device=env.device)  # (B, 8, 3)
    n, b = pos.shape[0], pos.shape[1]
    q = quat.unsqueeze(2).expand(n, b, 8, 4).reshape(n * b * 8, 4)
    c = corners_local.unsqueeze(0).expand(n, b, 8, 3).reshape(n * b * 8, 3)
    world_offsets = quat_apply(q, c).reshape(n, b, 8, 3)
    world_corners = pos.unsqueeze(2) + world_offsets  # (N, B, 8, 3)

    xy_dist = torch.norm(world_corners[..., :2] - cube_pos[:, None, None, :2], dim=-1)  # (N,B,8)
    closest_idx = xy_dist.argmin(dim=-1, keepdim=True)  # (N, B, 1)
    idx_expanded = closest_idx.unsqueeze(-1).expand(-1, -1, -1, 3)  # (N, B, 1, 3)
    return torch.gather(world_corners, 2, idx_expanded).squeeze(2)  # (N, B, 3)


def penalty_finger_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalizes FINGER_CLEARANCE_LINKS (right_hand_Link1_2/Link1_3) sitting over
    AND too low relative to the cube WHILE THE GRIPPER IS STILL OPEN AND THE
    APPROACH IS NOT (near-)ALIGNED -- the same origin-vs-mesh gap as
    penalty_base_clearance, empirically confirmed via check_finger_geometry.py
    (59 real disturbance events, only these two bodies implicated, all with the
    gripper open).

    FIXED: the first version had no alignment gate at all, only "still open".
    That structurally conflicted with reward_descend -- its verified target
    (grasp_offset, xy=2.88cm from the cube) sits INSIDE this penalty's own
    BASE_DANGER_XY_RADIUS (3cm), so the correct, deliberate handoff position was
    being penalized by construction, not just occasionally. No reward weight
    could fix that (confirmed: cutting the weight 3x made no measurable
    difference to the training collapse it caused). Now excludes the case where
    the EE is within ALIGN_XY + FINGER_ALIGN_MARGIN of the cube -- the same
    "is this a deliberate, centered approach" concept reward_descend already
    uses, with extra slack so the boundary isn't a hard cutoff -- while still
    catching a clumsy, off-center approach that happens to get close and low
    (the actual "unwanted knock" scenario the diagnostic found).

    UNLIKE penalty_base_clearance, this is ALSO gated on "still open" -- the
    base plate never legitimately touches the cube in any policy, but these two
    finger bodies must, once Policy 2 commits to a real grasp. Policy 1 never
    grasps anyway (gripper stays open all episode), so that gate costs it
    nothing while protecting Policy 2."""
    corners = _finger_closest_corners_w(env)  # (N, B, 3)
    cube_pos = _cube_pos_w(env)
    xy_dist = torch.norm(corners[..., :2] - cube_pos[:, None, :2], dim=-1)  # (N, B)
    cube_top_z = cube_pos[:, 2] + BLOCK_SIZE / 2.0
    clearance = corners[..., 2] - (cube_top_z[:, None] + BASE_CLEARANCE_MARGIN)  # (N,B)
    over_cube = (xy_dist < BASE_DANGER_XY_RADIUS).float()
    still_open = (_grip_mean(env) <= GRIP_CLOSED_THRESHOLD).float()  # see GRIP_CLOSED_THRESHOLD

    ee_xy_err = torch.norm(_ee_pos_w(env)[:, :2] - cube_pos[:, :2], dim=-1)
    not_aligned = (ee_xy_err >= ALIGN_XY + FINGER_ALIGN_MARGIN).float()

    per_body_penalty = torch.clamp(-clearance, min=0.0) * over_cube  # (N, B)
    return per_body_penalty.sum(dim=-1) * still_open * not_aligned


def penalty_table_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalizes any TABLE_CLEARANCE_LINKS body sweeping low over the TABLE's own
    surface (not the cube) -- empirically confirmed via check_table_contact.py:
    against the table's real 3D footprint (not just a Z-height comparison), most
    events were small (a few mm) but one mid-episode event reached 0.217m, on the
    same order as a genuine, uncapped collision. UNLIKE penalty_finger_clearance,
    this is UNCONDITIONAL (no gripper-state gate) -- the arm should never
    legitimately touch the table's surface in ANY of the 3 policies, unlike
    fingers-vs-cube where contact is the whole point of grasping.

    NOTE: this does NOT fix the separate, larger issue check_table_contact.py also
    found -- every episode starts with the reset pose (READY_ARM_POSE) already
    ~0.223m inside the table's volume, resolved by the physics engine's capped
    depenetration response before the policy ever acts. A reward term can only
    shape actions the policy actually takes; it has no way to influence the state
    at the instant right after env.reset(). That needs a direct fix to the reset
    pose itself, not reward shaping -- deliberately out of scope here per explicit
    instruction to add just the penalty and start training.

    Uses the same closest-corner approach as the other clearance penalties, but
    "over the table" is a box test (x AND y inside the table's real footprint),
    not a radius around a small object like the cube."""
    return _table_clearance_per_body(env).sum(dim=-1)


def _table_clearance_per_body(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Shared core for penalty_table_clearance and
    penalty_table_clearance_near_goal_excluded -- returns per-body penalty
    (N, B), not yet summed, so callers can gate/mask before summing."""
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_bodies(TABLE_CLEARANCE_LINKS, preserve_order=True)
    pos = robot.data.body_pos_w[:, ids, :]      # (N, B, 3)
    quat = robot.data.body_quat_w[:, ids, :]    # (N, B, 4)
    origins = env.scene.env_origins             # (N, 3)

    corners_local = torch.tensor(_TABLE_CORNERS_LOCAL, device=env.device)  # (B, 8, 3)
    n, b = pos.shape[0], pos.shape[1]
    q = quat.unsqueeze(2).expand(n, b, 8, 4).reshape(n * b * 8, 4)
    c = corners_local.unsqueeze(0).expand(n, b, 8, 3).reshape(n * b * 8, 3)
    world_offsets = quat_apply(q, c).reshape(n, b, 8, 3)
    world_corners = pos.unsqueeze(2) + world_offsets  # (N, B, 8, 3)
    local_corners = world_corners - origins[:, None, None, :]  # strip env offset -> table-local frame

    x_in = (local_corners[..., 0] >= _TABLE_X_RANGE[0]) & (local_corners[..., 0] <= _TABLE_X_RANGE[1])
    y_in = (local_corners[..., 1] >= _TABLE_Y_RANGE[0]) & (local_corners[..., 1] <= _TABLE_Y_RANGE[1])
    over_table = x_in & y_in  # (N, B, 8)

    z = local_corners[..., 2]
    # Corners outside the table's footprint shouldn't count toward "how low is
    # the lowest at-risk corner" -- push them to +inf so they're never picked as
    # the min (clamp(-inf, min=0) below then naturally resolves to 0, no NaN).
    z_over_table = torch.where(over_table, z, torch.full_like(z, float("inf")))
    min_z, _ = z_over_table.min(dim=-1)  # (N, B)
    clearance = min_z - (TABLE_TOP_Z + TABLE_CLEARANCE_MARGIN)  # (N, B)
    return torch.clamp(-clearance, min=0.0)  # (N, B)


NEAR_GOAL_PENETRATION_ALLOWANCE = TABLE_CLEARANCE_MARGIN  # m -- see
    # penalty_table_clearance_near_goal_excluded's docstring: exactly cancels
    # TABLE_CLEARANCE_MARGIN so the near-goal allowance is "down to the table
    # SURFACE", not "down to the safety margin above it".


def penalty_table_clearance_near_goal_excluded(env: ManagerBasedRLEnv) -> torch.Tensor:
    """FIXED (2nd time): the original fix (full exclusion near the goal) was
    itself too permissive. Measured directly (check_link2_table_contact.py) on
    a checkpoint trained under the full exclusion: right_hand_base_link showed
    100% contact rate near the goal with 2.01cm MEAN penetration BEYOND
    TABLE_CLEARANCE_MARGIN (1.5cm) -- i.e. genuinely ~0.5cm into the table's
    solid volume, not just "close to the surface", nearly every single step
    near the goal. Zeroing the penalty entirely gave no gradient against this
    at all, so nothing stopped it from getting arbitrarily deep. This is
    exactly what the user visually reported as the hand/finger link dragging
    through the table.

    FIXED (correctly this time): still allow legitimate near-goal contact
    (unconditionally excluding it was right in spirit -- placing an object
    down requires the hand to approach the table), but cap it PER-BODY at
    NEAR_GOAL_PENETRATION_ALLOWANCE instead of a full exclusion. Since
    _table_clearance_per_body's raw value is already measured relative to
    TABLE_CLEARANCE_MARGIN (not the raw table surface), subtracting
    NEAR_GOAL_PENETRATION_ALLOWANCE (== TABLE_CLEARANCE_MARGIN) converts the
    tolerance from "must stay above the safety margin" (too strict for
    placing) to "must not sink through the table's actual surface" (the real
    line that should never be crossed) -- touching is free, penetrating is
    not. Everywhere else (during transit/reach) is unchanged: the full,
    unconditional penalty from penalty_table_clearance still applies."""
    obj: RigidObject = env.scene["object"]
    goal_w = torch.tensor(GOAL_POS, device=env.device) + env.scene.env_origins
    xy_dist = torch.norm(obj.data.root_pos_w[:, :2] - goal_w[:, :2], dim=-1)
    near_goal = xy_dist < SETTLE_NEAR_RADIUS

    per_body = _table_clearance_per_body(env)  # (N, B), already clamped >=0
    capped_per_body = torch.clamp(per_body - NEAR_GOAL_PENETRATION_ALLOWANCE, min=0.0)
    near_goal_penalty = capped_per_body.sum(dim=-1)
    full_penalty = per_body.sum(dim=-1)

    return torch.where(near_goal, near_goal_penalty, full_penalty)


MIN_CARRY_HEIGHT = LIFT_CAP  # m -- reusing Policy 1/2's already-validated lift
                              # saturation threshold (0.12m), not a fresh guess.
                              # Policy 2 already hands off well above this
                              # (INSPECT_POS is 0.157m above _CUBE_REST_Z) so
                              # it's proven achievable from Policy 3's own
                              # reset state, not aspirational.


def penalty_low_carry(env: ManagerBasedRLEnv) -> torch.Tensor:
    """NEW: Policy 3 has nothing rewarding height while carrying -- unlike
    Policy 1/2's reward_lift, move_to_goal only scores the cube's xyz
    DISTANCE to GOAL_POS, so "skim it an inch above the table the whole way"
    and "carry it at a safe height" score identically as long as the final
    position matches. Measured directly (policy3_cube_height_timeline.py) on
    the narrowed checkpoint: cube starts ~0.134m above _CUBE_REST_Z at reset
    (matching Policy 2's own handoff height) but collapses to ~0.012-0.018m
    within the first ~10 steps and stays there for the rest of the episode --
    visually indistinguishable from dragging the cube along the table, per
    user report.

    FIXED: was a raw shortfall-in-meters penalty (0 to MIN_CARRY_HEIGHT range),
    copying _table_clearance_per_body's pattern -- but that penalty's typical
    violation is only 3-5cm (bounded by how far a body can physically
    penetrate before physics pushes back), while this one's naturally ranges
    up to the full MIN_CARRY_HEIGHT=0.12m. Same weight number on a ~2-3x
    larger typical raw value meant it wasn't actually comparable to
    table_clearance despite matching its weight -- closer to competing with
    move_to_goal outright than intended. Normalized into a [0,1] ratio
    instead, the same pattern reward_lift already uses (fitting, since
    MIN_CARRY_HEIGHT literally *is* LIFT_CAP) -- puts it on the same scale as
    move_to_goal/settle/place, so its weight below is now directly comparable
    to theirs instead of borrowed from a differently-scaled term.

    Near-goal exclusion (same as penalty_table_clearance_near_goal_excluded):
    this MUST turn off near the goal, or it directly fights place/settle,
    which *want* the cube to come down there."""
    obj: RigidObject = env.scene["object"]
    height_above_rest = torch.clamp(obj.data.root_pos_w[:, 2] - _CUBE_REST_Z, 0.0, MIN_CARRY_HEIGHT)
    shortfall_ratio = 1.0 - (height_above_rest / MIN_CARRY_HEIGHT)  # 0 at/above target, 1 at rest height

    goal_w = torch.tensor(GOAL_POS, device=env.device) + env.scene.env_origins
    xy_dist = torch.norm(obj.data.root_pos_w[:, :2] - goal_w[:, :2], dim=-1)
    near_goal = xy_dist < SETTLE_NEAR_RADIUS

    return shortfall_ratio * _is_grasping(env).float() * (~near_goal).float()


def penalty_torso_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalizes the right arm's distal links (elbow through hand base) getting
    closer than TORSO_CLEARANCE_MIN_DIST to torso_link/head_link -- the arm-vs-
    own-body self-collision risk visually confirmed at the carry-to-inspect
    pose (screenshot showed the held cube/gripper pressed against torso_link's
    UNITREE-branded panel). Origin-distance based, not mesh-based -- see
    TORSO_CLEARANCE_MIN_DIST's comment for why.

    Gated on _is_grasping(): the reach/approach phase (gripper open, extending
    toward the table) legitimately brings the arm this close to the torso as
    part of normal reaching -- see TORSO_CLEARANCE_MIN_DIST's comment. Only
    penalize once genuinely holding the cube, when closeness to the torso is
    actually the carry-to-inspect risk this exists for."""
    robot: Articulation = env.scene["robot"]
    arm_ids, _ = robot.find_bodies(TORSO_CLEARANCE_LINKS, preserve_order=True)
    torso_ids, _ = robot.find_bodies(TORSO_BODY_LINKS, preserve_order=True)
    arm_pos = robot.data.body_pos_w[:, arm_ids, :]      # (N, A, 3)
    torso_pos = robot.data.body_pos_w[:, torso_ids, :]  # (N, T, 3)
    dist = torch.norm(arm_pos[:, :, None, :] - torso_pos[:, None, :, :], dim=-1)  # (N, A, T)
    penalty = torch.clamp(TORSO_CLEARANCE_MIN_DIST - dist, min=0.0)  # (N, A, T)
    grasping = _is_grasping(env).float()
    return penalty.sum(dim=(-1, -2)) * grasping


def penalty_action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )


def penalty_joint_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_joints(ARM_JOINTS)
    return torch.sum(torch.square(robot.data.joint_vel[:, ids]), dim=-1)