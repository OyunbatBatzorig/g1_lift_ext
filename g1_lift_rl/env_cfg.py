# g1_lift_rl/env_cfg.py
"""Phase 0/2 content only: the physical scene (robot, table, cube) and the
robot-hold reset event. ActionsCfg / ObservationsCfg / RewardsCfg / TerminationsCfg
and the top-level G1LiftEnvCfg are deliberately NOT defined yet -- that's the MDP
design (Phase 1), agreed on paper before it's written here.
"""
import math

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp
from .constants import (
    ROBOT_USD, ROBOT_POS, ROBOT_ROT, READY_ARM_POSE, LEFT_ARM_STOW,
    ARM_JOINTS, GRIPPER_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSE,
    TABLE_POS, TABLE_SIZE, BLOCK_SIZE, BLOCK_INIT_POS, CUBE_ROT, CUBE_YAW_RAD,
)

# ---------------------------------------------------------------------------
# Robot: G1 + Dex1, self-collision ON, finger-collision-filtered USD.
# Actuator gains verified in g1_redblock_ext: gripper at stiffness=800/damping=3
# reaches full closure (+0.0245) on the patched USD; adding friction=200 (the
# production config's value) made the gripper seize near the open position --
# confirmed empirically, so it's intentionally left out here.
# ---------------------------------------------------------------------------
_DEFAULT_JOINTS = {
    ".*_hip_.*": 0.0, ".*_knee_joint": 0.0, ".*_ankle_.*": 0.0,
    "waist_.*": 0.0,
    "left_hand_Joint.*": GRIPPER_OPEN, "right_hand_Joint.*": GRIPPER_OPEN,
    **READY_ARM_POSE,
    **LEFT_ARM_STOW,
}

G1_DEX1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
        ),
        # Caps contact-resolution speed if a finger interpenetrates the cube --
        # values match the proven production config (unitree_sim_isaaclab/robots/
        # unitree.py). Without this, an uncapped depenetration response can launch
        # the cube at extreme velocity (measured: ~9.6 m/s / ~560 rad/s) the moment
        # the hand gets close enough to actually contact it.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=ROBOT_POS, rot=ROBOT_ROT,
        joint_pos=_DEFAULT_JOINTS, joint_vel={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(joint_names_expr=[".*_joint"], stiffness=150.0, damping=10.0),
        # REVERTED 200/20 -> 800/3 (back to the original). Tried softening to 200/20
        # to reduce contact force against the lightweight cube, but the evidence
        # didn't support it: 800/3 ran stable across 4+ training attempts (up to 987
        # iterations, zero numerical crashes) despite measured ~8.9-9.6 m/s contact
        # events; 200/20 crashed or diverged numerically in BOTH of its 2 attempts
        # (explicit NaN at iter 131, value-loss divergence to inf by iter 601). It
        # didn't reduce the violence and introduced a new failure mode -- reverting
        # rather than continuing to chase this lever. The real fix for contact
        # violence is more likely in the APPROACH GEOMETRY (see arm_teleop.py), not
        # actuator gains.
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[".*_hand_Joint.*"], stiffness=800.0, damping=3.0
        ),
    },
)


# ---------------------------------------------------------------------------
# Scene: ground, light, robot, table, cube.
# ---------------------------------------------------------------------------
@configclass
class LiftSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0)),
    )
    robot: ArticulationCfg = G1_DEX1_CFG
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.55, 0.5)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS),
    )
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=False, max_depenetration_velocity=1.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.01, rest_offset=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max", restitution_combine_mode="min",
                static_friction=10.0, dynamic_friction=1.5, restitution=0.0),
        ),
        # NEW: rot=CUBE_ROT -- yaws the cube so a FACE, not a corner/edge, faces
        # the gripper's actual approach direction. See CUBE_ROT's comment in
        # constants.py for the measurement behind this.
        init_state=RigidObjectCfg.InitialStateCfg(pos=BLOCK_INIT_POS, rot=CUBE_ROT),
    )


def reset_object_cube_frame_jitter(
    env, env_ids: torch.Tensor,
    perp_range: tuple[float, float], along_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
):
    """Jitter the cube's spawn position in ITS OWN rotated frame (CUBE_YAW_RAD),
    not independent world x/y -- see EventCfg.reset_object's comment for why
    an independent-world-axis narrowing doesn't correspond to "tight
    perpendicular to the approach, loose along it" once the cube itself is
    rotated. perp_range is the axis perpendicular to the approach direction
    (should be tight -- shifts here move the contact point off the face
    center toward an edge); along_range is along the approach direction
    (can be loose -- shifts here just change engagement depth, not which
    face is presented)."""
    asset = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()

    n = len(env_ids)
    perp = torch.empty(n, device=env.device).uniform_(*perp_range)
    along = torch.empty(n, device=env.device).uniform_(*along_range)
    cos_t, sin_t = math.cos(CUBE_YAW_RAD), math.sin(CUBE_YAW_RAD)
    dx = cos_t * perp - sin_t * along
    dy = sin_t * perp + cos_t * along

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
    positions[:, 0] += dx
    positions[:, 1] += dy

    asset.write_root_pose_to_sim(torch.cat([positions, root_states[:, 3:7]], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)


# ---------------------------------------------------------------------------
# Events: robot-hold reset (verified) + cube reset with small x/y jitter
# (MDP_SPEC.md does not pin an exact range; using the same small jitter as the
# old project's precedent -- flag if you want this tightened/widened).
# ---------------------------------------------------------------------------
@configclass
class EventCfg:
    reset_robot = EventTermCfg(
        func=mdp.reset_robot_to_default, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # FIXED: narrowed from +-0.05 -- measured directly (check_policy1_final_vs_
    # pregrasp.py) that Policy 1's actual converged arm pose diverges hugely
    # from PRE_GRASP_ARM_POSE (up to 1.23 rad / ~70 deg on the elbow) AND has
    # high variance across envs (std 0.17-0.45 rad on several joints) -- Policy
    # 1 uses a wide range of different arm configurations depending on where
    # the cube randomly lands. Rather than redesign Policy 2's reset (already
    # trained and working well, don't want to retrain it), narrowing the range
    # Policy 1 has to generalize across should reduce that variance and give
    # it a better chance of consistently converging near PRE_GRASP_ARM_POSE,
    # the SAME reference Policy 2's reset already assumes. Center kept at
    # BLOCK_INIT_POS unchanged (verified: PRE_GRASP_ARM_POSE's own implied
    # cube position, EE - GRASP_OFFSET, is only 3.9cm from BLOCK_INIT_POS --
    # already a reasonable center, no need to shift it).
    # REVERTED: +-0.02 -> +-0.01 -> back to +-0.02 (isotropic). check_contact_
    # correlation.py measured jitter directly against finger-cube contact across
    # 64 envs and found NO positive correlation for an ISOTROPIC (x AND y
    # together) narrowing -- mean jitter was actually slightly LOWER in contact
    # envs (0.699cm) than no-contact envs (0.785cm). The real driver is EE-
    # position imprecision at the converged hover pose (contact envs: 4.28cm
    # mean deviation from the ideal grasp-offset target; no-contact envs:
    # 3.51cm) -- narrowing BOTH axes equally was the wrong lever.
    #
    # NEW: reset_object_cube_frame_jitter, replacing reset_root_state_uniform's
    # naive independent world x/y ranges. CAUGHT BEFORE SHIPPING: an initial
    # version narrowed world-x alone (+-0.02->+-0.01) reasoning "x is the axis
    # perpendicular to the approach" -- WRONG, since world x/y no longer line
    # up with the cube's geometry now that CUBE_ROT rotates it -34.76 deg. The
    # axis that actually threatens face/corner alignment (perpendicular to the
    # approach direction) is the CUBE's rotated local-x, which decomposes into
    # world (82.2% x, 57.0% y) -- a correlated mix, not a pure x-only
    # narrowing. Jitters in the cube's own rotated frame (tight perpendicular
    # to approach, loose along it -- moving along the approach axis doesn't
    # change which face is presented, only how deep) and transforms to world
    # coordinates, instead of approximating with independent world ranges.
    reset_object = EventTermCfg(
        func=reset_object_cube_frame_jitter, mode="reset",
        params={
            "perp_range": (-0.01, 0.01),  # tight: perpendicular to approach (was world-x's role)
            "along_range": (-0.02, 0.02),  # loose: along the approach direction
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


# ---------------------------------------------------------------------------
# Actions: MDP_SPEC.md section 1. 7 right-arm joint targets + 1 binary gripper
# action (BinaryJointPositionActionCfg's raw action dim is 1, not 2 -- it drives
# both gripper joints from a single scalar). Total action dim = 8.
# ---------------------------------------------------------------------------
@configclass
class ActionsCfg:
    arm = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINTS, scale=0.5, use_default_offset=True)
    gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=GRIPPER_JOINTS,
        open_command_expr={j: GRIPPER_OPEN for j in GRIPPER_JOINTS},
        close_command_expr={j: GRIPPER_CLOSE for j in GRIPPER_JOINTS},
    )


# ---------------------------------------------------------------------------
# Observations: MDP_SPEC.md section 2, 7+7+2+3+3+3+3+8=36, PLUS hand_base_to_object
# (3) = 39. Added after visually confirming the mounting plate (HAND_BASE_LINK)
# strikes the cube's top surface during descent -- previously untracked anywhere,
# so the policy had no way to see this risk, only get penalized after a violent
# contact already happened (penalty_contact_disturbance, velocity-only).
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_joint_pos_rel = ObsTerm(func=mdp.arm_joint_pos_rel)
        arm_joint_vel = ObsTerm(func=mdp.arm_joint_vel)
        gripper_joint_pos = ObsTerm(func=mdp.gripper_joint_pos)
        object_position = ObsTerm(func=mdp.object_position)
        ee_position = ObsTerm(func=mdp.ee_position)
        ee_to_object = ObsTerm(func=mdp.ee_to_object)
        object_to_inspect = ObsTerm(func=mdp.object_to_inspect)
        hand_base_to_object = ObsTerm(func=mdp.hand_base_to_object)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Rewards -- POLICY 1 ONLY: reach to the verified pre-grasp point and hold there,
# gripper OPEN, cube undisturbed. This is now one of THREE single-purpose
# policies (each its own network, chained by a scripted handoff, no learned
# high-level controller):
#   Policy 1 (THIS env): reach + position at pre_grasp        -- gripper open
#   Policy 2 (future):   grasp (verified) + carry to inspection -- gripper closes
#   Policy 3 (future):   place at target + release             -- gripper opens
# grasp/lift/inspect/inspect_bonus/close_gradient are Policy 2's job, not
# Policy 1's -- removed from THIS RewardsCfg, but their functions stay defined
# and exported in mdp/rewards.py / mdp/__init__.py for Policy 2 to use as-is.
# early_close and contact_disturbance are KEPT: Policy 1 must still arrive with
# the gripper open (not pre-empt Policy 2 by closing early) and must not
# knock/disturb the cube on the way in -- Policy 2 needs an undisturbed cube to
# grasp. No success termination -- Policy 1 must REACH AND STAY, so the episode
# has to run long enough to reward holding position, not just passing through it.
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    hover         = RewTerm(func=mdp.reward_hover,         weight=1.0)
    descend       = RewTerm(func=mdp.reward_descend,       weight=1.5)
    # FIXED: reward_straddle_orientation -> reward_match_pregrasp_pose. The
    # straddle version fixed the palm-down tilt (worked well, straddle
    # reached 0.94/1.0) but only targeted "some horizontal straddle" -- the
    # resulting orientation still didn't match PRE_GRASP_ARM_POSE (visually
    # "upside down" relative to Policy 2's actual reset). For real robot
    # deployment (control switches from Policy 1 to Policy 2 using whatever
    # state Policy 1 actually left the arm in, not a synthetic reset), Policy
    # 2's observations include raw joint angles, so EE-position matching
    # alone isn't a strong enough guarantee -- see
    # reward_match_pregrasp_pose's docstring. Pulls joints directly toward
    # PRE_GRASP_ARM_POSE, which subsumes the straddle goal (that pose already
    # IS a genuine, verified grasp orientation).
    #
    # FIXED: 1.0 -> 2.5. Full-run TensorBoard trace (2026-07-13_13-37-13, K_JOINT
    # already fixed for that whole run) showed match_pregrasp PEAKING early
    # (0.253 @ iter 375) then DECLINING as descend climbed (0.088 @ iter 750),
    # then plateauing flat at 0.08-0.12 for the entire back half of training --
    # not an undertrained/still-improving curve, a real competing-objective
    # signature. Root cause: reward_descend only scores EE *position* (no
    # orientation term), so for this 7-DOF arm it has zero preference among the
    # infinitely many joint configs reaching the same EE point -- whatever
    # config PPO finds easiest while chasing descend's larger, redundancy-
    # tolerant reward (weight 1.5) apparently drives right_wrist_roll_joint to
    # its hard limit, and match_pregrasp at 1.0 wasn't strong enough to pull it
    # back. 2.5 makes match_pregrasp outweigh descend, testing whether this is
    # purely a weight-balance problem before trying anything more invasive
    # (re-deriving PRE_GRASP_ARM_POSE itself, or a weight curriculum).
    #
    # CONFIRMED: 2.5 worked -- direct per-joint replay (not just the reward
    # curve) on the resulting checkpoint (2026-07-16_09-18-57) showed
    # right_wrist_roll_joint at +0.12 rad, nowhere near its former hard-limit
    # saturation point (-1.972). Total joint-space distance from
    # PRE_GRASP_ARM_POSE dropped from ~2.67 rad to ~0.53 rad.
    #
    # TRIED AND REVERTED: 2.5 -> 3.3. The noisy training curve suggested this
    # helped (raw match_pregrasp 0.545->0.607), but a deterministic per-joint
    # replay (not the stochastic training average) showed it actually
    # regressed on every axis that matters: total joint-space distance from
    # PRE_GRASP_ARM_POSE got WORSE (0.53->0.65 rad), EE-position deviation
    # from the ideal grasp geometry got worse (3.0cm->5.5cm), and per-env
    # consistency collapsed (e.g. wrist_roll std went 0.027->0.138 rad --
    # envs stopped converging to one stable pose). User visually confirmed the
    # 3.3 checkpoint's gripper wasn't grasp-ready. Back to 2.5, the measurably
    # better value on every axis checked.
    match_pregrasp = RewTerm(func=mdp.reward_match_pregrasp_pose, weight=2.5)
    early_close   = RewTerm(func=mdp.penalty_early_close,  weight=-0.5)
    contact_disturbance = RewTerm(func=mdp.penalty_contact_disturbance, weight=-0.1)
    # base_clearance alone at -3.0 is the CONFIRMED-good configuration (the
    # original successful run: hover->0.6, descend engaged ~iter 280,
    # base_clearance itself settled at 0.0000; re-confirmed again in the
    # isolation test run 2026-07-07_14-48-43).
    base_clearance = RewTerm(func=mdp.penalty_base_clearance, weight=-3.0)
    # finger_clearance is DELIBERATELY NOT included here (removed after two
    # failed attempts to make it coexist with reward_descend -- see
    # mdp/rewards.py:penalty_finger_clearance's docstring for the full history).
    # table_clearance is ALSO NOT included, despite having no structural
    # conflict with the task the way finger_clearance did -- confirmed
    # empirically it broke training anyway when combined with base_clearance
    # (action_std collapsed to ~0.05 by iter ~250, descend never fired, same
    # symptom as before). Working theory: the table's danger zone is the whole
    # table footprint (60x50cm) vs. the cube's tiny 3cm radius, so early random
    # exploration (action_std starts at 0.60) sweeps into it far more often,
    # suppressing exploration before hover/descend are ever discovered --
    # table_clearance itself peaked at iter ~30, exactly when action_std was
    # still high. Not yet re-tested in isolation (table_clearance alone,
    # without base_clearance) to confirm this theory. penalty_contact_disturbance
    # (below) is a reactive, outcome-based signal instead for both fingers and
    # table -- it can't structurally conflict with the task, since it only
    # fires when the cube actually moves, not merely when the hand is nearby.
    action_rate   = RewTerm(func=mdp.penalty_action_rate,  weight=-0.01)
    joint_vel     = RewTerm(func=mdp.penalty_joint_vel,    weight=-1.0e-4)


# ---------------------------------------------------------------------------
# Terminations: MDP_SPEC.md section 4. No success termination -- Policy 1 must
# HOLD, so the episode has to run long enough to reward holding.
# ---------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    dropped = DoneTerm(func=mdp.object_dropped, time_out=False)
    launched = DoneTerm(func=mdp.object_launched, time_out=False)


@configclass
class G1LiftEnvCfg(ManagerBasedRLEnvCfg):
    scene = LiftSceneCfg(num_envs=2048, env_spacing=2.5, replicate_physics=True)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.003
        self.sim.physx.enable_ccd = True
        self.sim.physx.num_position_iterations = 12
        self.sim.physx.num_velocity_iterations = 4


@configclass
class G1LiftEnvCfg_PLAY(G1LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
