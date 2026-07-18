# g1_lift_rl/env_cfg_policy2.py
"""POLICY 2: grasp (verified) + carry to inspection.

Single purpose: starting near where Policy 1 is trained to hand off (arm at
PRE_GRASP_ARM_POSE +- small noise, gripper OPEN), close the gripper on the
cube, lift it, and carry it to INSPECT_POS. Reuses Policy 1's robot/scene
definition unchanged (G1_DEX1_CFG, LiftSceneCfg) -- only the reset pose and the
action/observation/reward/termination scope differ, since this is otherwise
the exact same physical setup.

Trained independently of Policy 1: the reset event below approximates Policy 1's
expected hand-off state directly, so Policy 1 does not need to exist or run first.
"""
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass

from . import mdp
from .env_cfg import LiftSceneCfg
from .constants import (
    ARM_JOINTS, GRIPPER_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSE, PRE_GRASP_ARM_POSE,
    EE_LINKS, GRASP_OFFSET,
)

# Small per-joint noise around the verified pre-grasp pose: Policy 1 won't land at
# the EXACT same arm configuration every episode, so a single fixed pose would
# only approximate the real hand-off state. If Policy 2 needs a bit of fine
# positioning to correct for the residual mismatch, that is a reasonable, small
# part of its own job (reward_descend, kept below at a smaller weight than
# Policy 1's, covers exactly this).
_ARM_POSE_NOISE = 0.03  # rad, per joint


def reset_robot_then_couple_cube(
    env, env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """FIXED (replaces separate reset_robot + independently-jittered
    reset_object): resets the right arm to PRE_GRASP_ARM_POSE (+- noise), THEN
    places the cube at the resulting ACTUAL EE position minus the verified
    GRASP_OFFSET -- i.e. the cube always starts in the same correct physical
    relationship to wherever the (noisy) arm actually ended up, instead of
    being placed at a fixed absolute point and independently jittered on top.

    This eliminates the root cause of a whole session's worth of reset-overlap
    problems: with two INDEPENDENT sources of randomness (fixed arm pose +-
    noise, cube +-0.05m jitter), the two would sometimes land in a bad
    relationship to each other purely by chance -- verified empirically to
    cause depenetration launches up to 232 m/s in the worst observed case, no
    matter how carefully the nominal arm pose was solved or how much margin
    was added. Coupling the cube's position to the arm's ACTUAL post-reset FK
    removes that failure mode structurally: the two are correlated by
    construction, not independently sampled, so there's no longer a
    "mismatch" scenario to guard against. Policy 2 keeps its cube-position
    observations and reward_descend/close_gradient -- it still has to learn to
    handle whatever real imprecision Policy 1 leaves it with (arm noise alone
    still varies the EE position meaningfully through the kinematic chain),
    it just no longer also has to survive an uncorrelated cube placement.

    Order matters: the arm reset must happen and its forward kinematics must
    be refreshed BEFORE the cube position is computed, so this is a single
    combined event (not two separate EventTermCfg entries) to make that
    ordering explicit and unambiguous rather than relying on manager event
    ordering."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    # 1. Reset the arm (same as before).
    default_pos = robot.data.default_joint_pos[env_ids].clone()
    arm_ids = [robot.find_joints([n])[0][0] for n in ARM_JOINTS]
    noise = (torch.rand(len(env_ids), len(arm_ids), device=env.device) - 0.5) * 2 * _ARM_POSE_NOISE
    for i, jname in enumerate(ARM_JOINTS):
        default_pos[:, arm_ids[i]] = PRE_GRASP_ARM_POSE[jname] + noise[:, i]
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINTS)
    for jid in gripper_ids:
        default_pos[:, jid] = GRIPPER_OPEN
    default_vel = robot.data.default_joint_vel[env_ids]
    robot.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)
    robot.set_joint_position_target(default_pos, env_ids=env_ids)
    robot.write_data_to_sim()

    # 2. Refresh kinematics (no physics step -- pure forward-kinematics
    # readback, same pattern used throughout this session's diagnostics) so
    # body_pos_w reflects the joint state just written, not stale data.
    env.sim.forward()
    env.scene.update(dt=0.0)

    # 3. Place the cube at the ACTUAL resulting EE position minus the
    # verified grasp offset (ee = cube + GRASP_OFFSET -> cube = ee - GRASP_OFFSET).
    ee_ids, _ = robot.find_bodies(EE_LINKS)
    ee_pos_w = robot.data.body_pos_w[env_ids][:, ee_ids, :].mean(dim=1)  # (len(env_ids), 3)
    grasp_offset = torch.tensor(GRASP_OFFSET, device=env.device)
    cube_pos_w = ee_pos_w - grasp_offset

    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, :3] = cube_pos_w
    obj.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    obj.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)


@configclass
class EventCfg:
    reset_robot_and_cube = EventTermCfg(
        func=reset_robot_then_couple_cube, mode="reset",
        params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("object")},
    )


@configclass
class ActionsCfg:
    arm = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINTS, scale=0.5, use_default_offset=True)
    gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=GRIPPER_JOINTS,
        open_command_expr={j: GRIPPER_OPEN for j in GRIPPER_JOINTS},
        close_command_expr={j: GRIPPER_CLOSE for j in GRIPPER_JOINTS},
    )


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
        # Same open-hand descend/grasp motion as Policy 1, reusing the identical
        # reward functions below -- the mounting-plate-strikes-cube risk isn't
        # specific to Policy 1's task, it's specific to this physical motion, and
        # Policy 2 still has the gripper open until it actually grasps.
        hand_base_to_object = ObsTerm(func=mdp.hand_base_to_object)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Rewards -- POLICY 2 ONLY: grasp + carry to inspection. No hover (starts already
# roughly positioned); descend kept at a SMALLER weight than Policy 1's purely for
# fine positioning/correcting the residual hand-off mismatch, not for reaching
# from scratch. grasp/lift/inspect/inspect_bonus/close_gradient/early_close/
# contact_disturbance are the SAME functions Policy 1 leaves defined but unused --
# reused as-is here, this is exactly their intended home.
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    descend       = RewTerm(func=mdp.reward_descend,         weight=1.0)
    close_gradient = RewTerm(func=mdp.reward_close_gradient, weight=1.0)
    grasp         = RewTerm(func=mdp.reward_grasp,           weight=2.0)
    lift          = RewTerm(func=mdp.reward_lift,            weight=4.0)
    inspect       = RewTerm(func=mdp.reward_inspect,         weight=2.0)
    inspect_bonus = RewTerm(func=mdp.reward_inspect_bonus,   weight=4.0)
    early_close   = RewTerm(func=mdp.penalty_early_close,    weight=-0.5)
    contact_disturbance = RewTerm(func=mdp.penalty_contact_disturbance, weight=-0.1)
    # Matches Policy 1 (see env_cfg.py for the full history): finger_clearance
    # removed entirely -- structurally conflicts with reward_descend/
    # close_gradient here for the same reason (fingers must legitimately get
    # close to the cube to grasp; the verified approach sits inside its own
    # danger zone by construction). table_clearance ALSO removed -- confirmed on
    # Policy 1 that base_clearance+table_clearance together broke training
    # (action_std collapse) even though table_clearance has no structural
    # conflict on its own; working theory is an early-exploration frequency
    # effect (table's danger zone is the whole table footprint, far more likely
    # to be swept by random early exploration than the cube's tiny 3cm radius),
    # which isn't specific to Policy 1's smaller reward budget -- not re-tested
    # on Policy 2 directly, but not worth risking the same failure here either.
    base_clearance = RewTerm(func=mdp.penalty_base_clearance, weight=-3.0)
    # NEW (2nd weight): arm-vs-own-body (torso/head) self-collision penalty --
    # added after a trained rollout screenshot showed the held cube/gripper
    # pressed against torso_link at the carry-to-inspect pose. Gated on
    # _is_grasping() so it cannot conflict with the reach phase (see
    # penalty_torso_clearance's and TORSO_CLEARANCE_MIN_DIST's comments in
    # rewards.py for why that gate is required, not optional, for this
    # specific penalty).
    #
    # FIXED: -1.0 (the first, conservative weight) trained cleanly (no
    # action_std collapse, dropped/launched stayed low) but wasn't STRONG
    # enough -- measured directly (check_torso_dist_per_body.py) that the
    # trained policy settled at right_wrist_roll_link<->torso_link ~0.150-
    # 0.163m across all 16 envs, consistently under TORSO_CLEARANCE_MIN_DIST
    # (0.19m), i.e. it ate a small ongoing penalty (~-0.0229 episode-mean)
    # rather than avoid it. This is NOT a structural conflict like
    # finger_clearance's original bug -- the teleop-verified reference pose
    # reaches the same INSPECT_POS region with wrist_pitch<->torso at 0.215m,
    # comfortably clear, proving a better solution exists -- so raising the
    # weight to match base_clearance's scale should push training toward it
    # instead of accepting the cheaper, closer local optimum.
    torso_clearance = RewTerm(func=mdp.penalty_torso_clearance, weight=-3.0)
    action_rate   = RewTerm(func=mdp.penalty_action_rate,    weight=-0.01)
    joint_vel     = RewTerm(func=mdp.penalty_joint_vel,      weight=-1.0e-4)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    dropped = DoneTerm(func=mdp.object_dropped, time_out=False)
    launched = DoneTerm(func=mdp.object_launched, time_out=False)


@configclass
class G1Policy2EnvCfg(ManagerBasedRLEnvCfg):
    # num_envs reduced from Policy 1's 2048: three policies training concurrently
    # share one GPU, so this keeps total load more balanced rather than 3x2048.
    scene = LiftSceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=True)
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
class G1Policy2EnvCfg_PLAY(G1Policy2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
