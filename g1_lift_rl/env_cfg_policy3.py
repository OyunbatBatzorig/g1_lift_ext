# g1_lift_rl/env_cfg_policy3.py
"""POLICY 3: move to goal + place. (release + return-to-ready split off into
Policy 4, see env_cfg_policy4.py -- see RewardsCfg's docstring for why.)

Single purpose: starting already holding the cube (arm at INSPECT_ARM_POSE,
gripper CLOSED, cube pre-positioned at INSPECT_POS -- a self-consistent pair,
both measured in the SAME teleop recording), carry it to GOAL_POS and set it
down there, still gripped -- Policy 4 takes over from there. Reuses Policy 1/2's
robot definition (G1_DEX1_CFG) unchanged; the SCENE differs only in the cube's
default position (INSPECT_POS instead of BLOCK_INIT_POS on the table), since
Policy 3 assumes a successful Policy-2 hand-off rather than reaching from
scratch.

Trained independently of Policy 1/2: the reset event below recreates their
expected hand-off state directly, so they do not need to exist or run first.
"""
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
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
    ARM_JOINTS, GRIPPER_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSE, INSPECT_ARM_POSE,
    INSPECT_GRIP_VALUE, INSPECT_POS, GOAL_POS, TABLE_TOP_Z, EE_LINKS, GRASP_OFFSET,
)


@configclass
class Policy3SceneCfg(LiftSceneCfg):
    """Same scene as Policy 1/2, except the cube's default position is
    INSPECT_POS (already-held point) instead of BLOCK_INIT_POS (table spawn) --
    Policy 3 starts from a hand-off state, not from scratch."""

    # Visual-only marker at GOAL_POS's xy -- same pattern as g1_redblock_ext's
    # env_cfg.py:goal (flat yellow disc, kinematic, massless, no collision, so
    # it can't be pushed or interacted with, purely a reference for where
    # reward_place/reward_release are actually checking). GOAL_POS here is a
    # fixed constant (unlike g1_redblock_ext's per-episode randomize_goal
    # event), so no reset event is needed to reposition it.
    #
    # FIXED: GOAL_POS's z (0.843) is the CUBE'S CENTER height when resting at
    # the goal (TABLE_TOP_Z + BLOCK_SIZE/2), which is what reward_place/
    # reward_release correctly compare against -- but placing the marker disc
    # at that same z put it floating ~3.3cm above the table surface (visually
    # confirmed). The marker itself uses TABLE_TOP_Z directly instead, so it
    # sits flush on the table like g1_redblock_ext's does; GOAL_POS itself is
    # untouched since the reward functions depend on its cube-center value.
    goal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        spawn=sim_utils.CylinderCfg(
            radius=0.05, height=0.002,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(GOAL_POS[0], GOAL_POS[1], TABLE_TOP_Z)),
    )

    def __post_init__(self):
        self.object.init_state.pos = INSPECT_POS


def reset_robot_then_couple_cube(
    env, env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """FIXED: replaces the old reset_robot_to_inspection + independently-jittered
    reset_object pair -- exact same bug class Policy 2 already hit and fixed (see
    env_cfg_policy2.py:reset_robot_then_couple_cube's docstring). Measured
    directly (check_policy3_reset_dist.py) after INSPECT_POS/INSPECT_ARM_POSE's
    2nd update: with the cube independently jittered around a fixed INSPECT_POS
    while the gripper was separately forced to the hardcoded GRIPPER_CLOSE, only
    5/16 envs landed inside GRASP_DIST at reset (mean dist 0.0445m, up to
    0.064m) -- worse than intended, because GRIPPER_CLOSE (+0.0245) is a
    physically-unreachable full closure whenever a real cube is actually in the
    hand (a genuine grip mechanically blocks at ~-0.007 to -0.017, see
    GRIP_CLOSED_THRESHOLD in rewards.py) -- so driving the gripper straight to
    GRIPPER_CLOSE moves the fingertip midpoint (EE_LINKS) to a different
    configuration than the teleop recording (a real, contact-blocked grip) that
    GRASP_OFFSET was measured against, independent of the cube-jitter question
    entirely.

    Fix: reset the arm to INSPECT_ARM_POSE (gripper closed), refresh kinematics,
    THEN place the cube at the ACTUAL resulting EE position minus GRASP_OFFSET --
    deterministically self-consistent by construction, same as Policy 2's grasp
    reset, regardless of exactly how the gripper's own joint value compares to a
    real physical grip.

    UPDATE: gripper now set to INSPECT_GRIP_VALUE (Policy 2's actual measured
    converged grip, ~-0.0144), not the hardcoded GRIPPER_CLOSE -- geometrically
    the cube's position never depended on this (GRASP_OFFSET-coupled either
    way), but Policy 3's observations include gripper_joint_pos, so this keeps
    the reset distribution-consistent with what Policy 2 really hands off."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    default_pos = robot.data.default_joint_pos[env_ids].clone()
    arm_ids = [robot.find_joints([n])[0][0] for n in ARM_JOINTS]
    for i, jname in enumerate(ARM_JOINTS):
        default_pos[:, arm_ids[i]] = INSPECT_ARM_POSE[jname]
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINTS)
    for jid in gripper_ids:
        default_pos[:, jid] = INSPECT_GRIP_VALUE
    default_vel = robot.data.default_joint_vel[env_ids]
    robot.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)

    # FIXED: the POSITION TARGET must be GRIPPER_CLOSE, not INSPECT_GRIP_VALUE.
    # INSPECT_GRIP_VALUE is the RESULTING joint position once mechanically
    # blocked by the cube -- it only holds because Policy 2's actual action
    # continuously commands full closure (BinaryJointPositionActionCfg's
    # "close" -> GRIPPER_CLOSE) every step; the ongoing push against the
    # blocked position is what generates real grip force. Setting the target
    # to the already-blocked value made the controller see "already at
    # target" -> zero corrective force -> nothing counteracting gravity.
    # Confirmed empirically (check_policy3_cube_falls.py): with target ==
    # INSPECT_GRIP_VALUE, the cube fell from 0.170m to 0.039m above the table
    # (back to resting height) within 20 steps and stayed there -- not a
    # render artifact, the cube was genuinely dropping. Initial STATE stays at
    # INSPECT_GRIP_VALUE (matches the geometry GRASP_OFFSET was measured
    # against); only the commanded TARGET changes.
    target_pos = default_pos.clone()
    for jid in gripper_ids:
        target_pos[:, jid] = GRIPPER_CLOSE
    robot.set_joint_position_target(target_pos, env_ids=env_ids)
    robot.write_data_to_sim()

    # Refresh kinematics (no physics step) so body_pos_w reflects the joint
    # state just written, not stale data -- same pattern as Policy 2.
    env.sim.forward()
    env.scene.update(dt=0.0)

    ee_ids, _ = robot.find_bodies(EE_LINKS)
    ee_pos_w = robot.data.body_pos_w[env_ids][:, ee_ids, :].mean(dim=1)
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
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Rewards -- POLICY 3 ONLY: move to goal + place. No grasp/lift/inspect (already
# assumed held at reset -- Policy 2's job, not this one's). No release either
# anymore -- release + return-to-ready moved to Policy 4 (see env_cfg_policy4.py),
# after a matched-checkpoint comparison showed clinging-at-goal was a stable,
# already-rewarded local optimum (settle+place alone pay 3.5 with the cube just
# held still) that release's sparse, narrow-target bonus could never reliably
# outcompete -- across two independent training runs (plain PPO and a self-
# imitation-learning variant), release only ever fired in 1-3 of 16-64 envs, and
# a plain-PPO resume erased it entirely (0.0000 for the whole run). Splitting the
# reliable skill (get the cube to the goal and set it down) from the fragile one
# (then let go) removes the reward conflict at its root: Policy 3 now has no
# reason to ever open the gripper, so there's nothing left to compete with
# settle/place, and no rare-event consolidation problem to solve here at all.
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    move_to_goal = RewTerm(func=mdp.reward_move_to_goal, weight=3.0)
    # Dense bridge between move_to_goal (rewards proximity) and place (sparse,
    # all-or-nothing "settled" bonus) -- added after a real training run showed
    # place stuck at ~0.000 for 873/1500 iterations despite move_to_goal
    # climbing steadily. Diagnosis: no reward gradient anywhere encouraged the
    # policy to decelerate/stop once near the goal, only to get close. See
    # reward_settle_near_goal's docstring. Weight below move_to_goal's (this is
    # a shaping aid, not the main task signal) but well above the smaller
    # penalty terms.
    settle       = RewTerm(func=mdp.reward_settle_near_goal, weight=1.5)
    place        = RewTerm(func=mdp.reward_place,        weight=2.0)
    # base_clearance/finger_clearance are deliberately NOT included here (unlike
    # Policy 1/2): both protect an UNGRASPED cube from an open, approaching hand,
    # and Policy 3 starts already holding the cube -- the gripper only opens again
    # at release, by which point the cube is already resting at GOAL_POS, not
    # something the hand could still knock off approach. table_clearance IS
    # relevant here though: the arm is still working close to the table surface
    # the whole time (descending to place the cube at GOAL_POS), same risk as
    # Policy 1/2's descend phase, just for a different reason (placing, not
    # reaching) -- see penalty_table_clearance's docstring.
    #
    # FIXED: unconditional penalty_table_clearance -> penalty_table_clearance_
    # near_goal_excluded. Measured directly (check_policy3_table_contact_at_
    # goal.py) that 16/16 envs show real table contact exactly when placing
    # the cube near the goal -- unconditionally penalizing that fights the
    # actual task (setting an object down requires the hand to be near the
    # table). See penalty_table_clearance_near_goal_excluded's docstring.
    table_clearance = RewTerm(func=mdp.penalty_table_clearance_near_goal_excluded, weight=-3.0)
    # NEW: nothing above rewards carry HEIGHT -- move_to_goal only scores xyz
    # distance to GOAL_POS, so skimming the cube an inch above the table the
    # whole way scores the same as carrying it safely. Measured directly
    # (policy3_cube_height_timeline.py) that the cube collapses from its
    # ~0.134m reset height down to ~0.012-0.018m within ~10 steps and stays
    # there -- visually read as dragging.
    #
    # FIXED: weight was -3.0, copying table_clearance's number -- but
    # penalty_low_carry is now normalized to [0,1] (matching move_to_goal/
    # settle/place/reward_lift's convention, not table_clearance's raw-meters
    # one, see penalty_low_carry's docstring for why that copy was wrong).
    # -2.0 puts it in the same ballpark as place (2.0) -- a real, felt
    # correction, but a shaping term riding alongside move_to_goal (3.0), not
    # sized to outright compete with it.
    low_carry    = RewTerm(func=mdp.penalty_low_carry, weight=-2.0)
    contact_disturbance = RewTerm(func=mdp.penalty_contact_disturbance, weight=-0.1)
    action_rate  = RewTerm(func=mdp.penalty_action_rate,  weight=-0.01)
    joint_vel    = RewTerm(func=mdp.penalty_joint_vel,    weight=-1.0e-4)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    dropped = DoneTerm(func=mdp.object_dropped, time_out=False)
    launched = DoneTerm(func=mdp.object_launched, time_out=False)


@configclass
class G1Policy3EnvCfg(ManagerBasedRLEnvCfg):
    # num_envs reduced from Policy 1's 2048: three policies training concurrently
    # share one GPU, so this keeps total load more balanced rather than 3x2048.
    scene = Policy3SceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=True)
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
class G1Policy3EnvCfg_PLAY(G1Policy3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
