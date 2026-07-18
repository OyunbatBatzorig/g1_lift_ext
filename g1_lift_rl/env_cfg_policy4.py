# g1_lift_rl/env_cfg_policy4.py
"""POLICY 4: release + return to ready.

Single purpose: starting already at the goal, holding the cube (arm at
GOAL_ARM_POSE, gripper closed at GOAL_GRIP_VALUE, cube coupled via GRASP_OFFSET
-- Policy 3's actual measured handoff state, same discipline as Policy 2->3's
INSPECT_ARM_POSE), open the gripper to release the cube, then pull the arm back
toward READY_ARM_POSE (Policy 1's own reset target, closing the loop for another
pick-and-place cycle).

Split off from what used to be Policy 3's release+return_to_ready terms: a
matched-checkpoint comparison (plain PPO vs. a self-imitation-learning variant)
showed release could never reliably outcompete clinging when settle+place paid
out 3.5 just for holding the cube still at the goal -- see env_cfg_policy3.py's
RewardsCfg docstring for the full diagnosis. Policy 4 has no move_to_goal/
settle/place reward at all, so clinging pays literally nothing here: releasing
is the only path to reward, which should make it trivial to learn instead of a
rare event to rediscover.

Trained independently of Policy 1/2/3: the reset event below recreates Policy
3's expected end state directly, so they do not need to exist or run first.
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
    ARM_JOINTS, GRIPPER_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSE, GOAL_ARM_POSE,
    GOAL_GRIP_VALUE, GOAL_POS, TABLE_TOP_Z, EE_LINKS, GRASP_OFFSET,
)


@configclass
class Policy4SceneCfg(LiftSceneCfg):
    """Same scene as Policy 3, except the cube's default position is GOAL_POS
    (already-placed point) instead of INSPECT_POS -- Policy 4 starts from
    Policy 3's hand-off state, not from a fresh carry."""

    # Same visual-only marker pattern as Policy 3's scene -- kept here too since
    # reward_release still checks distance to GOAL_POS (the cube shouldn't drift
    # away from it before/during release).
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
        self.object.init_state.pos = GOAL_POS


def reset_robot_then_couple_cube_at_goal(
    env, env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Same pattern as Policy 3's reset_robot_then_couple_cube (see its
    docstring for the full reasoning: arm reset to a measured pose, kinematics
    refreshed, THEN the cube placed at the ACTUAL resulting EE position minus
    GRASP_OFFSET -- deterministically self-consistent by construction), just
    keyed to GOAL_ARM_POSE/GOAL_GRIP_VALUE (Policy 3's measured end state)
    instead of INSPECT_ARM_POSE/INSPECT_GRIP_VALUE (Policy 2's).

    Gripper TARGET is commanded to GRIPPER_CLOSE (not GOAL_GRIP_VALUE) for the
    same reason as Policy 3's reset: GOAL_GRIP_VALUE is the RESULTING position
    once mechanically blocked by the cube, and only holds under continuous
    closure pressure -- setting the target to the already-blocked value would
    read as "already at target" and stop generating grip force, dropping the
    cube before the episode even starts."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]

    default_pos = robot.data.default_joint_pos[env_ids].clone()
    arm_ids = [robot.find_joints([n])[0][0] for n in ARM_JOINTS]
    for i, jname in enumerate(ARM_JOINTS):
        default_pos[:, arm_ids[i]] = GOAL_ARM_POSE[jname]
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINTS)
    for jid in gripper_ids:
        default_pos[:, jid] = GOAL_GRIP_VALUE
    default_vel = robot.data.default_joint_vel[env_ids]
    robot.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)

    target_pos = default_pos.clone()
    for jid in gripper_ids:
        target_pos[:, jid] = GRIPPER_CLOSE
    robot.set_joint_position_target(target_pos, env_ids=env_ids)
    robot.write_data_to_sim()

    # Refresh kinematics (no physics step) so body_pos_w reflects the joint
    # state just written, not stale data -- same pattern as Policy 2/3.
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
        func=reset_robot_then_couple_cube_at_goal, mode="reset",
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
# Rewards -- POLICY 4 ONLY: release + return to ready. No move_to_goal/settle/
# place -- the cube is already at the goal by construction (reset), and unlike
# Policy 3, nothing here rewards keeping it there or holding it. Clinging pays
# exactly zero in this reward set, so release is the only path to any reward at
# all -- see the module docstring above for why this is the point of the split.
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    release = RewTerm(func=mdp.reward_release, weight=6.0)
    return_to_ready = RewTerm(func=mdp.reward_return_to_ready, weight=1.0)
    # Same reasoning as Policy 3: the arm is still working close to the table
    # surface right at release, same near-goal contact exemption applies.
    table_clearance = RewTerm(func=mdp.penalty_table_clearance_near_goal_excluded, weight=-3.0)
    contact_disturbance = RewTerm(func=mdp.penalty_contact_disturbance, weight=-0.1)
    action_rate  = RewTerm(func=mdp.penalty_action_rate,  weight=-0.01)
    joint_vel    = RewTerm(func=mdp.penalty_joint_vel,    weight=-1.0e-4)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    dropped = DoneTerm(func=mdp.object_dropped, time_out=False)
    launched = DoneTerm(func=mdp.object_launched, time_out=False)


@configclass
class G1Policy4EnvCfg(ManagerBasedRLEnvCfg):
    # Same num_envs reasoning as Policy 3 -- multiple policies training
    # concurrently share one GPU.
    scene = Policy4SceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=True)
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
class G1Policy4EnvCfg_PLAY(G1Policy4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
