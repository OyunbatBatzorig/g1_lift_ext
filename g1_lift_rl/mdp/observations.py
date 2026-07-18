# g1_lift_rl/mdp/observations.py
"""Observation terms per MDP_SPEC.md section 2. Total dim = 36 (verified by
py_compile + a dimension sum check, not by running the env -- that's the
zero_agent run)."""
from __future__ import annotations
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply
from ..constants import ARM_JOINTS, GRIPPER_JOINTS, EE_LINKS, INSPECT_POS, HAND_BASE_LINK


def _ee_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """World-frame grasp centre = midpoint of the two fingertip links. (N, 3)"""
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_bodies(EE_LINKS)
    return robot.data.body_pos_w[:, ids, :].mean(dim=1)


# Base link mesh bounding box in its OWN local frame (probe_base_mesh_extent.py).
# The origin point alone undercounts the plate's real reach by ~9cm -- see
# mdp/rewards.py:penalty_base_clearance for the full story (verified empirically
# via check_hand_geometry_v2.py that the plate's actual edge, not its origin,
# grazes the cube during real disturbance events).
_HAND_BASE_BBOX_MIN = (-0.035, 0.000, -0.035)
_HAND_BASE_BBOX_MAX = (0.035, 0.0738, 0.089)
_HAND_BASE_CORNERS_LOCAL = [
    (x, y, z)
    for x in (_HAND_BASE_BBOX_MIN[0], _HAND_BASE_BBOX_MAX[0])
    for y in (_HAND_BASE_BBOX_MIN[1], _HAND_BASE_BBOX_MAX[1])
    for z in (_HAND_BASE_BBOX_MIN[2], _HAND_BASE_BBOX_MAX[2])
]


def _hand_base_closest_corner_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    """World position of whichever corner of the base plate's ACTUAL mesh extent
    (not just its origin) is closest, in xy, to the cube. (N, 3)"""
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_bodies([HAND_BASE_LINK])
    pos = robot.data.body_pos_w[:, ids[0], :]      # (N, 3)
    quat = robot.data.body_quat_w[:, ids[0], :]    # (N, 4)
    cube_pos = env.scene["object"].data.root_pos_w

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


def arm_joint_pos_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Right-arm joint positions relative to default pose. (N, 7)"""
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_joints(ARM_JOINTS)
    return (robot.data.joint_pos - robot.data.default_joint_pos)[:, ids]


def arm_joint_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Right-arm joint velocities. (N, 7)"""
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_joints(ARM_JOINTS)
    return robot.data.joint_vel[:, ids]


def gripper_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Raw gripper finger joint positions (continuous open/closed signal). (N, 2)"""
    robot: Articulation = env.scene["robot"]
    ids, _ = robot.find_joints(GRIPPER_JOINTS)
    return robot.data.joint_pos[:, ids]


# Defensive cap for observations derived from cube position: a violent contact
# event (measured directly: cube launched to +0.7m above the table) can feed an
# extreme value straight into the network's INPUT, not just a reward scalar --
# this is a second, independent layer of protection alongside the object_launched
# termination (which reacts one step later, after the observation was already
# read). NaN is guarded explicitly since torch.clamp alone does not sanitize it.
_OBS_POS_CAP = 5.0  # m -- generously larger than any legitimate scene coordinate


def _safe(t: torch.Tensor) -> torch.Tensor:
    t = torch.nan_to_num(t, nan=0.0, posinf=_OBS_POS_CAP, neginf=-_OBS_POS_CAP)
    return torch.clamp(t, min=-_OBS_POS_CAP, max=_OBS_POS_CAP)


def object_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cube position in the env-local frame. (N, 3)"""
    obj: RigidObject = env.scene["object"]
    return _safe(obj.data.root_pos_w - env.scene.env_origins)


def ee_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Grasp-centre (EE) position in the env-local frame. (N, 3)"""
    return _safe(_ee_pos_w(env) - env.scene.env_origins)


def ee_to_object(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Vector from EE to cube (origin-independent). (N, 3)"""
    obj: RigidObject = env.scene["object"]
    return _safe(obj.data.root_pos_w - _ee_pos_w(env))


def object_to_inspect(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Vector from cube to the inspection point (origin-independent). (N, 3)"""
    obj: RigidObject = env.scene["object"]
    inspect = torch.tensor(INSPECT_POS, device=env.device) + env.scene.env_origins
    return _safe(inspect - obj.data.root_pos_w)


def hand_base_to_object(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Vector from the mounting plate's CLOSEST CORNER (full mesh extent, not just
    its origin -- confirmed empirically to undercount the real risk by ~9cm) to
    the cube. (N, 3) Gives the policy direct, proactive visibility into the body
    that visually strikes the cube's top surface -- previously untracked
    anywhere; it could only be punished reactively (penalty_base_clearance),
    never actually observed."""
    obj: RigidObject = env.scene["object"]
    return _safe(obj.data.root_pos_w - _hand_base_closest_corner_w(env))
