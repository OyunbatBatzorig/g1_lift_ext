# g1_lift_rl/mdp/events.py
from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def reset_robot_to_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset joints to the default pose AND write that pose into the drive target.

    Writing joint_pos_target for ALL joints (not just actuated ones) is what
    holds passive joints -- the left arm, waist, legs -- at the rest pose. Without
    this the PD controller drives them back toward the USD default each step.
    """
    robot: Articulation = env.scene[asset_cfg.name]

    default_pos = robot.data.default_joint_pos[env_ids]
    default_vel = robot.data.default_joint_vel[env_ids]

    # set the physical state at reset
    robot.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)
    # CRITICAL: also write the drive target so passive joints are *held* here
    robot.set_joint_position_target(default_pos, env_ids=env_ids)
    robot.write_data_to_sim()
