# g1_lift_rl/mdp/__init__.py
"""MDP building blocks for the lift task.

Pulls in Isaac Lab's base mdp (JointPositionActionCfg, BinaryJointPositionActionCfg,
action_rate_l2, joint_vel_l2, last_action, time_out, reset_root_state_uniform, ...)
plus this package's event, observation, termination, and reward terms. Rewards are
implemented but NOT yet wired into env_cfg's RewardsCfg -- that's still pending
(weights are a separate Phase 3b decision).
"""
from isaaclab.envs.mdp import *  # noqa: F401, F403

# --- events ---
from .events import reset_robot_to_default

# --- observations ---
from .observations import (
    arm_joint_pos_rel,
    arm_joint_vel,
    gripper_joint_pos,
    object_position,
    ee_position,
    ee_to_object,
    object_to_inspect,
    hand_base_to_object,
)

# --- terminations ---
from .terminations import object_dropped, object_launched

# --- rewards ---
from .rewards import (
    reward_hover,
    reward_descend,
    reward_straddle_orientation,
    reward_match_pregrasp_pose,
    reward_close_gradient,
    reward_grasp,
    reward_lift,
    reward_inspect,
    reward_inspect_bonus,
    reward_move_to_goal,
    reward_settle_near_goal,
    reward_place,
    reward_release,
    reward_return_to_ready,
    penalty_action_rate,
    penalty_joint_vel,
    penalty_early_close,
    penalty_contact_disturbance,
    penalty_base_clearance,
    penalty_finger_clearance,
    penalty_table_clearance,
    penalty_table_clearance_near_goal_excluded,
    penalty_torso_clearance,
    penalty_low_carry,
)
