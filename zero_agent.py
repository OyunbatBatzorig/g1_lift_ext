#!/usr/bin/env python3
"""Numeric (no rendering) zero-agent scene check for g1_lift_ext.

Part A: builds the real scene, steps with all-zero actions (zero action == default
right-arm pose + gripper OPEN per BinaryJointPositionAction's "0 -> open" convention),
then asserts what a visual check would otherwise confirm: left arm held at
LEFT_ARM_STOW, right arm near READY_ARM_POSE, gripper open, cube resting on the
table, nothing NaN.

Part B: explicitly forces the cube to each of the 4 corners of the +-0.05 reset
jitter range (not relying on random sampling to eventually land there) and checks
each one stays on the table -- tested, not hoped. Does NOT check reachability from
the ready pose: zero action never moves the arm toward the cube (there's no policy
yet), so that question is out of scope for this script -- it's a Phase 3 question.

Run:
    cd ~/projects/g1_lift_ext
    python zero_agent.py --num_envs 4
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-Lift-Ext-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import g1_lift_rl  # noqa: registers task
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from g1_lift_rl.constants import (
    LEFT_ARM_STOW, READY_ARM_POSE, GRIPPER_OPEN, GRIPPER_JOINTS, TABLE_TOP_Z,
    TABLE_POS, TABLE_SIZE, BLOCK_SIZE, BLOCK_INIT_POS,
)

TOL_ARM = 0.05      # rad (~2.9deg) -- catches real drooping, tolerates minor PD sag under gravity
TOL_GRIPPER = 0.005
JITTER = 0.05        # matches EventCfg.reset_object's pose_range x/y
CORNERS = [(-JITTER, -JITTER), (-JITTER, JITTER), (JITTER, -JITTER), (JITTER, JITTER)]


def main():
    task = args.task.split(":")[-1]
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=env_cfg, render_mode=None)

    scene = env.unwrapped.scene
    robot = scene["robot"]
    obj = scene["object"]
    device = env.unwrapped.device

    checks = []

    def check(name, cond, detail):
        checks.append((name, bool(cond), detail))

    def step_zero(n):
        action_dim = env.unwrapped.action_manager.total_action_dim
        zero_actions = torch.zeros(args.num_envs, action_dim, device=device)
        for _ in range(n):
            env.step(zero_actions)

    # ---------------------------------------------------------------------
    # Part A: pose-holding + default scene check
    # ---------------------------------------------------------------------
    env.reset()
    step_zero(args.steps)

    jp = robot.data.joint_pos
    nan_any = torch.isnan(jp).any() or torch.isnan(obj.data.root_pos_w).any()
    check("no_nans", not nan_any, f"any NaN in joint_pos/object pos: {bool(nan_any)}")

    for name, target in LEFT_ARM_STOW.items():
        ids, _ = robot.find_joints([name])
        err = (jp[:, ids[0]] - target).abs().max().item()
        check(f"left_arm[{name}]", err < TOL_ARM, f"target={target:+.3f} max|err|={err:.4f}")

    for name, target in READY_ARM_POSE.items():
        ids, _ = robot.find_joints([name])
        err = (jp[:, ids[0]] - target).abs().max().item()
        check(f"right_arm[{name}]", err < TOL_ARM, f"target={target:+.3f} max|err|={err:.4f}")

    gids, _ = robot.find_joints(GRIPPER_JOINTS)
    gerr = (jp[:, gids] - GRIPPER_OPEN).abs().max().item()
    check("gripper_open", gerr < TOL_GRIPPER, f"target={GRIPPER_OPEN:+.4f} max|err|={gerr:.4f}")

    expect_z = TABLE_TOP_Z + BLOCK_SIZE / 2.0
    cube_err = (obj.data.root_pos_w[:, 2] - expect_z).abs().max().item()
    check("cube_on_table[default]", cube_err < 0.01, f"expect_z={expect_z:.3f} max|err|={cube_err:.4f}")

    # ---------------------------------------------------------------------
    # Part B: force the cube to each +-0.05 jitter corner explicitly. A random
    # reset would only rarely sample exactly at the extremes -- this tests them
    # every time instead of hoping enough resets eventually cover them.
    # ---------------------------------------------------------------------
    table_x_lo = TABLE_POS[0] - TABLE_SIZE[0] / 2.0
    table_x_hi = TABLE_POS[0] + TABLE_SIZE[0] / 2.0
    table_y_lo = TABLE_POS[1] - TABLE_SIZE[1] / 2.0
    table_y_hi = TABLE_POS[1] + TABLE_SIZE[1] / 2.0

    for dx, dy in CORNERS:
        target_x = BLOCK_INIT_POS[0] + dx
        target_y = BLOCK_INIT_POS[1] + dy
        label = f"corner(dx={dx:+.2f},dy={dy:+.2f})"

        # static geometric check: is the forced target even within the table footprint?
        within_table = (table_x_lo < target_x < table_x_hi) and (table_y_lo < target_y < table_y_hi)
        check(
            f"{label}_within_table_bounds", within_table,
            f"target=({target_x:.3f},{target_y:.3f}) "
            f"table_x=[{table_x_lo:.3f},{table_x_hi:.3f}] table_y=[{table_y_lo:.3f},{table_y_hi:.3f}]",
        )

        env.reset()
        local_pos = torch.tensor([target_x, target_y, BLOCK_INIT_POS[2]], device=device)
        world_pos = local_pos.unsqueeze(0).expand(args.num_envs, -1) + scene.env_origins
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(args.num_envs, -1)
        obj.write_root_pose_to_sim(torch.cat([world_pos, identity_quat], dim=-1))
        obj.write_root_velocity_to_sim(torch.zeros(args.num_envs, 6, device=device))

        step_zero(args.steps)

        obj_local = obj.data.root_pos_w - scene.env_origins
        xy_err = (obj_local[:, :2] - torch.tensor([target_x, target_y], device=device)).abs().max().item()
        z_err = (obj_local[:, 2] - expect_z).abs().max().item()
        check(
            f"{label}_settled_in_place", xy_err < 0.01 and z_err < 0.01,
            f"xy_max|err|={xy_err:.4f} z_max|err|={z_err:.4f} (expect both < 0.01 -- didn't slide/fall)",
        )

    print("\n" + "=" * 78)
    print(f"ZERO-AGENT SCENE CHECK  (task={task}, num_envs={args.num_envs}, steps={args.steps})")
    print("=" * 78)
    all_pass = True
    for name, ok, detail in checks:
        all_pass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:32s} {detail}")
    print("=" * 78)
    print("RESULT:", "ALL PASS" if all_pass else "FAILURES PRESENT")
    print("NOTE: reachability of the cube from READY_ARM_POSE is NOT tested here --")
    print("      zero action never moves the arm toward the cube. That's Phase 3.")
    print("=" * 78 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
