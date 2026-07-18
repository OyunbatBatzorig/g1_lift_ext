#!/usr/bin/env python3
"""Ground-truth asset check for the G1 + Dex1 lift task.

Prints the REAL joint names, gripper joint limits, fingertip/EE body names, and
cube info from the actual loaded scene -- so we stop trusting assumed values.
It also commands the gripper fully open then fully closed and reports the joint
positions reached, so we can SEE which direction closes and the true range.

Run:
    cd ~/projects/g1_lift_ext
    python check_assets.py --num_envs 4
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-Lift-Ext-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import g1_lift_rl  # noqa: registers task
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    task = args.task.split(":")[-1]
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=env_cfg, render_mode=None)

    scene = env.unwrapped.scene
    robot = scene["robot"]

    print("\n" + "=" * 70)
    print("ALL JOINT NAMES (index : name : [lower, upper] limits)")
    print("=" * 70)
    names = robot.data.joint_names
    limits = robot.data.joint_pos_limits[0]  # (num_joints, 2), env 0
    for i, n in enumerate(names):
        lo, hi = limits[i, 0].item(), limits[i, 1].item()
        tag = ""
        nl = n.lower()
        if "hand" in nl or "gripper" in nl or "finger" in nl:
            tag = "   <-- GRIPPER?"
        if "elbow" in nl or "shoulder" in nl or "wrist" in nl:
            tag = "   <-- ARM"
        print(f"  [{i:2d}] {n:35s} [{lo:+.4f}, {hi:+.4f}]{tag}")

    print("\n" + "=" * 70)
    print("ALL BODY / LINK NAMES (index : name)")
    print("=" * 70)
    for i, b in enumerate(robot.data.body_names):
        tag = "   <-- FINGERTIP?" if "hand_Link" in b else ""
        print(f"  [{i:2d}] {b}{tag}")

    # try the gripper joints we currently assume
    print("\n" + "=" * 70)
    print("GRIPPER SWEEP TEST (command open, then close, read back positions)")
    print("=" * 70)
    try:
        from g1_lift_rl.constants import GRIPPER_JOINTS, GRIPPER_OPEN, GRIPPER_CLOSE
        gids, gnames = robot.find_joints(GRIPPER_JOINTS)
        print(f"  constants GRIPPER_JOINTS = {GRIPPER_JOINTS} -> resolved {gnames} ids {gids}")
        print(f"  constants GRIPPER_OPEN  = {GRIPPER_OPEN}")
        print(f"  constants GRIPPER_CLOSE = {GRIPPER_CLOSE}")

        env.reset()
        # drive fully toward each limit using joint position targets
        full = robot.data.joint_pos.clone()

        def drive(target_val, label):
            tgt = full.clone()
            tgt[:, gids] = target_val
            for _ in range(60):
                robot.set_joint_position_target(tgt)
                scene.write_data_to_sim()
                env.unwrapped.sim.step()
                scene.update(env_cfg.sim.dt)
            pos = robot.data.joint_pos[0, gids]
            print(f"  commanded {label:18s} ({target_val:+.4f}) -> reached {[f'{p:+.4f}' for p in pos.tolist()]}")

        lo = limits[gids[0], 0].item()
        hi = limits[gids[0], 1].item()
        print(f"  gripper joint hard limits: [{lo:+.4f}, {hi:+.4f}]")
        drive(lo, "toward lower limit")
        drive(hi, "toward upper limit")
        drive(GRIPPER_OPEN, "GRIPPER_OPEN")
        drive(GRIPPER_CLOSE, "GRIPPER_CLOSE")
        print("\n  -> Compare reached values + watch the render to confirm which")
        print("     direction visually CLOSES the fingers, and the true range.")
    except Exception as e:
        print(f"  [!] gripper sweep failed: {e}")

    print("\n" + "=" * 70)
    print("CUBE / OBJECT")
    print("=" * 70)
    obj = scene["object"]
    print(f"  object root pos (env0): {obj.data.root_pos_w[0].tolist()}")
    print(f"  env origin     (env0): {scene.env_origins[0].tolist()}")

    print("\n[done] close the window / Ctrl+C to exit.\n")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()