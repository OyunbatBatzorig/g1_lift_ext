#!/usr/bin/env python3
"""Reachability check WITH collision ON (transfer-honest).

Question it answers: from the current robot base/ready pose, can the arm bring the
gripper to the cube WITHOUT the head camera (d435_link / head_link) hitting the table?

It does NOT use a trained policy. It directly drives the right arm toward a pose that
should reach over the cube, stepping physics with collision on, and every few steps
reports:
  - gripper(EE midpoint)-to-cube distance  (can it reach?)
  - head/camera height above the table top  (is the head clearing the table, or jammed
    at/under it?)
  - torso/pelvis pitch proxy (how far the body leaned forward)

Interpretation:
  - EE-cube dist gets small (<~0.03) AND head stays clearly ABOVE table top -> a clean,
    collision-free reach EXISTS; the problem is the policy, add a collision penalty.
  - EE-cube dist stalls (can't get small) while head is AT/BELOW table top -> the head is
    blocking; the workspace geometry forces a head-first lean. Fix base height/distance
    or table/cube placement so a real robot could reach -- do NOT disable collision.

Run:
  cd ~/projects/g1_lift_ext
  python check_reach.py --num_envs 1
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-Lift-Ext-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import g1_lift_rl  # noqa: registers task
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from g1_lift_rl.constants import TABLE_TOP_Z, EE_LINKS, ARM_JOINTS


def main():
    task = args.task.split(":")[-1]
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=env_cfg, render_mode=None)
    scene = env.unwrapped.scene
    robot = scene["robot"]
    sim = env.unwrapped.sim

    # resolve indices
    ee_bids, ee_names = robot.find_bodies(EE_LINKS)
    arm_jids, arm_names = robot.find_joints(ARM_JOINTS)
    # head / camera bodies (report whichever exist)
    head_bids, head_names = robot.find_bodies(["d435_link", "head_link"])

    print(f"\n[reach] EE links   : {ee_names}")
    print(f"[reach] arm joints : {arm_names}")
    print(f"[reach] head bodies: {head_names}")
    print(f"[reach] TABLE_TOP_Z = {TABLE_TOP_Z}\n")

    env.reset()

    def ee_pos():
        return robot.data.body_pos_w[:, ee_bids, :].mean(dim=1)

    def cube_pos():
        return scene["object"].data.root_pos_w

    def head_min_z():
        # lowest head/camera body height, env 0
        return robot.data.body_pos_w[0, head_bids, 2].min().item()

    # Build a target arm pose that reaches DOWN/FORWARD toward the cube.
    # We sweep the arm from its current pose toward a "reach over the cube" pose by
    # nudging shoulder pitch / elbow gradually, and just watch what collision allows.
    # (We don't solve IK; we push toward the cube and let physics+collision decide how
    #  far it gets -- that's exactly the constraint the policy faces.)
    default = robot.data.default_joint_pos.clone()
    target = default.clone()

    # crude "reach forward and down" delta on the right arm (positions are env-local
    # offsets; signs chosen to lower & extend the hand toward a -Y, table-height cube).
    # These are intentionally large so we PUSH into the workspace and see what blocks.
    reach_delta = {
        "right_shoulder_pitch_joint": 0.6,
        "right_shoulder_roll_joint": -0.3,
        "right_elbow_joint": 0.6,
        "right_wrist_pitch_joint": 0.5,
    }
    name_to_jid = {n: j for n, j in zip(arm_names, arm_jids)}
    for n, dv in reach_delta.items():
        if n in name_to_jid:
            target[:, name_to_jid[n]] = default[:, name_to_jid[n]] + dv

    with torch.inference_mode():
        for t in range(args.steps):
            # drive ALL joints to default, arm toward the reach target (hold everything
            # else so the body/left arm don't drift) -- collision on the whole time.
            robot.set_joint_position_target(target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(env_cfg.sim.dt)
            if t % 20 == 0:
                d = torch.norm(ee_pos() - cube_pos(), dim=-1)[0].item()
                hz = head_min_z()
                clear = hz - TABLE_TOP_Z
                flag = "HEAD CLEAR" if clear > 0.02 else "HEAD AT/UNDER TABLE <--"
                print(f"[t={t:3d}] EE-cube dist = {d:.3f} | head min z = {hz:.3f} "
                      f"(table top {TABLE_TOP_Z}) -> clearance {clear:+.3f}  {flag}")

    print("\n[reach] DONE. Read: if EE-cube dist stayed large WHILE head clearance went "
          "<= ~0 (HEAD AT/UNDER TABLE), the head blocks the reach -> fix workspace "
          "geometry (robot height/distance, table/cube), keep collision ON. If EE-cube "
          "got small with head clearance staying positive, a clean reach exists -> add a "
          "collision penalty and let the policy learn it.\n")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
