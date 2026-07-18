#!/usr/bin/env python3
"""Diagnostic: print what the probe actually sees, so we can find why
'grasping' never fires. Prints, every 10 steps:
  - min/mean/max gripper joint position across envs
  - min EE-to-cube distance across envs
  - min/max cube height above table
Also prints, once at start: resolved GRIPPER_JOINTS names+ids and EE_LINKS names+ids.

CONTACT/DISTURBANCE CHECK (every step, not gated to the %10 print): while the
gripper is still OPEN (grip_mean <= 0.0), tracks the running max cube linear
velocity, angular velocity, and xy drift-from-start across all envs. A spike in
any of these while the gripper hasn't started closing means a finger clipped the
cube during descent (physically knocking/tipping it) rather than the cube sitting
undisturbed until a deliberate grasp -- the finger-spread-vs-cube-width margin is
tight (~9cm open fingertip separation vs 6cm cube, ~1.5cm clearance per side), so
this is a real risk to check, not just a hypothetical. Printed as a final summary,
plus an inline note the first time a new max is set (so you can see roughly when
in the approach it happens). xy-drift baseline is captured right after env.reset()
at t=0; if RslRlVecEnvWrapper auto-resets an individual env mid-rollout (episode
timeout/drop) its baseline goes stale for the remainder of this run -- fine for
spotting drift during the initial approach (the steps we care about here), just
not a lifetime-accurate drift tracker per env.

Run exactly like the probe:
  python diag_grasp.py --num_envs 16 --policy_path <.../exported/policy.pt>
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-Lift-Ext-Play-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--policy_path", type=str, default=None)
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os, torch
import gymnasium as gym
import g1_lift_rl  # noqa: registers task
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from g1_lift_rl.constants import TABLE_TOP_Z, GRIPPER_JOINTS, EE_LINKS
from g1_lift_rl.mdp.rewards import GRASP_DIST


def main():
    task = args.task.split(":")[-1]
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    policy = torch.jit.load(args.policy_path, map_location=args.device); policy.eval()
    scene = env.unwrapped.scene
    robot = scene["robot"]

    gids, gnames = robot.find_joints(GRIPPER_JOINTS)
    bids, bnames = robot.find_bodies(EE_LINKS)
    print(f"\n[diag] GRIPPER_JOINTS -> {gnames}  (ids {gids})")
    print(f"[diag] EE links       -> {bnames}  (ids {bids})")
    print(f"[diag] GRASP_DIST={GRASP_DIST}  TABLE_TOP_Z={TABLE_TOP_Z}\n")

    def ee_pos_w():
        return robot.data.body_pos_w[:, bids, :].mean(dim=1)
    def gpos():
        return robot.data.joint_pos[:, gids]
    def get_obs(o):
        if isinstance(o, torch.Tensor): return o
        try: return o["policy"]
        except (KeyError, TypeError): return o

    obs, _ = env.reset()

    obj = scene["object"]
    xy_start = obj.data.root_pos_w[:, :2].clone()

    max_lin_vel = 0.0
    max_ang_vel = 0.0
    max_xy_drift = 0.0
    max_lin_vel_t, max_ang_vel_t, max_xy_drift_t = -1, -1, -1

    with torch.inference_mode():
        for t in range(args.steps):
            actions = policy(get_obs(obs))
            obs, _, _, _ = env.step(actions)

            g_all = gpos()
            open_mask = g_all.mean(dim=-1) <= 0.0  # gripper not (yet) closing

            if open_mask.any():
                lin_vel = torch.norm(obj.data.root_lin_vel_w[open_mask], dim=-1)
                ang_vel = torch.norm(obj.data.root_ang_vel_w[open_mask], dim=-1)
                xy_drift = torch.norm(obj.data.root_pos_w[open_mask, :2] - xy_start[open_mask], dim=-1)

                lv, av, xd = lin_vel.max().item(), ang_vel.max().item(), xy_drift.max().item()
                if lv > max_lin_vel:
                    max_lin_vel, max_lin_vel_t = lv, t
                if av > max_ang_vel:
                    max_ang_vel, max_ang_vel_t = av, t
                if xd > max_xy_drift:
                    max_xy_drift, max_xy_drift_t = xd, t

            if t % 10 == 0:
                obj_w = obj.data.root_pos_w
                d = torch.norm(ee_pos_w() - obj_w, dim=-1)
                g = g_all
                h = obj_w[:, 2] - TABLE_TOP_Z
                print(f"[t={t:3d}] grip min/mean/max = "
                      f"{g.min():.4f}/{g.mean():.4f}/{g.max():.4f} | "
                      f"EE-cube dist min = {d.min():.3f} (<{GRASP_DIST}?) | "
                      f"cube height min/max = {h.min():.3f}/{h.max():.3f}")

    print("\n" + "=" * 70)
    print("CONTACT/DISTURBANCE CHECK (cube motion while gripper still OPEN)")
    print("=" * 70)
    print(f"  max linear velocity while open : {max_lin_vel:.4f} m/s   (at t={max_lin_vel_t})")
    print(f"  max angular velocity while open: {max_ang_vel:.4f} rad/s (at t={max_ang_vel_t})")
    print(f"  max xy drift-from-start (open) : {max_xy_drift:.4f} m    (at t={max_xy_drift_t})")
    print("  -> near-zero across all three = cube undisturbed until a deliberate grasp.")
    print("  -> any clearly non-zero spike = a finger likely clipped the cube while")
    print("     still open (approach angle/alignment issue, not a reward-shaping one).")
    print("=" * 70 + "\n")

    env.close(); simulation_app.close()

if __name__ == "__main__":
    main()
