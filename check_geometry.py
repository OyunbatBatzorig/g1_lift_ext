#!/usr/bin/env python3
"""Minimal geometry check at RESET pose (no arm driving, no policy).
Reports, for env 0, the resting positions so we can see the workspace layout:
  - head camera (d435_link) and head_link world z vs table top
  - right hand base + EE (fingertip midpoint) world position
  - cube world position, table top z
  - pelvis/torso height
This avoids any flawed 'reach' guessing -- just the static layout truth.

Run:  cd ~/projects/g1_lift_ext && python check_geometry.py --num_envs 1
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-Lift-Ext-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import g1_lift_rl  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from g1_lift_rl.constants import TABLE_TOP_Z, EE_LINKS

def main():
    task = args.task.split(":")[-1]
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=env_cfg, render_mode=None)
    scene = env.unwrapped.scene
    robot = scene["robot"]
    env.reset()
    # let it settle a few steps at the reset/hold pose
    sim = env.unwrapped.sim
    for _ in range(30):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim(); sim.step(); scene.update(env_cfg.sim.dt)

    o = scene.env_origins[0]
    def w(name):
        ids, _ = robot.find_bodies([name])
        p = robot.data.body_pos_w[0, ids[0], :] - o
        return p.tolist()
    ee_ids, _ = robot.find_bodies(EE_LINKS)
    ee = (robot.data.body_pos_w[0, ee_ids, :].mean(dim=0) - o).tolist()
    cube = (scene["object"].data.root_pos_w[0] - o).tolist()

    print("\n========== GEOMETRY AT REST (env-local) ==========")
    print(f"  TABLE_TOP_Z            : {TABLE_TOP_Z:.3f}")
    print(f"  cube pos (x,y,z)       : {cube[0]:+.3f}, {cube[1]:+.3f}, {cube[2]:+.3f}")
    print(f"  pelvis  z              : {w('pelvis')[2]:.3f}")
    print(f"  torso_link z           : {w('torso_link')[2]:.3f}")
    print(f"  head_link   (x,y,z)    : {w('head_link')[0]:+.3f}, {w('head_link')[1]:+.3f}, {w('head_link')[2]:.3f}")
    print(f"  d435_link   (x,y,z)    : {w('d435_link')[0]:+.3f}, {w('d435_link')[1]:+.3f}, {w('d435_link')[2]:.3f}")
    print(f"  right_hand_base (x,y,z): {w('right_hand_base_link')[0]:+.3f}, {w('right_hand_base_link')[1]:+.3f}, {w('right_hand_base_link')[2]:.3f}")
    print(f"  EE midpoint (x,y,z)    : {ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:.3f}")
    print(f"  -- head camera vs table top: d435 z {w('d435_link')[2]:.3f} - table {TABLE_TOP_Z:.3f} = {w('d435_link')[2]-TABLE_TOP_Z:+.3f}")
    print(f"  -- EE-to-cube dist at rest : {((torch.tensor(ee)-torch.tensor(cube)).norm()).item():.3f}")
    print("==================================================\n")
    env.close(); simulation_app.close()

if __name__ == "__main__":
    main()
