import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import g1_lift_rl  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

task = "Isaac-G1-Lift-Ext-Play-v0"
env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
env = gym.make(task, cfg=env_cfg, render_mode=None)
robot = env.unwrapped.scene["robot"]

jid, _ = robot.find_joints(["right_wrist_roll_joint"])
limits = robot.data.joint_pos_limits[0, jid[0], :]
print(f"right_wrist_roll_joint limits: {limits.tolist()}", flush=True)

env.close(); simulation_app.close()
