"""What horizontal (XY) angle does the gripper actually approach the cube
from, at Policy 1's converged hover pose? Used to compute how much to rotate
the cube (around Z) so a FACE, not an edge/corner, faces the real approach
direction -- measured from the actual checkpoint, not derived from
GRASP_OFFSET alone (the approach trajectory can differ from the final offset
direction).
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--policy_path", type=str, required=True)
parser.add_argument("--steps", type=int, default=795)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import math
import gymnasium as gym
import g1_lift_rl  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from g1_lift_rl.mdp.rewards import _ee_pos_w, _cube_pos_w

task = "Isaac-G1-Lift-Ext-Play-v0"
env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
env = gym.make(task, cfg=env_cfg, render_mode=None)
env = RslRlVecEnvWrapper(env)
policy = torch.jit.load(args.policy_path, map_location=args.device)
policy.eval()
raw_env = env.unwrapped


def get_obs(o):
    try:
        return o["policy"]
    except (KeyError, TypeError):
        return o


obs, _ = env.reset()
with torch.inference_mode():
    for t in range(args.steps):
        actions = policy(get_obs(obs))
        obs, _, _, _ = env.step(actions)

ee = _ee_pos_w(raw_env)
cube = _cube_pos_w(raw_env)
offset = ee - cube  # vector FROM cube center TO gripper (approach direction)
offset_xy = offset[:, :2]

# Angle from world +Y axis (matching how GRASP_OFFSET's y-dominant convention reads)
angle_rad = torch.atan2(offset_xy[:, 0], offset_xy[:, 1])  # atan2(x, y): 0 = +Y, positive = toward +X
mean_angle_rad = angle_rad.mean().item()
mean_offset_xy = offset_xy.mean(dim=0).tolist()

print(f"\nMean EE-cube XY offset (approach direction proxy): {mean_offset_xy}", flush=True)
print(f"Per-env angle from +Y axis: mean={math.degrees(mean_angle_rad):.2f} deg, "
      f"std={math.degrees(angle_rad.std().item()):.2f} deg", flush=True)
print(f"individual angles (deg): {[round(math.degrees(a),2) for a in angle_rad.tolist()]}", flush=True)
print(f"\n=> To align a cube FACE (currently normal to world Y) with this approach "
      f"direction, rotate the cube by {math.degrees(mean_angle_rad):+.2f} deg around Z.", flush=True)

env.close()
simulation_app.close()
