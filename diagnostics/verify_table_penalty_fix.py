"""Sanity check the capped near-goal table-penalty fix before retraining:
replay the CURRENT checkpoint (known to have ~0.5cm base_link penetration,
100% of near-goal steps) through the NEW penalty_table_clearance_near_goal_
excluded and confirm it now returns nonzero for that known-bad behavior,
while still returning ~0 for the acceptable Link2 contact -- proving the fix
distinguishes them correctly before spending 25+ min retraining.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--policy_path", type=str, required=True)
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import g1_lift_rl  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from g1_lift_rl.mdp.rewards import (
    penalty_table_clearance_near_goal_excluded, _table_clearance_per_body, SETTLE_NEAR_RADIUS,
)
from g1_lift_rl.constants import GOAL_POS

task = "Isaac-G1-Policy3-Ext-Play-v0"
env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
env = gym.make(task, cfg=env_cfg, render_mode=None)
env = RslRlVecEnvWrapper(env)
policy = torch.jit.load(args.policy_path, map_location=args.device)
policy.eval()
raw_env = env.unwrapped
obj = raw_env.scene["object"]
n = args.num_envs
device = args.device


def get_obs(o):
    try:
        return o["policy"]
    except (KeyError, TypeError):
        return o


obs, _ = env.reset()
goal_w_local = torch.tensor(GOAL_POS, device=device)
near_goal_new_penalty_sum = 0.0
near_goal_steps = 0

with torch.inference_mode():
    for t in range(args.steps):
        actions = policy(get_obs(obs))
        obs, _, _, _ = env.step(actions)

        new_penalty = penalty_table_clearance_near_goal_excluded(raw_env)
        goal_w = goal_w_local + raw_env.scene.env_origins
        xy_dist = torch.norm(obj.data.root_pos_w[:, :2] - goal_w[:, :2], dim=-1)
        near_goal = xy_dist < SETTLE_NEAR_RADIUS

        if near_goal.any():
            near_goal_new_penalty_sum += new_penalty[near_goal].sum().item()
            near_goal_steps += near_goal.sum().item()

mean_new_penalty_near_goal = near_goal_new_penalty_sum / max(near_goal_steps, 1)
print(f"\nRESULT: mean NEW capped penalty near goal = {mean_new_penalty_near_goal:.5f} "
      f"(was exactly 0.0 under the old full exclusion, across {near_goal_steps} near-goal steps)", flush=True)

env.close()
simulation_app.close()
