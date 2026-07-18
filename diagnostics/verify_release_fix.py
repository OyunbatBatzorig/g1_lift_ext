"""Sanity check the RELEASE_HOLD_STEPS fix before spending 20+ min retraining:
replay the KNOWN-flickering checkpoint (2026-07-16_08-44-24, confirmed to open
the gripper for one instant then reclose without ever moving) through the new
reward_release logic. If the fix works, this checkpoint's flicker pattern
should now read reward_release=0 for the whole episode (it never sustains
RELEASE_HOLD_STEPS), proving the exploit is closed -- doesn't require the
checkpoint to actually succeed, just confirms the gate rejects the flicker.
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--policy_path", type=str, required=True)
parser.add_argument("--steps", type=int, default=800)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import g1_lift_rl  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from g1_lift_rl.mdp.rewards import _grip_mean, _at_goal_settled, RELEASE_HOLD_STEPS, GRIP_CLOSED_THRESHOLD
# NOTE: deliberately NOT importing/calling reward_release directly here -- the
# reward manager already calls it once internally inside env.step() (it's a
# registered RewTerm), so an extra explicit call here would double-increment
# the hold counter and inflate every number below. Read raw_env._release_hold_
# counter (which the internal call already updated) and raw_env.reward_buf
# instead of calling the function again.

task = "Isaac-G1-Policy4-Ext-Play-v0"
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
print(f"RELEASE_HOLD_STEPS = {RELEASE_HOLD_STEPS}", flush=True)

max_hold_seen = torch.zeros(args.num_envs, device=args.device)
ever_release_new = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)
ever_instantaneous = torch.zeros(args.num_envs, dtype=torch.bool, device=args.device)

with torch.inference_mode():
    for t in range(args.steps):
        actions = policy(get_obs(obs))
        obs, _, _, _ = env.step(actions)
        # Read the counter the reward manager's OWN internal call already
        # updated this step -- do not call reward_release() again here.
        new_release = raw_env._release_hold_counter >= RELEASE_HOLD_STEPS
        ever_release_new |= new_release
        gripper_open = _grip_mean(raw_env) <= GRIP_CLOSED_THRESHOLD
        instantaneous = _at_goal_settled(raw_env) & gripper_open
        ever_instantaneous |= instantaneous
        max_hold_seen = torch.maximum(max_hold_seen, raw_env._release_hold_counter)

print(f"\nInstantaneous condition ever true (old exploit trigger): {ever_instantaneous.sum().item()}/{args.num_envs}", flush=True)
print(f"NEW reward_release (sustained {RELEASE_HOLD_STEPS} steps) ever true: {ever_release_new.sum().item()}/{args.num_envs}", flush=True)
print(f"Max consecutive hold-counter reached per env: {max_hold_seen.tolist()}", flush=True)

env.close()
simulation_app.close()
