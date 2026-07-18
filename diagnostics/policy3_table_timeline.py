"""Plain per-step timeline (not bucketed) for Policy 3's narrowed checkpoint:
table-clearance violation and xy-distance-to-goal, step by step, for the first
part of one episode -- so it's clear exactly when contact starts relative to
reset vs. approach vs. settling at the goal, instead of a phase-bucketed
summary. Tracks a handful of envs individually (not averaged) so the timeline
per env is legible. Lightweight (small num_envs), runs alongside Policy 1's
training job on the same GPU.
"""
import argparse
from isaaclab.app import AppLauncher

RESULT_FILE = "policy3_table_timeline_result.txt"  # writes next to wherever this script is run from

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--policy_path", type=str, required=True)
parser.add_argument("--steps", type=int, default=150)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import g1_lift_rl  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from g1_lift_rl.mdp.rewards import _table_clearance_per_body, SETTLE_NEAR_RADIUS
from g1_lift_rl.constants import GOAL_POS

f = open(RESULT_FILE, "w")
def log(msg=""):
    print(msg, flush=True)
    f.write(str(msg) + "\n")
    f.flush()

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

log(f"Policy 3 checkpoint: {args.policy_path}")
log(f"Tracking {n} envs individually for {args.steps} steps (SETTLE_NEAR_RADIUS={SETTLE_NEAR_RADIUS}m)\n")
header = "step " + "".join(f"| env{i}: xy_dist  viol  " for i in range(n))
log(header)

with torch.inference_mode():
    for t in range(args.steps):
        actions = policy(get_obs(obs))
        obs, _, _, _ = env.step(actions)

        per_body = _table_clearance_per_body(raw_env)
        violation = per_body.sum(dim=-1)

        goal_w = goal_w_local + raw_env.scene.env_origins
        xy_dist = torch.norm(obj.data.root_pos_w[:, :2] - goal_w[:, :2], dim=-1)

        if t % 3 == 0 or (t < 20):  # dense early on, sparser later
            row = f"{t:4d} "
            for i in range(n):
                row += f"| {xy_dist[i].item():.3f}m {violation[i].item():.4f}m "
            log(row)

f.close()
env.close()
simulation_app.close()
