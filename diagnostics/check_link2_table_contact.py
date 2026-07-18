"""Is right_hand_Link2 (or any specific finger/hand body) dragging on the
TABLE during TRANSIT (not just the already-known, already-excluded near-goal
placing contact)? Uses the same _table_clearance_per_body geometry Policy 3's
own penalty_table_clearance_near_goal_excluded already computes, but reports
PER-BODY, PER-PHASE (transit vs near-goal) instead of summed into one reward
number -- so we can see exactly which body, and exactly when.
"""
import argparse
from isaaclab.app import AppLauncher

RESULT_FILE = "link2_table_contact_result.txt"  # writes next to wherever this script is run from

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
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
from g1_lift_rl.mdp.rewards import SETTLE_NEAR_RADIUS
from g1_lift_rl.constants import GOAL_POS, TABLE_CLEARANCE_LINKS

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
robot = raw_env.scene["robot"]
obj = raw_env.scene["object"]
n = args.num_envs
device = args.device

# Reuse the exact same per-body table clearance geometry as
# _table_clearance_per_body / penalty_table_clearance in rewards.py.
from g1_lift_rl.mdp.rewards import _table_clearance_per_body
body_ids, body_names = robot.find_bodies(TABLE_CLEARANCE_LINKS, preserve_order=True)


def get_obs(o):
    try:
        return o["policy"]
    except (KeyError, TypeError):
        return o


obs, _ = env.reset()
goal_w_local = torch.tensor(GOAL_POS, device=device)

sum_violation = {name: {"transit": 0.0, "near_goal": 0.0} for name in body_names}
nonzero_count = {name: {"transit": 0, "near_goal": 0} for name in body_names}
total_count = {"transit": 0, "near_goal": 0}

with torch.inference_mode():
    for t in range(args.steps):
        actions = policy(get_obs(obs))
        obs, _, _, _ = env.step(actions)

        per_body = _table_clearance_per_body(raw_env)  # (N, B)

        goal_w = goal_w_local + raw_env.scene.env_origins
        xy_dist = torch.norm(obj.data.root_pos_w[:, :2] - goal_w[:, :2], dim=-1)
        near_goal = xy_dist < SETTLE_NEAR_RADIUS

        n_transit = (~near_goal).sum().item()
        n_near = near_goal.sum().item()
        total_count["transit"] += n_transit
        total_count["near_goal"] += n_near

        for i, name in enumerate(body_names):
            v = per_body[:, i]
            sum_violation[name]["transit"] += v[~near_goal].sum().item()
            sum_violation[name]["near_goal"] += v[near_goal].sum().item()
            nonzero_count[name]["transit"] += (v[~near_goal] > 1e-6).sum().item()
            nonzero_count[name]["near_goal"] += (v[near_goal] > 1e-6).sum().item()

log(f"Policy 3 checkpoint: {args.policy_path}")
log(f"Ran {args.steps} steps across {n} envs. transit steps={total_count['transit']}  near_goal steps={total_count['near_goal']}\n")
log(f"{'body':28s} {'transit nz%':>12s} {'transit mean(cm)':>17s} {'near_goal nz%':>14s} {'near_goal mean(cm)':>19s}")
for name in body_names:
    tc = total_count["transit"] or 1
    ng = total_count["near_goal"] or 1
    t_nz = 100.0 * nonzero_count[name]["transit"] / tc
    t_mean = sum_violation[name]["transit"] / tc * 100
    n_nz = 100.0 * nonzero_count[name]["near_goal"] / ng
    n_mean = sum_violation[name]["near_goal"] / ng * 100
    log(f"{name:28s} {t_nz:11.2f}% {t_mean:16.4f} {n_nz:13.2f}% {n_mean:18.4f}")

f.close()
env.close()
simulation_app.close()
