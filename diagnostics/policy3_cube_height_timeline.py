"""Is Policy 3 carrying the cube through the air, or dragging it along the
table surface? Tracks cube height above _CUBE_REST_Z (table-resting height)
alongside xy-distance-to-goal and _is_grasping, same timeline style as
policy3_table_timeline.py. If cube height stays near 0 (near resting height)
the whole time it's being "carried" toward the goal, that's dragging, not
lifting-and-carrying.
"""
import argparse
from isaaclab.app import AppLauncher

RESULT_FILE = "policy3_cube_height_result.txt"  # writes next to wherever this script is run from

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
from g1_lift_rl.mdp.rewards import _is_grasping, _grip_mean, _CUBE_REST_Z
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
log(f"_CUBE_REST_Z = {_CUBE_REST_Z:.4f} m (cube center height when resting on table)")
log(f"Tracking {n} envs for {args.steps} steps: cube_height_above_rest, xy_dist_to_goal, is_grasping\n")
header = "step " + "".join(f"| env{i}: hgt(cm) xy_dist grasp " for i in range(n))
log(header)

with torch.inference_mode():
    for t in range(args.steps):
        actions = policy(get_obs(obs))
        obs, _, _, _ = env.step(actions)

        cube_z = obj.data.root_pos_w[:, 2]
        height_above_rest = (cube_z - _CUBE_REST_Z) * 100.0  # cm

        goal_w = goal_w_local + raw_env.scene.env_origins
        xy_dist = torch.norm(obj.data.root_pos_w[:, :2] - goal_w[:, :2], dim=-1)
        grasping = _is_grasping(raw_env)

        if t % 3 == 0 or (t < 20):
            row = f"{t:4d} "
            for i in range(n):
                row += f"| {height_above_rest[i].item():+6.2f}cm {xy_dist[i].item():.3f}m {'Y' if grasping[i].item() else 'n':>5s} "
            log(row)

f.close()
env.close()
simulation_app.close()
