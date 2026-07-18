"""Redo of check_finger_cube_contact.py: the first version only checked the
two fingertip BODY ORIGINS as zero-size points and found no contact -- but
that misses (a) the fingertip's actual mesh volume and (b) the other finger
link segments (middle knuckles etc.) entirely, any of which could be the part
actually touching the cube in the screenshot. This version checks all 7
Dex1 finger/hand link bodies using their real bounding boxes (same BBOXES
used by solve_pregrasp_anchored.py, already measured for this exact purpose),
each body's 8 corners transformed into the cube's local frame.
"""
import argparse
from isaaclab.app import AppLauncher

RESULT_FILE = "finger_cube_contact_v2_result.txt"  # writes next to wherever this script is run from

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--policy_path", type=str, required=True)
parser.add_argument("--steps", type=int, default=795)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import g1_lift_rl  # noqa
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from g1_lift_rl.constants import BLOCK_SIZE

RESULT_FILE_LOG = RESULT_FILE
f = open(RESULT_FILE, "w")
def log(msg=""):
    print(msg, flush=True)
    f.write(str(msg) + "\n")
    f.flush()

# Same bounding boxes solve_pregrasp_anchored.py already measured for these
# exact Dex1 finger/hand bodies (local frame, min/max corners in meters).
BBOXES = {
    "right_hand_base_link": ((-0.035, 0.000, -0.035), (0.035, 0.0738, 0.089)),
    "right_hand_Link1_1":   ((-0.080, -0.005, -0.008), (-0.000, 0.009, 0.0065)),
    "right_hand_Link1_2":   ((-0.006, -0.0068, -0.0304), (0.013, 0.075, 0.000)),
    "right_hand_Link1_3":   ((-0.000, -0.002, -0.0284), (0.004, 0.0475, 0.000)),
    "right_hand_Link2_1":   ((0.000, -0.005, -0.0065), (0.080, 0.009, 0.008)),
    "right_hand_Link2_2":   ((-0.013, -0.0068, 0.000), (0.006, 0.075, 0.0304)),
    "right_hand_Link2_3":   ((-0.004, -0.002, -0.0284), (0.000, 0.0475, 0.000)),
}
BODY_NAMES = list(BBOXES.keys())
half = BLOCK_SIZE / 2.0

task = "Isaac-G1-Lift-Ext-Play-v0"
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

body_ids, body_names = robot.find_bodies(BODY_NAMES, preserve_order=True)
corners_local = []
for name in body_names:
    mn, mx = BBOXES[name]
    pts = [(x, y, z) for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])]
    corners_local.append(pts)
corners_local = torch.tensor(corners_local, device=device)  # (B, 8, 3)
B = len(body_names)


def get_obs(o):
    try:
        return o["policy"]
    except (KeyError, TypeError):
        return o


obs, _ = env.reset()

max_penetration = {name: torch.zeros(n, device=device) for name in body_names}
ever_penetrating = {name: torch.zeros(n, dtype=torch.bool, device=device) for name in body_names}

with torch.inference_mode():
    for t in range(args.steps):
        actions = policy(get_obs(obs))
        obs, _, _, _ = env.step(actions)

        pos = robot.data.body_pos_w[:, body_ids, :]      # (N, B, 3)
        quat = robot.data.body_quat_w[:, body_ids, :]    # (N, B, 4)
        q = quat.unsqueeze(2).expand(n, B, 8, 4).reshape(n * B * 8, 4)
        c = corners_local.unsqueeze(0).expand(n, B, 8, 3).reshape(n * B * 8, 3)
        world_offsets = quat_apply(q, c).reshape(n, B, 8, 3)
        world_corners = pos.unsqueeze(2) + world_offsets  # (N, B, 8, 3)

        cube_pos = obj.data.root_pos_w  # (N, 3)
        cube_quat = obj.data.root_quat_w  # (N, 4)
        cq = cube_quat[:, None, None, :].expand(n, B, 8, 4).reshape(-1, 4)
        rel_world = (world_corners - cube_pos[:, None, None, :]).reshape(-1, 3)
        rel_local = quat_apply_inverse(cq, rel_world).reshape(n, B, 8, 3)

        pen_per_axis = half - rel_local.abs()  # (N, B, 8, 3)
        penetrating_corner = (pen_per_axis > 0).all(dim=-1)  # (N, B, 8)
        depth_per_corner = torch.clamp(pen_per_axis.min(dim=-1).values, min=0.0)  # (N, B, 8)
        depth_per_body = depth_per_corner.max(dim=-1).values  # (N, B) -- worst corner per body
        penetrating_body = penetrating_corner.any(dim=-1)  # (N, B)

        for i, name in enumerate(body_names):
            max_penetration[name] = torch.maximum(max_penetration[name], depth_per_body[:, i])
            ever_penetrating[name] |= penetrating_body[:, i]

log(f"Policy 1 checkpoint: {args.policy_path}")
log(f"Ran {args.steps} steps across {n} envs -- full finger-geometry cube penetration check.\n")
for name in body_names:
    log(f"{name:24s}  ever_penetrating={ever_penetrating[name].sum().item():3d}/{n}   "
        f"max_depth={max_penetration[name].max().item()*100:.3f}cm   "
        f"mean_max_depth={max_penetration[name].mean().item()*100:.3f}cm")

any_contact = torch.zeros(n, dtype=torch.bool, device=device)
for name in body_names:
    any_contact |= ever_penetrating[name]
log(f"\nEnvs with ANY finger-link penetration at some point: {any_contact.sum().item()}/{n}")

f.close()
env.close()
simulation_app.close()
