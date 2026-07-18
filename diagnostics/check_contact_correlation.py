"""Which explains the finger-cube contact found in check_finger_cube_contact_v2.py:
cube reset jitter, or the policy's own EE-position imprecision at its
converged hover pose? Tracks BOTH per env (not averaged) alongside whether
that env ever showed finger-link penetration, so they can be directly
compared side by side instead of guessed at.
"""
import argparse
from isaaclab.app import AppLauncher

RESULT_FILE = "contact_correlation_result.txt"  # writes next to wherever this script is run from

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
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
from g1_lift_rl.mdp.rewards import _ee_pos_w, _cube_pos_w
from g1_lift_rl.constants import BLOCK_SIZE, BLOCK_INIT_POS, GRASP_OFFSET

f = open(RESULT_FILE, "w")
def log(msg=""):
    print(msg, flush=True)
    f.write(str(msg) + "\n")
    f.flush()

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
corners_local = torch.tensor(corners_local, device=device)
B = len(body_names)


def get_obs(o):
    try:
        return o["policy"]
    except (KeyError, TypeError):
        return o


obs, _ = env.reset()

# Capture reset jitter immediately after reset (env-local, so origin-subtracted).
block_init = torch.tensor(BLOCK_INIT_POS, device=device)
cube_reset_pos = obj.data.root_pos_w - raw_env.scene.env_origins
jitter = cube_reset_pos[:, :2] - block_init[:2]
jitter_mag = torch.norm(jitter, dim=-1)

ever_penetrating = torch.zeros(n, dtype=torch.bool, device=device)

with torch.inference_mode():
    for t in range(args.steps):
        actions = policy(get_obs(obs))
        obs, _, _, _ = env.step(actions)

        pos = robot.data.body_pos_w[:, body_ids, :]
        quat = robot.data.body_quat_w[:, body_ids, :]
        q = quat.unsqueeze(2).expand(n, B, 8, 4).reshape(n * B * 8, 4)
        c = corners_local.unsqueeze(0).expand(n, B, 8, 3).reshape(n * B * 8, 3)
        world_offsets = quat_apply(q, c).reshape(n, B, 8, 3)
        world_corners = pos.unsqueeze(2) + world_offsets

        cube_pos = obj.data.root_pos_w
        cube_quat = obj.data.root_quat_w
        cq = cube_quat[:, None, None, :].expand(n, B, 8, 4).reshape(-1, 4)
        rel_world = (world_corners - cube_pos[:, None, None, :]).reshape(-1, 3)
        rel_local = quat_apply_inverse(cq, rel_world).reshape(n, B, 8, 3)
        pen_per_axis = half - rel_local.abs()
        penetrating_corner = (pen_per_axis > 0).all(dim=-1)
        ever_penetrating |= penetrating_corner.any(dim=(-1, -2))

# Final EE deviation from the ideal grasp-offset target (converged hover pose).
final_ee = _ee_pos_w(raw_env)
final_cube = _cube_pos_w(raw_env)
grasp_offset = torch.tensor(GRASP_OFFSET, device=device)
ee_target = final_cube + grasp_offset
ee_dev = torch.norm(final_ee - ee_target, dim=-1)

log(f"Policy 1 checkpoint: {args.policy_path}")
log(f"{n} envs, {args.steps} steps -- per-env jitter vs EE-deviation vs contact\n")
log(f"{'env':>4s} {'jitter(cm)':>11s} {'ee_dev(cm)':>11s} {'contact':>8s}")
# Sort by contact status so contact envs are easy to scan
order = sorted(range(n), key=lambda i: (not ever_penetrating[i].item(), i))
for i in order:
    log(f"{i:4d} {jitter_mag[i].item()*100:11.3f} {ee_dev[i].item()*100:11.3f} {'YES' if ever_penetrating[i].item() else 'no':>8s}")

contact_mask = ever_penetrating
log(f"\nMean jitter (contact envs): {jitter_mag[contact_mask].mean().item()*100 if contact_mask.any() else float('nan'):.3f}cm")
log(f"Mean jitter (no-contact envs): {jitter_mag[~contact_mask].mean().item()*100:.3f}cm")
log(f"Mean EE_dev (contact envs): {ee_dev[contact_mask].mean().item()*100 if contact_mask.any() else float('nan'):.3f}cm")
log(f"Mean EE_dev (no-contact envs): {ee_dev[~contact_mask].mean().item()*100:.3f}cm")

f.close()
env.close()
simulation_app.close()
