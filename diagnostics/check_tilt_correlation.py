"""Does the hand's approach TILT (deviation from pointing straight down)
correlate with finger-cube contact, as the user's visual read suggests (~85-86
deg instead of 90 -- i.e. not perfectly vertical)? Computes the hand's
pointing direction as the vector from right_hand_base_link to the fingertip
midpoint (EE_LINKS average), measures its angle from straight-down (-Z), and
correlates with contact the same way jitter/EE-deviation were checked.
"""
import argparse
from isaaclab.app import AppLauncher

RESULT_FILE = "tilt_correlation_result.txt"  # writes next to wherever this script is run from

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
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
from g1_lift_rl.constants import BLOCK_SIZE, EE_LINKS

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

ee_ids, _ = robot.find_bodies(EE_LINKS)


def get_obs(o):
    try:
        return o["policy"]
    except (KeyError, TypeError):
        return o


obs, _ = env.reset()
ever_penetrating = torch.zeros(n, dtype=torch.bool, device=device)
final_tilt_deg = torch.zeros(n, device=device)

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

        # Same straddle-tilt metric as reward_straddle_orientation: raw Z
        # height difference between the two fingertips -- 0 = level, larger
        # = one finger dips lower than the other.
        p1 = robot.data.body_pos_w[:, ee_ids[0], :]
        p2 = robot.data.body_pos_w[:, ee_ids[1], :]
        z_mag = torch.abs(p1[:, 2] - p2[:, 2])
        final_tilt_cm = z_mag * 100.0  # keep overwriting -- last step is the converged value

log(f"Policy 1 checkpoint: {args.policy_path}")
log(f"{n} envs -- fingertip height difference (straddle tilt) vs contact\n")
log(f"{'env':>4s} {'tilt(cm)':>9s} {'contact':>8s}")
order = sorted(range(n), key=lambda i: (not ever_penetrating[i].item(), i))
for i in order:
    log(f"{i:4d} {final_tilt_cm[i].item():9.3f} {'YES' if ever_penetrating[i].item() else 'no':>8s}")

contact_mask = ever_penetrating
log(f"\nMean tilt (contact envs): {final_tilt_cm[contact_mask].mean().item() if contact_mask.any() else float('nan'):.3f} cm")
log(f"Mean tilt (no-contact envs): {final_tilt_cm[~contact_mask].mean().item():.3f} cm")
log(f"(0 = fingertips level, larger = one dips lower than the other)")

f.close()
env.close()
simulation_app.close()
