#!/usr/bin/env python3
"""CORRECTED version of check_hand_base.py: the first version measured distance
from each body's ORIGIN POINT to the cube. But these bodies are not points --
their meshes extend well beyond their own origin (measured via USD bounding-box
query, see probe_base_mesh_extent.py):

    right_hand_base_link  local bbox  x=[-0.035,+0.035] y=[0,+0.074]  z=[-0.035,+0.089]
    right_hand_Link1_1                x=[-0.080,-0.000] y=[-0.005,+0.009] z=[-0.008,+0.0065]
    right_hand_Link1_2                x=[-0.006,+0.013] y=[-0.007,+0.075] z=[-0.030,+0.000]
    right_hand_Link1_3                x=[-0.000,+0.004] y=[-0.002,+0.048] z=[-0.028,+0.000]
    right_hand_Link2_1                x=[+0.000,+0.080] y=[-0.005,+0.009] z=[-0.007,+0.008]
    right_hand_Link2_2                x=[-0.013,+0.006] y=[-0.007,+0.075] z=[+0.000,+0.030]
    right_hand_Link2_3                x=[-0.004,+0.000] y=[-0.002,+0.048] z=[-0.028,+0.000]

The base link's mesh alone reaches up to ~9cm beyond its own tracked origin --
meaning the origin-only check could read "9cm away" while the plate's actual
edge is only 0-2cm from the cube. This version transforms all 8 corners of each
body's LOCAL bbox into WORLD space (using that body's actual position+orientation
at each step, via isaaclab.utils.math.quat_apply), then uses whichever corner is
closest+lowest -- the real physical extent, not an origin-point approximation.

Run:
    cd ~/projects/g1_lift_ext
    python check_hand_geometry_v2.py --num_envs 32 --policy_path <exported policy.pt>
"""
import argparse
from collections import deque
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-Lift-Ext-Play-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--policy_path", type=str, default=None)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--vel_spike", type=float, default=0.5)
parser.add_argument("--history", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import g1_lift_rl  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.utils.math import quat_apply
from g1_lift_rl.constants import BLOCK_SIZE

# Local-frame bounding boxes, measured via probe_base_mesh_extent.py. Each entry:
# body name -> (min_xyz, max_xyz) in that body's OWN link frame.
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


def main():
    task = args.task.split(":")[-1]
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    policy = torch.jit.load(args.policy_path, map_location=args.device); policy.eval()
    scene = env.unwrapped.scene
    robot = scene["robot"]
    device = args.device

    body_ids, body_names = robot.find_bodies(BODY_NAMES, preserve_order=True)
    print(f"[diag] tracked bodies -> {body_names} (ids {body_ids})")
    print(f"[diag] cube half-size = {BLOCK_SIZE/2:.3f}  vel_spike_threshold = {args.vel_spike} m/s\n")

    # Precompute the 8 corners of each body's local bbox, stacked as (B, 8, 3).
    corners_local = []
    for name in body_names:
        mn, mx = BBOXES[name]
        pts = [(x, y, z) for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])]
        corners_local.append(pts)
    corners_local = torch.tensor(corners_local, device=device)  # (B, 8, 3)

    def get_obs(o):
        if isinstance(o, torch.Tensor): raw = o
        else:
            try: raw = o["policy"]
            except (KeyError, TypeError): raw = o
        if raw.shape[-1] == 39:  # strip hand_base_to_object (indices 28:31); see check_hand_base.py
            raw = torch.cat([raw[:, :28], raw[:, 31:]], dim=-1)
        return raw

    def worst_corner_per_body():
        """For each tracked body, transform its 8 local-bbox corners into world
        space, then return the corner with the smallest xy-distance-to-cube AND
        the resulting (xy_dist, clearance) for that corner. Shapes: (N, B)."""
        pos = robot.data.body_pos_w[:, body_ids, :]     # (N, B, 3)
        quat = robot.data.body_quat_w[:, body_ids, :]   # (N, B, 4)
        cube_pos = scene["object"].data.root_pos_w      # (N, 3)
        cube_top_z = cube_pos[:, 2] + BLOCK_SIZE / 2.0

        N, B = pos.shape[0], pos.shape[1]
        # expand for broadcasting: (N, B, 8, 3)
        q = quat.unsqueeze(2).expand(N, B, 8, 4).reshape(N * B * 8, 4)
        c = corners_local.unsqueeze(0).expand(N, B, 8, 3).reshape(N * B * 8, 3)
        world_offsets = quat_apply(q, c).reshape(N, B, 8, 3)
        world_corners = pos.unsqueeze(2) + world_offsets  # (N, B, 8, 3)

        xy = torch.norm(world_corners[..., :2] - cube_pos[:, None, None, :2], dim=-1)  # (N,B,8)
        clear = world_corners[..., 2] - cube_top_z[:, None, None]                      # (N,B,8)

        # pick, per body, the corner that is closest in xy (the most "over the cube" one)
        closest_idx = xy.argmin(dim=-1, keepdim=True)  # (N,B,1)
        best_xy = torch.gather(xy, -1, closest_idx).squeeze(-1)      # (N,B)
        best_clear = torch.gather(clear, -1, closest_idx).squeeze(-1)  # (N,B)
        return best_xy, best_clear

    obs, _ = env.reset()
    n = args.num_envs
    history = [deque(maxlen=args.history) for _ in range(n)]
    already_flagged = torch.zeros(n, dtype=torch.bool, device=device)
    events = 0

    with torch.inference_mode():
        for t in range(args.steps):
            actions = policy(get_obs(obs))
            obs, _, _, _ = env.step(actions)

            cube_vel = torch.norm(scene["object"].data.root_lin_vel_w, dim=-1)
            xy, clear = worst_corner_per_body()  # (N, B) each

            for i in range(n):
                history[i].append((t, xy[i].tolist(), clear[i].tolist()))

            spiking = cube_vel > args.vel_spike
            newly_spiking = spiking & (~already_flagged)
            if newly_spiking.any():
                for i in torch.nonzero(newly_spiking).flatten().tolist():
                    events += 1
                    print(f"\n[EVENT {events}] env {i}  cube_vel={cube_vel[i]:.3f} m/s  at t={t}")
                    header = "  ".join(f"{nm:>18s}" for nm in body_names)
                    print(f"  {'t':>4s}  {header}")
                    for (ht, xy_row, clear_row) in history[i]:
                        cells = []
                        worst_name, worst_score = None, float("inf")
                        for nm, x, c in zip(body_names, xy_row, clear_row):
                            over_and_below = x < 0.02 and c < 0.005
                            score = c if x < 0.03 else float("inf")
                            if score < worst_score:
                                worst_score, worst_name = score, nm
                            mark = "*" if over_and_below else " "
                            cells.append(f"{x:5.3f}/{c:+6.3f}{mark}")
                        row = "  ".join(f"{c:>18s}" for c in cells)
                        tag = f"  <-- closest/lowest: {worst_name}" if worst_score < float("inf") else ""
                        print(f"  {ht:4d}  {row}{tag}")
            already_flagged = spiking

    print("\n" + "=" * 70)
    print(f"Total disturbance events (cube_vel > {args.vel_spike} m/s): {events}")
    print("Each cell is closest-corner xy_dist/clearance for that body's FULL mesh")
    print("extent (not just its origin). '*' = xy<0.02 AND clearance<0.005 (a real")
    print("overlap). 'closest/lowest' names whichever body's nearest corner was")
    print("nearest+lowest among those with xy<0.03.")
    print("=" * 70 + "\n")

    env.close(); simulation_app.close()


if __name__ == "__main__":
    main()
