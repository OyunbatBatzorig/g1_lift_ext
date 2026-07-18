#!/usr/bin/env python3
"""Does the mounting plate (right_hand_base_link) actually CAUSE the violent
contact events, or is that just a plausible-looking correlation?

The first version of this script only reported periodic snapshots and a global
"worst clearance," which was dominated by a meaningless t=0 reading (base far from
the cube, so "clearance" there is irrelevant) and couldn't tell us what was
happening AT THE MOMENT of a violent event, since disturbances are transient and
periodic (every-10-steps) sampling can miss them entirely.

This version is event-triggered: every step, per env, it tracks cube linear
velocity (same signal penalty_contact_disturbance/diag_grasp's contact check use)
AND both the mounting plate's and the fingertips' xy-distance/clearance relative
to the cube. Whenever an env's cube velocity spikes above VEL_SPIKE_THRESHOLD
(a real disturbance, not settling jitter), it prints that env's plate/fingertip
geometry for the steps immediately BEFORE and AT the spike -- directly answering
whether the plate was over-and-below the cube right when the disturbance
happened, or whether something else (e.g. the fingertips themselves) was the
proximate cause instead.

Run:
    cd ~/projects/g1_lift_ext
    python check_hand_base.py --num_envs 32 --policy_path <exported policy.pt>
"""
import argparse
from collections import deque
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-Lift-Ext-Play-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--policy_path", type=str, default=None)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--vel_spike", type=float, default=0.5,
                     help="cube linear speed (m/s) that counts as a real disturbance event")
parser.add_argument("--history", type=int, default=5,
                     help="how many preceding steps of geometry to show per event")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import g1_lift_rl  # noqa
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from g1_lift_rl.constants import EE_LINKS, BLOCK_SIZE


def main():
    task = args.task.split(":")[-1]
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    policy = torch.jit.load(args.policy_path, map_location=args.device); policy.eval()
    scene = env.unwrapped.scene
    robot = scene["robot"]

    ee_ids, ee_names = robot.find_bodies(EE_LINKS, preserve_order=True)
    base_ids, base_names = robot.find_bodies(["right_hand_base_link"], preserve_order=True)
    # ALL hand bodies, not just the two distal fingertips + base -- the first two
    # rounds of this check found neither of those overlapping the cube right
    # before a spike, so this checks every remaining candidate (proximal/middle
    # finger segments, wrist) in one pass instead of guessing which one next.
    ALL_HAND_NAMES = [
        "right_hand_base_link",
        "right_hand_Link1_1", "right_hand_Link1_2", "right_hand_Link1_3",
        "right_hand_Link2_1", "right_hand_Link2_2", "right_hand_Link2_3",
        "right_wrist_yaw_link",
    ]
    all_ids, all_names = robot.find_bodies(ALL_HAND_NAMES, preserve_order=True)
    print(f"[diag] EE links -> {ee_names} (ids {ee_ids})")
    print(f"[diag] base link -> {base_names} (ids {base_ids})")
    print(f"[diag] all hand bodies -> {all_names} (ids {all_ids})")
    print(f"[diag] cube half-size = {BLOCK_SIZE/2:.3f}  vel_spike_threshold = {args.vel_spike} m/s\n")

    def get_obs(o):
        if isinstance(o, torch.Tensor): raw = o
        else:
            try: raw = o["policy"]
            except (KeyError, TypeError): raw = o
        # The live env now emits 39-dim obs (hand_base_to_object was added AFTER
        # this checkpoint was trained on 36-dim). Strip that term back out before
        # feeding the OLD network -- this diagnostic reads hand-base/fingertip
        # geometry directly from robot.data regardless, so the policy's own input
        # doesn't need to include it; we just need the policy to actually run so
        # the arm moves the way it was really trained to.
        if raw.shape[-1] == 39:
            raw = torch.cat([raw[:, :28], raw[:, 31:]], dim=-1)
        return raw

    def all_body_geometry():
        """xy-dist and clearance-above-cube-top for EVERY tracked hand body at
        once. Returns (xy, clear) each shaped (num_envs, num_bodies)."""
        pos = robot.data.body_pos_w[:, all_ids, :]              # (N, B, 3)
        cube_pos = scene["object"].data.root_pos_w               # (N, 3)
        cube_top_z = cube_pos[:, 2] + BLOCK_SIZE / 2.0
        xy = torch.norm(pos[:, :, :2] - cube_pos[:, None, :2], dim=-1)      # (N, B)
        clear = pos[:, :, 2] - cube_top_z[:, None]                          # (N, B)
        return xy, clear

    obs, _ = env.reset()
    n = args.num_envs
    history = [deque(maxlen=args.history) for _ in range(n)]
    already_flagged = torch.zeros(n, dtype=torch.bool, device=args.device)  # avoid spamming one long event repeatedly
    events = 0

    with torch.inference_mode():
        for t in range(args.steps):
            actions = policy(get_obs(obs))
            obs, _, _, _ = env.step(actions)

            cube_vel = torch.norm(scene["object"].data.root_lin_vel_w, dim=-1)
            xy, clear = all_body_geometry()  # (N, B) each

            for i in range(n):
                history[i].append((t, xy[i].tolist(), clear[i].tolist()))

            spiking = cube_vel > args.vel_spike
            newly_spiking = spiking & (~already_flagged)
            if newly_spiking.any():
                for i in torch.nonzero(newly_spiking).flatten().tolist():
                    events += 1
                    print(f"\n[EVENT {events}] env {i}  cube_vel={cube_vel[i]:.3f} m/s  at t={t}")
                    header = "  ".join(f"{nm:>14s}" for nm in all_names)
                    print(f"  {'t':>4s}  {header}")
                    for (ht, xy_row, clear_row) in history[i]:
                        cells = []
                        worst_name, worst_score = None, float("inf")
                        for nm, x, c in zip(all_names, xy_row, clear_row):
                            over_and_below = x < 0.03 and c < 0.01
                            score = c if x < 0.05 else float("inf")  # only compete if roughly over the cube
                            if score < worst_score:
                                worst_score, worst_name = score, nm
                            mark = "*" if over_and_below else " "
                            cells.append(f"{x:5.3f}/{c:+6.3f}{mark}")
                        row = "  ".join(f"{c:>14s}" for c in cells)
                        tag = f"  <-- closest/lowest: {worst_name}" if worst_score < float("inf") else ""
                        print(f"  {ht:4d}  {row}{tag}")
            already_flagged = spiking  # reset the "newly" edge once it drops back down

    print("\n" + "=" * 70)
    print(f"Total disturbance events (cube_vel > {args.vel_spike} m/s): {events}")
    print("Each cell is xy_dist/clearance -- '*' marks xy<0.03 AND clearance<0.01")
    print("(over-and-below the cube). 'closest/lowest' names whichever tracked")
    print("body was nearest+lowest (among those with xy<0.05) at that step.")
    print("=" * 70 + "\n")

    env.close(); simulation_app.close()


if __name__ == "__main__":
    main()
