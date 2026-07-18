#!/usr/bin/env python3
"""Interactive right-arm teleop with gripper control, auto-reset on drop, and
LABELED 3-POINT recording (pre-grasp, inspection, target/place) with per-finger
contact-geometry diagnostics.

Keyboard control of the 7 right-arm joints PLUS gripper open/close, live EE/cube
readout after every press. Physics + collision ON. If the cube falls below 0.50m
(dropped off table), scene auto-resets.

RUN (keep the TERMINAL focused):
  cd ~/projects/g1_lift_ext
  python arm_teleop.py

KEYS (lowercase = increase, uppercase-row key = decrease):
  q / a : shoulder pitch  +/-        r / f : elbow        +/-
  w / s : shoulder roll   +/-        t / g : wrist roll   +/-
  e / d : shoulder yaw    +/-        y / h : wrist pitch  +/-
                                     u / j : wrist yaw    +/-
  c : close gripper (move to +0.0245)
  o : open gripper (move to -0.0200)

  [ / ] : step size halve / double   (default 0.1 rad)
  p     : print current pose + contact-geometry readout (NOT saved)
  1     : RECORD this pose as "pre_grasp"
  2     : RECORD this pose as "inspection"
  3     : RECORD this pose as "target"      (no real cube there -- reports the
          ESTIMATED held-cube position, i.e. EE_mid - verified grasp offset)
  x     : quit -- prints a summary table of every recorded point

CONTACT-GEOMETRY READOUT (new): reports each fingertip (Link1_3, Link2_3)
individually, not just their midpoint, plus a "straddle" check -- is the cube's
projection between the two fingertips along the closing axis (t in [0,1]), and
how far off that line is it (perp_dist)? Low perp_dist + t near 0.5 = a clean,
centered, face-on grip. t outside [0,1] or a large perp_dist = an off-centre
approach, consistent with catching an edge/corner rather than straddling a face
-- exactly the geometry that produces violent, unstable contact resolution,
independent of actuator stiffness or depenetration caps.
"""
import argparse
import sys, select, termios, tty
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-Lift-Ext-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import g1_lift_rl  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from g1_lift_rl.constants import (
    TABLE_TOP_Z, EE_LINKS, READY_ARM_POSE, GRIPPER_JOINTS, GRIPPER_OPEN,
    GRIPPER_CLOSE, BLOCK_SIZE,
)

ARM = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
KEYMAP = {  # key -> (joint index, direction)
    "q": (0, +1), "a": (0, -1),
    "w": (1, +1), "s": (1, -1),
    "e": (2, +1), "d": (2, -1),
    "r": (3, +1), "f": (3, -1),
    "t": (4, +1), "g": (4, -1),
    "y": (5, +1), "h": (5, -1),
    "u": (6, +1), "j": (6, -1),
}
GRIPPER_CLOSE_VAL = GRIPPER_CLOSE  # +0.0245
GRIPPER_OPEN_VAL = GRIPPER_OPEN    # -0.0200
DROP_HEIGHT_THRESHOLD = 0.50  # if cube z < this, reset scene

# Verified via earlier teleop sessions (mirrors mdp/rewards.py:reward_descend's
# grasp_offset -- keep in sync if that ever changes). Used ONLY to estimate where
# a held cube would be for the "target" point, where no real cube is present.
VERIFIED_GRASP_OFFSET = (-0.010, +0.027, +0.012)

RECORD_LABELS = {"1": "pre_grasp", "2": "inspection", "3": "target"}


def main():
    task = args.task.split(":")[-1]
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=env_cfg, render_mode=None)
    scene = env.unwrapped.scene
    robot = scene["robot"]
    sim = env.unwrapped.sim
    # preserve_order=True: EE_LINKS[0]="right_hand_Link1_3" must map to ee_ids[0],
    # EE_LINKS[1]="right_hand_Link2_3" to ee_ids[1] -- find_bodies defaults to
    # natural body-index order (preserve_order=False), which only coincidentally
    # matched EE_LINKS' listed order before. Explicit now since finger IDENTITY
    # (not just their average) matters for the straddle geometry below.
    ee_ids, ee_names = robot.find_bodies(EE_LINKS, preserve_order=True)
    arm_jids = [robot.find_joints([n])[0][0] for n in ARM]
    grip_jids, _ = robot.find_joints(GRIPPER_JOINTS)
    jlim = robot.data.joint_pos_limits[0]  # (J, 2) lower/upper

    # current commanded angles: start from READY_ARM_POSE
    angles = [READY_ARM_POSE.get(n, 0.0) for n in ARM]
    grip_cmd = GRIPPER_OPEN_VAL  # start with gripper open
    step = 0.1
    records = {}  # label -> readout dict, filled by keys 1/2/3

    old_attrs = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    def fingertip_positions():
        """Individual (NOT averaged) env-local fingertip positions.
        ee_ids[0] = right_hand_Link1_3 (finger 1), ee_ids[1] = right_hand_Link2_3 (finger 2)."""
        o = scene.env_origins[0]
        p1 = robot.data.body_pos_w[0, ee_ids[0], :] - o
        p2 = robot.data.body_pos_w[0, ee_ids[1], :] - o
        return p1, p2

    def straddle_geometry(p1, p2, cube):
        """Where does the cube sit relative to the line connecting the two
        fingertips? t in [0,1] = between them along the closing axis; perp_dist =
        how far off that line (small = clean face-on contact; large = off to the
        side -- an edge/corner-catching approach)."""
        axis = p2 - p1
        axis_len = torch.norm(axis).item()
        denom = torch.dot(axis, axis)
        t = (torch.dot(cube - p1, axis) / denom).item() if denom > 1e-9 else float("nan")
        perp = (cube - p1) - t * axis
        perp_dist = torch.norm(perp).item()
        return axis_len, t, perp_dist

    def fingertip_velocities():
        """Individual fingertip world linear velocities (m/s)."""
        v1 = robot.data.body_lin_vel_w[0, ee_ids[0], :]
        v2 = robot.data.body_lin_vel_w[0, ee_ids[1], :]
        return v1, v2

    def gather_readout(label=None):
        p1, p2 = fingertip_positions()
        ee_mid = (p1 + p2) / 2.0
        cube = scene["object"].data.root_pos_w[0] - scene.env_origins[0]
        d_mid = torch.norm(ee_mid - cube).item()
        d1 = torch.norm(p1 - cube).item()
        d2 = torch.norm(p2 - cube).item()
        axis_len, t, perp_dist = straddle_geometry(p1, p2, cube)
        offset = torch.tensor(VERIFIED_GRASP_OFFSET, device=ee_mid.device)
        est_cube_if_held = ee_mid - offset
        # velocity-matching: is the cube moving WITH the hand (real coupling) or
        # independently (not actually held)? -- see VEL_MATCH_THRESHOLD discussion.
        v1, v2 = fingertip_velocities()
        ee_vel = (v1 + v2) / 2.0
        cube_vel = scene["object"].data.root_lin_vel_w[0]
        rel_speed = torch.norm(ee_vel - cube_vel).item()
        ee_speed = torch.norm(ee_vel).item()
        cube_speed = torch.norm(cube_vel).item()
        return {
            "label": label,
            "angles": list(angles),
            "grip_cmd": grip_cmd,
            "ee_mid": ee_mid.tolist(),
            "fingertip1": p1.tolist(),
            "fingertip2": p2.tolist(),
            "cube": cube.tolist(),
            "dist_mid": d_mid,
            "dist_f1": d1,
            "dist_f2": d2,
            "finger_span": axis_len,
            "straddle_t": t,
            "straddle_perp_dist": perp_dist,
            "est_cube_if_held": est_cube_if_held.tolist(),
            "ee_speed": ee_speed,
            "cube_speed": cube_speed,
            "rel_speed": rel_speed,
        }

    def status():
        r = gather_readout()
        ang = " ".join(f"{a:+.2f}" for a in angles)
        grip_str = "CLOSED" if grip_cmd > 0.0 else "OPEN  "
        print(f"\r  EE({r['ee_mid'][0]:+.3f},{r['ee_mid'][1]:+.3f},{r['ee_mid'][2]:+.3f})"
              f" cube({r['cube'][0]:+.3f},{r['cube'][1]:+.3f},{r['cube'][2]:+.3f})"
              f" dist {r['dist_mid']:.3f} grip {grip_str}"
              f" t={r['straddle_t']:+.2f} perp={r['straddle_perp_dist']:.3f}"
              f" |ee_v|={r['ee_speed']:.3f} |cube_v|={r['cube_speed']:.3f} REL={r['rel_speed']:.3f}"
              f" step {step:.3f} | {ang}   ", end="", flush=True)

    def print_readout(r, header):
        print(f"\n\n===== {header} =====")
        print("READY_ARM_POSE = {")
        for n, a in zip(ARM, r["angles"]):
            print(f'    "{n}": {a:+.3f},')
        print("}")
        print(f"Gripper state:          {'CLOSED' if r['grip_cmd'] > 0.0 else 'OPEN'}")
        print(f"EE midpoint position:   ({r['ee_mid'][0]:+.3f}, {r['ee_mid'][1]:+.3f}, {r['ee_mid'][2]:+.3f})")
        print(f"Fingertip 1 (Link1_3):  ({r['fingertip1'][0]:+.3f}, {r['fingertip1'][1]:+.3f}, {r['fingertip1'][2]:+.3f})   dist-to-cube {r['dist_f1']:.3f}")
        print(f"Fingertip 2 (Link2_3):  ({r['fingertip2'][0]:+.3f}, {r['fingertip2'][1]:+.3f}, {r['fingertip2'][2]:+.3f})   dist-to-cube {r['dist_f2']:.3f}")
        print(f"Finger span (open width): {r['finger_span']:.3f} m")
        print(f"Cube position (actual): ({r['cube'][0]:+.3f}, {r['cube'][1]:+.3f}, {r['cube'][2]:+.3f})")
        print(f"EE-to-cube dist (mid):  {r['dist_mid']:.3f}")
        print(f"Straddle t (0..1 = between the two fingertips): {r['straddle_t']:+.3f}")
        print(f"Straddle perp-dist (0 = on the finger-to-finger line): {r['straddle_perp_dist']:.3f}"
              f"  (cube half-size = {BLOCK_SIZE/2:.3f})")
        print(f"  -> t in [0,1] AND perp_dist < ~{BLOCK_SIZE/2:.3f} = clean, centred, face-on straddle.")
        print(f"  -> t outside [0,1], or perp_dist large = off-centre; likely catching an edge/corner.")
        print(f"Estimated cube-if-held: ({r['est_cube_if_held'][0]:+.3f}, {r['est_cube_if_held'][1]:+.3f}, {r['est_cube_if_held'][2]:+.3f})"
              f"  (EE_mid - verified grasp offset {VERIFIED_GRASP_OFFSET})")
        print(f"EE speed:   {r['ee_speed']:.3f} m/s")
        print(f"Cube speed: {r['cube_speed']:.3f} m/s")
        print(f"REL speed (|ee_vel - cube_vel|): {r['rel_speed']:.3f} m/s"
              f"  -- low + gripper CLOSED + moving = real coupling (a genuine hold).")
        print()

    def check_cube_dropped():
        """Check if cube fell off table; reset if so."""
        cube_z = scene["object"].data.root_pos_w[0, 2].item()
        if cube_z < DROP_HEIGHT_THRESHOLD:
            print("\n[DROP DETECTED] Resetting scene...")
            env.reset()
            return True
        return False

    # Tracks the relative-speed range observed WHILE the gripper is closed and the
    # cube is actually moving (ee_speed above a tiny noise floor) -- i.e. during a
    # genuine carry attempt, not while sitting still (where rel_speed is trivially
    # ~0 for both a real and a fake grasp, telling us nothing). Reported at the end
    # so you don't have to read/note live numbers during the whole test.
    carry_rel_speed_samples = []
    MOVING_FLOOR = 0.01  # m/s -- ee must be genuinely moving for a sample to count

    print(__doc__)
    env.reset()
    try:
        with torch.inference_mode():
            tick = 0
            while simulation_app.is_running():
                # apply current command
                target = robot.data.default_joint_pos.clone()
                for jid, a in zip(arm_jids, angles):
                    lo, hi = jlim[jid, 0].item(), jlim[jid, 1].item()
                    target[:, jid] = max(lo, min(hi, a))
                # set gripper
                for jid in grip_jids:
                    target[:, jid] = grip_cmd
                robot.set_joint_position_target(target)
                scene.write_data_to_sim()
                sim.step()
                scene.update(env_cfg.sim.dt)

                # check for drop
                if check_cube_dropped():
                    continue

                # sample relative speed during any genuine closed-gripper motion
                if grip_cmd > 0.0:
                    r_now = gather_readout()
                    if r_now["ee_speed"] > MOVING_FLOOR:
                        carry_rel_speed_samples.append(r_now["rel_speed"])

                # non-blocking key read
                if select.select([sys.stdin], [], [], 0)[0]:
                    k = sys.stdin.read(1)
                    if k == "x":
                        break
                    elif k == "[":
                        step = max(0.0125, step / 2)
                    elif k == "]":
                        step = min(0.4, step * 2)
                    elif k == "p":
                        print_readout(gather_readout(), "CURRENT (not saved)")
                    elif k in RECORD_LABELS:
                        label = RECORD_LABELS[k]
                        r = gather_readout(label)
                        records[label] = r
                        print_readout(r, f"RECORDED as '{label}'")
                    elif k == "c":
                        grip_cmd = GRIPPER_CLOSE_VAL
                    elif k == "o":
                        grip_cmd = GRIPPER_OPEN_VAL
                    elif k in KEYMAP:
                        idx, sgn = KEYMAP[k]
                        angles[idx] += sgn * step
                tick += 1
                if tick % 10 == 0:
                    status()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)
        print()
        if records:
            print("\n" + "=" * 70)
            print("SESSION SUMMARY -- all recorded points")
            print("=" * 70)
            for label in ("pre_grasp", "inspection", "target"):
                if label in records:
                    print_readout(records[label], f"RECORDED: {label}")
        else:
            print_readout(gather_readout(), "FINAL (nothing recorded with 1/2/3)")

        print("=" * 70)
        print("CARRY TEST -- relative speed while gripper CLOSED and hand moving")
        print("=" * 70)
        if carry_rel_speed_samples:
            n = len(carry_rel_speed_samples)
            lo, hi = min(carry_rel_speed_samples), max(carry_rel_speed_samples)
            mean = sum(carry_rel_speed_samples) / n
            print(f"  samples: {n}   min={lo:.4f}  mean={mean:.4f}  max={hi:.4f}  (m/s)")
            print("  -> low and stable throughout = cube genuinely followed the hand (real grasp).")
            print("  -> large or growing values = cube lagged/slipped/separated (not really held).")
        else:
            print("  no samples -- gripper was never CLOSED while the hand was moving.")
            print("  (close the gripper at pre_grasp, then move the arm slowly, to run this test)")
        print("=" * 70 + "\n")

        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
