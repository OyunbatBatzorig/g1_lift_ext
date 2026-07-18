#!/usr/bin/env python3
"""Right-arm pose finder -- BATCH MODE.

Instead of one pose per launch, this evaluates a LIST of candidate right-arm poses in
a single run and prints a comparison table sorted by EE-to-cube distance, so you can
see which joint moves which axis and which pose wins -- without relaunching.

Each candidate is held (collision ON) for --settle steps, then we record the resulting
EE (fingertip-midpoint) position, the per-axis error vs the cube, the distance, and the
camera clearance. Between candidates the env is reset so they don't contaminate.

Cube is ~(-0.08, -0.34, 0.84). TARGET: EE just above the cube --
  x ~ cube x, y ~ cube y, z ~ cube z + 0.03, i.e. dist ~0.03-0.08, dz slightly positive.

The candidate list is defined in CANDIDATES below as dicts of joint-flag -> radians:
  sp=shoulder_pitch sr=shoulder_roll sy=shoulder_yaw el=elbow
  wr=wrist_roll wp=wrist_pitch wy=wrist_yaw
Edit CANDIDATES to explore; unspecified joints fall back to READY_ARM_POSE.

Run:  cd ~/projects/g1_lift_ext && python pose_finder.py
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-G1-Lift-Ext-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import g1_lift_rl  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from g1_lift_rl.constants import TABLE_TOP_Z, EE_LINKS, READY_ARM_POSE

FLAG_TO_JOINT = {
    "sp": "right_shoulder_pitch_joint",
    "sr": "right_shoulder_roll_joint",
    "sy": "right_shoulder_yaw_joint",
    "el": "right_elbow_joint",
    "wr": "right_wrist_roll_joint",
    "wp": "right_wrist_pitch_joint",
    "wy": "right_wrist_yaw_joint",
}

# ---------------------------------------------------------------------------
# CANDIDATE POSES TO COMPARE. Edit freely. Each is {flag: radians}.
# This first batch holds sp/sr/el/wp near the good values found so far
# (z and x were already close) and SWEEPS shoulder_yaw (sy) across a range,
# to find which yaw brings EE y from +0.41 toward the cube's y (-0.34).
# A couple also vary sr to disentangle roll vs yaw on the y axis.
# ---------------------------------------------------------------------------
CANDIDATES = [
    {"name": "rest (README pose)", },
    {"name": "fwd sp-0.5",  "sp": -0.5, "sr": -0.2, "sy": 0.0, "el": 0.8, "wp": 0.4},
    {"name": "fwd sp-0.8",  "sp": -0.8, "sr": -0.2, "sy": 0.0, "el": 0.8, "wp": 0.4},
    {"name": "fwd sp-1.1",  "sp": -1.1, "sr": -0.2, "sy": 0.0, "el": 0.8, "wp": 0.4},
    {"name": "fwd sp-0.8 el0.4", "sp": -0.8, "sr": -0.2, "sy": 0.0, "el": 0.4, "wp": 0.4},
    {"name": "fwd sp-0.8 el1.2", "sp": -0.8, "sr": -0.2, "sy": 0.0, "el": 1.2, "wp": 0.4},
    {"name": "fwd sp-0.8 sr-0.5", "sp": -0.8, "sr": -0.5, "sy": 0.0, "el": 0.8, "wp": 0.4},
    {"name": "fwd sp-1.1 el1.2", "sp": -1.1, "sr": -0.2, "sy": 0.0, "el": 1.2, "wp": 0.4},
]


def build_target(robot, cand):
    target = robot.data.default_joint_pos.clone()
    for flag, jname in FLAG_TO_JOINT.items():
        val = cand.get(flag, READY_ARM_POSE.get(jname, 0.0))
        jids, _ = robot.find_joints([jname])
        target[:, jids[0]] = val
    return target


def main():
    task = args.task.split(":")[-1]
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=env_cfg, render_mode=None)
    scene = env.unwrapped.scene
    robot = scene["robot"]
    sim = env.unwrapped.sim
    ee_ids, _ = robot.find_bodies(EE_LINKS)
    cam_ids, _ = robot.find_bodies(["d435_link"])

    results = []
    with torch.inference_mode():
        for cand in CANDIDATES:
            env.reset()
            target = build_target(robot, cand)
            for _ in range(args.settle):
                robot.set_joint_position_target(target)
                scene.write_data_to_sim(); sim.step(); scene.update(env_cfg.sim.dt)
            o = scene.env_origins[0]
            ee = (robot.data.body_pos_w[0, ee_ids, :].mean(dim=0) - o)
            cube = (scene["object"].data.root_pos_w[0] - o)
            cam_z = (robot.data.body_pos_w[0, cam_ids[0], 2] - o[2]).item()
            dx = (ee[0] - cube[0]).item()
            dy = (ee[1] - cube[1]).item()
            dz = (ee[2] - cube[2]).item()
            dist = torch.norm(ee - cube).item()
            results.append({
                "name": cand["name"], "cand": cand,
                "ee": [ee[0].item(), ee[1].item(), ee[2].item()],
                "dx": dx, "dy": dy, "dz": dz, "dist": dist, "cam_z": cam_z,
            })

    # cube is the same across resets; grab last
    print("\n================== POSE COMPARISON (sorted by dist, best first) ==================")
    print(f"  cube ~ ({cube[0].item():+.3f}, {cube[1].item():+.3f}, {cube[2].item():+.3f})   "
          f"table top {TABLE_TOP_Z}   target: dist small, dz ~ +0.03\n")
    print(f"  {'pose':24s} {'EE x':>7s} {'EE y':>7s} {'EE z':>7s} "
          f"{'dx':>7s} {'dy':>7s} {'dz':>7s} {'dist':>7s}")
    for r in sorted(results, key=lambda r: r["dist"]):
        print(f"  {r['name']:24s} {r['ee'][0]:+7.3f} {r['ee'][1]:+7.3f} {r['ee'][2]:+7.3f} "
              f"{r['dx']:+7.3f} {r['dy']:+7.3f} {r['dz']:+7.3f} {r['dist']:7.3f}")
    print("\n  Best pose's joint angles (copy into READY_ARM_POSE if dist is small):")
    best = sorted(results, key=lambda r: r["dist"])[0]
    for flag, jname in FLAG_TO_JOINT.items():
        val = best["cand"].get(flag, READY_ARM_POSE.get(jname, 0.0))
        print(f"    {jname:30s} = {val:+.3f}")
    print("==================================================================================\n")
    env.close(); simulation_app.close()


if __name__ == "__main__":
    main()