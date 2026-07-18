import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import g1_lift_rl  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from g1_lift_rl.constants import ARM_JOINTS, PRE_GRASP_ARM_POSE

task = "Isaac-G1-Lift-Ext-Play-v0"
env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
env = gym.make(task, cfg=env_cfg, render_mode=None)
robot = env.unwrapped.scene["robot"]

# Policy 1's actual converged values (from the last diagnostic).
converged = {
    "right_shoulder_pitch_joint": -1.0231,
    "right_shoulder_roll_joint": -0.7229,
    "right_shoulder_yaw_joint": 1.1715,
    "right_elbow_joint": 0.1624,
    "right_wrist_roll_joint": -1.9723,
    "right_wrist_pitch_joint": -0.5185,
    "right_wrist_yaw_joint": -0.0234,
}

for jname in ARM_JOINTS:
    jid, _ = robot.find_joints([jname])
    limits = robot.data.joint_pos_limits[0, jid[0], :].tolist()
    val = converged[jname]
    target = PRE_GRASP_ARM_POSE[jname]
    near_lower = abs(val - limits[0]) < 0.05
    near_upper = abs(val - limits[1]) < 0.05
    flag = "AT LOWER LIMIT" if near_lower else ("AT UPPER LIMIT" if near_upper else "")
    print(f"{jname:32s} limits=[{limits[0]:+.4f},{limits[1]:+.4f}]  converged={val:+.4f}  "
          f"target={target:+.4f}  {flag}", flush=True)

env.close(); simulation_app.close()
