"""Does Policy 1's ACTUAL converged final state match PRE_GRASP_ARM_POSE
(Policy 2's reset target, CEM/teleop-derived, never directly measured against
Policy 1's own trained behavior)? Same class of question already answered for
Policy 2 vs Policy 3 -- that comparison found the cube position matched well
but the ARM CONFIGURATION diverged substantially. Checking the same here.
"""
import argparse
from isaaclab.app import AppLauncher

RESULT_FILE = "policy1_final_vs_pregrasp_result.txt"  # writes next to wherever this script is run from

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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from g1_lift_rl.mdp.rewards import _ee_pos_w, _cube_pos_w
from g1_lift_rl.constants import ARM_JOINTS, PRE_GRASP_ARM_POSE, GRASP_OFFSET

f = open(RESULT_FILE, "w")
def log(msg=""):
    print(msg, flush=True)
    f.write(str(msg) + "\n")
    f.flush()

task = "Isaac-G1-Lift-Ext-Play-v0"
env_cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
env = gym.make(task, cfg=env_cfg, render_mode=None)
env = RslRlVecEnvWrapper(env)
policy = torch.jit.load(args.policy_path, map_location=args.device); policy.eval()
raw_env = env.unwrapped
robot = raw_env.scene["robot"]
device = args.device
n = args.num_envs

def get_obs(o):
    if isinstance(o, torch.Tensor): raw = o
    try: raw = o["policy"]
    except (KeyError, TypeError): raw = o
    return raw

obs, _ = env.reset()
with torch.inference_mode():
    for t in range(args.steps):
        actions = policy(get_obs(obs))
        obs, _, _, _ = env.step(actions)

arm_ids = [robot.find_joints([n_])[0][0] for n_ in ARM_JOINTS]
final_joint_pos = robot.data.joint_pos[:, arm_ids]
final_ee = _ee_pos_w(raw_env)
final_cube = _cube_pos_w(raw_env) - raw_env.scene.env_origins

log(f"Policy 1 checkpoint: {args.policy_path}")
log(f"Ran {args.steps} steps across {n} envs -- reading final state.\n")

log("=" * 78)
log("PER-JOINT: Policy 1's actual converged state vs PRE_GRASP_ARM_POSE")
log("=" * 78)
for i, jname in enumerate(ARM_JOINTS):
    p1_mean = final_joint_pos[:, i].mean().item()
    p1_std = final_joint_pos[:, i].std().item()
    p2_val = PRE_GRASP_ARM_POSE[jname]
    diff = p1_mean - p2_val
    log(f"  {jname:32s}  Policy1={p1_mean:+.4f} (std={p1_std:.4f})  "
        f"PRE_GRASP_ARM_POSE={p2_val:+.4f}  diff={diff:+.4f}")

log("\n" + "=" * 78)
log("EE / CUBE POSITION")
log("=" * 78)
log(f"  Policy 1 final EE pos (mean): {final_ee.mean(dim=0).tolist()}")
log(f"  Policy 1 final cube pos (env-local, mean): {final_cube.mean(dim=0).tolist()}")
grasp_offset = torch.tensor(GRASP_OFFSET, device=device)
expected_ee_if_pregrasp_correct = final_cube.mean(dim=0) + grasp_offset
log(f"  If PRE_GRASP_ARM_POSE's own geometry (cube + GRASP_OFFSET) held, EE should be: "
    f"{expected_ee_if_pregrasp_correct.tolist()}")
ee_diff = final_ee.mean(dim=0) - expected_ee_if_pregrasp_correct
log(f"  EE diff (actual - expected): {ee_diff.tolist()}  magnitude={torch.norm(ee_diff).item():.4f} m")

f.close()
env.close(); simulation_app.close()
