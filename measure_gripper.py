# measure_gripper.py
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

USD_PATH = "/home/virtual-acc/projects/unitree_sim_isaaclab/assets/robots/g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1.usd"


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    g = sim_utils.GroundPlaneCfg(); g.func("/World/ground", g)
    l = sim_utils.DomeLightCfg(intensity=2000.0); l.func("/World/Light", l)

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=200.0, damping=20.0)},
    )
    robot = Articulation(robot_cfg)
    sim.reset()

    grip_ids, _ = robot.find_joints(["right_hand_Joint1_1", "right_hand_Joint2_1"])
    tip_ids, _ = robot.find_bodies(["right_hand_Link1_3", "right_hand_Link2_3"])

    def measure(val, label):
        pos = torch.full((1, len(grip_ids)), float(val), device=sim.device)
        vel = torch.zeros_like(pos)
        robot.write_joint_state_to_sim(pos, vel, joint_ids=grip_ids)
        robot.set_joint_position_target(pos, joint_ids=grip_ids)
        for _ in range(60):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
        tips = robot.data.body_pos_w[:, tip_ids, :]
        gap = torch.norm(tips[:, 0, :] - tips[:, 1, :], dim=-1).item()
        print(f"  {label:18s} fingertip separation = {gap * 100:6.2f} cm")

    print("\n========== GRIPPER OPENING ==========")
    measure(+0.0245, "joint = +0.0245")
    measure(0.0,      "joint =  0.0000")
    measure(-0.0200, "joint = -0.0200")
    simulation_app.close()


if __name__ == "__main__":
    main()