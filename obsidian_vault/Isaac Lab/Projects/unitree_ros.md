# unitree_ros — ROS/Gazebo Packages & URDFs (vendored infrastructure)

`/home/virtual-acc/projects/unitree_ros`. Clean, unmodified vendored clone of `github.com/unitreerobotics/unitree_ros`, `master`, no local changes.

## What it is
ROS/Gazebo simulation stack for low-level joint control (not high-level walking), plus URDF robot descriptions for the full Unitree lineup. `robots/g1_description/` has extensive G1 URDF variants (29dof, 23dof, with dex1_1/inspire hands, lock-waist) plus MJCF `.xml` files.

## Usage evidence — none found
No `catkin_ws`, build artifacts, or ROS environment sourcing anywhere on the system; no mentions in `journal.txt`. Appears purely vendored for reference/potential asset source, never actually run.

## Relation to other projects
Per the outer workspace `CLAUDE.md`, this is the `UNITREE_ROS_DIR` URDF source [[unitree_rl_lab]]'s `assets/robots/unitree.py` can point at (alternative to `UNITREE_MODEL_DIR`'s USD assets). None of the custom `g1_*` training projects reference it or any `.urdf` file — they all use USD assets from `unitree_sim_isaaclab`/`unitree_model` instead. See [[overview]].
