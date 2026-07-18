# unitree_sdk2 — Real-Hardware C++ SDK (vendored infrastructure)

`/home/virtual-acc/projects/unitree_sdk2`. Clean, unmodified vendored clone of `github.com/unitreerobotics/unitree_sdk2`, `main`, no local changes.

## What it is
Official Unitree C++ SDK v2 for DDS-based communication with real hardware: low-level motor control (LowCmd/LowState) plus high-level sport/loco clients. Per-robot examples under `example/` including a `g1/` folder (`audio`, `dex3`, `g1d`, `high_level`, `low_level` subfolders).

## Usage evidence — actually built and installed
`build/` exists with `CMakeCache.txt` (`CMAKE_INSTALL_PREFIX=/opt/unitree_robotics`) and a root-owned `install_manifest.txt` — `sudo make install` was run. `/opt/unitree_robotics/{include,lib}` is populated (`libunitree_sdk2.a`, DDS libs, headers). Not mentioned in `journal.txt`, but filesystem confirms the build happened.

## Relation to other projects
The core DDS transport dependency for both [[unitree_mujoco]] and [[unitree_rl_lab]]'s `deploy/` C++ stack. No references found in any of the custom `g1_*` training projects — expected, since those are pure Isaac Lab Python training code with no hardware deployment layer yet. See [[overview]].
