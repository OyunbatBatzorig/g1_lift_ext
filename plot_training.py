#!/usr/bin/env python3
"""Plot the training curves that actually matter, as clean labeled line graphs.

Reads the TensorBoard event file that rsl_rl writes (RslRlOnPolicyRunnerCfg.logger
defaults to "tensorboard" -- G1LiftPPORunnerCfg never overrides it, so this is
already being written, no training-config change needed), and draws three stacked
panels:
  1. mean total reward        -> the headline: is learning happening at all
  2. the reward-term ladder   -> the diagnostic: which skill is being learned
  3. mean episode length      -> stability / how long the cube stays in play

Usage:
    python plot_training.py                       # auto-find the latest Policy 1 run
    python plot_training.py --policy 2             # latest Policy 2 run
    python plot_training.py --policy 3 --watch     # Policy 3, redraw every 30s
    python plot_training.py --logdir <run_dir>    # a specific run folder

Needs: matplotlib, tensorboard  (pip install matplotlib if missing)
"""
import argparse
import glob
import os
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Matches each PPORunnerCfg.experiment_name in agents/rsl_rl_ppo_cfg.py; training is
# run from ~/projects/g1_lift_ext (logs write relative to launch dir).
LOGS_ROOT = os.path.expanduser("~/projects/g1_lift_ext/logs/rsl_rl")
POLICY_DIRS = {
    "1": os.path.join(LOGS_ROOT, "g1_lift_policy1"),
    "2": os.path.join(LOGS_ROOT, "g1_policy2_grasp_carry"),
    "3": os.path.join(LOGS_ROOT, "g1_policy3_place_release"),
    "4": os.path.join(LOGS_ROOT, "g1_policy4_release_return"),
}
DEFAULT_BASE = POLICY_DIRS["1"]

# preferred display order for the reward ladder (others appended alphabetically).
# Matches each env_cfg*.py RewardsCfg term names; unrecognized terms still show up,
# just sorted after these by the fallback in term_key().
LADDER_ORDER = [
    "reach", "grasp", "lift", "holding", "inspect", "inspect_bonus", "close_grip",
    "descend", "close_gradient", "early_close", "contact_disturbance",
    "move_to_goal", "place", "release", "return_to_ready",
]


def latest_run(base):
    runs = [d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)]
    return max(runs, key=os.path.getmtime) if runs else None


def load_scalars(run_dir):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    out = {}
    for t in tags:
        ev = ea.Scalars(t)
        out[t] = ([e.step for e in ev], [e.value for e in ev])
    return out


def pick(tags, *needles):
    for t in tags:
        low = t.lower()
        if all(n in low for n in needles):
            return t
    return None


def plot(run_dir, save_path):
    data = load_scalars(run_dir)
    if not data:
        print(f"No scalar data yet in {run_dir} (training may have just started).")
        return
    tags = list(data.keys())

    reward_tag = pick(tags, "mean_reward") or pick(tags, "mean", "reward")
    eplen_tag = pick(tags, "mean_episode_length") or pick(tags, "episode_length")
    term_tags = [t for t in tags if t.lower().startswith("episode_reward")]

    def term_key(t):
        name = t.split("/")[-1]
        return (LADDER_ORDER.index(name) if name in LADDER_ORDER else 99, name)

    term_tags.sort(key=term_key)

    experiment_name = os.path.basename(os.path.dirname(run_dir))
    policy_label = {v: f"Policy {k}" for k, v in POLICY_DIRS.items()}.get(
        os.path.join(LOGS_ROOT, experiment_name), experiment_name
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f"G1 lift training ({policy_label})\n{os.path.basename(run_dir)}", fontsize=13)

    # 1. headline reward
    ax = axes[0]
    if reward_tag:
        x, y = data[reward_tag]
        ax.plot(x, y, color="tab:blue", lw=2)
    ax.set_title("Mean total reward  (should trend up)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("reward")
    ax.grid(True, alpha=0.3)

    # 2. reward ladder
    ax = axes[1]
    for t in term_tags:
        x, y = data[t]
        ax.plot(x, y, lw=1.8, label=t.split("/")[-1])
    ax.set_title("Reward terms  (watch 'lift' / 'holding' rise off zero)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("per-episode term reward")
    ax.grid(True, alpha=0.3)
    if term_tags:
        ax.legend(loc="upper left", fontsize=9, ncol=2)

    # 3. episode length
    ax = axes[2]
    if eplen_tag:
        x, y = data[eplen_tag]
        ax.plot(x, y, color="tab:green", lw=2)
    ax.set_title("Mean episode length  (drops = cube falling early)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("steps")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=110)
    plt.close(fig)
    print(f"saved {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default=None, help="specific run folder")
    p.add_argument("--base", default=None, help="folder holding all runs for one policy")
    p.add_argument("--policy", choices=["1", "2", "3", "4"], default=None,
                    help="shortcut for --base, e.g. --policy 2")
    p.add_argument("--watch", action="store_true", help="redraw every 30s")
    args = p.parse_args()

    base = args.base or (POLICY_DIRS[args.policy] if args.policy else DEFAULT_BASE)
    run_dir = args.logdir or latest_run(base)
    if not run_dir:
        print(f"No runs found under {base}. Start training first.")
        return
    save_path = os.path.join(run_dir, "training_curves.png")

    if args.watch:
        print("watching — Ctrl+C to stop")
        try:
            while True:
                plot(run_dir, save_path)
                time.sleep(30)
        except KeyboardInterrupt:
            print("stopped")
    else:
        plot(run_dir, save_path)


if __name__ == "__main__":
    main()
