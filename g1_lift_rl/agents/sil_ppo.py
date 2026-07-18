# g1_lift_rl/agents/sil_ppo.py
"""Self-imitation-learning PPO variant for Policy 3 (place + release).

Why this exists: Policy 3's `release` behaviour (settling the cube at the goal AND
opening the gripper) has only ever been produced once in this project -- 1/16 envs,
in run 2026-07-13_11-49-26 -- and resuming training from that checkpoint lost it
(regressed to exactly 0.0000 by 2026-07-13_13-58-44) instead of reinforcing it. This
matches the known PPO failure mode where a rare, locally-discovered good behaviour is
outweighed by the batch-averaged gradient from the far more common (but locally
sub-optimal) behaviour. Fix: capture (obs, action) pairs from any step where release
fires into a small ring buffer, and add an auxiliary behaviour-cloning loss toward
those actions, independent of the PPO gradient, so a rare success has a standing
chance to be reinforced even while it stays rare in the on-policy rollout batch.

Wired in via RslRlPpoAlgorithmCfg.class_name = "g1_lift_rl.agents.sil_ppo.SILPPO"
(resolve_callable supports fully-qualified project paths, same mechanism already used
for RND/symmetry) -- no fork of rsl_rl needed.

Design choices:
  - SIL_* hyperparameters below are hardcoded constants, not threaded through
    RslRlPpoAlgorithmCfg's @configclass. This is a one-off experimental technique for
    Policy 3 specifically, not a general library feature -- consistent with how this
    project already hardcodes comparable one-off constants directly in rewards.py
    (GRIP_CLOSED_THRESHOLD, K_JOINT, etc.) rather than pushing them through config
    plumbing.
  - The BC step uses its own dedicated Adam optimizer over self.actor's parameters
    only, run *after* the unmodified PPO update -- not mixed into PPO's shared
    actor+critic optimizer, so it can't perturb desired_kl's adaptive
    learning-rate schedule (which is calibrated purely from the PPO surrogate KL).
  - BC target is deterministic MSE against the actor's mean action
    (stochastic_output=False), not a log-prob loss -- avoids touching the
    distribution/entropy machinery PPO's own update already uses.
  - Release detection re-derives reward_release(env) directly from live sim state
    inside process_env_step, instead of trying to decompose the scalar combined
    reward -- reward_release is already a stateless, per-step function of live scene
    state (see rewards.py), so this is exactly what the reward manager itself would
    have just computed for that step, not an approximation.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms import PPO

from ..mdp.rewards import reward_release

# Seed demos captured offline by replaying the one checkpoint that ever produced
# release (2026-07-13_11-49-26, 1/16 envs) -- see
# scratchpad/capture_policy3_release_seed.py. Only 3 (obs, action) pairs exist
# (env 13, steps 131/134/138) because the release window was brief before the
# episode moved on. Loaded into the buffer at construct_algorithm() time so the BC
# loss has real signal from iteration 0, instead of depending on release firing
# again live before the buffer fills.
SIL_SEED_DEMO_PATH = Path(__file__).parent / "sil_seed_demos.pt"


class DemoBuffer:
    """Fixed-capacity ring buffer of (obs, action) pairs captured from steps where
    reward_release fired. Stores a flat obs tensor, not a TensorDict, because
    Policy 3's obs resolves to a single "policy" group shared by actor/critic
    (confirmed empirically: obs.keys() == ['policy'], shape [num_envs, 33] --
    see scratchpad/check_obs_groups.py), so there is no multi-group structure to
    preserve.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str):
        self.capacity = capacity
        self.device = device
        self.obs_buf = torch.zeros(capacity, obs_dim, device=device)
        self.act_buf = torch.zeros(capacity, action_dim, device=device)
        self._next = 0
        self.size = 0

    def add(self, obs: torch.Tensor, actions: torch.Tensor) -> None:
        n = obs.shape[0]
        if n == 0:
            return
        idx = (self._next + torch.arange(n, device=self.device)) % self.capacity
        self.obs_buf[idx] = obs.detach()
        self.act_buf[idx] = actions.detach()
        self._next = (self._next + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.size, (min(batch_size, self.size),), device=self.device)
        return self.obs_buf[idx], self.act_buf[idx]


class SILPPO(PPO):
    """PPO + a small self-imitation auxiliary BC loss on demonstrated release events."""

    SIL_BUFFER_CAPACITY = 2048  # generous relative to how rarely release has fired
    SIL_MIN_BUFFER_SIZE = 3     # matches the 3 seed demos captured offline -- see
                                 # SIL_SEED_DEMO_PATH; was 256 but that was an
                                 # unmeasured guess made before checking how much
                                 # data actually exists (only 3 samples, ever)
    SIL_BATCH_SIZE = 64
    SIL_LOSS_WEIGHT = 0.1       # keep well below the main RL objective
    SIL_BC_LR = 1.0e-4          # conservative, fixed -- gentle nudges only

    def __init__(self, actor, critic, storage, **kwargs) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self._env = None  # set by construct_algorithm() below
        self.demo_buffer: DemoBuffer | None = None  # lazily built on first add()
        self.bc_optimizer = optim.Adam(self.actor.parameters(), lr=self.SIL_BC_LR)

    def _ensure_buffer(self, obs_dim: int, action_dim: int) -> None:
        if self.demo_buffer is None:
            self.demo_buffer = DemoBuffer(self.SIL_BUFFER_CAPACITY, obs_dim, action_dim, self.device)

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        # Must read self.transition.observations/.actions (the pre-step obs/action
        # pair set by act()) *before* the base class call below clears self.transition.
        # The `obs` parameter here is the post-step observation -- not what the
        # action was conditioned on -- so it is not usable as the BC input.
        pre_obs = self.transition.observations
        pre_actions = self.transition.actions
        if pre_obs is not None and pre_actions is not None and self._env is not None:
            released = reward_release(self._env.unwrapped).bool()
            if released.any():
                obs_flat = pre_obs["policy"][released]
                self._ensure_buffer(obs_flat.shape[-1], pre_actions.shape[-1])
                self.demo_buffer.add(obs_flat, pre_actions[released])

        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        loss_dict = super().update()

        if self.demo_buffer is not None and self.demo_buffer.size >= self.SIL_MIN_BUFFER_SIZE:
            obs_batch, act_batch = self.demo_buffer.sample(self.SIL_BATCH_SIZE)
            obs_td = TensorDict({"policy": obs_batch}, batch_size=[obs_batch.shape[0]])
            pred_actions = self.actor(obs_td, stochastic_output=False)
            bc_loss = torch.nn.functional.mse_loss(pred_actions, act_batch)

            self.bc_optimizer.zero_grad()
            (self.SIL_LOSS_WEIGHT * bc_loss).backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.bc_optimizer.step()

            loss_dict["sil_bc"] = bc_loss.item()
        loss_dict["sil_buffer_size"] = float(self.demo_buffer.size) if self.demo_buffer is not None else 0.0

        return loss_dict

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        # PPO.construct_algorithm resolves alg_class from cfg["algorithm"]["class_name"]
        # (== "g1_lift_rl.agents.sil_ppo.SILPPO" for Policy 3), so `alg` below is
        # already a SILPPO instance -- this override only adds capturing `env`,
        # which the base construct_algorithm receives but never stores on self.
        alg = PPO.construct_algorithm(obs, env, cfg, device)
        alg._env = env

        if SIL_SEED_DEMO_PATH.exists():
            seed = torch.load(SIL_SEED_DEMO_PATH, map_location=device)
            obs_seed, act_seed = seed["obs"].to(device), seed["actions"].to(device)
            alg._ensure_buffer(obs_seed.shape[-1], act_seed.shape[-1])
            alg.demo_buffer.add(obs_seed, act_seed)
            print(f"[SILPPO] Seeded demo buffer with {obs_seed.shape[0]} pairs from {SIL_SEED_DEMO_PATH}")

        return alg
