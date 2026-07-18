# g1_lift_rl/__init__.py
"""G1 + Dex1 lift (Policy 1) -- RL task registration.

Entry points are forward references (strings) -- env_cfg.G1LiftEnvCfg and
agents.rsl_rl_ppo_cfg.G1LiftPPORunnerCfg don't exist yet (Phase 1 / Phase 3), and
don't need to: gym.register() never resolves them until gym.make() is called.
"""

import gymnasium as gym

gym.register(
    id="Isaac-G1-Lift-Ext-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1LiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1LiftPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Lift-Ext-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1LiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1LiftPPORunnerCfg",
    },
)

# Policy 2: grasp (verified) + carry to inspection.
gym.register(
    id="Isaac-G1-Policy2-Ext-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_policy2:G1Policy2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1Policy2PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Policy2-Ext-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_policy2:G1Policy2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1Policy2PPORunnerCfg",
    },
)

# Policy 3: move to goal + place (release + return_to_ready split off into
# Policy 4 below -- see env_cfg_policy3.py's RewardsCfg docstring).
gym.register(
    id="Isaac-G1-Policy3-Ext-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_policy3:G1Policy3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1Policy3PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Policy3-Ext-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_policy3:G1Policy3EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1Policy3PPORunnerCfg",
    },
)

# Policy 4: release + return to ready.
gym.register(
    id="Isaac-G1-Policy4-Ext-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_policy4:G1Policy4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1Policy4PPORunnerCfg",
    },
)

gym.register(
    id="Isaac-G1-Policy4-Ext-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_policy4:G1Policy4EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1Policy4PPORunnerCfg",
    },
)
