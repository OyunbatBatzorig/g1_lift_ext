# g1_lift_rl/agents/rsl_rl_ppo_cfg.py
"""PPO runner config for Phase 3 -- Policy 1 (lift), fresh training from scratch.

Static only: not wired into env_cfg, not run. gamma/lam come from the agreed MDP
spec; entropy_coef is raised above the library default to keep exploration alive
and avoid premature policy collapse while the grasp behavior is still forming.
"""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class G1LiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    resume = False
    save_interval = 50
    experiment_name = "g1_lift_policy1"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,        # lowered from 0.008: that value grew action_std
                                   # from 0.76->1.06 over iterations 635->913 for
                                   # Policy 1's narrower task (hover+descend only,
                                   # no grasp/lift payoff to counterbalance ongoing
                                   # exploration pressure) -- entropy pressure was
                                   # winning over task reward instead of the policy
                                   # converging, showing up as worsening dropped/
                                   # launched/contact_disturbance over training.
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",       # self-corrects lr against desired_kl
        gamma=0.99,                # from the MDP spec: long-horizon, values the held hold
        lam=0.95,                  # from the MDP spec: GAE bias/variance trade
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1Policy2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Policy 2 (grasp + carry to inspection), fresh training from scratch."""
    num_steps_per_env = 24
    # Reduced from 1500 -- the reset now couples the cube's position to the
    # arm's actual post-reset EE position (see env_cfg_policy2.py's
    # reset_robot_then_couple_cube), giving 100% aligned_frac and 100% within-
    # GRASP_DIST at reset instead of the ~55-60% typical of an independently-
    # jittered cube -- expected to need less exploration to find the
    # descend/grasp reward signal, so trying a shorter run first.
    # Reduced again, 1200 -> 800: observed across multiple prior runs that
    # reward curves stabilize by ~iteration 400, so 1200 was spending its back
    # half mostly flat -- 800 keeps a 2x margin past the observed
    # stabilization point instead of ~3x.
    max_iterations = 800
    resume = False
    save_interval = 50
    experiment_name = "g1_policy2_grasp_carry"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Starting at the same conservative value that fixed Policy 1's
        # entropy-vs-task-reward regression, not the untested 0.008 -- this is a
        # brand-new environment, so no reason to gamble on a higher value we
        # already know can cause growing action_std/worsening contact behavior.
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1Policy3PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Policy 3 (move to goal + place -- release + return_to_ready split off
    into Policy 4, see G1Policy4PPORunnerCfg below), fresh training from
    scratch."""
    num_steps_per_env = 24
    max_iterations = 1500
    resume = False
    save_interval = 50
    experiment_name = "g1_policy3_place_release"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        # SILPPO (agents/sil_ppo.py) was tried and reverted: it filled its demo
        # buffer and ran every iteration, but a matched replay comparison against
        # the pre-SIL checkpoint showed no reliable improvement on release (3/64
        # vs 1/64 envs ever releasing) and real regressions elsewhere -- place
        # -38%, action_rate +78%, joint_vel +50% -- so it was net harmful, not
        # neutral. Back to plain PPO (class_name defaults to "PPO").
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,        # same reasoning as G1Policy2PPORunnerCfg above
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1Policy4PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Policy 4 (release + return to ready), fresh training from scratch.

    Same hyperparameters as Policy 2/3 as a starting point -- no reason yet to
    believe this task needs different ones.

    FIXED: 800 -> 1500. First run (2026-07-16_08-44-24, 800 iters) learned
    release reliably (sustained ~770/800 steps per episode, confirmed via
    replay -- not the single-frame flicker it first looked like) but never
    moved the arm at all during the ~770-step not_grasping window
    return_to_ready had open the whole time (dist to READY_ARM_POSE identical
    to the reset value, std=0.0000 across envs). Gating confirmed correct
    (_is_grasping's AND-condition opens the window immediately on release, not
    after some separate condition) -- this reads as release (an easy,
    single-threshold behavior) getting discovered well within 800 iterations
    while return_to_ready (moving 7 joints ~2 rad) simply hadn't been
    discovered yet, not a reward-structure bug. Matching Policy 3's full 1500
    budget instead of Policy 2's 800 gives the harder half of this composite
    skill more room.
    """
    num_steps_per_env = 24
    max_iterations = 1500
    resume = False
    save_interval = 50
    experiment_name = "g1_policy4_release_return"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.6,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
