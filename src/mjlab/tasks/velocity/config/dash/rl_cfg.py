"""RL config for the Dash velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def dash_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the RL runner config for the Dash velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="dash_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def dash_run_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the RL runner config for the Dash running task.

  Shares the walking task's experiment folder so a walking run can be loaded as
  the starting point with `--agent.load-run`; the `run` suffix tells the two
  apart.
  """
  cfg = dash_ppo_runner_cfg()
  cfg.run_name = "run"
  # Halved from walking: in the first running run the action std jumped from
  # ~0.37 to ~0.6 several times, with reward drops each time, and the collapse
  # at 3.5 m/s started right after one of them.
  cfg.algorithm.entropy_coef = 0.005
  return cfg
