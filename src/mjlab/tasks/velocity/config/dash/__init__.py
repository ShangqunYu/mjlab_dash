"""Dash velocity tracking task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import dash_flat_env_cfg, dash_rough_env_cfg
from .rl_cfg import dash_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Dash",
  env_cfg=dash_rough_env_cfg(),
  play_env_cfg=dash_rough_env_cfg(play=True),
  rl_cfg=dash_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Dash",
  env_cfg=dash_flat_env_cfg(),
  play_env_cfg=dash_flat_env_cfg(play=True),
  rl_cfg=dash_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
