"""Dash motion tracking task registration."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import dash_flat_tracking_env_cfg
from .rl_cfg import dash_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Dash",
  env_cfg=dash_flat_tracking_env_cfg(),
  play_env_cfg=dash_flat_tracking_env_cfg(play=True),
  rl_cfg=dash_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
