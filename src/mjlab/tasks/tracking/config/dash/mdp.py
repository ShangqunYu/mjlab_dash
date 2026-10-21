"""Dash-specific tracking MDP terms.

Import explicitly; the task auto-importer skips `.mdp`.
"""

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class torque_near_limit:
  """Penalize actuator torque above a fraction of each actuator's effort limit.

  Free below `threshold` x limit; above it, the excess is scaled so the cost per
  actuator goes from 0 at the threshold to 1 at the limit, then squared and
  summed. Unlike `joint_torques_l2` it leaves the torque needed to hold the
  robot up alone and only pushes back where the policy rides the cap.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    asset: Entity = env.scene[asset_cfg.name]
    # Effort limits are set as actuator_forcerange and are not randomized.
    ctrl_ids = asset.indexing.ctrl_ids.cpu().numpy()
    limit = env.sim.mj_model.actuator_forcerange[ctrl_ids, 1]
    self._limit = torch.tensor(limit, dtype=torch.float, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    threshold: float = 0.9,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    ratio = asset.data.actuator_force.abs() / self._limit
    excess = torch.clamp(ratio - threshold, min=0.0) / (1.0 - threshold)
    return torch.sum(torch.square(excess), dim=1)
