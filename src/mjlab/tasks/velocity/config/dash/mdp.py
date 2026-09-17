"""Dash-specific MDP terms.

Kept separate from mjlab's `mjlab.tasks.velocity.mdp` so upstream stays
untouched; import explicitly, since the task auto-importer skips `.mdp`.
"""

from collections.abc import Sequence

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor


def feet_air_time_asymmetry(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Penalize one foot doing all the stepping.

  mjlab's `feet_air_time` pays +1 per foot airborne for 0.05-0.5 s and
  sums over feet, w/ no requirement that the feet take turns. Standing on one
  leg and pumping the other collects that reward every step, which is exactly
  the gait it converges to if nothing else constrains timing.

  Returns the spread in *last completed* swing duration across feet.
  A foot that never leaves the ground holds `last_air_time` at 0 while the
  swinging foot sits near its step period, so the spread stays large. In an
  alternating gait both feet swing for a similar duration and it goes to ~0.

  Uses last_air_time rather than current_air_time deliberately: the current
  value is out of phase by construction during normal walking (one foot is
  always airborne while the other is planted), so penalizing it would fight
  the gait rather than shape it.

  Gated on cmd speed, since standing still shouldn't be penalized for
  a lack of stepping.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  last_air_time = sensor.data.last_air_time
  assert last_air_time is not None, "sensor needs track_air_time=True"

  spread = last_air_time.max(dim=1).values - last_air_time.min(dim=1).values
  env.extras["log"]["Metrics/air_time_asymmetry"] = spread.mean()

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      spread = spread * (total > command_threshold).float()
  return spread


def feet_stance_imbalance(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_stance_time: float = 1.0,
  command_name: str | None = None,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Penalize a foot staying planted beyond a normal stance duration.

  Complements asymmetry term. That one only reacts once a foot has
  completed a swing; this one bites while a foot is still glued down, since
  `current_contact_time` grows w/o bound on a permanently planted foot.
  Clipped at `max_stance_time` so a robot that has genuinely stopped isn't
  punished w/o limit.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  contact_time = sensor.data.current_contact_time
  assert contact_time is not None, "sensor needs track_air_time=True"

  # max, not min: the failure case is one foot glued down while the other keeps
  # stepping, so the stepping foot's excess is 0 and a min would never fire.
  excess = (contact_time - max_stance_time).clamp(min=0.0)
  cost = excess.max(dim=1).values
  env.extras["log"]["Metrics/max_stance_time"] = contact_time.max(dim=1).values.mean()

  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
      cost = cost * (total > command_threshold).float()
  return cost


class arm_swing_symmetry:
  """Penalize one arm swinging while the other hangs.

  Measures how far each arm sits from its nominal pose, joint by matched joint,
  and charges the mismatch between the two sides. A left shoulder sweeping
  0.3 rad while the right stays put reads as a 0.3 rad mismatch; two arms doing
  the same amount of work read as ~0, whichever way each one is moving.

  Deliberately built on |deviation|, which makes it blind to *phase*: arms in
  natural opposition (left forward while right is back) and arms moving
  together both score 0. Only the amounts have to match. That is what makes it
  a symmetry term rather than a counter-swing term -- penalizing in-phase
  swinging would also penalize a mirror-symmetric static pose (both arms held
  forward), which is not the failure being described here. It also means the
  term does not care about each joint's sign convention, which is not uniform:
  under a mirror across the sagittal plane Dash's pitch axes keep their sign
  while roll and yaw flip.

  Position rather than velocity: a swing's velocity passes through zero at the
  extremes of the stroke, which is exactly where the two arms differ most,
  whereas the deviation peaks there.

  Note this cannot *create* arm swing -- two arms hanging still are perfectly
  symmetric. It only equalizes whatever swing the other terms produce.

  Implemented as a class so the left/right pairing is resolved once, at
  startup, rather than per step.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]

    # preserve_order, because the two lists are paired positionally: entry i on
    # the left is charged against entry i on the right. Resolution order would
    # otherwise follow model order, which need not interleave the sides evenly.
    left_ids, left_names = asset.find_joints(
      cfg.params["left_joint_names"], preserve_order=True
    )
    right_ids, right_names = asset.find_joints(
      cfg.params["right_joint_names"], preserve_order=True
    )
    if len(left_ids) != len(right_ids):
      raise ValueError(
        "arm_swing_symmetry needs one right joint per left joint, got "
        f"{left_names} against {right_names}."
      )
    # Catches a silently mispaired list (l_elbow charged against r_shoulder),
    # which would otherwise train quietly against the wrong target.
    for left_name, right_name in zip(left_names, right_names, strict=True):
      if left_name.removeprefix("l_") != right_name.removeprefix("r_"):
        raise ValueError(
          f"arm_swing_symmetry paired {left_name!r} with {right_name!r}; the "
          "lists are matched by position, so they must name the same joint on "
          "each side, in the same order."
        )

    self._left_ids = torch.tensor(left_ids, device=env.device, dtype=torch.long)
    self._right_ids = torch.tensor(right_ids, device=env.device, dtype=torch.long)

    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self._default_joint_pos = default_joint_pos

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    left_joint_names: Sequence[str],
    right_joint_names: Sequence[str],
    command_name: str | None = None,
    command_threshold: float = 0.5,
  ) -> torch.Tensor:
    del left_joint_names, right_joint_names  # Resolved in __init__.
    asset: Entity = env.scene[asset_cfg.name]

    deviation = (asset.data.joint_pos - self._default_joint_pos).abs()
    mismatch = (deviation[:, self._left_ids] - deviation[:, self._right_ids]).abs()
    # Summed, not averaged: averaging would dilute a single dead joint by the
    # number of pairs, so adding wrist joints later would quietly shrink the
    # penalty for a dead shoulder. Cost is in radians, so the weight is the knob.
    cost = mismatch.sum(dim=1)
    env.extras["log"]["Metrics/arm_swing_asymmetry"] = cost.mean()

    # Off by default: unlike stepping, symmetric arms are just as expected
    # standing still as walking, and both arms sit at nominal there anyway.
    if command_name is not None:
      command = env.command_manager.get_command(command_name)
      if command is not None:
        total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        cost = cost * (total > command_threshold).float()
    return cost


def torso_height_below_minimum(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Terminate when torso sinks too close to feet.

  mjlab ships `root_height_below_minimum`, but it thresholds the root's *world*
  z. That is only meaningful on a plane: on generated terrain the ground height
  varies per sub-terrain, so a fixed world-z cutoff fires on a robot standing
  perfectly well in a low patch and misses one collapsed on a high one.

  Measuring the torso relative to the lowest foot site is terrain-independent,
  and is really what "has it collapsed" means: the distance is just how far the
  legs are extended. For reference, on Dash this is 0.685 m at the nominal
  stance and 0.462 m at the deepest crouch the joint limits allow, so a
  threshold below ~0.46 can't fire on a legit pose.

  Fires when torso below threshold, which also covers the robot
  ending up under or level with its own feet.
  """
  asset: Entity = env.scene[asset_cfg.name]
  assert asset_cfg.body_ids is not None
  assert asset_cfg.site_ids is not None

  torso_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(-1)
  feet_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  return (torso_z - feet_z.min(dim=1).values) < minimum_height
