"""Dash-specific MDP terms.

Kept separate from mjlab's `mjlab.tasks.velocity.mdp` so upstream stays
untouched; import explicitly, since the task auto-importer skips `.mdp`.
"""

from collections.abc import Sequence
from typing import cast

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg


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


##
# Running: speed-dependent gait terms and the speed curriculum.
##


class capsule_pair_clearance:
  """Penalize two capsules coming within `margin` of each other.

  Built for the ankle linkages: on hardware the robot brings its feet closer
  together than the sim does, so a contact-only penalty leaves no room for that
  gap. This one starts charging `margin` before contact and keeps growing
  linearly through it: (margin - gap) / margin, so 1 at touching and more
  when overlapping.

  The surface gap is the distance between the two capsule axes minus both
  radii. The axis distance is taken over a grid of points along each segment,
  which is accurate to well under a millimetre for capsules this length.
  """

  _N_SAMPLES = 16

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset: Entity = env.scene[asset_cfg.name]
    local_ids = asset_cfg.geom_ids
    assert isinstance(local_ids, list) and len(local_ids) == 2
    self._geom_ids = asset.indexing.geom_ids[local_ids]
    size = env.sim.mj_model.geom_size[self._geom_ids.cpu().numpy()]
    self._radius = torch.tensor(size[:, 0], device=env.device)
    self._half_len = torch.tensor(size[:, 1], device=env.device)
    self._t = torch.linspace(-1.0, 1.0, self._N_SAMPLES, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    margin: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del asset_cfg  # resolved in __init__
    pos = env.sim.data.geom_xpos[:, self._geom_ids]  # (N, 2, 3)
    axis = env.sim.data.geom_xmat[:, self._geom_ids].reshape(-1, 2, 3, 3)[..., 2]
    offsets = (self._half_len[:, None] * self._t[None, :])[None, :, :, None]
    pts = pos[:, :, None] + offsets * axis[:, :, None]  # (N, 2, S, 3)
    axis_dist = torch.cdist(pts[:, 0], pts[:, 1]).flatten(1).min(dim=1).values
    gap = axis_dist - self._radius.sum()
    env.extras["log"]["Metrics/linkage_gap_min"] = gap.min()
    return (margin - gap).clamp(min=0.0) / margin


def _run_blend(
  env: ManagerBasedRlEnv, command_name: str, v_walk: float, v_run: float
) -> torch.Tensor:
  """How far each env's command is into the running regime, in [0, 1].

  0 at or below `v_walk` planar command speed, 1 at or above `v_run`, linear in
  between. Shared by every speed-scaled term, so they all switch from the
  walking to the running shaping over the same band.
  """
  command = env.command_manager.get_command(command_name)
  assert command is not None
  speed = torch.norm(command[:, :2], dim=1)
  return ((speed - v_walk) / (v_run - v_walk)).clamp(0.0, 1.0)


def track_linear_velocity_speed_scaled(
  env: ManagerBasedRlEnv,
  command_name: str,
  std_min: float,
  std_ratio: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """`track_linear_velocity` with a tolerance that grows with commanded speed.

  The stock term's tolerance is fixed (std 0.5 m/s), so at a 3.5 m/s command a
  policy that gives up and jogs at 0.3 m/s earns exactly the same ~0 reward as
  one that tries and reaches 2.5 m/s. Nothing pulls it back, while pose,
  upright, action rate and landing terms all favor moving less -- the stable
  "give up at top speed" state the first running run collapsed into.

  std = max(std_min, std_ratio * |command_xy|) keeps the stock tolerance at
  walking speeds and makes partial progress at top speed pay.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  std = (std_ratio * torch.norm(command[:, :2], dim=1)).clamp(min=std_min)
  return torch.exp(-(xy_error + z_error) / std.square())


def _lerp(a: float, b: float, t: torch.Tensor) -> torch.Tensor:
  return a + (b - a) * t


def forward_progress(
  env: ManagerBasedRlEnv,
  command_name: str,
  v_walk: float,
  v_run: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Fraction of the commanded planar speed achieved, along the command.

  `clip(v . c_hat / |c|, 0, 1)`: linear in speed, so unlike the exponential
  tracking kernel it never goes flat. A policy that has stopped at a fast
  command still gains reward for every bit of speed it adds back, which is
  what pulls it out of the "give up at top speed" trap (see
  track_linear_velocity_speed_scaled). Faded in with `_run_blend`, so walking
  speeds see none of it.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd_xy = command[:, :2]
  speed = torch.norm(cmd_xy, dim=1).clamp(min=1e-6)
  along = torch.sum(asset.data.root_link_lin_vel_b[:, :2] * cmd_xy, dim=1) / speed
  progress = (along / speed).clamp(0.0, 1.0)
  blend = _run_blend(env, command_name, v_walk, v_run)
  env.extras["log"]["Metrics/forward_progress"] = (
    progress * blend
  ).sum() / blend.sum().clamp(min=1e-6)
  return progress * blend


def action_rate_l2_speed_scaled(
  env: ManagerBasedRlEnv,
  command_name: str,
  run_scale: float,
  v_walk: float,
  v_run: float,
) -> torch.Tensor:
  """`action_rate_l2`, scaled down to `run_scale` of its cost at running speed.

  A running stride needs fast action changes. At the walking weight this term
  cost ~0.3/s more running at 4 m/s than standing still, part of what made
  giving up pay there.
  """
  cost = envs_mdp.action_rate_l2(env)
  blend = _run_blend(env, command_name, v_walk, v_run)
  return cost * _lerp(1.0, run_scale, blend)


def track_angular_velocity_speed_scaled(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  run_scale: float,
  v_walk: float,
  v_run: float,
) -> torch.Tensor:
  """`track_angular_velocity`, scaled down to `run_scale` at running speed.

  A running torso yaws through every stride, so at 4 m/s yaw tracking paid
  ~0.34/s less than standing still -- the largest single term in favor of
  giving up. Scaling the reward down shrinks that gap by the same factor.
  """
  reward = vel_mdp.track_angular_velocity(env, std=std, command_name=command_name)
  blend = _run_blend(env, command_name, v_walk, v_run)
  return reward * _lerp(1.0, run_scale, blend)


def feet_air_time_speed_scaled(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  threshold_min: float,
  threshold_max_walk: float,
  threshold_max_run: float,
  v_walk: float,
  v_run: float,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """`feet_air_time` with a cap that grows with commanded speed.

  The walking cap (0.3 s) was cut to stop long, floaty swings, but a running
  stride needs a longer swing, and a fixed cap would stop paying for it.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  assert current_air_time is not None
  blend = _run_blend(env, command_name, v_walk, v_run)
  threshold_max = _lerp(threshold_max_walk, threshold_max_run, blend).unsqueeze(1)
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = in_range.float().sum(dim=1)

  in_air = current_air_time > 0
  env.extras["log"]["Metrics/air_time_mean"] = (
    current_air_time * in_air.float()
  ).sum() / in_air.float().sum().clamp(min=1)

  command = env.command_manager.get_command(command_name)
  assert command is not None
  total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return reward * (total > command_threshold).float()


def feet_clearance_speed_scaled(
  env: ManagerBasedRlEnv,
  height_sensor_name: str,
  command_name: str,
  target_height_walk: float,
  target_height_run: float,
  v_walk: float,
  v_run: float,
  asset_cfg: SceneEntityCfg,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """`feet_clearance` with a target height that grows with commanded speed.

  The stock cost is |height - target| weighted by foot speed, so it grows with
  running speed on its own: the swing foot moves ~2x the body speed. It's
  divided by the command speed (floored at 1 m/s) so that walking costs are
  unchanged and a 5 m/s run isn't dominated by this term.
  """
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor)
  command = env.command_manager.get_command(command_name)
  assert command is not None

  blend = _run_blend(env, command_name, v_walk, v_run)
  target = _lerp(target_height_walk, target_height_run, blend).unsqueeze(1)
  foot_vel = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
  delta = torch.abs(height_sensor.data.heights - target)

  speed = torch.norm(command[:, :2], dim=1)
  cost = (delta * foot_vel).sum(dim=1) / speed.clamp(min=1.0)
  total = speed + torch.abs(command[:, 2])
  return cost * (total > command_threshold).float()


class feet_swing_height_speed_scaled(vel_mdp.feet_swing_height):
  """`feet_swing_height` with a target height that grows with commanded speed.

  The 5 cm walking target came from hardware (higher read as a high-stepping
  gait), but a running swing needs more ground clearance for the shin to fold.
  """

  def __call__(  # type: ignore[override]
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    command_name: str,
    target_height_walk: float,
    target_height_run: float,
    v_walk: float,
    v_run: float,
    command_threshold: float = 0.05,
  ) -> torch.Tensor:
    blend = _run_blend(env, command_name, v_walk, v_run)
    target = _lerp(target_height_walk, target_height_run, blend).unsqueeze(1)
    # The parent only uses target_height elementwise (peak / target - 1), so a
    # per-env column broadcasts the same as a scalar does.
    return super().__call__(
      env,
      sensor_name=sensor_name,
      height_sensor_name=height_sensor_name,
      target_height=cast(float, target),
      command_name=command_name,
      command_threshold=command_threshold,
    )


def soft_landing_speed_scaled(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  run_scale: float,
  v_walk: float,
  v_run: float,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """`soft_landing`, scaled down to `run_scale` of its cost at running speed.

  Landing forces in a run are inherently 2-3x body weight. At the walking weight
  the cost would push the policy away from flight phases altogether.
  """
  cost = vel_mdp.soft_landing(env, sensor_name, command_name, command_threshold)
  blend = _run_blend(env, command_name, v_walk, v_run)
  return cost * _lerp(1.0, run_scale, blend)


class velocity_level_curriculum:
  """Raise the max forward speed command one level at a time, on performance.

  mjlab's `commands_vel` switches ranges at fixed step counts whether or not
  the policy keeps up. This one promotes from `levels[i]` to `levels[i + 1]`
  only once the policy tracks the current range well, and demotes back down
  if it stops tracking it.

  The score is the episode's sum of `reward_name` divided by its maximum
  possible value (weight x episode length), the same normalization as the
  logged `Episode_Reward/*`. Falls end episodes early, so they score low too.
  Only episodes that started after the latest promotion count, so the score
  always describes the current range. Scores are averaged over windows of
  `min_episodes` episodes; a window averaging `promote_threshold` promotes, one
  below `demote_threshold` demotes, and either way the next window starts
  fresh, so the score tracks the current policy rather than everything since
  the last level change. Without demotion, a policy that gives up on the top
  of its range is held there indefinitely.

  The score alone can't see a policy giving up at the top of its range: one
  that stops at fast commands still tracks every slow one, and averaged ~0.51
  while collapsed. So it is also gated on top-speed tracking: across envs
  commanded at least `top_frac` of the current max forward speed, the mean
  achieved/commanded forward speed. Promotion needs it at `top_promote`, and
  it falling below `top_demote` demotes regardless of the score. Sampling
  starts one episode length after a level change, so the brief dip while the
  policy first meets the new top speed doesn't count.

  For calibration: the converged walking policy scored ~0.71 at +/-1 m/s. The
  score drifts down as the range widens even for a policy that tracks well,
  since resampled commands jump further (e.g. -1 to +3 m/s) and every jump
  costs an acceleration transient. A policy holding 2.9 m/s at a 3.0 m/s
  command scored ~0.62 at that level.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._levels: list[float] = list(cfg.params["levels"])
    if any(b <= a for a, b in zip(self._levels, self._levels[1:], strict=False)):
      raise ValueError(f"levels must be increasing, got {self._levels}")
    self._level = int(cfg.params.get("start_level", 0))
    self._level_start_step = 0
    self._score_sum = 0.0
    self._score_count = 0
    self._last_score = 0.0
    self._top_sum = torch.zeros((), device=env.device)
    self._top_count = torch.zeros((), device=env.device)
    self._last_top = 1.0
    self._apply(env, cfg.params["command_name"], cfg.params["min_lin_vel_x"])

  def _apply(
    self, env: ManagerBasedRlEnv, command_name: str, min_lin_vel_x: float
  ) -> None:
    term = env.command_manager.get_term(command_name)
    assert term is not None
    cfg = cast(UniformVelocityCommandCfg, term.cfg)
    cfg.ranges.lin_vel_x = (min_lin_vel_x, self._levels[self._level])

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    reward_name: str,
    levels: Sequence[float],
    min_lin_vel_x: float,
    promote_threshold: float,
    demote_threshold: float,
    min_episodes: int,
    top_frac: float,
    top_promote: float,
    top_demote: float,
    asset_cfg: SceneEntityCfg,
    start_level: int = 0,
  ) -> dict[str, torch.Tensor]:
    del levels, start_level  # Resolved in __init__.
    # Called before the episode counters are reset, so these are the lengths
    # of the episodes ending now.
    start_step = env.common_step_counter - env.episode_length_buf[env_ids]
    fresh = start_step >= self._level_start_step
    if env.common_step_counter > 0 and fresh.any():
      weight = env.reward_manager.get_term_cfg(reward_name).weight
      sums = env.reward_manager._episode_sums[reward_name][env_ids][fresh]
      scores = sums / (weight * env.max_episode_length_s)
      self._score_sum += scores.sum().item()
      self._score_count += int(fresh.sum().item())

    # Top-speed tracking, sampled across all envs (not just the resetting ones)
    # every call; kept on device to avoid a sync per step.
    if env.common_step_counter - self._level_start_step > env.max_episode_length:
      command = env.command_manager.get_command(command_name)
      assert command is not None
      cmd_x = command[:, 0]
      top = cmd_x >= top_frac * self._levels[self._level]
      vx = env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, 0]
      ratio = (vx / cmd_x.clamp(min=1e-6)).clamp(0.0, 1.5)
      self._top_sum += (ratio * top).sum()
      self._top_count += top.sum()

    if self._score_count >= min_episodes:
      self._last_score = self._score_sum / self._score_count
      if self._top_count.item() > 0:
        self._last_top = (self._top_sum / self._top_count).item()
      self._score_sum = 0.0
      self._score_count = 0
      self._top_sum.zero_()
      self._top_count.zero_()
      step = 0
      if self._last_score < demote_threshold or self._last_top < top_demote:
        step = -1 if self._level > 0 else 0
      elif self._last_score >= promote_threshold and self._last_top >= top_promote:
        step = 1 if self._level + 1 < len(self._levels) else 0
      if step:
        self._level += step
        self._level_start_step = env.common_step_counter
        self._apply(env, command_name, min_lin_vel_x)

    return {
      "level": torch.tensor(float(self._level)),
      "max_lin_vel_x": torch.tensor(self._levels[self._level]),
      # Score of the last completed window.
      "score": torch.tensor(self._last_score),
      "top_speed_ratio": torch.tensor(self._last_top),
      "episodes": torch.tensor(float(self._score_count)),
    }
