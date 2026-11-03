"""Dash velocity tracking environment configurations.

Structured after mjlab's Unitree G1 velocity config.
"""

from mjlab.asset_zoo.robots.dash import DASH_ACTION_SCALE, get_dash_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from .hardware_dr import add_hardware_randomization
from .mdp import (
  action_rate_l2_speed_scaled,
  arm_swing_symmetry,
  feet_air_time_asymmetry,
  feet_air_time_speed_scaled,
  feet_clearance_speed_scaled,
  feet_stance_imbalance,
  feet_swing_height_speed_scaled,
  forward_progress,
  soft_landing_speed_scaled,
  torso_height_below_minimum,
  track_angular_velocity_speed_scaled,
  track_linear_velocity_speed_scaled,
  velocity_level_curriculum,
)


def dash_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Dash rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  # Dash's box feet and box torso generate more contact points on rough terrain
  # than the G1's capsule-heavy collider set; 70 (the G1 value) overflows.
  cfg.sim.nconmax = 250

  cfg.scene.entities = {"robot": get_dash_robot_cfg()}

  # The real robot has no state estimator, so the policy can't get true base
  # linear velocity the way sim does (Unitree's own G1 sim-to-real recipe
  # drops it from the observation for the same reason -- their low-level IMU
  # state exposes only gyroscope/accelerometer/quaternion, no velocity). Kept
  # in the critic: it's privileged sim-only information, discarded after
  # training, and asymmetric actor-critic is exactly what it's for.
  del cfg.observations["actor"].terms["base_lin_vel"]

  # Dash's root body is the torso; mjlab's default terrain scan frame is not.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "torso"

  site_names = ("l_foot", "r_foot")
  geom_names = ("l_foot_collision", "r_foot_collision")

  # Wire the foot height scan to the per-foot sole sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="subtree", pattern=r"^[lr]_foot$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  # Every collider except the feet. Touching the ground with any of these means
  # the robot has fallen (or is dragging a limb), so the episode ends.
  nonfoot_geom_names = (
    "torso_collision",
    "l_hip_collision",
    "r_hip_collision",
    "l_dist_hip_collision",
    "r_dist_hip_collision",
    "l_upper_leg_collision",
    "r_upper_leg_collision",
    "l_lower_leg_collision",
    "r_lower_leg_collision",
    "l_ankle_linkage_collision",
    "r_ankle_linkage_collision",
    "l_knee_linkage_collision",
    "r_knee_linkage_collision",
    "l_upper_arm_collision",
    "r_upper_arm_collision",
    "l_lower_arm_collision",
    "r_lower_arm_collision",
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_contact",
    primary=ContactMatch(mode="geom", pattern=nonfoot_geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    # `illegal_contact` prefers force_history when present, which lets it apply
    # a force threshold instead of firing on any grazing touch.
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = DASH_ACTION_SCALE

  cfg.viewer.body_name = "torso"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.05

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)

  add_hardware_randomization(cfg)

  # mjlab adds a random 10-50 mm to the spawn height on every reset, on top of
  # init_state.pos. With init_state.pos[2] tuned so the soles sit exactly on the
  # ground, that offset is the only thing left making the robot start airborne,
  # and it drops in with downward velocity to absorb before it can do anything.
  # Zeroed so it starts in contact. Restore a small range (e.g. (0.0, 0.02))
  # once a gait exists -- spawn-height variation is useful robustness DR.
  cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)

  # Pose reward tolerances, following the G1 rationale: hips and knees get the
  # most freedom for stride, ankles are tighter, and arms get moderate freedom
  # for swing.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  # Leg tolerances are looser than the G1's: `pose` scores exp(-err^2/std^2)
  # against the nominal stance, so a tight std on hip_pitch and knee penalizes
  # exactly the joint excursions a stride is made of. The G1 can afford tighter
  # values because its ankle roll contributes to balance; Dash has to get its
  # stability from the legs, so it needs room to swing them.
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.45,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.55,
    r".*ankle_pitch.*": 0.3,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.7,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.9,
    r".*ankle_pitch.*": 0.4,
    r".*shoulder_pitch.*": 0.5,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow.*": 0.35,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso",)

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # Leave `track_linear_velocity`, `pose` and `action_rate_l2` at mjlab's
  # defaults (2.0, 1.0, -0.1). Raising tracking to 10.0 and zeroing the other
  # two was an attempt to force movement out of a robot that was actually
  # torque-limited (see get_spec in dash_constants); with the clamp lifted the
  # stock balance applies again. `action_rate_l2` in particular must stay
  # non-zero: it is the only term penalizing action jitter, and without it PPO's
  # entropy bonus drives the policy's action std up without bound.

  cfg.rewards["dof_pos_limits"].weight = 0.0
  # cfg.rewards["track_linear_velocity"].weight = 4.0

  # soft_landing's cost is raw landing force in N (unsquared, unnormalized);
  # the stock -1e-5 weight is 4-5 orders of magnitude too small to matter
  # against landing forces of ~60-240 N. -0.001 puts its per-step
  # contribution in the same ballpark as foot_slip, enough to actually shape
  # behavior. Watch Metrics/landing_force_mean and Episode_Reward/soft_landing
  # and retune from there.
  cfg.rewards["soft_landing"].weight = -0.01

  # cfg.rewards["body_ang_vel"].weight = -0.05
  # cfg.rewards["angular_momentum"].weight = -0.02

  # Sized to sit under the tracking terms (weight 2.0) without dominating them.
  # This is the main knob: raise it if the gait stays flat-footed.
  cfg.rewards["air_time"].weight = 0.75
  # air_time pays every step a foot is airborne up to threshold_max, so the cap
  # sets how long a swing it is worth holding. At the stock 0.5 s the policy
  # swung ~270 ms with almost no double support, which looked long on hardware.
  cfg.rewards["air_time"].params["threshold_max"] = 0.3

  cfg.rewards["action_rate_l2"].weight = -0.1

  cfg.rewards["foot_clearance"].weight = -0.5
  # mjlab's 0.1 m swing target is tuned for the G1; on hardware Dash's ~9 cm
  # lift read as too high.
  cfg.rewards["foot_clearance"].params["target_height"] = 0.05
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.05

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  # Asymmetry is the shaping term: it compares the last completed swing of each
  # foot, so a hopping gait (one foot never swinging) reads as a large spread.
  # Stance imbalance is the backstop: it bites while a foot is still planted,
  # before it has completed any swing for the asymmetry term to compare.
  cfg.rewards["air_time_asymmetry"] = RewardTermCfg(
    func=feet_air_time_asymmetry,
    weight=-0.5,
    params={"sensor_name": feet_ground_cfg.name, "command_name": "twist"},
  )
  cfg.rewards["stance_imbalance"] = RewardTermCfg(
    func=feet_stance_imbalance,
    weight=-0.5,
    params={
      "sensor_name": feet_ground_cfg.name,
      "max_stance_time": 1.0,
      "command_name": "twist",
    },
  )

  # Equalizes arm swing across the two sides; it cannot create swing, so if the
  # arms end up symmetric but dead, the thing to loosen is the `pose` std on
  # shoulder_pitch above, which is what holds them at nominal in the first place.
  # Cost is a sum of per-joint mismatches in radians, so a fully dead arm during
  # a 0.3 rad shoulder sweep costs ~0.3 * weight. Raise the weight if one arm
  # keeps lagging.
  cfg.rewards["arm_swing_symmetry"] = RewardTermCfg(
    func=arm_swing_symmetry,
    weight=-0.5,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "left_joint_names": (
        "l_shoulder_pitch",
        "l_shoulder_roll",
        "l_shoulder_yaw",
        "l_elbow_pitch",
      ),
      "right_joint_names": (
        "r_shoulder_pitch",
        "r_shoulder_roll",
        "r_shoulder_yaw",
        "r_elbow_pitch",
      ),
    },
  )

  # End episode as soon as anything but a foot hits ground. The stock
  # `fell_over` only fires on torso orientation, so a robot that drops onto a
  # knee or drags a hand can keep collecting reward while visibly failing.
  # 10 N filters out grazing brushes; a 38 kg robot putting real weight on a
  # limb is far above that.
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Catch collapses that keep the torso upright and the limbs off ground --
  # a deep sag that neither `fell_over` (orientation) nor `illegal_contact`
  # (ground touch) detects. The designer's mechanical joint limits (knee to
  # 150 deg) fold the robot down to 0.02 m, well below this 0.40 m threshold,
  # so it's reachable rather than unreachable -- this is a judgement call
  # about how deep a crouch to allow, not a bound that can't fire. Revisit if
  # policies get terminated while legitimately squatting.
  cfg.terminations["torso_too_low"] = TerminationTermCfg(
    func=torso_height_below_minimum,
    params={
      "minimum_height": 0.40,
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=("torso",), site_names=site_names
      ),
    },
  )

  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def dash_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Dash flat terrain velocity configuration."""
  cfg = dash_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # No terrain to scan on flat ground.
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  # Disabled while tuning for a gentler, more stable gait: stages 1-2 push
  # commands up to 3.0 m/s / 0.7 rad/s, which rewards aggressive, high-effort
  # motion. Pinned at stage 0's range (lin_vel_x/y +/-1.0, ang_vel_z +/-0.5)
  # for the whole run instead. Re-enable once the gait is stable at that range.
  cfg.curriculum.pop("command_vel", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg


# Speed band over which the gait terms switch from walking to running shaping.
# The lower end matches the `pose` reward's running_threshold (1.5 m/s), above
# which it already uses std_running.
RUN_BLEND_V_WALK = 1.5
RUN_BLEND_V_RUN = 3.0


def dash_run_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Dash flat-ground running configuration.

  Same as the flat walking task, except:
  - The max forward command steps up from 1.0 to 5.0 m/s through a
    performance-gated curriculum (`velocity_level_curriculum`).
  - The walking-tuned gait terms (swing cap, clearance height, landing cost)
    relax with commanded speed between RUN_BLEND_V_WALK and RUN_BLEND_V_RUN.
    Below that band they match the walking task, so a walking checkpoint
    starts on the reward it was trained on.

  WARNING: the actuators clamp torque only; there is no torque-speed curve, so
  at high speed the policy can rely on motors spinning faster than the real ones
  can at that torque. Add one before putting anything above ~2 m/s on hardware.
  """
  cfg = dash_flat_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # Lateral and yaw stay at walking magnitudes: a sharp turn or a sidestep at
  # 4 m/s is a separate skill, not part of learning to run.
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
  # More straight-line commands than walking (0.2), since that's where the
  # top speeds get practiced.
  twist_cmd.rel_forward_envs = 0.4

  blend = {"v_walk": RUN_BLEND_V_WALK, "v_run": RUN_BLEND_V_RUN}

  # Tolerance grows with commanded speed above 2.5 m/s (0.2 x 2.5 = the stock
  # 0.5), reaching 1.0 m/s at a 5 m/s command. See
  # track_linear_velocity_speed_scaled for why a fixed tolerance collapses.
  track = cfg.rewards["track_linear_velocity"]
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=track_linear_velocity_speed_scaled,
    weight=track.weight,
    params={
      "command_name": "twist",
      "std_min": track.params["std"],
      "std_ratio": 0.2,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  site_names = ("l_foot", "r_foot")

  # The terms below exist because, at 4 m/s, running at 3.2 m/s earned ~0.17/s
  # *less* than standing still: tracking paid +1.25/s, while yaw tracking,
  # pose, action rate, landing and slip together cost ~1.4/s. Giving up won
  # at the top of the range, then spread down to 3 m/s, where running still
  # won but a stopped policy got no tracking signal to find its way back.
  # forward_progress gives that signal and pays ~+0.7/s for running at 4 m/s;
  # halving action rate and yaw tracking at speed closes another ~0.3/s.
  cfg.rewards["forward_progress"] = RewardTermCfg(
    func=forward_progress,
    weight=1.0,
    params={
      "command_name": "twist",
      "asset_cfg": SceneEntityCfg("robot"),
      **blend,
    },
  )
  action_rate = cfg.rewards["action_rate_l2"]
  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=action_rate_l2_speed_scaled,
    weight=action_rate.weight,
    params={"command_name": "twist", "run_scale": 0.5, **blend},
  )
  track_ang = cfg.rewards["track_angular_velocity"]
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=track_angular_velocity_speed_scaled,
    weight=track_ang.weight,
    params={
      "command_name": "twist",
      "std": track_ang.params["std"],
      "run_scale": 0.5,
      **blend,
    },
  )

  air_time = cfg.rewards["air_time"]
  cfg.rewards["air_time"] = RewardTermCfg(
    func=feet_air_time_speed_scaled,
    weight=air_time.weight,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "threshold_min": air_time.params["threshold_min"],
      "threshold_max_walk": air_time.params["threshold_max"],
      "threshold_max_run": 0.45,
      **blend,
    },
  )
  foot_clearance = cfg.rewards["foot_clearance"]
  cfg.rewards["foot_clearance"] = RewardTermCfg(
    func=feet_clearance_speed_scaled,
    weight=foot_clearance.weight,
    params={
      "height_sensor_name": "foot_height_scan",
      "command_name": "twist",
      "target_height_walk": foot_clearance.params["target_height"],
      "target_height_run": 0.12,
      "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
      **blend,
    },
  )
  swing_height = cfg.rewards["foot_swing_height"]
  cfg.rewards["foot_swing_height"] = RewardTermCfg(
    func=feet_swing_height_speed_scaled,
    weight=swing_height.weight,
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
      "command_name": "twist",
      "target_height_walk": swing_height.params["target_height"],
      "target_height_run": 0.12,
      **blend,
    },
  )
  soft_landing = cfg.rewards["soft_landing"]
  cfg.rewards["soft_landing"] = RewardTermCfg(
    func=soft_landing_speed_scaled,
    weight=soft_landing.weight,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "run_scale": 0.25,
      **blend,
    },
  )

  if play:
    twist_cmd.ranges.lin_vel_x = (-1.0, 5.0)
    return cfg

  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=velocity_level_curriculum,
    params={
      "command_name": "twist",
      "reward_name": "track_linear_velocity",
      "levels": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
      "min_lin_vel_x": -1.0,
      # Below the walking policy's 0.71 at +/-1 m/s, since the score drifts
      # down as the range widens (see velocity_level_curriculum). 0.65 stalled
      # a policy that tracked 3.0 m/s to within 4% at ~0.62.
      "promote_threshold": 0.6,
      # Well under the ~0.56 the collapsed policy settled at, so ordinary noise
      # around the promote threshold doesn't bounce levels.
      "demote_threshold": 0.45,
      # Several thousand episodes per level: one full turnover of 4096 envs.
      "min_episodes": 4096,
      # Top-speed gate. A policy that ran 3.18 m/s at a 4.0 m/s command
      # holds ~0.8 steady (lower while accelerating after a resample); a
      # collapsed one sits near 0.1.
      "top_frac": 0.8,
      "top_promote": 0.7,
      "top_demote": 0.5,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  return cfg
