"""Dash flat tracking environment configurations.

Structured after mjlab's Unitree G1 tracking config.
"""

import mujoco

from mjlab.asset_zoo.robots.dash import DASH_ACTION_SCALE, get_dash_robot_cfg
from mjlab.asset_zoo.robots.dash.dash_constants import get_spec
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.tasks.velocity.config.dash.hardware_dr import add_hardware_randomization

from .mdp import torque_near_limit


def get_spec_with_forearm_thigh_collision() -> mujoco.MjSpec:
  """Dash spec with forearm-thigh contact restored via mesh hulls.

  dash_v1.xml excludes lower_arm/upper_leg because their capsules report false
  overlap through the walking arm swing. Mocap motions can still drive the hand
  into the thigh, which hardware then hits, so tracking adds a hull of each
  visual mesh and an explicit pair between them. Pairs bypass the exclude and
  the contype/conaffinity masks, so the hulls touch nothing else.
  """
  spec = get_spec()
  for side in ("l", "r"):
    for body in (f"{side}_lower_arm", f"{side}_upper_leg"):
      visual = spec.geom(f"{body}_visual")
      spec.body(body).add_geom(
        name=f"{body}_hull",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=visual.meshname,
        pos=visual.pos.tolist(),
        quat=visual.quat.tolist(),
        group=3,
        contype=0,
        conaffinity=0,
        density=0,
      )
    spec.add_pair(
      geomname1=f"{side}_lower_arm_hull",
      geomname2=f"{side}_upper_leg_hull",
      condim=1,
    )
  return spec


def dash_flat_tracking_env_cfg(
  has_state_estimation: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Dash flat terrain tracking configuration.

  `has_state_estimation` defaults to False: the real robot has no state
  estimator, so the actor can't observe base linear velocity or the motion
  anchor's position relative to the robot (same reason the velocity task drops
  `base_lin_vel`). The critic keeps both as privileged sim-only information.
  """
  cfg = make_tracking_env_cfg()

  robot_cfg = get_dash_robot_cfg()
  robot_cfg.spec_fn = get_spec_with_forearm_thigh_collision
  cfg.scene.entities = {"robot": robot_cfg}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  # Same contact budget as the Dash flat velocity task; the base tracking
  # config's nconmax=35 is sized for the G1's collider set.
  cfg.sim.njmax = 300
  cfg.sim.nconmax = None

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = DASH_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  # Dash's torso is both the root and the IMU body, so it is the anchor.
  motion_cmd.anchor_body_name = "torso"
  # One body per G1 counterpart: dist_hip ~ hip_roll_link, lower_leg ~
  # knee_link, foot ~ ankle_roll_link, upper_arm ~ shoulder_roll_link (it sits
  # after all three shoulder joints), lower_arm ~ elbow_link. Dash has no wrist,
  # so lower_arm doubles as the arm end effector.
  motion_cmd.body_names = (
    "torso",
    "l_dist_hip",
    "l_lower_leg",
    "l_foot",
    "r_dist_hip",
    "r_lower_leg",
    "r_foot",
    "l_upper_arm",
    "l_lower_arm",
    "r_upper_arm",
    "r_lower_arm",
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^[lr]_foot_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)
  # Same hardware-derived COM, mass, PD gain, armature and friction ranges as
  # the walking policy. Replaces the stock base_com ranges.
  add_hardware_randomization(cfg)
  # Wider than walking's x range: the dance policy leaned on hardware, and the
  # measured 2-3.5 cm rearward offset sits well inside +/-6 cm. Tracking only.
  cfg.events["base_com"].params["ranges"][0] = (-0.06, 0.06)

  # Nothing else in the tracking reward looks at torque, so the policy rode the
  # hip pitch and knee caps through the hardest moves. Free below 90% of each
  # effort limit; a joint at its cap costs `weight` per step, against the six
  # tracking terms' combined maximum of 5.
  cfg.rewards["torque_near_limit"] = RewardTermCfg(
    func=torque_near_limit,
    weight=-1.0,
    params={"threshold": 0.9},
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "l_foot",
    "r_foot",
    "l_lower_arm",
    "r_lower_arm",
  )

  cfg.viewer.body_name = "torso"

  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
