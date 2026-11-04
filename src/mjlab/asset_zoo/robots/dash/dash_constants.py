"""Dash humanoid constants.

18 DOF: 5 per leg (hip yaw/roll/pitch, knee pitch, ankle pitch) and 4 per arm
(shoulder pitch/roll/yaw, elbow pitch). Total mass ~34.1 kg.
"""

import functools
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

# Generated from the designer's robotDataPackage URDF by
# scripts/import_dash_urdf.py (dash_bar_trajopt repo).
DASH_XML: Path = Path(__file__).parent / "xmls" / "dash_v1.xml"
assert DASH_XML.exists()


def get_spec(xml: Path = DASH_XML) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(xml))

  # Every joint in the MJCF carries `actuatorfrcrange="-10 10"`, inherited from
  # the URDF's placeholder 10 Nm. MuJoCo applies that clamp to qfrc_actuator
  # *independently* of the actuator's own forcerange, so the tighter of the two
  # wins and the effort_limit values below become dead code. At 10 Nm the robot
  # cannot stand: holding the default pose needs ~42 Nm at hip_pitch, so the hip
  # saturates and sags 0.58 rad, and every episode begins mid-collapse.
  #
  # Clearing the joint-level limit makes effort_limit the single source of truth.
  # Re-derive both from real motor specs before deploying to hardware.
  #
  # The designer's URDF actually leaves effort="0" on every joint rather than
  # carrying an actuatorfrcrange, which is no more usable as a limit than the
  # 10 Nm placeholder -- so this loop is a no-op today and effort_limit already
  # rules. Kept in case a future URDF revision reintroduces a real clamp.
  for joint in spec.joints:
    joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE

  _add_ankle_linkage_colliders(spec)
  _add_forearm_thigh_collision(spec)
  _add_knee_linkage_colliders(spec)
  _add_realsense(spec)

  return spec


def _add_forearm_thigh_collision(spec: mujoco.MjSpec) -> None:
  """Restore forearm-thigh contact via mesh hulls.

  dash_v1.xml excludes lower_arm/upper_leg because their capsules report false
  overlap through the arm swing. Without any contact, though, the running
  policy swings the forearm straight through the thigh (mesh overlap in ~25% of
  steps at 2-5 m/s), which hardware then hits. So add a hull of each visual
  mesh and an explicit pair between them. Pairs bypass the exclude and the
  contype/conaffinity masks, so the hulls touch nothing else. The thigh hull
  sits ~10 mm proud of the real surface, which errs toward clearance.
  """
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


# The knee is also driven through a linkage the meshes leave out, running down
# the outer side of the thigh. The thigh's outer face is 78 mm from the
# upper_leg axis, and the forearm swings just outside it, so the linkage is
# what the arm would hit first. Modeled like the ankle linkage: a massless
# capsule from the hip-pitch axis (y=0) to the knee axis (y=0.28). In the
# upper_leg frame +y points down the thigh and z is lateral: outer is +z on
# the left leg, -z on the right. Size is a first guess pending a measurement.
KNEE_LINKAGE_OUTER_EXTENT = 0.078 + 0.030
KNEE_LINKAGE_RADIUS = 0.020
KNEE_LINKAGE_Y_RANGE = (0.0, 0.28)


def _add_knee_linkage_colliders(spec: mujoco.MjSpec) -> None:
  """Add the knee linkages; needs the forearm hulls to pair against.

  Named *_collision, so it collides with everything not excluded (ground, the
  other leg). The forearm is excluded from the whole upper_leg body, so it
  gets an explicit pair with the forearm hull, as the thigh hull does.
  """
  z = KNEE_LINKAGE_OUTER_EXTENT - KNEE_LINKAGE_RADIUS
  y0, y1 = KNEE_LINKAGE_Y_RANGE
  for side, outer in (("l", 1.0), ("r", -1.0)):
    spec.body(f"{side}_upper_leg").add_geom(
      name=f"{side}_knee_linkage_collision",
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      fromto=[0.0, y0, outer * z, 0.0, y1, outer * z],
      size=[KNEE_LINKAGE_RADIUS, 0, 0],
      group=3,
      density=0.0,
    )
    spec.add_pair(
      geomname1=f"{side}_lower_arm_hull",
      geomname2=f"{side}_knee_linkage_collision",
      condim=1,
    )


# The ankle is driven through a linkage that the designer's meshes leave out.
# On hardware it runs the whole shank on the inner side, and at running speed
# the two linkages hit each other. Modeled as a massless capsule on each lower
# leg from the knee axis (y=0) down to just above the ankle axis (y=0.28);
# stopping at 0.27 keeps the rounded end ~18 mm above the ground with the foot
# flat. The capsule axis sits ANKLE_LINKAGE_OFFSET inward of the shank axis,
# so its inner face is OFFSET + RADIUS = 73 mm from it -- 50 mm past the
# shank's own inner face (23 mm). Deliberately larger than the ~30 mm the
# linkage measures, for margin: the real robot brings its feet closer together
# than the sim does. In the lower_leg frame +y points down the shank and z is
# lateral: inner is -z on the left leg, +z on the right.
ANKLE_LINKAGE_OFFSET = 0.038
ANKLE_LINKAGE_RADIUS = 0.035
ANKLE_LINKAGE_Y_RANGE = (0.0, 0.27)


def _add_ankle_linkage_colliders(spec: mujoco.MjSpec) -> None:
  z = ANKLE_LINKAGE_OFFSET
  y0, y1 = ANKLE_LINKAGE_Y_RANGE
  for side, inner in (("l", -1.0), ("r", 1.0)):
    spec.body(f"{side}_lower_leg").add_geom(
      name=f"{side}_ankle_linkage_collision",
      type=mujoco.mjtGeom.mjGEOM_CAPSULE,
      fromto=[0.0, y0, inner * z, 0.0, y1, inner * z],
      size=[ANKLE_LINKAGE_RADIUS, 0, 0],
      group=3,
      density=0.0,
    )


# Intel RealSense D435 on a printed jig at the bottom of the torso's front
# face. Pose from the Onshape "Dash" document, Assembly 1 (whose frame matches
# the torso body frame to ~2 mm): the jig's seat faces 50 deg below
# horizontal, and the camera's back sits flush on it. The realsense body's
# frame has +x along the optical axis, +y = torso +y, +z = away from the
# torso. Both meshes are expressed in that frame. The D435 sits upright: its
# 1/4-20 tripod hole faces -z (down/back). Mass: 72 g is Intel's
# D435 spec, plus ~20 g for the printed jig.
REALSENSE_POS = (0.10607, 0.0, 0.03069)
REALSENSE_QUAT = (0.90631, 0.0, 0.42261, 0.0)
REALSENSE_HALF_SIZE = (0.0121, 0.0454, 0.0124)
REALSENSE_MASS = 0.072 + 0.020
REALSENSE_MESH_DIR = Path(__file__).parent / "xmls" / "assets_sensors"
# Depth stream at 848x480: 89.9 x 58.9 deg, from this unit's factory
# calibration (serial 825312071982); render at 848x480 to match.
REALSENSE_FOVY = 58.9


def _add_realsense(spec: mujoco.MjSpec) -> None:
  """Add the camera + jig as a welded body with a depth camera at the lens.

  The box collider is named *_collision, so it collides with the ground and
  the legs; the forearm is excluded from the torso subtree, so it gets an
  explicit pair with each forearm hull.
  """
  for name in ("realsense_d435", "realsense_jig"):
    spec.add_mesh(name=name, file=str(REALSENSE_MESH_DIR / f"{name}.stl"))
  hx, hy, hz = REALSENSE_HALF_SIZE
  body = spec.body("torso").add_body(
    name="realsense",
    pos=REALSENSE_POS,
    quat=REALSENSE_QUAT,
    mass=REALSENSE_MASS,
    # Solid box over the camera; the jig is a small fraction of the mass.
    inertia=[
      REALSENSE_MASS / 3 * (hy**2 + hz**2),
      REALSENSE_MASS / 3 * (hx**2 + hz**2),
      REALSENSE_MASS / 3 * (hx**2 + hy**2),
    ],
  )
  # Group 1, not the robot's visual group 2: still drawn by the viewer, but
  # left out of the depth render, since the camera sits on its own front
  # glass and a position perturbation would put it inside the housing.
  for name in ("realsense_d435", "realsense_jig"):
    body.add_geom(
      name=f"{name}_visual",
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=name,
      group=1,
      contype=0,
      conaffinity=0,
      density=0.0,
    )
  body.add_geom(
    name="realsense_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=list(REALSENSE_HALF_SIZE),
    group=3,
    density=0.0,
  )
  # MuJoCo cameras look along their -z. The image is rendered upright (image
  # up = +z, right = robot right), which matches the raw D435 stream since
  # the camera is mounted upright.
  body.add_camera(
    name="realsense_depth",
    pos=[hx, 0.0, 0.0],
    xyaxes=[0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    fovy=REALSENSE_FOVY,
  )
  for side in ("l", "r"):
    spec.add_pair(
      geomname1=f"{side}_lower_arm_hull",
      geomname2="realsense_collision",
      condim=1,
    )


##
# Actuator config.
##

# WARNING: the effort limits below are ESTIMATES. Neither
# the URDF (placeholder 10 Nm on every joint) nor the DashMotorControl firmware
# (KT and gear ratio are runtime-configurable registers) pins them down. They
# are scaled from a 34 kg mass (the robot weighs ~38 kg, measured 2026-10-09;
# the model totals 40.0) and typical humanoid ratios. Replace
# with real values before deploying -- they bound the
# torque the policy can rely on, so a policy trained against optimistic limits
# will not reproduce on hardware.
# Grouped by physical motor, not by limb: hip_yaw, hip_roll, hip_pitch and
# shoulder_pitch share one motor, knee_pitch uses a different (higher-torque)
# one of its own, and the remaining arm joints share a third. Same motor gets
# the same effort limit, stiffness and damping below -- there's no reason for
# software gains to differ across joints the hardware can't tell apart.
EFFORT_LIMIT_HIP = 50.0
EFFORT_LIMIT_KNEE = 80.0
EFFORT_LIMIT_ANKLE = 40.0
EFFORT_LIMIT_ARM = 30.0

# Reflected rotor inertia, also carried over from IsaacLab config.
ARMATURE = 0.01

# Command delay, in 5 ms physics steps: 0-25 ms, resampled every step.  The
# 2026-09-17 policy, trained without any, falls walking at 35-40 ms of delay
# in plain MuJoCo and cannot stand at 60 ms (DARoS-Core
# Systems/DashSystem/CLAUDE.md).
DELAY_MIN_LAG = 0
DELAY_MAX_LAG = 5

# Gains matched to the Unitree G1's published sim-to-real config
# (unitree_rl_gym deploy/deploy_real/configs/g1.yaml: legs 100/150 stiffness,
# 2/4 damping; arms 20-100 stiffness, 1-2 damping) rather than derived from
# Dash's own load, on the user's call that the earlier per-joint tuning
# below -- sized to hold gravity sag under 0.12-0.18 rad -- ran stiffer than
# the real actuators. Trade-off to watch: hip_pitch measured at 42 Nm holding
# the default stance (see the settle test in scripts/), so at stiffness 80 it
# will sag ~0.5 rad under gravity alone -- the policy has to actively fight
# that rather than get it for free from the PD term, unlike the old 350
# setting. If training stalls on just holding a standing pose, this is the
# first thing to revisit.
DASH_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_hip_yaw",
    ".*_hip_roll",
    ".*_hip_pitch",
    ".*_shoulder_pitch",
  ),
  stiffness=80.0,
  damping=2.0,
  effort_limit=EFFORT_LIMIT_HIP,
  armature=ARMATURE,
  delay_min_lag=DELAY_MIN_LAG,
  delay_max_lag=DELAY_MAX_LAG,
)
DASH_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_pitch",),
  stiffness=80.0,
  damping=2.0,
  effort_limit=EFFORT_LIMIT_KNEE,
  armature=ARMATURE,
  delay_min_lag=DELAY_MIN_LAG,
  delay_max_lag=DELAY_MAX_LAG,
)
DASH_ANKLE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch",),
  stiffness=80.0,
  damping=2.0,
  effort_limit=EFFORT_LIMIT_ANKLE,
  armature=ARMATURE,
  delay_min_lag=DELAY_MIN_LAG,
  delay_max_lag=DELAY_MAX_LAG,
)
DASH_ARM_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_roll",
    ".*_shoulder_yaw",
    ".*_elbow_pitch",
  ),
  stiffness=40.0,
  damping=1.0,
  effort_limit=EFFORT_LIMIT_ARM,
  armature=ARMATURE,
  delay_min_lag=DELAY_MIN_LAG,
  delay_max_lag=DELAY_MAX_LAG,
)

# Root height (0.6735) measured, not guessed: fold the keyframe into the model
# and read off the foot sites so the sole sits exactly on the ground at this
# pose. shoulder_pitch is 0.0 rather than swept back, chosen to keep the
# forearm clear of the thigh at rest -- see the capsule/collision notes in
# dash_v1.xml. Six of the eighteen joints have the opposite sign convention in
# the designer's URDF (l_hip_yaw, l_shoulder_pitch, r_hip_roll,
# r_shoulder_pitch, r_shoulder_roll, r_shoulder_yaw); shoulder_pitch is the
# only one of those with a non-zero default here, so keep that in mind before
# giving it a nonzero value again -- both sides must flip together to stay
# mirror-symmetric.
KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.6735),
  joint_pos={
    ".*_hip_pitch": -0.349,
    ".*_knee_pitch": 0.698,
    ".*_ankle_pitch": -0.349,
    ".*_elbow_pitch": -0.8,
    ".*_shoulder_pitch": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^[lr]_foot_collision$"

# Everything collides. Feet get condim=3 and friction priority; every other
# collider is frictionless (condim=1) so grazing self-contacts don't fight the
# gait.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1, ".*": 0},
  friction={_foot_regex: (0.6,)},
)

# Feet only, no self collision. Cheaper; useful for flat-ground training.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

DASH_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    DASH_HIP_ACTUATOR_CFG,
    DASH_KNEE_ACTUATOR_CFG,
    DASH_ANKLE_ACTUATOR_CFG,
    DASH_ARM_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_dash_robot_cfg() -> EntityCfg:
  """Get a fresh Dash robot config instance.

  Returns a new EntityCfg each time so callers that mutate it don't affect
  other tasks sharing the config.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=functools.partial(get_spec, DASH_XML),
    articulation=DASH_ARTICULATION,
  )


# Action scale in rad of joint target per unit of policy output.
#
# mjlab's built-in robots derive this as 0.25 * effort_limit / stiffness, which
# only means something when the effort limit is a real motor spec. Dash's are
# estimates, so the scale is set directly instead: 0.25 rad is the
# usual starting point for humanoid locomotion. Revisit once real torque limits
# are calculated.
DASH_ACTION_SCALE: dict[str, float] = {".*": 0.25}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_dash_robot_cfg())
  viewer.launch(robot.spec.compile())
