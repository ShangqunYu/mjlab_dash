"""Dash sim-to-real domain randomization, shared by its velocity and tracking tasks."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg


def add_hardware_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add Dash's hardware-derived randomization to cfg's events.

  Expects the stock `base_com` event with its body names already set to the torso.
  """
  # Sim-to-real randomization, from the 2026-10-09 hardware runs (see
  # DARoS-Core Systems/DashSystem/CLAUDE.md, "Torque figures, measured").
  #
  # COM: standing ankle torque on the robot (-10.5 and -13.1 Nm, both ankles)
  # matches this model with the torso COM moved 2-3.5 cm in -x; nominal gives
  # -3.8. The stock +/-2.5 cm stops short of that, so the x range is skewed.
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.04, 0.04),
    1: (-0.025, 0.025),
    2: (-0.03, 0.03),
  }
  # Mass: the robot weighs ~38 kg (scale, 2026-10-09); this model totals 40.0.
  # Per-body scale 0.88-1.05, so the mean total is ~38.6 kg. pseudo_inertia,
  # not body_mass, so inertia scales with mass. It rebuilds body_ipos from the
  # defaults, so it must run BEFORE base_com or it erases the torso COM offset:
  # base_com is re-inserted below to move it after this term.
  cfg.events["body_mass"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
      "alpha_range": (-0.064, 0.024),  # mass x e^(2 alpha) = 0.88-1.05
    },
  )
  cfg.events["base_com"] = cfg.events.pop("base_com")
  # The hardware runs the policy's own kp/kd (--policy-gains), but what the
  # motor delivers per unit of either has never been measured.
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_names=[".*"]),
      "kp_range": (0.8, 1.2),
      "kd_range": (0.7, 1.3),
      "operation": "scale",
    },
  )
  # Reflected rotor inertia is unmeasured. ARMATURE (0.01) came from an IsaacLab
  # config; reflected inertia grows with gear ratio squared, and the knee (18:1)
  # and ankle (36:1) use the same motor as the 10:1 hips, so if 0.01 is right
  # for a hip the ankle is ~0.13. Ranges span that scaling, widely.
  cfg.events["armature"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "ranges": {
        r".*_hip_.*": (0.01, 0.03),
        r".*_shoulder_pitch": (0.01, 0.03),
        r".*_knee_pitch": (0.02, 0.05),
        r".*_ankle_pitch": (0.01, 0.03),
        r".*_(shoulder_roll|shoulder_yaw|elbow_pitch)": (0.008, 0.01),
      },
      "operation": "abs",
    },
  )
  # Coulomb friction at the output, read from dash_home's sweep torque at
  # steady speed: ankles 0.6-1.1 Nm, knee <= 1.4, hip yaw <= 2-2.7. Hip
  # pitch/roll and the arms are unmeasured and borrow the hip yaw bound.
  cfg.events["joint_friction"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "ranges": {
        r".*_hip_.*": (0.0, 2.5),
        r".*_knee_pitch": (0.0, 1.5),
        r".*_ankle_pitch": (0.0, 1.5),
        r".*_(shoulder_pitch|shoulder_roll|shoulder_yaw|elbow_pitch)": (0.0, 2.0),
      },
      "operation": "abs",
    },
  )
