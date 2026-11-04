"""Forward depth from the torso-mounted RealSense D435.

Follows the Go1 setup of Zhuang et al.'s parkour code
(github.com/ZiwenZhuang/parkour, go1_field_distill_config.py), whose camera is a
D435i -- the same depth hardware as Dash's D435. The camera itself
(`realsense_depth`, its pose and FOV) lives in `dash_constants._add_realsense`.

`process_depth` is the one preprocessing path for both sim and hardware: the
deploy side streams 424x240 depth through RealSense's spatial + temporal
filters (`ONBOARD_FILTERS`), converts to metres and calls it, so the network
sees identically cropped, clipped and resized images in both.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensor, CameraSensorCfg


@dataclass(frozen=True)
class DepthCfg:
  # Hardware stream: 424x240, the parkour Go1 setting. Lower resolution lets
  # the stereo match closer (0.12 m vs 0.175 m at 640x480), which matters with
  # the ground this near the lens.
  stream_width: int = 424
  stream_height: int = 240
  # Sim renders at a quarter of that, with the same aspect and FOV.
  render_width: int = 106
  render_height: int = 60
  # Columns dropped from each side, as fractions of the width. The left band is
  # the D435's left-imager-only region with no stereo match (parkour Go1:
  # 60 and 46 px of 424).
  crop_left: float = 60 / 424
  crop_right: float = 46 / 424
  out_height: int = 48
  out_width: int = 64
  depth_range: tuple[float, float] = (0.0, 2.0)
  # Stereo noise, parkour Go1 values (metres): uniform in [0, noise) added
  # to every pixel, and no depth (0) closer than min_distance. Their block
  # and sky artifacts are left out for now.
  min_distance: float = 0.12
  far_distance: float = 2.0
  near_noise: float = 0.02
  far_noise: float = 0.08
  # Frames arrive at 10 Hz with 80-140 ms latency at the 50 Hz policy rate.
  # Latency is a placeholder (parkour Go2, USB 3): measure it on the robot.
  refresh_steps: int = 5
  latency_steps: tuple[int, int] = (4, 7)


DEPTH = DepthCfg()

ONBOARD_FILTERS = {
  "spatial": dict(filter_magnitude=5, filter_smooth_alpha=0.75, filter_smooth_delta=1),
  "spatial_holes_fill": 4,
  "temporal": dict(filter_smooth_alpha=0.75, filter_smooth_delta=1),
}


def process_depth(depth: torch.Tensor, cfg: DepthCfg = DEPTH) -> torch.Tensor:
  """Metric depth (B, H, W) -> network input (B, 1, out_h, out_w) in [0, 1].

  Invalid pixels (0) stay at 0, matching the D435's no-depth value.
  """
  lo, hi = cfg.depth_range
  x = ((depth.clamp(lo, hi) - lo) / (hi - lo)).unsqueeze(1)
  w = x.shape[-1]
  x = x[..., round(cfg.crop_left * w) : w - round(cfg.crop_right * w)]
  x = F.interpolate(x, size=(cfg.out_height, cfg.out_width), mode="bilinear")
  return x.clamp(0.0, 1.0)


class forward_depth:
  """Noisy, 10 Hz forward depth from the RealSense camera sensor.

  Between refreshes the last frame is held, as on hardware; refresh phases
  are staggered across envs. Latency is added by the observation manager's
  delay buffer (see `add_forward_depth`).
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self._sensor_name: str = cfg.params["sensor_name"]
    self._cfg: DepthCfg = cfg.params.get("depth_cfg", DEPTH)
    n = env.num_envs
    self._phase = torch.randint(0, self._cfg.refresh_steps, (n,), device=env.device)
    self._step = 0
    self._frame = torch.zeros(
      n, 1, self._cfg.out_height, self._cfg.out_width, device=env.device
    )
    self._fresh = torch.ones(n, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._fresh[slice(None) if env_ids is None else env_ids] = True

  def _corrupt(self, d: torch.Tensor) -> torch.Tensor:
    c = self._cfg
    far = d > c.far_distance
    close = d < c.min_distance
    noise = torch.rand_like(d) * torch.where(far, c.far_noise, c.near_noise)
    return torch.where(close, torch.zeros_like(d), d + noise)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    depth_cfg: DepthCfg = DEPTH,
    corrupt: bool = True,
  ) -> torch.Tensor:
    del sensor_name, depth_cfg
    sensor: CameraSensor = env.scene[self._sensor_name]
    assert sensor.data.depth is not None
    due = ((self._step + self._phase) % self._cfg.refresh_steps == 0) | self._fresh
    self._step += 1
    if due.any():
      d = sensor.data.depth[due, ..., 0]
      # The renderer returns 0 where a ray hits nothing; that is open space,
      # so far, not the D435's 0 = no stereo match (too close or dropout).
      d = torch.where(d > 0, d, self._cfg.depth_range[1])
      if corrupt:
        d = self._corrupt(d)
      self._frame[due] = process_depth(d, self._cfg)
      self._fresh[due] = False
    return self._frame.flatten(1)


def add_forward_depth(
  cfg: ManagerBasedRlEnvCfg, play: bool = False, depth_cfg: DepthCfg = DEPTH
) -> None:
  """Add the D435 camera sensor, a `depth` observation group and camera DR."""
  cam = CameraSensorCfg(
    name="forward_depth",
    camera_name="robot/realsense_depth",
    width=depth_cfg.render_width,
    height=depth_cfg.render_height,
    data_types=("depth",),
    # Terrain (0) and the robot's visual meshes (2). Not group 1, the camera's
    # own housing, nor the oversized group-3 colliders (linkage capsules,
    # camera box), which the real lens can't see.
    enabled_geom_groups=(0, 2),
    use_textures=False,
    use_shadows=False,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam,)

  lo, hi = depth_cfg.latency_steps
  cfg.observations["depth"] = ObservationGroupCfg(
    terms={
      "forward_depth": ObservationTermCfg(
        func=forward_depth,
        params={
          "sensor_name": cam.name,
          "depth_cfg": depth_cfg,
          "corrupt": not play,
        },
        delay_min_lag=0 if play else lo,
        delay_max_lag=0 if play else hi,
        delay_update_period=depth_cfg.refresh_steps,
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  if play:
    return
  # Mount tolerance of a printed jig: a few mm and ~1 deg, plus per-unit FOV
  # spread (this unit measures 58.9 deg vertical at 16:9).
  cam_asset = SceneEntityCfg("robot", camera_names=("realsense_depth",))
  cfg.events["realsense_pos"] = EventTermCfg(
    func=dr.cam_pos,
    mode="startup",
    params={
      "asset_cfg": cam_asset,
      "operation": "add",
      "ranges": {0: (-0.003, 0.003), 1: (-0.003, 0.003), 2: (-0.003, 0.003)},
    },
  )
  cfg.events["realsense_quat"] = EventTermCfg(
    func=dr.cam_quat,
    mode="startup",
    params={
      "asset_cfg": cam_asset,
      "roll_range": (-0.01, 0.01),
      "pitch_range": (-0.02, 0.02),
      "yaw_range": (-0.01, 0.01),
    },
  )
  cfg.events["realsense_fovy"] = EventTermCfg(
    func=dr.cam_fovy,
    mode="startup",
    params={"asset_cfg": cam_asset, "operation": "add", "ranges": (-1.0, 1.0)},
  )
