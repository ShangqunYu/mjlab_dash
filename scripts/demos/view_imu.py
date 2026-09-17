"""Show where the IMU site sits on the torso.

The site is a 1 cm sphere inside a solid mesh, so it is invisible in a normal
viewer -- this fades the visual meshes to a shell and paints the IMU red, then
prints its coordinates and how much room is left to the torso wall.

On a hybrid-graphics laptop, prefix with:
  __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
"""

import argparse

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.dash import get_dash_robot_cfg
from mjlab.entity.entity import Entity

# Same scripts/demos/ directory; see the note in pose_joints.py.
from pose_joints import launch_frozen
from view_default_pose import add_ground, resolve_pose

SITE_NAME = "imu_in_torso"
BODY_NAME = "torso"
VISUAL_GROUP = 2  # The `visual` class in dash.xml.


def torso_wall_at(model: mujoco.MjModel, height: float, slab: float = 0.015):
  """Left/right extent of the torso mesh at a given height, in torso frame.

  Read off the mesh rather than the collision box because the box is a padded
  stand-in (0.13 half-width against the mesh's ~0.119 here), and a site placed
  against the box would float outside the shell you see on screen.
  """
  mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "Torso")
  geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "torso_visual")
  adr, num = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]

  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, model.geom_quat[geom_id])
  verts = (
    model.mesh_vert[adr : adr + num] @ rot.reshape(3, 3).T + model.geom_pos[geom_id]
  )

  band = verts[np.abs(verts[:, 2] - height) < slab]
  if not len(band):
    return None
  return float(band[:, 1].min()), float(band[:, 1].max())


def main() -> None:
  parser = argparse.ArgumentParser()
  args = parser.parse_args()

  cfg = get_dash_robot_cfg()
  spec = Entity(cfg).spec
  add_ground(spec)
  model = spec.compile()
  data = mujoco.MjData(model)

  site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
  if site_id < 0:
    raise SystemExit(f"no site named {SITE_NAME!r} in the model")

  # Fade the shell and light up the site. Model-side, not a render flag, so the
  # IMU stays legible from any angle and the feet sites stay distinguishable.
  for geom_id in range(model.ngeom):
    if model.geom_group[geom_id] == VISUAL_GROUP:
      model.geom_rgba[geom_id, 3] = 0.25
  model.site_rgba[:] = [1.0, 0.85, 0.1, 0.6]  # Other sites: dim amber.
  model.site_size[:, 0] = 0.008
  model.site_rgba[site_id] = [1.0, 0.1, 0.1, 1.0]
  model.site_size[site_id] = [0.018, 0.018, 0.018]

  data.qpos[0:3] = cfg.init_state.pos
  data.qpos[3:7] = cfg.init_state.rot
  assert cfg.init_state.joint_pos is not None
  for name, value in resolve_pose(model, cfg.init_state.joint_pos).items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = value
  mujoco.mj_forward(model, data)

  local = model.site_pos[site_id]
  side = "right (-y)" if local[1] < 0 else "left (+y)" if local[1] > 0 else "centreline"
  print(f"\n{SITE_NAME}")
  print(
    f"  in {BODY_NAME} frame : x={local[0]:+.4f}  y={local[1]:+.4f}  z={local[2]:+.4f}"
  )
  print(f"  side               : {side}")
  print(f"  world, default pose: {np.round(data.site_xpos[site_id], 4)}")

  wall = torso_wall_at(model, float(local[2]))
  if wall is not None:
    right, left = wall
    near = right if local[1] < 0 else left
    print(f"  torso wall at z={local[2]:.3f}: y from {right:+.4f} to {left:+.4f}")
    print(f"  clearance to near wall: {abs(near - local[1]) * 1000:+.1f} mm inside")
    if abs(local[1]) > abs(near):
      print("  -> OUTSIDE the shell. Reduce |y|.")

  print("\nRed ball is the IMU; amber are the other sites. Meshes are at 25% alpha.")
  launch_frozen(model, data)


if __name__ == "__main__":
  main()
