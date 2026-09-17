"""Pose the robot by hand in MuJoCo's viewer.

A scratchpad for "what does this joint actually do" and for building a pose by
eye. Separate from ``view_default_pose.py``, which is a check -- it asserts one
specific pose. Nothing here is authoritative about what training spawns in.

Time never advances. The viewer is the full `simulate` GUI, but the physics
thread driving it is ours and only ever calls ``mj_forward``, so the robot
cannot fall, sag, or be dragged back to zero by its actuators. It opens at the
configured init_state pose and only what you drag moves.

  - Sliders: right-hand panel, "Joint" section. Tab / Shift-Tab toggle the
    left and right panels.
  - The play/pause button and Space do nothing here, by design. To watch the
    pose actually behave, use ``view_default_pose.py --physics``.

On a hybrid-graphics laptop, prefix with:
  __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia
"""

import argparse
import atexit
import threading
import time

import mujoco
import mujoco.viewer
from mjlab.asset_zoo.robots.dash import get_dash_robot_cfg
from mjlab.entity.entity import Entity
from mujoco._simulate import Simulate

# Same repo, same scripts/demos/ directory -- this file is run as a script,
# so that directory is on sys.path. Reused rather than copied so the floor
# and the pattern resolution cannot drift apart between the two tools.
from view_default_pose import add_ground, geom_group_option, resolve_pose


def launch_frozen(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  opt: mujoco.MjvOption | None = None,
) -> None:
  """The managed viewer, driven by a physics loop that never steps.

  Neither public entry point can do this. `launch_passive` leaves the joint
  sliders read-only -- editable sliders are a property of the *managed* GUI,
  which is what the `run_physics_thread=True` argument below selects. `launch`
  gives editable sliders but starts its loop running, and `Simulate.run` is a
  read-only property, so there is no way to ask it to start paused.

  What is left is to keep the managed GUI and swap out only the loop behind it:
  `mj_forward` instead of `mj_step`, which is exactly what MuJoCo's own loop
  does while paused (see `viewer._physics_loop`) -- it recomputes kinematics so
  a dragged slider redraws, without integrating time.
  """
  viewer = mujoco.viewer
  simulate = Simulate(
    mujoco.MjvCamera(),
    opt if opt is not None else mujoco.MjvOption(),
    mujoco.MjvPerturb(),
    None,  # user_scn: the managed GUI owns its own scene.
    True,  # run_physics_thread: selects the managed (non-passive) GUI.
    None,  # key_callback.
  )

  if viewer._MJPYTHON is None:
    if not viewer.glfw.init():
      raise mujoco.FatalError("could not initialize GLFW")
    atexit.register(viewer.glfw.terminate)

  def frozen_loop() -> None:
    # The render thread only learns about the model through this handshake, so
    # it has to happen here, on the physics thread, exactly as upstream does.
    viewer._reload(simulate, lambda: (model, data))
    while not simulate.exitrequest:
      time.sleep(0.001)  # Yield to the render thread; upstream does the same.
      with simulate.lock():
        mujoco.mj_forward(model, data)

  physics = threading.Thread(target=frozen_loop)
  atexit.register(simulate.exit)
  physics.start()
  simulate.render_loop()
  atexit.unregister(simulate.exit)
  physics.join()
  simulate.destroy()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--collision",
    action="store_true",
    help="Show the collision primitives (geom group 3) instead of the visual "
    "meshes (group 2). MuJoCo hides group 3 by default, so this is the only "
    "way to see the fitted capsules and foot boxes without hunting through the "
    "Rendering panel.",
  )
  args = parser.parse_args()

  cfg = get_dash_robot_cfg()
  spec = Entity(cfg).spec
  add_ground(spec)
  model = spec.compile()
  data = mujoco.MjData(model)

  # Belt and braces: nothing integrates, so there is nothing for gravity to act
  # on, but this keeps the state honest for anything that reads the model.
  model.opt.gravity[:] = 0.0

  data.qpos[0:3] = cfg.init_state.pos
  data.qpos[3:7] = cfg.init_state.rot
  assert cfg.init_state.joint_pos is not None
  for name, value in resolve_pose(model, cfg.init_state.joint_pos).items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = value
  mujoco.mj_forward(model, data)

  print(__doc__)
  launch_frozen(model, data, geom_group_option(collision=args.collision))


if __name__ == "__main__":
  main()
