# Development Workflow

**Always use `uv run`, not python**.

```sh

# 1. Make changes.

# 2. Type check.
uv run ty check  # Fast
uv run pyright  # More thorough, but slower

# 3. Run tests.
uv run pytest tests/  # Single suite
uv run pytest tests/<test_file>.py  # Specific file

# 4. Format and lint before committing.
uv run ruff format
uv run ruff check --fix
```

We've bundled common commands into a Makefile for convenience.

```sh
make format     # Format and lint
make type       # Type-check
make check      # make format && make type
make test-fast  # Run tests excluding slow ones
make test       # Run the full test suite
make docs       # Build documentation
```

Always run `make check` before committing. This runs formatting, linting,
and type checking. Do not commit code that fails type checking.

Before creating a PR, ensure all checks pass with `make test`.

When making user-facing changes, add an entry to `docs/source/changelog.rst`
under the "Upcoming version (not yet released)" section using
Added/Changed/Fixed categories. Reference issues with `:issue:\`123\``
(renders as a link to the GitHub issue).

# Commits and PRs

- Put `Fixes #<number>` at the end of the commit message body, not in
  the title.
- PR body should be plain, concise prose. No section headers, checklists,
  or structured templates. Describe the problem, what the change does, and
  any non-obvious tradeoffs. A good PR description reads like a short
  paragraph to a colleague, not a form.
- PR and commit messages are rendered on GitHub, so don't hard-wrap them
  at 88 columns. Let each sentence flow on one line.

Some style guidelines to follow:
- Line length limit is 88 columns. This applies to code, comments, and docstrings.
- Avoid local imports unless they are strictly necessary (e.g. circular imports).
- Tests should follow these principles:
  - Use functions and fixtures; do not use test classes.
  - Favor targeted, efficient tests over exhaustive edge-case coverage.
  - Prefer running individual tests rather than the full test suite to improve iteration speed.

# This fork: Dash humanoid

This is a fork of upstream mjlab (`ShangqunYu/mjlab_dash`), extended with the
Dash humanoid robot and its locomotion task, following the exact same layout
mjlab uses for its own robots (G1, Go1, YAM).

It's actively co-developed alongside a **separate sibling repo**,
`~/Repositories/dash_bar_trajopt` ("Dash humanoid robot tasks for mjlab" --
named for its original focus, bar/trajectory-optimization work, which still
lives there). That repo depends on this one via a local editable `uv` path
(`[tool.uv.sources]` -> `{ path = "../mjlab_dash", editable = true }`), so any
change here is immediately live there on the next run -- no version bump or
publish step. `dash_bar_trajopt` has its own `CLAUDE.md`/memory; this repo's
memory and session history are entirely separate (Claude Code scopes both
per git repo) and neither syncs across machines, so a session started here
has no visibility into work done in that repo unless it's written down in a
file like this one.

**What lives here** (added on top of stock mjlab):

- `src/mjlab/asset_zoo/robots/dash/` -- the Dash humanoid (18 DOF: 5/leg
  hip yaw/roll/pitch + knee + ankle, 4/arm shoulder pitch/roll/yaw + elbow).
  `dash_constants.py` defines:
  - Actuators grouped **by physical motor**, not by limb: hip yaw, hip roll,
    hip pitch, and shoulder_pitch all share one motor category
    (`DASH_HIP_ACTUATOR_CFG`); knee_pitch has its own, higher-torque motor
    (`DASH_KNEE_ACTUATOR_CFG`); ankle and the remaining arm joints
    (shoulder_roll/yaw, elbow) are their own categories. `EFFORT_LIMIT_*` and
    `ARMATURE` are explicitly flagged as estimates (scaled from robot mass,
    not real motor datasheets) -- replace before any hardware deployment.
  - PD gains (stiffness 80/damping 2 on legs, 40/1 on arms) are matched to
    Unitree G1's published sim-to-real config
    (`unitree_rl_gym/deploy/deploy_real/configs/g1.yaml`) rather than derived
    from Dash's own load -- deliberately softer than an earlier
    load-derived tuning (350 stiffness), on the read that the real actuators
    can't hit that stiffness. Revisit if training stalls just holding a
    stance.
  - One default pose, `KNEES_BENT_KEYFRAME`, and one model, `xmls/dash_v1.xml`
    (imported from the designer's URDF by `dash_bar_trajopt`'s
    `scripts/import_dash_urdf.py`, which writes its output directly into this
    repo). Named `v1`, not because a second model exists today, but to leave
    `dash_v2.xml` free for whenever the designer sends the next URDF
    revision -- an older IsaacLab-exported model and its own keyframe were
    retired since nothing trains against it anymore.
- `src/mjlab/tasks/velocity/config/dash/` -- the locomotion task
  (`Mjlab-Velocity-Rough-Dash`, `Mjlab-Velocity-Flat-Dash`). The actor's
  observations deliberately exclude `base_lin_vel` (kept in the critic only,
  as privileged sim-only information) -- the real robot has no state
  estimator and can't get true base linear velocity, matching Unitree's own
  G1 sim-to-real recipe, which drops it for the same reason.
- `scripts/demos/{view_default_pose,pose_joints,view_imu}.py` -- MuJoCo
  viewer tools for inspecting the Dash robot's pose and collision geometry by
  hand. Robot-specific, so they live here rather than in `dash_bar_trajopt`.

**What does *not* live here**: the upper-body-only Dash variant (used for
manipulation and bar-angle tasks) and the bar-trajectory-optimization work
the sibling repo is named for. Both stay in `dash_bar_trajopt` -- this fork
is scoped to full-body locomotion.
