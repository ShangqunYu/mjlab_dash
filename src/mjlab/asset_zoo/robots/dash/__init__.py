"""Dash humanoid.

Moved from the dash_bar_trajopt repo's dash_mjlab package (Sep 2026), which
keeps the upper-body-only variant (manipulation/bar tasks) and depends on
this package for the full-body robot used by the velocity task.

Only the designer's URDF-imported model (dash_v1.xml, formerly distinguished
as "v2" against an older IsaacLab export -- that older model and its
KNEES_BENT_KEYFRAME were retired since every policy in this repo trains
against the URDF-imported one) is supported. Named v1 rather than dropping
the version entirely, since a future URDF revision is expected to become v2.
"""

from mjlab.asset_zoo.robots.dash.dash_constants import (
  DASH_ACTION_SCALE,
  DASH_ARTICULATION,
  DASH_XML,
  FEET_ONLY_COLLISION,
  FULL_COLLISION,
  get_dash_robot_cfg,
)

__all__ = [
  "DASH_ACTION_SCALE",
  "DASH_ARTICULATION",
  "DASH_XML",
  "FEET_ONLY_COLLISION",
  "FULL_COLLISION",
  "get_dash_robot_cfg",
]
