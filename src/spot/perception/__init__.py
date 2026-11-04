"""Perception helpers built on Spot's built-in world model.

This package wraps the ``LocalGridService`` and ``RayCastService`` with
numpy-friendly helpers:

* :mod:`local_grid_helpers` — fetch + decode ``LocalGrid`` responses (RAW
  and RLE encodings, every documented ``cell_format``) into numpy arrays,
  plus a module-level cache of discovered grid type names per robot.
* :mod:`obstacle_query` — "is there something near me?" helpers layered
  on top of the decoded grids, returning body-frame ``(distance, bearing)``
  pairs.
* :mod:`raycast_helpers` — thin wrapper around ``RayCastClient`` that
  handles the Vec3 boilerplate, service-unavailable errors, and hit-type
  name decoding.

All helpers are read-only and free of side effects on the robot.
"""
