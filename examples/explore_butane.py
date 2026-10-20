"""Explore the butane dihedral scan, with trails and tangents."""
import numpy as np

from butane import butane
from landau import Viewer
from landau.explorer import explore

positions, numbers = butane()
viewer = Viewer().load(positions, numbers)
viewer.set_sizes(radius_scale=0.6)
viewer.add_vectors("tangent", np.gradient(positions, axis=0), scale=8.0)
explore(viewer)
