"""Explore trp-cage's NMR models as a trajectory: python explore_trpcage.py 1l2y.pdb"""
import sys

import numpy as np

import riemann_viewer
from riemann_viewer import Viewer, pdb
from riemann_viewer.explorer import explore

positions, numbers, roles = pdb.read(sys.argv[1])
frames, atoms, _ = positions.shape
viewer = Viewer().load(positions, numbers, roles=roles)
viewer.style = "protein"

tangents = np.gradient(positions, axis=0)
tangents /= np.linalg.norm(tangents.reshape(frames, -1), axis=1)[:, None, None]
viewer.add_vectors("tangent", tangents * np.sqrt(atoms), scale=1.5, visible=False)
t = np.linspace(0, 1, frames)
viewer.set_trails(segment_ids=riemann_viewer.segment_ids(t, np.linspace(0, 1, 7)))
viewer.set_ghost(positions[-1])
# every fifth model as "discrete" states: dots, and an alternative to scrub through
discrete = positions[::5]
viewer.add_points("discrete", discrete.reshape(-1, 3), color=(0.9, 0.45, 0.1), radius=0.08)
viewer.set_discrete_frames(discrete)
viewer.set_visible("clashes", True)

deviation = np.linalg.norm(positions - positions[-1], axis=-1)  # per atom, from the ghost
viewer.color_by("deviation", deviation, cmap="viridis")
rmsd = np.sqrt((deviation**2).mean(axis=1))
clashes = []
for frame in range(frames):
    viewer.frame = frame
    clashes.append(len(viewer.clashes()))
viewer.frame = 0
explore(viewer, plots={"rmsd to last": rmsd, "clashes": np.array(clashes, float)}, t=t)
