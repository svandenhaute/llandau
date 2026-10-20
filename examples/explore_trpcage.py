"""Explore trp-cage's NMR models as a trajectory: python explore_trpcage.py 1l2y.pdb"""
import sys

import numpy as np

import molview
from molview import Viewer, pdb
from molview.explorer import explore

positions, numbers, roles = pdb.read(sys.argv[1])
frames, atoms, _ = positions.shape
viewer = Viewer().load(positions, numbers, roles=roles)
viewer.style = "protein"

tangents = np.gradient(positions, axis=0)
tangents /= np.linalg.norm(tangents.reshape(frames, -1), axis=1)[:, None, None]
viewer.add_vectors("tangent", tangents * np.sqrt(atoms), scale=1.5, visible=False)
t = np.linspace(0, 1, frames)
viewer.set_trails(segment_ids=molview.segment_ids(t, np.linspace(0, 1, 7)))
viewer.set_ghost(positions[-1])
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
