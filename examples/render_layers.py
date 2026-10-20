"""Offscreen renders of each feature, into a folder: python render_layers.py OUT_DIR"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import landau
from landau import Viewer, pdb

out = Path(sys.argv[1])
pdb_path = sys.argv[2]
positions, numbers, roles = pdb.read(pdb_path)
frames, atoms, _ = positions.shape
print(f"{frames} frames, {atoms} atoms")

viewer = Viewer().load(positions, numbers, roles=roles)
# tangents: finite differences, unit system norm, times sqrt(atoms) as in riemann
tangents = np.gradient(positions, axis=0)
tangents /= np.linalg.norm(tangents.reshape(frames, -1), axis=1)[:, None, None]
tangents *= np.sqrt(atoms)
t = np.linspace(0, 1, frames)
viewer.set_trails(segment_ids=landau.segment_ids(t, np.linspace(0, 1, 7)))
viewer.frame = 10


def save(name):
    Image.fromarray(viewer.render_image(1000, 750)).save(out / f"{name}.png")


viewer.set_sizes(radius_scale=0.5)
save("1_atoms_trails")
viewer.add_vectors("tangent", tangents, scale=1.5)
viewer.set_visible("trails", False)
save("2_atoms_tangents")
viewer.set_visible("tangent", False)
viewer.set_visible("trails", True)

viewer.style = "protein"
save("3_protein_sticks")
viewer.set_ghost(positions[-1])
save("4_protein_ghost")
viewer.set_ghost(None)
viewer.sidechains = "spheres"
viewer.set_visible("clashes", True)
viewer.clash_overlap = 0.25
print("clashes in frame 10:", len(viewer.clashes()))
save("5_protein_spheres_clashes")
viewer.set_visible("clashes", False)
viewer.sidechains = "sticks"
viewer.set_visible("trails", False)
viewer.set_visible("surface", True)
save("5b_protein_surface")
viewer.set_visible("surface", False)
viewer.set_visible("trails", True)
displacement = np.linalg.norm(positions - positions.mean(axis=0), axis=-1)
viewer.color_by("displacement", displacement, cmap="fire")
viewer.coloring = "displacement"
viewer.style = "atoms"
save("6_atoms_coloring")
viewer.coloring = None
viewer.theme = "dark"
viewer.style = "protein"
viewer.set_ghost(positions[-1])
save("7_dark_protein_ghost")
