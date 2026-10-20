"""Live 3D steps on a light deck: the butane dihedral scan.

    llandau play examples/live_demo.py
"""
import numpy as np
from ase.build import molecule
from llandau import *


def dihedral_scan(num_frames=120):
    """butane turning about its central bond: positions (frames, atoms, 3), numbers (atoms,)"""
    atoms = molecule("trans-butane")
    carbons = sorted((i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "C"),
                     key=lambda i: atoms.positions[i, 0])
    a, b, c, d = carbons
    moving = atoms.positions[:, 0] > (atoms.positions[b, 0] + atoms.positions[c, 0]) / 2
    start = atoms.get_dihedral(a, b, c, d)
    frames = []
    for angle in np.linspace(0, 360, num_frames):
        atoms.set_dihedral(a, b, c, d, start + angle, mask=moving.astype(int))
        frames.append(atoms.positions.copy())
    return np.asarray(frames), atoms.numbers


@live_scene("butane")
def butane(viewer):
    positions, numbers = dihedral_scan()
    viewer.load(positions, numbers)
    viewer.set_sizes(radius_scale=0.6)
    viewer.add_vectors("tangent", np.gradient(positions, axis=0), scale=8.0, visible=False)
    viewer.set_discrete_frames(positions[::20])


class Live(Deck):
    theme = LIGHT

    def construct(self):
        self.title("a dihedral scan")
        before = self.reveal(place(text("before the 3D step"), R.body))
        body_left, body_right = R.body.hsplit(0.4)
        first = self.live("butane", body_right, frame=0)  # a still image; the live view covers it
        notes = self.reveal(place(text("drag, scroll, pinch"), body_left), replace=before)
        second = self.live("butane", body_right, camera={"azimuth": 0, "elevation": 60}, frame=60,
                           show=["tangent"])
        tangents = self.reveal(place(text("tangents, from above"), body_left), replace=[notes, first])
        self.reveal(place(text("after"), R.body), replace=[second, tangents])


SLIDES = order(Live)
