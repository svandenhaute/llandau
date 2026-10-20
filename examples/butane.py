"""Test trajectory: the central dihedral of butane turning a full circle."""
import numpy as np
from ase.build import molecule

from riemann_viewer import elements


def butane(num_frames=120):
    """positions (frames, atoms, 3) and atomic numbers (atoms,)."""
    atoms = molecule("trans-butane")
    carbons = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "C"]
    a, b, c, d = sorted(carbons, key=lambda i: atoms.positions[i, 0])
    edges = elements.infer_bonds(atoms.positions, atoms.numbers)
    # atoms on the d side of the b-c bond turn with the dihedral
    side = np.zeros(len(atoms), dtype=bool)
    side[d] = True
    side |= np.isin(np.arange(len(atoms)), edges[(edges == d).any(axis=1)].ravel())
    side[c] = False
    start = atoms.get_dihedral(a, b, c, d)
    frames = []
    for angle in np.linspace(0, 360, num_frames):
        atoms.set_dihedral(a, b, c, d, start + angle, mask=side.astype(int))
        frames.append(atoms.positions.copy())
    return np.asarray(frames, dtype=np.float32), atoms.numbers
