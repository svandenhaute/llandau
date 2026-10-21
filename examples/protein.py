"""The README's 3D example: trp-cage's NMR models (PDB 1L2Y) as a trajectory, explored.

    python examples/protein.py      # downloads 1l2y.pdb next to this file the first time
"""
import urllib.request
from pathlib import Path

import numpy as np
from llandau import Viewer, pdb, segment_ids
from llandau.explorer import explore

path = Path(__file__).with_name("1l2y.pdb")
if not path.exists():
    urllib.request.urlretrieve("https://files.rcsb.org/download/1L2Y.pdb", path)
positions, numbers, roles = pdb.read(path)
frames = len(positions)

viewer = Viewer().load(positions, numbers, roles=roles)
viewer.style = "protein"
t = np.linspace(0, 1, frames)
viewer.set_trails(segment_ids=segment_ids(t, np.linspace(0, 1, 7)))
viewer.color_by("deviation", np.linalg.norm(positions - positions[-1], axis=-1))
rmsd = np.sqrt(((positions - positions[-1]) ** 2).sum(axis=-1).mean(axis=1))
explore(viewer, plots={"rmsd": rmsd}, t=t)
