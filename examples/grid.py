"""A grid of live 3D views: trp-cage's NMR models, ten of them side by side; a click
opens one up for a closer look (drag to turn it, H for the panel of controls, 0 for the
step's view again), the next click puts it back.

    llandau play examples/grid.py       # downloads 1l2y.pdb next to this file the first time
"""
import urllib.request
from pathlib import Path

from llandau import *
from llandau import pdb

path = Path(__file__).with_name("1l2y.pdb")
if not path.exists():
    urllib.request.urlretrieve("https://files.rcsb.org/download/1L2Y.pdb", path)
positions, numbers, roles = pdb.read(path)

MODELS = list(range(0, len(positions), max(1, len(positions) // 10)))[:10]

for k, model in enumerate(MODELS):
    @live_scene(f"model{k}")
    def scene(viewer, model=model):
        viewer.load(positions[model], numbers, roles=roles)
        viewer.style = "protein"


class Models(Deck):
    theme = LIGHT

    def construct(self):
        self.title("NMR models of trp-cage: click one")
        for k, cell in enumerate(R.body.grid(2, 5, gap=0.2)):
            self.live(f"model{k}", cell, opens=R.body, camera={"azimuth": 36 * k})


SLIDES = order(Models)
