# llandau

Scientific slides drawn live, with interactive geometric models inside them. A deck is
written like a [manim](https://www.manim.community) scene and plays straight from the
code (wgpu): no videos, no render step. Everything on screen is live while you present:
step back, scrub, turn a molecule, change its rendering. The same plots save as paper
figures.

```
pip install llandau            # macOS and Linux wheels; text needs the typst CLI
```

## A deck

![a deck playing: a plot drawing itself, a formula being written, a histogram fading in](https://raw.githubusercontent.com/svandenhaute/llandau/main/docs/slides.gif)

```python
from llandau import *

class Results(Deck):
    def construct(self):
        self.title("sampling a double well")
        ax = Plot(R.body.left, xlabel="$x$", ylabel="energy")
        profile = ax.plot(x, energy, label="$U(x)$")
        self.reveal(ax)                                   # frame and ticks
        self.reveal(profile, anim=Create)                 # the line draws itself
        self.reveal(place(text("$p(x) prop e^(-U \\/ k T)$"), R.body.right), anim=Write)

SLIDES = order(Results)
```

`llandau play examples/slides.py` presents it: the arrows step forward and back, the wheel
scrubs through every animation, F goes full screen. `llandau pdf` and `llandau png` export one
page per step.

## Geometric models

![trp-cage's NMR models playing in a wgpu view: all atoms in flat shading, then the protein tube and side chains, lit](https://raw.githubusercontent.com/svandenhaute/llandau/main/docs/protein.gif)

```python
viewer = Viewer().load(positions, numbers, roles=roles)    # (frames, atoms, 3), (atoms,)
viewer.style = "protein"
viewer.set_trails(segment_ids=segment_ids(t, boundaries))
viewer.color_by("deviation", np.linalg.norm(positions - positions[-1], axis=-1))
explore(viewer, plots={"rmsd": rmsd}, t=t)                 # a window; ? shows every key
```

`python examples/protein.py` is this, on trp-cage's NMR models from the PDB.

A model is drawn with wgpu, and the window is interactive: drag to turn, scroll through the frames, and every setting has a
key (lit or flat shading, protein or atoms, layers, a light or dark theme); R records what you
do into an MP4. A deck shows the same viewer live inside a slide, just as interactive, with
`self.live("name", R.body.right, camera={"azimuth": 30}, frame=60)`.

## Paper figures

![a figure saved by llandau: a damped oscillation, its envelope and noisy samples with error bars](https://raw.githubusercontent.com/svandenhaute/llandau/main/docs/figure.png)

```python
fig, ax = subplots(figsize=(3.4, 2.2))                      # inches, points, Typst math
ax.plot(t, signal, color="k", label="model")
ax.plot(t, envelope, color="#888888", ls="--", label="envelope")
ax.errorbar(ts, samples, yerr=0.05, fmt="o", ms=4, capsize=2, label="samples")
ax.set_xlabel("time $t$ (ps)")
ax.legend()
fig.savefig("figure.pdf")                                   # or .svg, .png
```

`examples/` has the decks and figures above; the modules' docstrings document the rest
(`python -c "import llandau; help(llandau)"`).
