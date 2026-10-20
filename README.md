# present

Minimal, layered scientific slides, drawn live. Decks are written like manim scenes (with
manim-slides' steps), but nothing is rendered ahead: [riemann_viewer](https://github.com/svandenhaute/riemann_viewer)
draws every animation as it plays, so there are no videos, no render step and no Qt.

```python
from present import *

class Results(Deck):
    theme = LIGHT
    def construct(self):
        self.title("train 'likelihood' functions!")
        reaction = self.reveal(image("figures/reaction.png", R.body.left))
        self.reveal(below(text("cost $tilde N^3$"), reaction), anim=Write)
        self.show_svg("figures/logits.svg", R.body.right)   # one slide step per Inkscape layer

SLIDES = order(Results)
```

```
present play slides.py              # present: arrows step, F full screen, the wheel scrubs
present play slides.py --only Results --start 3
present pdf slides.py               # one page per step
present png slides.py OUT_DIR
```

## Install

```
pip install riemann_viewer --find-links https://github.com/svandenhaute/riemann_viewer/releases/expanded_assets/v0.4.0
pip install -e .
```

External tools:
- **typst** CLI: all text and math
- **rsvg-convert** (librsvg) or **Inkscape**: rasterizing layered SVGs (or `pip install present[cairosvg]`)
- fonts: Linux Biolinum O (text), Libertinus Math (math; falls back to New Computer Modern Math)
- optional: matplotlib (its colormaps), contourpy (contour plots)

## Modules

| module | contents |
|---|---|
| `mobject` | manim's model: `Mobject VMobject VGroup Group ImageMobject SVGMobject`, colors, directions |
| `shapes` | `Line Arrow Circle Dot Rectangle Polygon RoundedRectangle SurroundingRectangle ...` |
| `animation` | `FadeIn Write Create Transform ReplacementTransform .animate LaggedStart ...`, rate functions |
| `scene` | `Scene` (`add play wait next_slide`), updaters, `ValueTracker`, `always_redraw`, `build` |
| `axes` | manim's `NumberLine Axes NumberPlane ParametricFunction FunctionGraph` |
| `plot` | `Plot`: matplotlib-style plots of numpy data, revealed in steps |
| `theme` | `DARK`/`LIGHT`, `FONT`, type scale `TITLE BODY SMALL LABEL` |
| `text` | `text bold italic eq typst title bullets` (Typst) |
| `layout` | `R` regions (`R.body.left.top`, `hsplit`, `grid`, `right_of(mob)`...), `place`, `below above left_of right_of align stack row`, gaps `XS S M L XL` |
| `media` | `image svg layered_svg`; layered SVGs are split when building, cached in `.present_cache/` |
| `deck` | `Deck` (`step reveal title show_svg live`), `order`, `typewrite` |
| `live` | `live_scene`, `Deck.live`: interactive 3D (riemann_viewer) on chosen steps |

Animations follow manim's definitions (timing, lag, rate functions); those without a native
clip (rotations, `path_arc`, updaters, custom `interpolate_mobject`) are stepped at 60 fps when
the deck is built.

## Plots

```python
ax = Plot(R.body.right, xlabel="time $t$ (ps)", ylabel="energy (eV)")
md = ax.plot(t, e, label="MD")                       # nothing shown yet
with ax.group() as samples:                          # several commands, one step
    ax.scatter(ts, es, c=w, cmap="viridis", s=30, label="samples")
    ax.fill_between(t, lo, hi, alpha=0.2)
legend = ax.legend()

self.reveal(ax)                                      # frame, ticks and labels first
self.reveal(md, anim=Create)                         # the line draws itself
self.reveal(samples, legend, anim=FadeIn)
```

`plot scatter fill_between axhline axvline bar hist errorbar imshow contour contourf text
legend colorbar`, with matplotlib's arguments; sizes in points (times `scale=2`). `ax.c2p(x, y)`
places other mobjects at data coordinates.

## Live 3D steps

```python
@live_scene("butane")
def butane(viewer):                          # a riemann_viewer.Viewer, in the deck's theme
    viewer.load(positions, numbers)          # (frames, atoms, 3), (atoms,)

class Results(Deck):
    def construct(self):
        self.live("butane", R.body.right)    # the next step shows it over R.body.right
        self.reveal(place(text("3D here"), R.body.left))
        self.live("butane", R.body.right, camera={"azimuth": 30}, frame=60)  # moves there
        self.reveal(...)
```

In the player, over a live view: drag rotates, right drag pans, the wheel scrubs frames, pinch
zooms; `,` `.` step frames, `P` plays, `D` toggles the discrete states, `0` resets the camera.
Exports show a still image of each live view. See `present.live` for the view settings.

Layer conventions for `show_svg`: steps are ordered by the first number in each Inkscape layer
label (layers sharing a number appear together), saved visibility is ignored, and layers whose
label starts with `_` are skipped.
