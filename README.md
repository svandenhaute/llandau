# present

Thin helpers on top of [manim](https://www.manim.community) and
[manim-slides](https://github.com/jeertmans/manim-slides) for minimal, layered scientific slides.

```python
from manim import *
from present import *

class Results(Deck):
    theme = LIGHT
    def construct(self):
        self.title("train 'likelihood' functions!")
        reaction = self.reveal(image("figures/reaction.png", R.body.left))
        self.reveal(below(text("cost $tilde N^3$"), reaction))
        self.show_svg("figures/logits.svg", R.body.right)   # one slide step per Inkscape layer

SLIDES = order(Results)
```

```
present render slides.py            # render all SLIDES, write slides.html
present render slides.py --only Results -q l
```

## Install

```
pip install -e .
```

External tools:
- **typst** CLI: for any text containing `$...$`, and `eq()` / `typst()`
- **rsvg-convert** (librsvg) or **Inkscape**: rasterizing layered SVGs (or `pip install present[cairosvg]`)
- fonts: Linux Biolinum O (text), Libertinus Math (math; falls back to New Computer Modern Math)

## Modules

| module | contents |
|---|---|
| `theme` | `DARK`/`LIGHT`, `FONT`, type scale `TITLE BODY SMALL LABEL` |
| `text` | `text bold italic eq typst title bullets` |
| `layout` | `R` regions (`R.body.left.top`, `hsplit`, `grid`, `right_of(mob)`...), `place`, `below above left_of right_of align stack row`, gaps `XS S M L XL` |
| `media` | `image svg layered_svg`; layered SVGs are split at render time, cached in `.present_cache/` |
| `deck` | `Deck` (`step reveal title show_svg live`), `order`, `typewrite` |
| `live` | `live_scene`, `present live`: interactive 3D (riemann_viewer) on selected steps |

## Live 3D steps

With `pip install -e ".[live]"` and [riemann_viewer](https://github.com/svandenhaute/riemann_viewer), chosen steps show an interactive
3D view over a region of the slide, in manim-slides' Qt player:

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

```
present render slides.py && present live slides.py
```

Drag rotates, the wheel scrubs frames, pinch zooms; `,` `.` step frames, `P` plays, `D` toggles
the discrete states, `0` resets the camera, the arrows change slides. The rendered slides (video,
HTML, PDF) show a still image of each live view. See `present.live` for the view settings.

Layer conventions for `show_svg`: steps are ordered by the first number in each Inkscape layer
label (layers sharing a number appear together), saved visibility is ignored, and layers whose
label starts with `_` are skipped.
