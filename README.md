# llandau

Smooth scientific slides, drawn live. Decks are written like manim scenes (with manim-slides'
steps), but nothing is rendered ahead: every animation is drawn as it plays (wgpu), so there are
no videos, no render step and no Qt. Built for presenting [riemann](https://github.com/rotskoff-group/riemann):
interactive 3D views of atomic trajectories, splines, tangents and proteins sit inside the
slides, and plots come straight from numpy arrays.

```python
from llandau import *

class Results(Deck):
    theme = LIGHT
    def construct(self):
        self.title("sampling a double well")
        energy = self.reveal(image("figures/energy.png", R.body.left))
        self.reveal(below(text("$p(x) prop e^(-U(x) \/ k T)$"), energy), anim=Write)
        self.show_svg("figures/sketch.svg", R.body.right)   # one slide step per Inkscape layer

SLIDES = order(Results)
```

The figures above are placeholders; [examples/slides.py](examples/slides.py) is the same deck with
plots drawn from numpy, runnable as is.

```
llandau play slides.py              # present: arrows step, F full screen, the wheel scrubs
llandau play slides.py --only Results --start 3
llandau pdf slides.py               # one page per step
llandau png slides.py OUT_DIR
```

## Install

Wheels for macOS and Linux (Python >= 3.10) are on the GitHub releases (llandau, after Lev
Landau: `landau` on PyPI is another project):

```
pip install llandau --find-links https://github.com/svandenhaute/landau/releases/expanded_assets/v0.3.0
```

or build it (needs Rust): `pip install maturin && maturin develop --release`
(`MACOSX_DEPLOYMENT_TARGET=15.0` on macOS).

External tools:
- **typst** CLI: all text and math
- **rsvg-convert** (librsvg) or **Inkscape**: rasterizing layered SVGs (or `pip install cairosvg`)
- fonts: Linux Biolinum O (text), Libertinus Math (math; falls back to New Computer Modern Math)
- optional: pillow (`llandau pdf | png`), matplotlib (its colormaps), contourpy (contour plots)

## Slides

| module | contents |
|---|---|
| `mobject` | manim's model: `Mobject VMobject VGroup Group ImageMobject SVGMobject`, colors, directions |
| `shapes` | `Line Arrow Circle Dot Rectangle Polygon RoundedRectangle SurroundingRectangle ...` |
| `animation` | `FadeIn Write Create Transform ReplacementTransform .animate LaggedStart ...`, rate functions |
| `scene` | `Scene` (`add play wait next_slide`), updaters, `ValueTracker`, `always_redraw`, `build` |
| `axes` | manim's `NumberLine Axes NumberPlane ParametricFunction FunctionGraph` |
| `plot` | `Plot`: matplotlib-style plots of numpy data, revealed in steps |
| `figure` | `subplots Figure`: plots for papers, saved as SVG, PDF or PNG |
| `theme` | `DARK`/`LIGHT`, `FONT`, type scale `TITLE BODY SMALL LABEL` |
| `text` | `text bold italic eq typst title bullets` (Typst) |
| `layout` | `R` regions (`R.body.left.top`, `hsplit`, `grid`, `right_of(mob)`...), `place`, `below above left_of right_of align stack row`, gaps `XS S M L XL` |
| `media` | `image svg layered_svg`; layered SVGs are split when building, cached in `.llandau_cache/` |
| `deck` | `Deck` (`step reveal title show_svg live`), `order`, `typewrite` |
| `live` | `live_scene`, `Deck.live`: interactive 3D on chosen steps |

Animations follow manim's definitions (timing, lag, rate functions); those without a native
clip (rotations, `path_arc`, updaters, custom `interpolate_mobject`) are stepped at 60 fps when
the deck is built. A deck is one timeline of clips; the player evaluates it at any time, so
playing, stepping back and scrubbing (the wheel) are the same thing.

### Plots

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

### Figures for papers

The same plots, saved as files, in print sizes (inches and points) and the thesis
matplotlibrc style:

```python
from llandau import subplots

fig, ax = subplots(figsize=(3.0, 2.0))                    # inches
ax.plot(lam, exact, color="k", label="exact")
ax.plot(lam, ode, color="#b8b8b8", label="ODE ($A -> B$)")
ax.scatter(lam[i], ode[i], s=20, color=CB0, zorder=50)
ax.set_xlabel("$lambda_(A -> B)$ [-]")                   # math in Typst syntax
ax.set_xlim([-1.0, 1.0])
ax.legend(bbox_to_anchor=(0.5, 1.0))
fig.savefig("ode.svg")                                    # vector; also .pdf, .png (dpi=)
```

SVG and PDF are vector, with text as outlines (it looks the same in any document);
`bbox_inches="tight"` and a transparent background are the defaults. `fig.show()` opens a
window. `subplots(2, 3)` gives an array of plots; `fig.add(mobject)` puts anything else on the
figure; `place(fig, R.body, fit="fill")` puts a finished figure on a slide. matplotlib's
setters work (`set_xlabel set_ylabel set_title set_xlim set_ylim set_xscale set_xticks
set_aspect grid`), as do `zorder`, `colorbar(..., format="%.1f", labelpad=12)` and colors like
`"k"`, `"C0"`. Math is Typst's: `$\lambda_{A\rightarrow B}$` becomes `$lambda_(A -> B)$`.
See `examples/paper_figures.py`.

### Live 3D steps

```python
@live_scene("butane")
def butane(viewer):                          # a llandau.Viewer, in the deck's theme
    viewer.load(positions, numbers)          # (frames, atoms, 3), (atoms,)

class Results(Deck):
    def construct(self):
        self.live("butane", R.body.right)    # the next step shows it over R.body.right
        self.reveal(place(text("3D here"), R.body.left))
        self.live("butane", R.body.right, camera={"azimuth": 30}, frame=60)  # moves there
        self.reveal(...)
```

Over a live view: drag rotates, right drag pans, the wheel scrubs frames, pinch zooms; `,` `.`
step frames, `P` plays, `D` toggles the discrete states, `0` resets the camera. Exports show a
still image of each live view. See `llandau.live` for the view settings.

Layer conventions for `show_svg`: steps are ordered by the first number in each Inkscape layer
label (layers sharing a number appear together), saved visibility is ignored, and layers whose
label starts with `_` are skipped.

## The viewer

```python
from llandau import Viewer, segment_ids
from llandau.explorer import explore

viewer = Viewer(theme="light").load(positions, numbers, roles=roles)   # (frames, atoms, 3), (atoms,)
viewer.style = "protein"                      # or "atoms" (ball and stick)
viewer.sidechains = "spheres"                 # van der Waals spheres; or "sticks"
viewer.shading = "flat"                       # plain colors with a dark outline (cinemol-like); or "lit"
viewer.set_visible("surface", True)           # translucent molecular surface around the heavy atoms
viewer.set_surface(opacity=0.4, colors="uniform", sharpness=2.0)   # colors="atoms": nearest atom's color
viewer.set_visible("clashes", True)           # vdW overlap > viewer.clash_overlap (Å), beyond 3 bonds
viewer.set_trails(segment_ids=segment_ids(t, spline.segment_boundaries))
viewer.add_vectors("tangent", tangents, scale=1.5)
viewer.color_by("kinetic energy", per_atom_values, cmap="fire")   # then viewer.coloring = "kinetic energy"
viewer.set_ghost(target_positions)            # translucent, in the same style
viewer.set_discrete_frames(states)            # (states, atoms, 3): same atoms, e.g. the discrete trajectory
viewer.scrub_discrete = True                  # `frame` now walks those states instead of the path
viewer.frame = 30
image = viewer.render_image(800, 600)         # (600, 800, 4) uint8
explore(viewer, plots={"action": action}, t=t)
```

Roles (as riemann's `atom_role`): 0 hydrogen, 1 side chain heavy atom, 2 backbone N C O, 3 CA.
The protein style draws a smooth tube through the CAs and the side chains, no hydrogens;
trails follow the CAs only. Themes ("light", "dark") set the background and every
non-element color; element colors come from `llandau.elements`.

`explore(viewer, plots=..., t=...)` opens a native window (winit + egui) with the 3D view, a
panel of controls and plots; it works on macOS, X11 and Wayland. Keys: space plays, `,` `.`
step frames, `D` toggles the discrete states, `0` resets the camera, Escape closes. Without a
GPU, a software Vulkan driver (Mesa's lavapipe) works; `WGPU_BACKEND=vulkan|gl|metal` picks the
backend, `Viewer().adapter` tells which GPU is in use.

```
python examples/explore_butane.py                     # explorer window
python examples/explore_trpcage.py 1l2y.pdb           # protein: NMR models as frames
python examples/render_layers.py OUT_DIR 1l2y.pdb     # offscreen PNG of each feature
llandau play examples/live_demo.py                     # a deck with live 3D steps
```

## Layout

- `src/slides/`: decks: `timeline` (clips, marks, manim's animation logic), `paint` (lyon
  tessellation, supersampled wgpu drawing), `svg` (usvg), `player` (the window), `live` (3D
  views in decks)
- `src/`: the viewer: `camera` (z-up turntable), `scene` (structure and layers -> instances),
  `protein`, `surface`, `clash`, `theme`, `render` (sphere impostors, instanced lathe meshes,
  two-pass ghost, overlay), `window` (the explorer), `lib` (Python bindings)
- `python/llandau/`: the Python side of both
