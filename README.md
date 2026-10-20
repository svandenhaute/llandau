# riemann_viewer

A small wgpu viewer for atomic trajectories. It draws straight into a native
window handle, so it can float over a slide in the manim-slides player, sit in
an explorer window, or render offscreen to numpy.

Install a wheel from the GitHub releases (macOS and Linux, Python >= 3.10):

```
pip install "riemann_viewer[qt] @ https://github.com/svandenhaute/riemann_viewer/releases/download/v0.2.0/riemann_viewer-0.2.0-cp310-abi3-manylinux_2_28_x86_64.whl"
pip install "riemann_viewer[qt] @ https://github.com/svandenhaute/riemann_viewer/releases/download/v0.2.0/riemann_viewer-0.2.0-cp310-abi3-macosx_11_0_arm64.whl"
```

or build it (needs Rust):

```
pip install maturin && maturin develop --release     # MACOSX_DEPLOYMENT_TARGET=15.0 on macOS
python examples/explore_butane.py                     # explorer window
python examples/explore_trpcage.py 1l2y.pdb           # protein: NMR models as frames
python examples/slides_player.py ~/sandbox/polyscope-slides/slides/Dummy.json
python examples/render_layers.py OUT_DIR 1l2y.pdb     # offscreen PNG of each feature
```

```python
from riemann_viewer import Viewer, segment_ids
from riemann_viewer.explorer import explore

viewer = Viewer(theme="light").load(positions, numbers, roles=roles)   # (frames, atoms, 3), (atoms,)
viewer.style = "protein"                      # or "atoms" (ball and stick)
viewer.sidechains = "spheres"                 # van der Waals spheres; or "sticks"
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
non-element color; element colors come from `riemann_viewer.elements`.

On Linux (X11 or Wayland) the Qt widget renders offscreen and paints the image
(`RIEMANN_VIEWER_QT=raster`, the default there); on macOS it draws into its own native
window (`native`), which the overlay over the manim-slides player needs. Without a
GPU, a software Vulkan driver (Mesa's lavapipe) works; `WGPU_BACKEND=vulkan|gl|metal`
picks the backend, `Viewer().adapter` tells which GPU is in use.

`riemann_viewer.qt.ViewerWidget`: left drag rotates, right drag (or shift + drag) pans,
wheel scrubs frames, ctrl + wheel or pinch zooms, double click resets the camera.
Keys stay with the parent window. Over the manim-slides player it has to be an
overlay (`ViewerWidget(player, overlay=True)`): as a native child, the video stops showing.

Layout: `src/camera.rs` (z-up turntable), `src/scene.rs` (structure and layers -> instances),
`src/protein.rs` (chains, backbone spline), `src/clash.rs`, `src/theme.rs`,
`src/render.rs` (wgpu: sphere impostors, instanced lathe meshes, two-pass ghost, overlay),
`src/lib.rs` (Python bindings); `python/riemann_viewer/` (elements, colormaps, PDB reading,
the Qt widget and the explorer).
