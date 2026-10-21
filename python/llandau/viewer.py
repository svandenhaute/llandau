"""The 3D viewer for atomic trajectories: paths, spline trails, tangents, ghosts, protein
styles and surfaces; drawn offscreen, in the explorer window, or live in a deck."""
import numpy as np

from . import colormaps, elements
from ._core import Viewer as _Viewer

__all__ = ["Viewer", "colormaps", "elements", "segment_ids"]


def _f32(array):
    return None if array is None else np.ascontiguousarray(array, dtype=np.float32)


def _i64(array):
    return None if array is None else np.ascontiguousarray(array, dtype=np.int64)


class Viewer(_Viewer):
    """The Rust viewer plus conveniences that know about elements and themes.

    Array arguments may be any numpy dtype (or torch tensors on the CPU); they
    are converted to what the core expects.
    """

    def __init__(self, theme="light"):
        super().__init__()
        self.theme = theme

    def load(self, positions, numbers, edges=None, roles=None, backbone_colormap="rainbow"):
        """Set a trajectory from positions (frames, atoms, 3) and atomic numbers (atoms,).

        Bonds are inferred from the first frame unless given. With `roles` (0 hydrogen,
        1 side chain, 2 backbone N C O, 3 CA) the protein style becomes available and
        the backbone is colored N to C terminus with `backbone_colormap`.
        """
        positions = _f32(positions)
        if positions.ndim == 2:
            positions = positions[None]
        numbers = np.asarray(numbers)
        if edges is None:
            edges = elements.infer_bonds(positions[0], numbers)
        self.set_structure(
            positions,
            elements.covalent_radii(numbers),
            elements.vdw_radii(numbers),
            elements.colors(numbers, self.theme),
            edges,
            roles,
        )
        for theme in ("light", "dark"):  # element colors follow the theme
            self.set_colors(elements.colors(numbers, theme), theme)
        if roles is not None and backbone_colormap:
            num_alpha = int((np.asarray(roles) == 3).sum())
            self.set_backbone_colors(colormaps.apply(np.linspace(0, 1, num_alpha), backbone_colormap, normalize=False))
        return self

    def set_structure(self, positions, radii, vdw_radii, colors, edges, roles=None):
        super().set_structure(_f32(positions), _f32(radii), _f32(vdw_radii), _f32(colors), _i64(edges), _i64(roles))

    def set_colors(self, colors, theme=None):
        super().set_colors(_f32(colors), theme)

    def set_backbone_colors(self, colors=None):
        super().set_backbone_colors(_f32(colors))

    def set_trails(self, colors=None, segment_ids=None, opacity=1.0):
        super().set_trails(_f32(colors), _i64(segment_ids), opacity)

    def add_vectors(self, name, vectors, color=None, scale=1.0, visible=True):
        super().add_vectors(name, _f32(vectors), color, scale, visible)

    def add_arrows(self, name, origins, vectors, color=None, scale=1.0, visible=True):
        super().add_arrows(name, _f32(origins), _f32(vectors), color, scale, visible)

    def add_points(self, name, positions, colors=None, color=None, radius=0.05, visible=True):
        super().add_points(name, _f32(positions), _f32(colors), color, radius, visible)

    def set_discrete_frames(self, positions=None):
        super().set_discrete_frames(_f32(positions))

    def add_coloring(self, name, colors):
        super().add_coloring(name, _f32(colors))

    def set_ghost(self, positions, opacity=0.35, fade=0.6):
        super().set_ghost(_f32(positions), opacity, fade)

    def set_plots(self, plots: dict, t=None):
        """Scalars to plot along the trajectory, with a cursor at the current frame: name ->
        values per path frame, or name -> (xs, values) on the scale of `t`, the parameter of
        each path frame (default: the frame index). The explorer shows them (G: large); a
        deck's live view shows them large on G, or with plots=True."""
        frames = max(self.num_path_frames, 1)
        t = np.arange(frames, dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64)
        super().set_plots([_curve(name, values, t) for name, values in plots.items()], t.tolist())

    def color_by(self, name, values, cmap="viridis", normalize=True):
        """Add a coloring from per-atom scalars (atoms,) or (frames, atoms), normalized over all frames."""
        self.add_coloring(name, colormaps.apply(values, cmap, normalize))


def _curve(name, values, t):
    """(name, xs, values): values per frame, or (xs, values) on the scale of t."""
    if isinstance(values, tuple):
        xs, values = values
    else:
        values = np.asarray(values, dtype=np.float64)
        xs = t[: len(values)] if len(values) <= len(t) else np.arange(len(values))
    xs = np.asarray(xs, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(xs) != len(values):
        raise ValueError(f"plot {name!r}: {len(xs)} x values for {len(values)} values")
    return name, xs.tolist(), values.tolist()


def segment_ids(t, boundaries):
    """Spline segment of each trail segment (frame i to i + 1), from the frame parameters `t`
    and the segment `boundaries`; for `Viewer.set_trails(segment_ids=...)`."""
    t, boundaries = np.asarray(t), np.asarray(boundaries)
    return np.searchsorted(boundaries[1:-1], t[:-1], side="right").astype(np.int64)
