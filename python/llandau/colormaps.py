"""Scalars to colors, without a plotting library."""
import numpy as np

# a few anchor colors per map, interpolated linearly
MAPS = {
    "rainbow": [(0.27, 0.35, 0.80), (0.20, 0.65, 0.85), (0.30, 0.75, 0.45), (0.95, 0.80, 0.25), (0.90, 0.35, 0.25)],
    "fire": [(0.0, 0.0, 0.0), (0.55, 0.05, 0.05), (0.9, 0.35, 0.05), (1.0, 0.75, 0.2), (1.0, 1.0, 0.85)],
    "viridis": [(0.27, 0.00, 0.33), (0.23, 0.32, 0.55), (0.13, 0.57, 0.55), (0.37, 0.79, 0.38), (0.99, 0.91, 0.14)],
    "coolwarm": [(0.23, 0.30, 0.75), (0.55, 0.69, 1.0), (0.87, 0.87, 0.87), (0.96, 0.60, 0.48), (0.71, 0.02, 0.15)],
}


def apply(values, cmap="viridis", normalize=True):
    """Colors (..., 3) for `values`; min-max normalized over the finite entries unless `normalize` is False.

    `cmap` is a name from MAPS, a colorcet name, or a list of colors (rgb tuples or hex strings).
    """
    values = np.asarray(values, dtype=np.float64)
    if normalize:
        finite = np.isfinite(values)
        lo, hi = (values[finite].min(), values[finite].max()) if finite.any() else (0.0, 1.0)
        values = (values - lo) / (hi - lo) if hi > lo else np.full_like(values, 0.5)
    values = np.clip(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    lut = _lut(cmap)
    index = values * (len(lut) - 1)
    lo = np.floor(index).astype(int)
    hi = np.minimum(lo + 1, len(lut) - 1)
    frac = (index - lo)[..., None]
    return (lut[lo] * (1 - frac) + lut[hi] * frac).astype(np.float32)


def _lut(cmap):
    if isinstance(cmap, str) and cmap in MAPS:
        return np.asarray(MAPS[cmap], dtype=np.float64)
    if isinstance(cmap, str):
        import colorcet
        cmap = getattr(colorcet, cmap, None) or colorcet.palette[cmap]
    return np.asarray([_rgb(c) for c in cmap], dtype=np.float64)


def _rgb(color):
    if isinstance(color, str):
        h = color.lstrip("#")
        return tuple(int(h[k:k + 2], 16) / 255 for k in (0, 2, 4))
    return tuple(float(c) for c in color[:3])
