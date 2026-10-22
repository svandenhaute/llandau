"""Per-element radii and colors (light and dark themes)."""
import numpy as np

# covalent radii (Å, Cordero et al. 2008), index = atomic number
COVALENT = np.array([
    0.2, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66, 1.41, 1.21, 1.11, 1.07,
    1.05, 1.02, 1.06, 2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22,
    1.20, 1.19, 1.20, 1.20, 1.16,
])

# van der Waals radii (Å, Bondi 1964; H from Rowland & Taylor), others fall back to 2.0
VDW = {1: 1.10, 2: 1.40, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 10: 1.54, 11: 2.27, 12: 1.73,
       14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 18: 1.88, 19: 2.75, 35: 1.85, 53: 1.98}

# element colors in [0, 1]; light: on a white background (as in cinemol figures)
COLORS = {
    "light": {1: (0.82, 0.82, 0.82), 6: (0.39, 0.39, 0.39), 7: (0.21, 0.58, 0.79),
              8: (0.87, 0.25, 0.20), 16: (0.90, 0.75, 0.20), 15: (0.95, 0.55, 0.15)},
    "dark": {1: (0.92, 0.92, 0.92), 6: (0.62, 0.62, 0.62), 7: (0.35, 0.65, 0.95),
             8: (0.95, 0.35, 0.30), 16: (0.95, 0.82, 0.30), 15: (1.0, 0.6, 0.2)},
}
OTHER = (0.85, 0.45, 0.75)


def covalent_radii(numbers):
    numbers = np.asarray(numbers)
    return np.where(numbers < len(COVALENT), COVALENT[np.minimum(numbers, len(COVALENT) - 1)], 1.5)


def vdw_radii(numbers):
    return np.array([VDW.get(int(z), 2.0) for z in numbers])


def colors(numbers, theme="light"):
    table = COLORS[theme]
    return np.array([table.get(int(z), OTHER) for z in numbers])


def infer_bonds(positions, numbers, tolerance=1.15):
    """Bonds (edges, 2) between atoms closer than `tolerance` times the sum of their covalent radii."""
    from ._core import infer_bonds

    positions = np.ascontiguousarray(positions, dtype=np.float32)
    return infer_bonds(positions, np.ascontiguousarray(covalent_radii(numbers), dtype=np.float32), tolerance)
