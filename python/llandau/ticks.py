"""Tick placement and labels for linear and log axes (matplotlib's choices)."""
from __future__ import annotations

import math

import numpy as np

from .text import _minus

def nice_ticks(lo: float, hi: float, target: int) -> np.ndarray:
    """Ticks at 1, 2, 2.5 or 5 times a power of ten, about `target` of them in [lo, hi]."""
    if hi <= lo:
        return np.array([lo])
    raw = (hi - lo) / max(target, 1)
    power = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        step = m * power
        if (hi - lo) / step <= target:
            break
    first = math.ceil(lo / step - 1e-9) * step
    ticks = np.arange(first, hi + step * 1e-9, step)
    return np.round(ticks / step) * step


def log_ticks(lo: float, hi: float):
    """Major ticks at powers of ten in [lo, hi], minor ones at 2..9 times them."""
    a, b = math.floor(math.log10(lo) + 1e-9), math.ceil(math.log10(hi) - 1e-9)
    every = max(1, math.ceil((b - a + 1) / 8))
    major = [10.0 ** k for k in range(a, b + 1) if (k - a) % every == 0 and lo * (1 - 1e-9) <= 10.0 ** k <= hi * (1 + 1e-9)]
    minor = [m * 10.0 ** k for k in range(a - 1, b + 1) for m in range(2, 10)
             if lo <= m * 10.0 ** k <= hi] if every == 1 else []
    return np.array(major), np.array(minor)


def tick_labels(ticks) -> tuple[list[str], int]:
    """Labels for linear ticks with as few decimals as tell them apart, and the
    power of ten factored out of very large or small ones."""
    ticks = np.asarray(ticks, dtype=float)
    biggest = np.abs(ticks).max(initial=0)
    power = 0
    if biggest >= 1e4 or 0 < biggest < 1e-2:
        power = math.floor(math.log10(biggest))
        ticks = ticks / 10.0 ** power
    places = 0
    while places < 8 and np.any(np.abs(np.round(ticks, places) - ticks) > 1e-9 * max(biggest / 10.0 ** power, 1)):
        places += 1
    labels = []
    for v in ticks:
        s = f"{v:.{places}f}"
        labels.append(_minus(s.lstrip("-") if float(s) == 0 else s))  # no "-0"
    return labels, power
