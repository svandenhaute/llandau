"""The README's figure: a damped oscillation and noisy samples, as a paper figure.

    python examples/figure.py       # figure.png
"""
import numpy as np
from llandau import subplots

rng = np.random.default_rng(1)
t = np.linspace(0, 10, 300)
signal = np.exp(-t / 4) * np.cos(2 * t)
ts = t[::15]
samples = np.exp(-ts / 4) * np.cos(2 * ts) + rng.normal(0, 0.05, len(ts))

fig, ax = subplots(figsize=(3.4, 2.2))
ax.plot(t, signal, color="k", label="model")
ax.plot(t, np.exp(-t / 4), color="#888888", ls="--", label="envelope")
ax.errorbar(ts, samples, yerr=0.05, fmt="o", ms=4, color="C0", capsize=2, label="samples")
ax.set_xlabel("time $t$ (ps)")
ax.set_ylabel("$x$ (Å)")
ax.legend()
fig.savefig("figure.png", dpi=220, transparent=False)
