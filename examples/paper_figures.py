"""Paper figures with landau instead of matplotlib: four typical figures, on synthetic data.

    python examples/paper_figures.py      # ode.svg quasipotentials.svg paths.svg energy.svg/.pdf/.png
"""
import numpy as np
from landau import subplots

CB = ["#1b9e77", "#d95f02", "#7570b3"]
rng = np.random.default_rng(0)

# --- ode.svg
lam = np.linspace(-0.9, 0.9, 200)
exact = 0.8 * np.tanh(1.3 * lam) + 0.05 * np.sin(6 * lam)
reverse = 0.8 * np.tanh(1.3 * lam) - 0.08 * np.cos(3 * lam) + 0.03
fig, ax = subplots(figsize=(3.0, 2))
ax.plot(lam, exact, color="k", marker="", label="exact")
ax.plot(lam, reverse, color="#b8b8b8", marker="", label="ODE ($A -> B$)")
for i, index in enumerate([20, 57, 112]):
    ax.scatter(lam[index], reverse[index], s=20, color=CB[i], zorder=50)
ax.legend(bbox_to_anchor=(0.5, 1.0))
ax.set_xlabel("$lambda_(A -> B)$ [-]")
ax.set_ylabel("coordination number [-]")
ax.set_xlim([-1.0, 1.0])
fig.savefig("ode.svg")

# --- quasipotentials.svg
fig, ax = subplots(figsize=(3.0, 2))
highlight = {1: CB[0], 4: CB[1], 8: CB[2]}
for k, center in enumerate(np.linspace(-0.7, 0.75, 12)):
    x = np.linspace(-0.9, 0.9, 200)
    y = 3 * (x - center) ** 2 + 0.1 * np.sin(8 * x + k)
    ax.plot(x, y - y.min(), color=highlight.get(k, "#dadada"), linewidth=1.0 if k in highlight else 0.5,
            marker="", zorder=50 if k in highlight else 0)
ax.set_xlabel("$lambda_(A -> B)$ [-]")
ax.set_ylabel("potential energy [eV]")
ax.set_ylim([-0.05, 1.0])
ax.set_xlim([-1.0, 1.0])
fig.savefig("quasipotentials.svg")


# --- paths.svg: the Müller-Brown surface and three paths
def muller_brown(x, y):
    A, a, b, c = (-200, -100, -170, 15), (-1, -1, -6.5, 0.7), (0, 0, 11, 0.6), (-10, -10, -6.5, 0.7)
    x0, y0 = (1, 0, -0.5, -1), (0, 0.5, 1.5, 1)
    return sum(A[i] * np.exp(a[i] * (x - x0[i]) ** 2 + b[i] * (x - x0[i]) * (y - y0[i]) + c[i] * (y - y0[i]) ** 2)
               for i in range(4))


xs, ys = np.linspace(-1.6, 1.2, 200), np.linspace(-0.4, 2.1, 200)
X, Y = np.meshgrid(xs, ys)
V = (muller_brown(X, Y) + 147) / 100
levels = np.arange(0, 1.51, 0.1)
fig, ax = subplots(figsize=(3.5, 2.5))
surface = ax.contourf(X, Y, V, levels=levels, cmap="Greys_r", alpha=0.65)
ax.contour(X, Y, V, levels=levels, colors="white", alpha=0.2, linewidths=0.4)
colors = ["#ac442f", "#2fac52", "#2f76ac"]
t = np.linspace(0, 1, 100)
start, end = np.array([-0.56, 1.44]), np.array([0.62, 0.03])
for k, (label, bend) in enumerate(zip(["$epsilon$ = 2", "$epsilon$ = 1", "$epsilon$ = 0.1"], [0.0, 0.4, 0.6])):
    path = start + t[:, None] * (end - start) + bend * np.sin(np.pi * t)[:, None] * np.array([-0.8, -0.6])
    ax.plot(path[:, 0], path[:, 1], "-", color=colors[k], linewidth=1.0, zorder=3, label=label)
ax.set_xlim((-1.6, 1.2))
ax.set_ylim((-0.4, 2.1))
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
ax.colorbar(surface, label="energy", format="%.1f", labelpad=12)
ax.legend()
fig.savefig("paths.svg")

# --- energy.svg
fig, ax = subplots(figsize=(3.0, 2.5))
for k, length in enumerate([1.9, 2.5, 2.7]):
    s = np.linspace(0, length, 100)
    ax.plot(s, (1.5 - 0.45 * k) * np.exp(-((s - 0.55 - 0.4 * k) / 0.3) ** 2) + 0.4 * (s / length) ** 2,
            color=colors[k], marker="")
ax.grid()
ax.set_xlabel("Euclidean arc length")
ax.set_ylabel("energy")
fig.savefig("energy.svg")
fig.savefig("energy.pdf")
fig.savefig("energy.png", dpi=200)
