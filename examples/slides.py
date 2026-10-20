"""The README deck, with plots drawn from numpy instead of figure files.

    llandau play examples/slides.py
"""
import numpy as np
from llandau import *


class Results(Deck):
    theme = LIGHT

    def construct(self):
        self.title("sampling a double well")
        top, bottom = R.body.vsplit(0.88)
        left, right = top.hsplit(0.5, gap=0.4)

        # a double well and its Boltzmann density
        x = np.linspace(-2, 2, 400)
        energy = (x ** 2 - 1) ** 2 + 0.3 * x
        boltzmann = np.exp(-energy / 0.3)
        ax = Plot(left, xlabel="reaction coordinate $x$", ylabel="energy")
        profile = ax.plot(x, energy, label="$U(x)$")
        density = ax.fill_between(x, 0, boltzmann, alpha=0.3, label="$e^(-U \\/ k T)$")
        legend = ax.legend()

        self.reveal(ax)
        self.reveal(profile, anim=Create)
        self.reveal(density, legend, anim=FadeIn)
        self.reveal(place(text("$p(x) prop e^(-U(x) \\/ k T)$"), bottom), anim=Write)

        # samples from the density, then their histogram
        rng = np.random.default_rng(0)
        samples = rng.choice(x, size=2000, p=boltzmann / boltzmann.sum())
        hist = Plot(right, xlabel="$x$", ylabel="count")
        bars = hist.hist(samples, bins=30, color=COLORS[1])
        self.reveal(hist)
        self.reveal(bars, anim=FadeIn)


SLIDES = order(Results)
