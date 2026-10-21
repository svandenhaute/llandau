"""llandau: smooth scientific slides, drawn live (no videos are rendered), with
interactive 3D views of atomic trajectories (riemann's paths, splines, proteins).

    from llandau import *

    class Results(Deck):
        def construct(self):
            self.title("results")
            self.reveal(image("figures/a.png", R.body.left), anim=FadeIn)

    SLIDES = order(Results)

    llandau play slides.py

modules
    mobject    manim's model: Mobject, VMobject, VGroup, Group, ImageMobject, SVGMobject, colors, directions
    shapes     Line, Arrow, Circle, Dot, Rectangle, Polygon, ...
    animation  FadeIn, Write, Create, Transform, .animate, LaggedStart, ... and rate functions
    scene      Scene (add, play, wait, next_slide), build: scenes -> one timeline
    axes       manim's NumberLine, Axes, NumberPlane, ParametricFunction, FunctionGraph
    plot       Plot: matplotlib-style plots of numpy data (plot, scatter, fill_between, bar, hist,
               errorbar, imshow, contour(f), legend, colorbar), revealed in steps
    ticks, geometry, markers   its tick placement, clipping and dashes, marker painting
    figure     subplots, Figure: plots for papers, saved as SVG, PDF or PNG
    theme      DARK / LIGHT, font, type scale (TITLE, BODY, SMALL, LABEL)
    text       text, bold, italic, eq, typst, title, bullets (Typst)
    layout     R (screen regions), place, below/above/left_of/right_of, align, stack, row, gaps XS..XL
    media      image, svg, layered_svg (Inkscape layers, cached in .llandau_cache/)
    deck       Deck (a slide: step, reveal, title, show_svg, live), order, typewrite
    live       live_scene, Deck.live: interactive 3D (Viewer) on chosen steps
    cli        llandau play | pdf | png
    viewer     Viewer: atomic trajectories in 3D (trails, tangents, protein styles, surfaces),
               offscreen images and an explorer window (llandau.explorer)
    video      Video: RGBA frames to MP4 through ffmpeg (the explorer's record button)
    elements, colormaps, pdb
"""
from .mobject import *
from .shapes import *
from .animation import *
from .scene import *
from .axes import *
from .plot import *
from .figure import Figure, subplots
from .theme import *
from .layout import *
from .text import *
from .media import *
from .deck import *
from .live import live_scene
from .viewer import Viewer, segment_ids
from . import colormaps, elements
