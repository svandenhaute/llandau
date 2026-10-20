"""Minimal, layered scientific slides, drawn live: no videos are rendered.

    from present import *

    class Results(Deck):
        def construct(self):
            self.title("results")
            self.reveal(image("figures/a.png", R.body.left), anim=FadeIn)

    SLIDES = order(Results)

    present play slides.py

modules
    mobject    manim's model: Mobject, VMobject, VGroup, Group, ImageMobject, SVGMobject, colors, directions
    shapes     Line, Arrow, Circle, Dot, Rectangle, Polygon, ...
    animation  FadeIn, Write, Create, Transform, .animate, LaggedStart, ... and rate functions
    scene      Scene (add, play, wait, next_slide), build: scenes -> one timeline
    theme      DARK / LIGHT, font, type scale (TITLE, BODY, SMALL, LABEL)
    text       text, bold, italic, eq, typst, title, bullets (Typst)
    layout     R (screen regions), place, below/above/left_of/right_of, align, stack, row, gaps XS..XL
    media      image, svg, layered_svg (Inkscape layers, cached in .present_cache/)
    deck       Deck (a slide: step, reveal, title, show_svg, live), order, typewrite
    live       live_scene, Deck.live: interactive 3D (riemann_viewer) on chosen steps
    render     present play | pdf | png
"""
from .mobject import *
from .shapes import *
from .animation import *
from .scene import *
from .theme import *
from .layout import *
from .text import *
from .media import *
from .deck import *
from .live import live_scene
