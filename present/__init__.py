"""Helpers for building manim-slides decks with less code.

    from manim import *
    from present import *

modules
    theme   DARK / LIGHT, font, type scale (TITLE, BODY, SMALL, LABEL)
    text    text, bold, italic, eq, typst, title, bullets (math via the typst CLI)
    layout  R (screen regions), place, below/above/left_of/right_of, align, stack, row, gaps XS..XL
    media   image, svg, layered_svg (Inkscape layers split at render time, cached in .present_cache/)
    deck    Deck (base slide: step, reveal, title, show_svg), order, typewrite
    live    live_scene, Deck.live: interactive 3D (molview) on chosen steps (`present live`)
    render  present render scene.py
"""
from .theme import *
from .layout import *
from .text import *
from .media import *
from .deck import *
from .live import live_scene
