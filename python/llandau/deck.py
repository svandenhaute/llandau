"""Deck: a slide (a Scene with steps) with theme, slide numbers and short reveal helpers.

    class Results(Deck):
        theme = LIGHT
        def construct(self):
            self.title("results")
            self.cite("10.1038/s41586-021-03819-2")       # in the footer: Jumper et al., Nature ...
            self.reveal(image("images/a.png", R.body.left))
            self.show_svg("images/b.svg", R.body.right)   # one slide step per layer

    SLIDES = order(Title, Results, ...)   # numbers the slides; read by `llandau play`
"""
from __future__ import annotations

from .animation import AddTextLetterByLetter, Write, LaggedStart
from .mobject import Group, DL, DR, ImageMobject, config
from .scene import MIN_STEP, Scene

from .theme import DARK, use
from .layout import R, Region, place
from .text import text, title as _title
from .cite import cite as _cite
from .media import layered_svg
from . import live as _live

__all__ = ["Deck", "order", "typewrite"]


def typewrite(mob, **kwargs):
    """Letter by letter for plain text (text() without math), Write for anything else."""
    return AddTextLetterByLetter(mob, **kwargs) if getattr(mob, "plain_text", False) else Write(mob, **kwargs)


class Deck(Scene):
    theme = DARK
    numbered = True           # False: no number shown and not counted (title slide, full-screen photos)
    number: int | None = None  # assigned by order()
    run_time = 0.2             # default for reveal(anim=...)

    def setup(self):
        super().setup()
        use(self.theme)
        self.camera.background_color = self.theme.bg
        if self.numbered and self.number is not None:
            self.add(text(str(self.number), size=self.theme.label).to_corner(DR, buff=0.3))

    # the step's live views play until here (a recording, the move to a camera): the step ends after
    _live_until = 0.0

    def step(self, notes: str = ""):
        """End the current slide step. The short wait guarantees the step isn't empty.
        notes belong to the step that starts here (manim-slides attaches them to the next step)."""
        self.wait(MIN_STEP)
        self.next_slide(notes=notes)

    def next_slide(self, notes: str = "", **kwargs):
        if self.time < self._live_until:
            self.wait(self._live_until - self.time)
        return super().next_slide(notes, **kwargs)

    def reveal(self, *mobs, anim=None, run_time: float | None = None, lag: float = 0,
               replace=None, pause: bool = True, notes: str = ""):
        """Show mobs (instantly, or with anim e.g. FadeIn / Write / typewrite), then end the step.

        lag      >0: stagger the animations (LaggedStart lag_ratio)
        replace  mobject(s) shown instead: removed before the new ones appear
        pause    False: don't end the step (combine several reveals into one step)
        notes    speaker notes for the step that starts after this one
        """
        if replace is not None:
            self.remove(*(replace if isinstance(replace, (list, tuple)) else [replace]))
        if anim is None:
            self.add(*mobs)
        else:
            anims = [anim(m) for m in mobs]
            if lag and len(anims) > 1:
                anims = [LaggedStart(*anims, lag_ratio=lag)]
            self.play(*anims, run_time=run_time or self.run_time)
        if pause:
            self.step(notes)
        return mobs[0] if len(mobs) == 1 else Group(*mobs)

    def live(self, scene: str, region: Region | None = None, still: bool = True, opens: Region | None = None,
             state=None, play=None, **view):
        """The next step shows the 3D scene registered with @live_scene(scene) over region
        (default R.body), live and interactive in the player. Right after a reveal/step this only
        marks the upcoming step; otherwise it also ends the current one, so several calls in a row
        put several views on one step (a grid: one call per cell of R.body.grid(2, 5, gap=0.2)).

        view     settings applied when the step starts (see llandau.live): camera, frame, style,
                 sidechains, shading, fog, occlusion, coloring, scrub_discrete, show, hide, plots
        state    the whole view as the explorer copies it: a dict or a JSON file; the keywords
                 override its entries. Its camera is moved to, like camera=.
        play     keyframes the explorer recorded (a JSON file, or a list of dicts with "t"),
                 played on the step's clock from the step's start; the step lasts at least as long
        still    also draw a still image of the view into the region (what exports show; the live
                 view covers it). Returned, e.g. to replace it later; None without.
        opens    a click on the view opens it up to this region (e.g. R.body), over the other
                 views; the next click closes it. A drag still turns it.
        """
        _live.check(view)
        region = region if region is not None else R.body
        theme = "light" if self.theme.bg.to_rgb().mean() > 0.5 else "dark"
        state = _live.load_state(state) if state is not None else None
        play = _live.load_recording(play) if play is not None else None
        # ends the step as it is (no wait: no empty step), not after the live views' play: several
        # live() calls in a row share a step
        Scene.next_slide(self)
        self._builder.live(self, scene, theme, region, view, opens, state, play)
        # the step lasts as long as what plays on its clock: a recording, or the move to a camera
        if play:
            self._live_until = max(self._live_until, self.time + play[-1][0])
        elif "camera" in view or (state and "camera" in state):
            self._live_until = max(self._live_until, self.time + _live.CAMERA_MOVE)
        if not still:
            return None
        pixels = (round(2 * config.pixel_width * region.width / config.frame_width),
                  round(2 * config.pixel_height * region.height / config.frame_height))
        image = ImageMobject(_live.still(scene, theme, view, pixels, state, play))
        place(image, region, fit="fill")
        self.add(image)
        return image

    def title(self, s: str, anim=None, pause: bool = False, **kwargs):
        """Slide title in the title band. By default added without ending the step."""
        mob = _title(s, **kwargs)
        return self.reveal(mob, anim=anim, pause=pause)

    def cite(self, *sources: str, region: Region | None = None, anim=None, pause: bool = False, sep: str = "; ",
             title: bool = False, authors: int | str = 1, **kwargs):
        """Citation(s) of DOIs, arXiv ids or article URLs, left in the footer strip (default; it
        leaves the slide number room) or in region, shrunk to fit. Added without ending the step.
        title, authors: see llandau.cite.citation; kwargs go to text(): size, color, slant, ..."""
        if region is None:  # shrunk only to its width: the strip is lower than a line of text
            f = R.footer
            region = Region(f.x0, f.x1 - 0.8, f.y0, f.y0 + 1)
        mob = _cite(*sources, sep=sep, title=title, authors=authors, **kwargs)
        return self.reveal(place(mob, region, anchor=DL), anim=anim, pause=pause)  # on the number's baseline

    def show_svg(self, path, region: Region | None = None, layered: bool = True, anim=None,
                 run_time: float | None = None, pause: bool = True, accumulate: bool = True, **kwargs):
        """Show an Inkscape SVG, one slide step per layer (see llandau.media for the layer conventions).

        layered     False: show the whole drawing at once
        accumulate  False: each step replaces the previous one instead of adding to it
        kwargs      go to layered_svg: mode, order, layers, dpi, renderer, anchor, fit, buff, scale
        Returns the Group of steps.
        """
        steps = layered_svg(path, region if region is not None else R.body, accumulate=accumulate, **kwargs)
        if not layered:
            full = steps[-1] if steps.replace_previous and accumulate else Group(*steps)
            self.reveal(full, anim=anim, run_time=run_time, pause=pause)
            return steps
        previous = None
        for s in steps:
            self.reveal(s, anim=anim, run_time=run_time, pause=False)
            if previous is not None and steps.replace_previous:
                self.remove(previous)  # only now: the previous picture stays visible under a fade-in
            if pause:
                self.step()
            previous = s
        return steps


def order(*slides):
    """Fix the order of the talk and number the slides. Repeats are allowed (shown twice, numbered once)."""
    n = 0
    for cls in dict.fromkeys(slides):
        if getattr(cls, "numbered", False):
            n += 1
            cls.number = n
    return list(slides)
