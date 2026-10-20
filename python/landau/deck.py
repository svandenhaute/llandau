"""Deck: a slide (a Scene with steps) with theme, slide numbers and short reveal helpers.

    class Results(Deck):
        theme = LIGHT
        def construct(self):
            self.title("results")
            self.reveal(image("images/a.png", R.body.left))
            self.show_svg("images/b.svg", R.body.right)   # one slide step per layer

    SLIDES = order(Title, Results, ...)   # numbers the slides; read by `landau play`
"""
from __future__ import annotations

from .animation import AddTextLetterByLetter, Write, LaggedStart
from .mobject import Group, DR, ImageMobject, config
from .scene import Scene

from .theme import DARK, LABEL, use
from .layout import R, Region, place
from .text import text, title as _title
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
    run_time = 0.4             # default for reveal(anim=...)

    def setup(self):
        super().setup()
        use(self.theme)
        self.wait_time_between_slides = 0.05
        self.camera.background_color = self.theme.bg
        if self.numbered and self.number is not None:
            self.add(text(str(self.number), size=LABEL).to_corner(DR, buff=0.3))

    def step(self, notes: str = ""):
        """End the current slide step. The short wait guarantees the step isn't empty.
        notes belong to the step that starts here (manim-slides attaches them to the next step)."""
        self.wait(0.05)
        self.next_slide(notes=notes)

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

    def live(self, scene: str, region: Region | None = None, still: bool = True, **view):
        """The next step shows the 3D scene registered with @live_scene(scene) over region
        (default R.body), live and interactive in the player. Right after a reveal/step this only
        marks the upcoming step; otherwise it also ends the current one.

        view     settings applied when the step starts (see landau.live): camera, frame, style,
                 sidechains, shading, coloring, scrub_discrete, show, hide
        still    also draw a still image of the view into the region (what exports show; the live
                 view covers it). Returned, e.g. to replace it later; None without.
        """
        _live.check(view)
        region = region if region is not None else R.body
        theme = "light" if self.theme.bg.to_rgb().mean() > 0.5 else "dark"
        self.next_slide()  # no wait: no empty step
        self._builder.live(self, scene, theme, region, view)
        if not still:
            return None
        pixels = (round(2 * config.pixel_width * region.width / config.frame_width),
                  round(2 * config.pixel_height * region.height / config.frame_height))
        image = ImageMobject(_live.still(scene, theme, view, pixels))
        place(image, region, fit="fill")
        self.add(image)
        return image

    def title(self, s: str, anim=None, pause: bool = False, **kwargs):
        """Slide title in the title band. By default added without ending the step."""
        mob = _title(s, **kwargs)
        return self.reveal(mob, anim=anim, pause=pause)

    def show_svg(self, path, region: Region | None = None, layered: bool = True, anim=None,
                 run_time: float | None = None, pause: bool = True, accumulate: bool = True, **kwargs):
        """Show an Inkscape SVG, one slide step per layer (see landau.media for the layer conventions).

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
