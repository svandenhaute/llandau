"""Scenes compiled into one timeline that is drawn live.

A Scene runs like a manim scene (add, remove, play, wait) plus manim-slides'
next_slide; instead of rendering frames it records what is on screen when:
each drawn part as a clip of the timeline (shown as it is, morphing, being
created or written), and each next_slide as a mark where the presentation
pauses. build() runs a list of scenes into one deck; play() presents it.
"""
from __future__ import annotations

import hashlib
import inspect
import math
from types import SimpleNamespace

import numpy as np

from .animation import AnimationGroup, Clock, Wait, _prepare, linear
from .mobject import BLACK, Color, ImageMobject, Mobject, VMobject, config

__all__ = ["Scene", "build", "always_redraw", "ValueTracker"]

# frames per second when stepping animations through Python (baked animations, updaters)
FPS = 60
# samples of a rate function
RATE_SAMPLES = 257


class ValueTracker(Mobject):
    """A number to animate (with .animate.set_value) and read from updaters."""

    def __init__(self, value: float = 0, **kwargs):
        super().__init__(**kwargs)
        self.points = np.array([[float(value), 0.0, 0.0]])

    def get_value(self) -> float:
        return float(self.points[0, 0])

    def set_value(self, value: float):
        self.points = np.array([[float(value), 0.0, 0.0]])
        return self

    def increment_value(self, delta: float):
        return self.set_value(self.get_value() + delta)

    def interpolate(self, start, end, alpha):
        return self.set_value(start.get_value() + alpha * (end.get_value() - start.get_value()))


def _add_updater(self, function, index: int | None = None, call_updater: bool = False):
    updaters = self.__dict__.setdefault("updaters", [])
    updaters.insert(len(updaters) if index is None else index, function)
    if call_updater:
        _call(function, self, 0)
    return self


def _remove_updater(self, function):
    if function in self.__dict__.get("updaters", []):
        self.updaters.remove(function)
    return self


def _clear_updaters(self, recursive: bool = True):
    for m in self.get_family() if recursive else [self]:
        m.__dict__["updaters"] = []
    return self


def _get_updaters(self):
    return self.__dict__.get("updaters", [])


def _call(function, mobject, dt):
    try:
        wants_dt = len(inspect.signature(function).parameters) > 1
    except (TypeError, ValueError):
        wants_dt = False
    return function(mobject, dt) if wants_dt else function(mobject)


Mobject.add_updater = _add_updater
Mobject.remove_updater = _remove_updater
Mobject.clear_updaters = _clear_updaters
Mobject.get_updaters = _get_updaters


def always_redraw(function):
    mobject = function()
    mobject.add_updater(lambda m: m.become(function()))
    return mobject


def _visible(m: VMobject) -> bool:
    fill = m.fill_opacity * m.fill_color.alpha > 0
    stroke = m.stroke_opacity * m.stroke_color.alpha > 0 and m.stroke_width > 0
    return fill or stroke


def _drawable(m) -> bool:
    return (isinstance(m, VMobject) and len(m.points) >= 4) or isinstance(m, ImageMobject)


class _Builder:
    """Collects the clips of all scenes into a Slides timeline."""

    def __init__(self, background):
        from ._core import Slides
        self.slides = Slides(frame=(config.frame_width, config.frame_height), background=tuple(background))
        self.background = tuple(background)
        self.shapes: dict[bytes, int] = {}
        self.images: dict = {}
        self.placements: dict = {}
        self.rates: dict[int, tuple] = {}
        # drawn part -> (part, kind, geometry id, z, start)
        self.open: dict[int, tuple] = {}
        self.claimed: set[int] = set()
        self.play_span = (0.0, 0.0)
        self.scene = None
        self.marks = 0
        # live views: one viewer per (scene, theme); spans waiting for their step's end
        self.viewers = {}
        self.pending_lives = []

    # --- geometry ---------------------------------------------------------------------

    def shape(self, m: VMobject, visible_only: bool = False) -> int | None:
        n = 4 * (len(m.points) // 4)
        if n == 0:
            return None
        if visible_only and not _visible(m):
            return None
        points = np.ascontiguousarray(m.points[:n, :2], dtype=np.float32)
        style = m.style_array()
        key = hashlib.blake2b(points.tobytes() + style.tobytes(), digest_size=16).digest()
        shape = self.shapes.get(key)
        if shape is None:
            shape = self.shapes[key] = self.slides.add_shape(points, style)
        return shape

    def image(self, m: ImageMobject) -> int:
        key = m.path if m.path is not None else id(m.pixels)
        image = self.images.get(key)
        if image is None:
            if m.path is not None:
                image = self.slides.add_image_file(m.path)[0]
            else:
                image = self.slides.add_image(m.pixels)
            # keep arrays alive so their ids stay unique
            self.images[key] = image
            if m.path is None:
                self.images[("pixels", key)] = m.pixels
        return image

    def placement(self, m: ImageMobject) -> int:
        lo, hi = m.points.min(axis=0), m.points.max(axis=0)
        rect = (float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(hi[0] - lo[0]), float(hi[1] - lo[1]))
        key = (self.image(m), rect, float(m.opacity))
        placement = self.placements.get(key)
        if placement is None:
            placement = self.placements[key] = self.slides.add_placement(key[0], rect, key[2])
        return placement

    def geometry(self, m) -> tuple[str, int] | None:
        if isinstance(m, ImageMobject):
            return ("image", self.placement(m))
        if isinstance(m, VMobject):
            shape = self.shape(m, visible_only=True)
            return None if shape is None else ("shape", shape)
        return None

    def rate(self, function) -> int | None:
        if function is linear:
            return None
        entry = self.rates.get(id(function))
        if entry is None:
            samples = [float(function(t)) for t in np.linspace(0, 1, RATE_SAMPLES)]
            entry = self.rates[id(function)] = (self.slides.add_rate(samples), function)
        return entry[0]

    # --- what is on screen ------------------------------------------------------------------

    def z_index(self, scene, top) -> float:
        order = scene._order.get(id(top))
        if order is None:
            order = scene._order[id(top)] = scene._next_order()
        return getattr(top, "z_index", 0) * 1e6 + order

    @staticmethod
    def lift(top, part) -> float:
        """A part with a higher z_index than its top (e.g. a plot's spines) draws above it."""
        return max(getattr(part, "z_index", 0) - getattr(top, "z_index", 0), 0) * 1e6

    def drawn(self, scene):
        """(part, z) of everything on screen, bottom to top."""
        for top in scene.mobjects:
            parts = [m for m in top.family_members_with_points() if _drawable(m)]
            base = self.z_index(scene, top)
            for k, m in enumerate(parts):
                yield m, base + self.lift(top, m) + (k + 0.5) / (len(parts) + 1)

    def z_of(self, mobject):
        """z(i, n) for the i-th of n parts that `mobject` turns into."""
        scene = self.scene
        for top in scene.mobjects:
            parts = [m for m in top.family_members_with_points() if _drawable(m)]
            ids = [id(m) for m in parts]
            mine = [m for m in mobject.family_members_with_points() if _drawable(m)]
            if top is mobject or (mine and id(mine[0]) in ids):
                base = self.z_index(scene, top)
                first = ids.index(id(mine[0])) if mine and id(mine[0]) in ids else 0
                count, size = len(parts), max(len(mine), 1)
                lifts = [self.lift(top, m) for m in mine] or [0.0]

                def z(i, n):
                    lift = lifts[min(int(i * len(lifts) / max(n, 1)), len(lifts) - 1)]
                    return base + lift + (first + (i + 0.5) * size / max(n, 1)) / (count + 1)
                return z
        base = getattr(mobject, "z_index", 0) * 1e6 + scene._next_order()
        return lambda i, n: base + (i + 0.5) / (n + 1)

    def sync(self, scene, t: float):
        """Make the clips match the scene at time t."""
        wanted = {}
        for m, z in self.drawn(scene):
            if id(m) in self.claimed:
                continue
            geometry = self.geometry(m)
            if geometry is not None:
                wanted[id(m)] = (m, geometry, z)
        for key in list(self.open):
            part, kind, geometry, z, start = self.open[key]
            now = wanted.get(key)
            if now is None or now[1] != (kind, geometry) or now[2] != z:
                self.close(key, t)
        for key, (m, (kind, geometry), z) in wanted.items():
            if key not in self.open:
                self.open[key] = (m, kind, geometry, z, t)

    def close(self, key, t: float):
        part, kind, geometry, z, start = self.open.pop(key)
        if t > start:
            clip_kind = "image" if kind == "image" else "show"
            self.slides.add_clip(clip_kind, geometry, geometry, z, start, t, (start, start))

    def close_all(self, t: float):
        for key in list(self.open):
            self.close(key, t)

    # --- animations ------------------------------------------------------------------

    def claim(self, mobject):
        """The parts of `mobject` are animated now: their clips end."""
        for m in mobject.get_family():
            self.claimed.add(id(m))
            if id(m) in self.open:
                self.close(id(m), self.play_span[0])

    def _timing(self, clock: Clock, rate, reverse: bool):
        """(window, rate id, reverse) of a clip running on `clock`."""
        if clock.affine:
            return (clock.start, clock.end), self.rate(rate), reverse
        start, end = self.play_span
        times = np.linspace(start, end, RATE_SAMPLES)
        samples = [float(rate(1 - clock(t) if reverse else clock(t))) for t in times]
        return (start, end), self.slides.add_rate(samples), False

    def morph(self, a, b, z, clock, rate, reverse=False):
        start, end = self.play_span
        window, rate_id, reverse = self._timing(clock, rate, reverse)
        if isinstance(a, ImageMobject) and isinstance(b, ImageMobject):
            self.slides.add_clip("image", self.placement(a), self.placement(b), z, start, end, window, rate_id, reverse)
            return
        if isinstance(a, VMobject) and isinstance(b, VMobject):
            ga, gb = self.shape(a), self.shape(b)
            if ga is not None and gb is not None:
                self.slides.add_clip("morph", ga, gb, z, start, end, window, rate_id, reverse)
            return
        # a raster image and a vector shape: switch halfway
        mid = (window[0] + window[1]) / 2
        for m, (t0, t1) in ((a, (start, mid)), (b, (mid, end))):
            geometry = self.geometry(m)
            if geometry is not None and t1 > t0:
                kind = "image" if geometry[0] == "image" else "show"
                self.slides.add_clip(kind, geometry[1], geometry[1], z, t0, t1, (t0, t0))

    def partial(self, kind, leaf, z, clock, rate, reverse=False):
        start, end = self.play_span
        shape = self.shape(leaf) if isinstance(leaf, VMobject) else None
        if shape is None:
            return
        window, rate_id, reverse = self._timing(clock, rate, reverse)
        self.slides.add_clip(kind, shape, shape, z, start, end, window, rate_id, reverse)

    def show(self, leaf, z, t0: float, t1: float):
        geometry = self.geometry(leaf)
        if geometry is not None and t1 > t0:
            kind = "image" if geometry[0] == "image" else "show"
            self.slides.add_clip(kind, geometry[1], geometry[1], z, t0, t1, (t0, t0))

    def mark(self, t: float, slide: int, notes: str):
        self.slides.add_mark(t, slide, notes)
        self.marks += 1
        for viewer, rect, start, enter, camera in self.pending_lives:
            self.slides.add_live(viewer, rect, start, t, enter, camera)
        self.pending_lives = []

    def live(self, scene, name: str, theme: str, region, view: dict):
        """Show live scene `name` over `region` until the next mark."""
        from . import live
        viewer = self.viewers.get((name, theme))
        if viewer is None:
            from .viewer import Viewer
            viewer = self.viewers[(name, theme)] = Viewer(theme=theme)
            live.build(viewer, name)
        rect = ((region.x0 + region.x1) / 2, (region.y0 + region.y1) / 2, region.width, region.height)
        enter = lambda v, view=dict(view): live.apply_view(v, view, camera=False)  # noqa: E731
        self.pending_lives.append((viewer, rect, scene.time, enter, view.get("camera")))


class Scene:
    """A manim-like scene; also manim-slides' Slide (next_slide)."""

    def __init__(self, **kwargs):
        self.mobjects: list[Mobject] = []
        self.camera = SimpleNamespace(background_color=BLACK)
        self.time = 0.0
        self.updaters = []
        self._order: dict[int, float] = {}
        self._order_counter = 0
        self._builder: _Builder | None = None
        self._slide = 0
        self._slide_start = 0.0
        self._last_mark = None
        self._notes = ""

    def setup(self):
        pass

    def construct(self):
        pass

    def tear_down(self):
        pass

    def _next_order(self) -> float:
        self._order_counter += 1
        return float(self._order_counter)

    # --- mobjects -----------------------------------------------------------------------

    def add(self, *mobjects):
        for m in mobjects:
            if m in self.mobjects:
                self.mobjects.remove(m)
            self.mobjects.append(m)
            self._order[id(m)] = self._next_order()
        return self

    def add_foreground_mobjects(self, *mobjects):
        for m in mobjects:
            m.z_index = max(getattr(m, "z_index", 0), 1e3)
        return self.add(*mobjects)

    def remove(self, *mobjects):
        for m in mobjects:
            if m in self.mobjects:
                self.mobjects.remove(m)
                continue
            # a part of something on screen: its parent splits into the other parts
            for top in list(self.mobjects):
                if m in top.get_family():
                    index = self.mobjects.index(top)
                    self.mobjects[index:index + 1] = _without(top, m)
                    for part in self.mobjects[index:index + 1 + len(top.submobjects)]:
                        self._order.setdefault(id(part), self._order.get(id(top), 0))
        return self

    def replace(self, old, new):
        if old in self.mobjects:
            index = self.mobjects.index(old)
            if new in self.mobjects:
                self.mobjects.remove(new)
                index = self.mobjects.index(old)
            self.mobjects[index] = new
            self._order[id(new)] = self._order.get(id(old), self._next_order())
        else:
            self.remove(old)
            self.add(new)
        return self

    def clear(self):
        self.mobjects = []
        return self

    def bring_to_front(self, *mobjects):
        return self.add(*mobjects)

    def bring_to_back(self, *mobjects):
        for m in mobjects:
            if m in self.mobjects:
                self.mobjects.remove(m)
        self.mobjects[0:0] = list(mobjects)
        low = min((self._order.get(id(m), 0) for m in self.mobjects), default=0)
        for k, m in enumerate(mobjects):
            self._order[id(m)] = low - len(mobjects) + k
        return self

    def get_top_level_mobjects(self):
        return list(self.mobjects)

    def add_updater(self, function):
        self.updaters.append(function)

    def remove_updater(self, function):
        self.updaters = [f for f in self.updaters if f is not function]

    # --- time ---------------------------------------------------------------------------

    def _has_updaters(self) -> bool:
        return bool(self.updaters) or any(
            m.get_updaters() for top in self.mobjects for m in top.get_family()
        )

    def _update(self, dt: float):
        for top in list(self.mobjects):
            for m in top.get_family():
                for f in list(m.get_updaters()):
                    _call(f, m, dt)
        for f in self.updaters:
            f(dt)

    def play(self, *animations, run_time: float | None = None, rate_func=None, lag_ratio: float | None = None,
             **kwargs):
        builder = self._builder
        animations = [_prepare(a) for a in animations]
        for a in animations:
            if run_time is not None:
                a.run_time = run_time
            if rate_func is not None:
                a.rate_func = rate_func
            if lag_ratio is not None:
                a.lag_ratio = lag_ratio
                if isinstance(a, AnimationGroup):
                    a._build_timings()
        duration = max((a.get_run_time() for a in animations), default=0.0)
        start, end = self.time, self.time + duration
        builder.scene = self
        builder.sync(self, start)
        for a in animations:
            a._setup_scene(self)
        builder.play_span = (start, end)

        natives, baked = [], []
        bake_all = self._has_updaters()

        def compile_tree(a, clock):
            if isinstance(a, AnimationGroup):
                for child, child_clock in a.children(clock):
                    compile_tree(child, child_clock)
            elif bake_all or not a.compile(builder, clock):
                baked.append((a, clock))
            else:
                natives.append(a)
        for a in animations:
            compile_tree(a, Clock(start, start + a.get_run_time()))

        if baked or bake_all:
            for a, _ in baked:
                a.begin()
            steps = max(1, math.ceil(duration * FPS))
            previous = start
            for k in range(steps + 1):
                t = start + duration * k / steps
                for a, clock in baked:
                    a.interpolate(clock(t))
                if k > 0:
                    self._update(t - previous)
                previous = t
                builder.sync(self, t)
            for a, _ in baked:
                a.finish()
        for a in natives:
            a.finish_native(self)
        for a in animations:
            a.clean_up_from_scene(self)
        builder.claimed.clear()
        self.time = end
        builder.sync(self, end)
        return self

    def wait(self, duration: float = 1.0, stop_condition=None, frozen_frame=None, **kwargs):
        builder = self._builder
        builder.scene = self
        builder.sync(self, self.time)
        if self._has_updaters() and not frozen_frame:
            steps = max(1, math.ceil(duration * FPS))
            for k in range(1, steps + 1):
                self._update(duration / steps)
                builder.sync(self, self.time + duration * k / steps)
        self.time += duration
        builder.sync(self, self.time)
        return self

    def pause(self, duration: float = 1.0):
        return self.wait(duration, frozen_frame=True)

    def next_slide(self, notes: str = "", **kwargs):
        """End the current step: the presentation pauses here. `notes` belong to
        the step that starts here (as in manim-slides)."""
        builder = self._builder
        builder.scene = self
        builder.sync(self, self.time)
        if self.time > self._slide_start and self.time != self._last_mark:
            builder.mark(self.time, self._slide, self._notes)
            self._last_mark = self.time
            self._notes = notes
        else:
            self._notes = notes or self._notes
        return self

    # manim-slides' other names
    next_section = next_slide


def _without(top: Mobject, part: Mobject) -> list[Mobject]:
    """top's submobjects, recursively without `part`."""
    if top is part:
        return []
    if part not in top.get_family():
        return [top]
    out = []
    for m in top.submobjects:
        out.extend(_without(m, part))
    return out


def build(scenes, background=None):
    """Run scene classes (or instances) one after another into one deck: a
    Slides timeline (llandau._core). Each scene is one slide."""
    builder = None
    t = 0.0
    for index, scene in enumerate(scenes):
        scene = scene() if isinstance(scene, type) else scene
        scene._slide = index
        scene.time = scene._slide_start = t
        if builder is None:
            scene.setup()
            bg = background if background is not None else Color(scene.camera.background_color).to_rgba()
            builder = _Builder(bg)
        else:
            scene.setup()
        scene._builder = builder
        builder.scene = scene
        builder.sync(scene, t)
        scene.construct()
        if scene.time <= scene._slide_start or builder.pending_lives:
            scene.wait(0.05)  # a slide (or live step) that only adds things still shows them
        scene.next_slide()  # the end of the last step
        # a background other than the deck's (set in setup or construct): a rectangle behind everything
        bg = Color(scene.camera.background_color).to_rgba()
        if not np.allclose(bg, builder.background):
            builder.show(_backdrop(bg), -1e12, scene._slide_start, scene.time)
        t = scene.time
        builder.close_all(t)
        scene.tear_down()
    if builder is None:
        raise ValueError("no scenes")
    return builder.slides


def _backdrop(color):
    from .shapes import Rectangle
    return Rectangle(width=config.frame_width, height=config.frame_height, stroke_width=0,
                     fill_color=Color(color[:3]), fill_opacity=float(color[3]))
