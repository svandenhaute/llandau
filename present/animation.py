"""Animations, as manim defines them, compiled into timeline clips.

Most animations compile to clips the renderer interpolates itself (morphs,
partial curves, outlines then fills), so playback needs no Python. Any other
animation runs the manim way: `interpolate(alpha)` on the mobjects, sampled
at FPS while compiling ("baked"); this covers rotations, arcs, updaters and
animations written for manim.
"""
from __future__ import annotations

import math

import numpy as np

from .mobject import (
    DOWN, ORIGIN, PI, TAU, UP, YELLOW, Color, Group, ImageMobject, Mobject, VMobject, OUT,
)

__all__ = [
    "Animation", "Wait", "AnimationGroup", "LaggedStart", "LaggedStartMap", "Succession",
    "Transform", "ReplacementTransform", "TransformFromCopy", "ClockwiseTransform", "CounterclockwiseTransform",
    "MoveToTarget", "ApplyMethod", "ApplyFunction", "ApplyMatrix", "Restore", "FadeToColor", "ScaleInPlace",
    "ShrinkToCenter", "FadeTransform", "FadeTransformPieces", "TransformMatchingShapes",
    "FadeIn", "FadeOut", "GrowFromCenter", "GrowFromPoint", "GrowFromEdge", "GrowArrow", "SpinInFromNothing",
    "Create", "Uncreate", "DrawBorderThenFill", "Write", "Unwrite", "ShowPartial", "ShowPassingFlash",
    "ShowIncreasingSubsets", "AddTextLetterByLetter", "RemoveTextLetterByLetter", "Indicate", "Flash",
    "Circumscribe", "Wiggle", "Rotate", "Rotating", "MoveAlongPath", "UpdateFromFunc", "UpdateFromAlphaFunc",
    "rate_functions", "linear", "smooth", "smoothstep", "rush_into", "rush_from", "slow_into", "double_smooth",
    "there_and_back", "there_and_back_with_pause", "running_start", "not_quite_there", "wiggle", "lingering",
    "exponential_decay", "ease_in_sine", "ease_out_sine", "ease_in_out_sine", "ease_in_quad", "ease_out_quad",
    "ease_in_out_quad", "ease_in_cubic", "ease_out_cubic", "ease_in_out_cubic", "ease_out_back", "ease_in_back",
    "ease_out_elastic", "ease_out_bounce",
]


# --- rate functions (manim's) ---------------------------------------------------------

def linear(t):
    return t


def _sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


def smooth(t, inflection: float = 10.0):
    error = _sigmoid(-inflection / 2)
    return min(max((_sigmoid(inflection * (t - 0.5)) - error) / (1 - 2 * error), 0), 1)


def smoothstep(t):
    return 0 if t <= 0 else 3 * t ** 2 - 2 * t ** 3 if t < 1 else 1


def rush_into(t, inflection: float = 10.0):
    return 2 * smooth(t / 2.0, inflection)


def rush_from(t, inflection: float = 10.0):
    return 2 * smooth(t / 2.0 + 0.5, inflection) - 1


def slow_into(t):
    return math.sqrt(1 - (1 - t) * (1 - t))


def double_smooth(t):
    return 0.5 * smooth(2 * t) if t < 0.5 else 0.5 * (1 + smooth(2 * t - 1))


def there_and_back(t, inflection: float = 10.0):
    return smooth(2 * t if t < 0.5 else 2 * (1 - t), inflection)


def there_and_back_with_pause(t, pause_ratio: float = 1.0 / 3):
    a = 2.0 / (1.0 - pause_ratio)
    if t < 0.5 - pause_ratio / 2:
        return smooth(a * t)
    if t < 0.5 + pause_ratio / 2:
        return 1
    return smooth(a - a * t)


def _bezier(values):
    values = list(values)
    n = len(values) - 1
    return lambda t: sum(math.comb(n, k) * (1 - t) ** (n - k) * t ** k * v for k, v in enumerate(values))


def running_start(t, pull_factor: float = -0.5):
    return _bezier([0, 0, pull_factor, pull_factor, 1, 1, 1])(t)


def not_quite_there(func=smooth, proportion: float = 0.7):
    return lambda t, *args: proportion * func(t, *args)


def wiggle(t, wiggles: float = 2):
    return there_and_back(t) * math.sin(wiggles * math.pi * t)


def lingering(t):
    return min(max(t / 0.8, 0), 1)


def exponential_decay(t, half_life: float = 0.1):
    return 1 - math.exp(-t / half_life)


def ease_in_sine(t):
    return 1 - math.cos(t * math.pi / 2)


def ease_out_sine(t):
    return math.sin(t * math.pi / 2)


def ease_in_out_sine(t):
    return -(math.cos(math.pi * t) - 1) / 2


def ease_in_quad(t):
    return t * t


def ease_out_quad(t):
    return 1 - (1 - t) * (1 - t)


def ease_in_out_quad(t):
    return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2


def ease_in_cubic(t):
    return t ** 3


def ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def ease_in_out_cubic(t):
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_in_back(t):
    c1 = 1.70158
    return (c1 + 1) * t ** 3 - c1 * t ** 2


def ease_out_back(t):
    c1 = 1.70158
    return 1 + (c1 + 1) * (t - 1) ** 3 + c1 * (t - 1) ** 2


def ease_out_elastic(t):
    c4 = (2 * math.pi) / 3
    return 0 if t == 0 else 1 if t == 1 else 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * c4) + 1


def ease_out_bounce(t):
    n1, d1 = 7.5625, 2.75
    if t < 1 / d1:
        return n1 * t * t
    if t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    if t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    t -= 2.625 / d1
    return n1 * t * t + 0.984375


class _RateFunctions:
    """manim.rate_functions as a namespace."""

    def __getattr__(self, name):
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(name) from None


rate_functions = _RateFunctions()


# --- timing ------------------------------------------------------------------------------

class Clock:
    """An animation's linear progress (0 to 1) as a function of time; affine
    when its group has a linear rate, which the renderer handles exactly."""

    def __init__(self, start: float, end: float, outer=None):
        self.start, self.end = start, end
        # outer: (clock, rate, start_fraction, end_fraction) of the enclosing group
        self.outer = outer

    @property
    def affine(self) -> bool:
        return self.outer is None

    def __call__(self, t: float) -> float:
        if self.outer is None:
            if self.end <= self.start:
                return 1.0 if t >= self.end else 0.0
            return min(max((t - self.start) / (self.end - self.start), 0.0), 1.0)
        clock, rate, a, b = self.outer
        u = rate(clock(t))
        return min(max((u - a) / (b - a), 0.0), 1.0) if b > a else float(u >= b)

    def span(self) -> tuple[float, float]:
        if self.outer is None:
            return self.start, self.end
        return self.outer[0].span()

    def sub(self, a: float, b: float, rate=linear) -> "Clock":
        """The clock running over fractions [a, b] of this one, after `rate`."""
        if self.outer is None and rate is linear:
            d = self.end - self.start
            return Clock(self.start + a * d, self.start + b * d)
        return Clock(*self.span(), outer=(self, rate, a, b))


# --- base -------------------------------------------------------------------------------

class Animation:
    """manim's Animation: run_time seconds, rate_func applied to each
    submobject's progress, lag_ratio staggering the submobjects."""

    introducer = False

    def __init__(self, mobject: Mobject | None = None, run_time: float = 1.0, rate_func=smooth, lag_ratio: float = 0.0,
                 reverse_rate_function: bool = False, remover: bool = False, introducer: bool | None = None,
                 suspend_mobject_updating: bool = True, name: str | None = None, **kwargs):
        self.mobject = mobject if mobject is not None else Mobject()
        self.run_time = run_time
        self.rate_func = rate_func
        self.lag_ratio = lag_ratio
        self.reverse_rate_function = reverse_rate_function
        self.remover = remover
        if introducer is not None:
            self.introducer = introducer
        self.suspend_mobject_updating = suspend_mobject_updating
        self.name = name
        self.starting_mobject = None

    def __repr__(self):
        return f"{type(self).__name__}({type(self.mobject).__name__})"

    # --- manim's protocol (used when baked) --------------------------------------------

    def begin(self):
        self.starting_mobject = self.create_starting_mobject()
        self.interpolate(0)

    def create_starting_mobject(self):
        return self.mobject.copy()

    def get_all_mobjects(self):
        return [self.mobject, self.starting_mobject]

    def get_all_families_zipped(self):
        return zip(*[m.family_members_with_points() for m in self.get_all_mobjects()])

    def interpolate(self, alpha: float):
        self.interpolate_mobject(alpha)

    def interpolate_mobject(self, alpha: float):
        families = list(self.get_all_families_zipped())
        for i, mobs in enumerate(families):
            self.interpolate_submobject(*mobs, self.get_sub_alpha(alpha, i, len(families)))

    def interpolate_submobject(self, submobject, starting_submobject, alpha: float):
        pass

    def get_sub_alpha(self, alpha: float, index: int, num_submobjects: int) -> float:
        full = (num_submobjects - 1) * self.lag_ratio + 1
        value = alpha * full
        lower = index * self.lag_ratio
        if self.reverse_rate_function:
            return self.rate_func(1 - min(max(value - lower, 0), 1))
        return self.rate_func(min(max(value - lower, 0), 1))

    def finish(self):
        self.interpolate(1)

    def clean_up_from_scene(self, scene):
        if self.remover:
            scene.remove(self.mobject)

    def _setup_scene(self, scene):
        if self.introducer:
            scene.add(self.mobject)

    def get_run_time(self) -> float:
        return self.run_time

    def set_run_time(self, run_time: float):
        self.run_time = run_time
        return self

    def set_rate_func(self, rate_func):
        self.rate_func = rate_func
        return self

    def get_rate_func(self):
        return self.rate_func

    # --- compiling ------------------------------------------------------------------------

    def compile(self, ctx, clock: Clock) -> bool:
        """Emit native clips through ctx; False: bake this animation instead."""
        return False

    def finish_native(self, scene):
        """The state at the end, after compile() emitted clips."""
        self.finish()

    def _lag_clocks(self, clock: Clock, n: int):
        """Each submobject's clock and whether its rate runs backwards."""
        full = (n - 1) * self.lag_ratio + 1
        return [clock.sub(i * self.lag_ratio / full, (i * self.lag_ratio + 1) / full) for i in range(n)]


class Wait(Animation):
    def __init__(self, run_time: float = 1, **kwargs):
        super().__init__(None, run_time=run_time, rate_func=linear, **kwargs)

    def compile(self, ctx, clock):
        return True

    def finish_native(self, scene):
        pass


# --- groups ------------------------------------------------------------------------------

class AnimationGroup(Animation):
    def __init__(self, *animations, group=None, run_time: float | None = None, rate_func=linear,
                 lag_ratio: float = 0, **kwargs):
        self.animations = [_prepare(a) for a in animations]
        mobjects = [a.mobject for a in self.animations]
        super().__init__(group if group is not None else Group(*mobjects), rate_func=rate_func, lag_ratio=lag_ratio,
                         **kwargs)
        self._build_timings()
        self.run_time = self.max_end_time if run_time is None else run_time

    def _build_timings(self):
        run_times = np.array([a.get_run_time() for a in self.animations])
        lags = run_times[:-1] * self.lag_ratio
        starts = np.concatenate([[0.0], np.cumsum(lags)]) if len(run_times) else np.zeros(0)
        self.timings = [(a, s, s + r) for a, s, r in zip(self.animations, starts, run_times)]
        self.max_end_time = max((e for _, _, e in self.timings), default=0.0)

    def _setup_scene(self, scene):
        for a in self.animations:
            a._setup_scene(scene)

    def begin(self):
        for a in self.animations:
            a.begin()

    def interpolate(self, alpha: float):
        time = self.rate_func(alpha) * self.max_end_time
        for a, start, end in self.timings:
            run = end - start
            a.interpolate(min(max((time - start) / run, 0), 1) if run else float(time >= end))

    def finish(self):
        for a in self.animations:
            a.finish()

    def clean_up_from_scene(self, scene):
        for a in self.animations:
            a.clean_up_from_scene(scene)

    def children(self, clock: Clock):
        """Each animation with its clock."""
        out = []
        for a, start, end in self.timings:
            m = self.max_end_time or 1.0
            out.append((a, clock.sub(start / m, end / m, self.rate_func) if self.rate_func is not linear
                        else clock.sub(start / m, end / m)))
        return out


class LaggedStart(AnimationGroup):
    def __init__(self, *animations, lag_ratio: float = 0.05, **kwargs):
        super().__init__(*animations, lag_ratio=lag_ratio, **kwargs)


class Succession(AnimationGroup):
    def __init__(self, *animations, lag_ratio: float = 1, **kwargs):
        super().__init__(*animations, lag_ratio=lag_ratio, **kwargs)


def LaggedStartMap(anim_class, mobject, arg_creator=None, run_time: float = 2, **kwargs):  # noqa: N802
    anim_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k not in ("lag_ratio", "rate_func")}
    args = [arg_creator(m) if arg_creator else (m,) for m in mobject]
    return LaggedStart(*[anim_class(*a, **anim_kwargs) for a in args], run_time=run_time, **kwargs)


def _prepare(animation):
    if isinstance(animation, _AnimationBuilder):
        return animation.build()
    return animation


# --- transforms ------------------------------------------------------------------------

class Transform(Animation):
    """The mobject turns into target_mobject: points and styles are aligned
    (manim's align_data, at the level of drawn parts) and interpolated."""

    def __init__(self, mobject, target_mobject=None, path_arc: float = 0, path_arc_axis=OUT,
                 replace_mobject_with_target_in_scene: bool = False, **kwargs):
        self.path_arc = path_arc
        self.path_arc_axis = path_arc_axis
        self.target_mobject = target_mobject if target_mobject is not None else Mobject()
        self.replace_mobject_with_target_in_scene = replace_mobject_with_target_in_scene
        super().__init__(mobject, **kwargs)

    def create_target(self):
        return self.target_mobject

    def begin(self):
        self.target_mobject = self.create_target()
        self.target_copy = self.target_mobject.copy()
        _align_families(self.mobject, self.target_copy)
        super().begin()

    def get_all_mobjects(self):
        return [self.mobject, self.starting_mobject, self.target_copy]

    def interpolate_submobject(self, submobject, starting, target, alpha):
        _interpolate(submobject, starting, target, alpha, self.path_arc, self.path_arc_axis)

    def clean_up_from_scene(self, scene):
        super().clean_up_from_scene(scene)
        if self.replace_mobject_with_target_in_scene:
            scene.replace(self.mobject, self.target_mobject)

    def compile(self, ctx, clock):
        if self.path_arc:
            return False
        self.target_mobject = self.create_target()
        self.target_copy = self.target_mobject.copy()
        start = self.mobject.copy()
        _align_families(start, self.target_copy)
        pairs = list(zip(start.family_members_with_points(), self.target_copy.family_members_with_points()))
        clocks = self._lag_clocks(clock, len(pairs))
        z = ctx.z_of(self.mobject)
        ctx.claim(self.mobject)
        for i, ((a, b), c) in enumerate(zip(pairs, clocks)):
            ctx.morph(a, b, z(i, len(pairs)), c, self.rate_func, self.reverse_rate_function)
        self._aligned_start = start
        return True

    def finish_native(self, scene):
        end = self.rate_func(0.0 if self.reverse_rate_function else 1.0)
        if abs(end - 1) < 1e-6:
            _become_keeping_parts(self.mobject, self.target_copy)
        elif abs(end) > 1e-6:
            # e.g. there_and_back ends where it started; others in between
            state = self._aligned_start.copy()
            for m, a, b in zip(state.family_members_with_points(), self._aligned_start.family_members_with_points(),
                               self.target_copy.family_members_with_points()):
                _interpolate(m, a, b, end)
            _become_keeping_parts(self.mobject, state)


class ReplacementTransform(Transform):
    def __init__(self, mobject, target_mobject, **kwargs):
        super().__init__(mobject, target_mobject, replace_mobject_with_target_in_scene=True, **kwargs)

    def finish_native(self, scene):
        pass  # the target replaces the mobject (clean_up_from_scene)


class TransformFromCopy(Transform):
    def __init__(self, mobject, target_mobject, **kwargs):
        super().__init__(mobject.copy(), target_mobject, replace_mobject_with_target_in_scene=True, **kwargs)
        self.introducer = True

    def finish_native(self, scene):
        pass


class ClockwiseTransform(Transform):
    def __init__(self, mobject, target_mobject, path_arc: float = -PI, **kwargs):
        super().__init__(mobject, target_mobject, path_arc=path_arc, **kwargs)


class CounterclockwiseTransform(Transform):
    def __init__(self, mobject, target_mobject, path_arc: float = PI, **kwargs):
        super().__init__(mobject, target_mobject, path_arc=path_arc, **kwargs)


class MoveToTarget(Transform):
    def __init__(self, mobject, **kwargs):
        if getattr(mobject, "target", None) is None:
            raise ValueError("MoveToTarget needs mobject.generate_target() first")
        super().__init__(mobject, mobject.target, **kwargs)


class _MethodAnimation(Transform):
    def __init__(self, mobject, methods, **kwargs):
        self.methods = methods
        super().__init__(mobject, **kwargs)

    def create_target(self):
        target = self.mobject.copy()
        for method, args, kwargs in self.methods:
            getattr(target, method)(*args, **kwargs)
        return target


def ApplyMethod(method, *args, **kwargs):  # noqa: N802
    anim_kwargs = {k: kwargs.pop(k) for k in ("run_time", "rate_func", "lag_ratio") if k in kwargs}
    return _MethodAnimation(method.__self__, [(method.__name__, args, kwargs)], **anim_kwargs)


class ApplyFunction(Transform):
    def __init__(self, function, mobject, **kwargs):
        self.function = function
        super().__init__(mobject, **kwargs)

    def create_target(self):
        target = self.function(self.mobject.copy())
        if not isinstance(target, Mobject):
            raise TypeError("ApplyFunction's function must return a mobject")
        return target


class ApplyMatrix(_MethodAnimation):
    def __init__(self, matrix, mobject, about_point=ORIGIN, **kwargs):
        super().__init__(mobject, [("apply_matrix", (matrix,), {"about_point": about_point})], **kwargs)


class Restore(Transform):
    def __init__(self, mobject, **kwargs):
        if getattr(mobject, "_saved_state", None) is None:
            raise ValueError("Restore needs mobject.save_state() first")
        super().__init__(mobject, mobject._saved_state, **kwargs)


class FadeToColor(_MethodAnimation):
    def __init__(self, mobject, color, **kwargs):
        super().__init__(mobject, [("set_color", (color,), {})], **kwargs)


class ScaleInPlace(_MethodAnimation):
    def __init__(self, mobject, scale_factor: float, **kwargs):
        super().__init__(mobject, [("scale", (scale_factor,), {})], **kwargs)


class ShrinkToCenter(ScaleInPlace):
    def __init__(self, mobject, **kwargs):
        super().__init__(mobject, 0, **kwargs)


class _FromStart(Transform):
    """Animations that bring a mobject in from a start state (an introducer) or
    take it out to an end state (a remover)."""

    def __init__(self, mobject, **kwargs):
        super().__init__(mobject, **kwargs)

    def start_state(self):
        return self.mobject.copy()

    def end_state(self):
        return self.mobject.copy()

    def begin(self):
        # mobject goes from start_state to end_state
        target = self.end_state()
        start = self.start_state()
        self.target_mobject = target
        self.mobject.become(start)
        self.target_copy = target.copy()
        _align_families(self.mobject, self.target_copy)
        self.starting_mobject = self.mobject.copy()
        self.interpolate(0)

    def create_target(self):
        return self.end_state()

    def compile(self, ctx, clock):
        if self.path_arc:
            return False
        start, end = self.start_state(), self.end_state()
        _align_families(start, end)
        pairs = list(zip(start.family_members_with_points(), end.family_members_with_points()))
        clocks = self._lag_clocks(clock, len(pairs))
        z = ctx.z_of(self.mobject)
        ctx.claim(self.mobject)
        for i, ((a, b), c) in enumerate(zip(pairs, clocks)):
            ctx.morph(a, b, z(i, len(pairs)), c, self.rate_func, self.reverse_rate_function)
        self._final = end
        return True

    def finish_native(self, scene):
        if not self.remover:
            _become_keeping_parts(self.mobject, self._final)


class FadeIn(_FromStart):
    introducer = True

    def __init__(self, *mobjects, shift=None, target_position=None, scale: float = 1, **kwargs):
        mobject = mobjects[0] if len(mobjects) == 1 else Group(*mobjects)
        self.shift_vector = _fade_shift(mobject, shift, target_position)
        self.scale_factor = scale
        self._point_target = target_position is not None
        super().__init__(mobject, **kwargs)

    def start_state(self):
        start = self.mobject.copy().fade(1)
        start.shift(-self.shift_vector if not self._point_target else self.shift_vector)
        return start.scale(self.scale_factor)

    def _setup_scene(self, scene):
        # FadeIn(a, b) adds each of them
        for m in (self.mobject.submobjects if type(self.mobject) is Group else [self.mobject]):
            scene.add(m)


class FadeOut(_FromStart):
    def __init__(self, *mobjects, shift=None, target_position=None, scale: float = 1, **kwargs):
        mobject = mobjects[0] if len(mobjects) == 1 else Group(*mobjects)
        self.shift_vector = _fade_shift(mobject, shift, target_position)
        self.scale_factor = scale
        self._mobjects = mobjects
        super().__init__(mobject, remover=True, **kwargs)

    def end_state(self):
        return self.mobject.copy().fade(1).shift(self.shift_vector).scale(self.scale_factor)

    def clean_up_from_scene(self, scene):
        scene.remove(*self._mobjects)


def _fade_shift(mobject, shift, target_position):
    if shift is not None:
        return np.asarray(shift, dtype=float)
    if target_position is not None:
        if isinstance(target_position, Mobject):
            target_position = target_position.get_center()
        return np.asarray(target_position, dtype=float) - mobject.get_center()
    return np.zeros(3)


class GrowFromPoint(_FromStart):
    introducer = True

    def __init__(self, mobject, point, point_color=None, **kwargs):
        self.point = np.asarray(point, dtype=float)
        self.point_color = point_color
        super().__init__(mobject, **kwargs)

    def start_state(self):
        start = self.mobject.copy().scale(0).move_to(self.point)
        if self.point_color is not None:
            start.set_color(self.point_color)
        return start


class GrowFromCenter(GrowFromPoint):
    def __init__(self, mobject, point_color=None, **kwargs):
        super().__init__(mobject, mobject.get_center(), point_color=point_color, **kwargs)


class GrowFromEdge(GrowFromPoint):
    def __init__(self, mobject, edge, point_color=None, **kwargs):
        super().__init__(mobject, mobject.get_critical_point(edge), point_color=point_color, **kwargs)


class GrowArrow(GrowFromPoint):
    def __init__(self, arrow, point_color=None, **kwargs):
        super().__init__(arrow, arrow.get_start(), point_color=point_color, **kwargs)


class SpinInFromNothing(GrowFromCenter):
    def __init__(self, mobject, angle: float = PI / 2, point_color=None, **kwargs):
        super().__init__(mobject, point_color=point_color, path_arc=angle, **kwargs)


class FadeTransform(AnimationGroup):
    """The mobject fades out while stretching onto the target, which fades in
    from the mobject's shape."""

    def __init__(self, mobject, target_mobject, stretch: bool = True, dim_to_match: int = 1, **kwargs):
        self.to_add_on_completion = target_mobject
        out_target = mobject.copy().replace(target_mobject, dim_to_match, stretch).fade(1)
        start = target_mobject.copy().replace(mobject, dim_to_match, stretch).fade(1)
        anim_kwargs = {k: kwargs.pop(k) for k in ("run_time", "rate_func", "lag_ratio") if k in kwargs}
        fade_out = Transform(mobject, out_target, remover=True, **anim_kwargs)
        fade_in = _FadeInFrom(target_mobject, start, **anim_kwargs)
        super().__init__(fade_out, fade_in, **kwargs)


class FadeTransformPieces(FadeTransform):
    pass


class _FadeInFrom(_FromStart):
    introducer = True

    def __init__(self, mobject, start, **kwargs):
        self._start = start
        super().__init__(mobject, **kwargs)

    def start_state(self):
        return self._start.copy()


class TransformMatchingShapes(AnimationGroup):
    """Parts of the same shape move to their place in the target; the rest fade."""

    def __init__(self, mobject, target_mobject, **kwargs):
        anim_kwargs = {k: kwargs.pop(k) for k in ("run_time", "rate_func") if k in kwargs}
        sources, targets = mobject.family_members_with_points(), target_mobject.family_members_with_points()
        unmatched = list(range(len(targets)))
        anims, faded = [], []
        for s in sources:
            key = _shape_key(s)
            match = next((j for j in unmatched if _shape_key(targets[j]) == key), None)
            if match is None:
                faded.append(s)
            else:
                unmatched.remove(match)
                anims.append(Transform(s, targets[match], **anim_kwargs))
        if faded:
            anims.append(FadeOut(*faded, **anim_kwargs))
        if unmatched:
            anims.append(FadeIn(*[targets[j] for j in unmatched], **anim_kwargs))
        self._source, self._target = mobject, target_mobject
        super().__init__(*anims, **kwargs)

    def clean_up_from_scene(self, scene):
        super().clean_up_from_scene(scene)
        scene.remove(self._source)
        for m in self._source.family_members_with_points():
            scene.remove(m)
        for m in self._target.family_members_with_points():
            scene.remove(m)
        scene.add(self._target)


def _shape_key(m):
    p = m.points - m.get_center()
    scale = max(m.width, m.height, 1e-9)
    return (len(p), np.round(p / scale, 2).tobytes())


# --- partial and outlines ----------------------------------------------------------

class ShowPartial(Animation):
    def _bounds(self, alpha):
        raise NotImplementedError

    def interpolate_submobject(self, submobject, starting, alpha):
        submobject.pointwise_become_partial(starting, *self._bounds(alpha))


class Create(ShowPartial):
    introducer = True

    def __init__(self, mobject, lag_ratio: float = 1.0, **kwargs):
        super().__init__(mobject, lag_ratio=lag_ratio, **kwargs)

    def _bounds(self, alpha):
        return 0, alpha

    def compile(self, ctx, clock):
        return _compile_leaves(self, ctx, clock, "create")

    def finish_native(self, scene):
        pass


class Uncreate(Create):
    introducer = False

    def __init__(self, mobject, reverse_rate_function: bool = True, remover: bool = True, **kwargs):
        super().__init__(mobject, reverse_rate_function=reverse_rate_function, remover=remover, **kwargs)


class DrawBorderThenFill(Animation):
    introducer = True

    def __init__(self, vmobject, run_time: float = 2, rate_func=double_smooth, stroke_width: float = 2,
                 stroke_color=None, **kwargs):
        super().__init__(vmobject, run_time=run_time, rate_func=rate_func, **kwargs)
        self.stroke_width = stroke_width
        self.stroke_color = stroke_color

    def begin(self):
        self.outline = self.mobject.copy().set_fill(opacity=0)
        for m in self.outline.family_members_with_points():
            if isinstance(m, VMobject):
                color = self.stroke_color or (m.stroke_color if m.stroke_width > 0 else m.fill_color)
                m.set_stroke(color=color, width=self.stroke_width, opacity=1)
        super().begin()

    def get_all_mobjects(self):
        return [self.mobject, self.starting_mobject, self.outline]

    def interpolate_submobject(self, submobject, starting, outline, alpha):
        if alpha < 0.5:
            submobject.pointwise_become_partial(outline, 0, 2 * alpha)
            submobject.match_style(outline, family=False)
        else:
            submobject.interpolate(outline, starting, 2 * alpha - 1)

    def compile(self, ctx, clock):
        # the renderer's outline is manim's default one
        if self.stroke_width != 2 or self.stroke_color is not None:
            return False
        return _compile_leaves(self, ctx, clock, "write")

    def finish_native(self, scene):
        pass


class Write(DrawBorderThenFill):
    def __init__(self, vmobject, rate_func=linear, reverse: bool = False, run_time: float | None = None,
                 lag_ratio: float | None = None, **kwargs):
        n = len(vmobject.family_members_with_points())
        if run_time is None:
            run_time = 1 if n < 15 else 2
        if lag_ratio is None:
            lag_ratio = min(4.0 / max(1.0, n), 0.2)
        self.reverse = reverse
        kwargs.setdefault("remover", reverse)
        super().__init__(vmobject, rate_func=rate_func, run_time=run_time, lag_ratio=lag_ratio,
                         introducer=not reverse, **kwargs)

    def compile(self, ctx, clock):
        if self.reverse:
            return False
        return super().compile(ctx, clock)


class Unwrite(Write):
    def __init__(self, vmobject, rate_func=linear, reverse: bool = True, **kwargs):
        super().__init__(vmobject, rate_func=rate_func, reverse=reverse, reverse_rate_function=True, **kwargs)

    def compile(self, ctx, clock):
        return _compile_leaves(self, ctx, clock, "write", order=-1 if self.reverse else 1)


def _compile_leaves(anim, ctx, clock, kind, order: int = 1):
    leaves = anim.mobject.family_members_with_points()
    if any(isinstance(m, ImageMobject) for m in leaves):
        return False
    z = ctx.z_of(anim.mobject)
    ctx.claim(anim.mobject)
    n = len(leaves)
    clocks = anim._lag_clocks(clock, n)
    if order < 0:
        clocks = clocks[::-1]
    for i, (leaf, c) in enumerate(zip(leaves, clocks)):
        ctx.partial(kind, leaf, z(i, n), c, anim.rate_func, anim.reverse_rate_function)
    return True


class ShowPassingFlash(ShowPartial):
    def __init__(self, mobject, time_width: float = 0.1, **kwargs):
        self.time_width = time_width
        super().__init__(mobject, remover=True, introducer=True, **kwargs)

    def _bounds(self, alpha):
        tw = self.time_width
        upper = alpha * (1 + tw)
        return min(max(upper - tw, 0), 1), min(max(upper, 0), 1)


class ShowIncreasingSubsets(Animation):
    introducer = True

    def __init__(self, group, suspend_mobject_updating: bool = False, int_func=np.floor,
                 reverse_rate_function: bool = False, **kwargs):
        self.all_submobs = list(group.submobjects)
        self.int_func = int_func
        super().__init__(group, suspend_mobject_updating=suspend_mobject_updating,
                         reverse_rate_function=reverse_rate_function, **kwargs)

    def interpolate_mobject(self, alpha):
        n = len(self.all_submobs)
        value = 1 - self.rate_func(alpha) if self.reverse_rate_function else self.rate_func(alpha)
        self.mobject.submobjects = self.all_submobs[: int(self.int_func(value * n))]

    def finish_native(self, scene):
        self.mobject.submobjects = self.all_submobs
        if self.remover:
            self.mobject.submobjects = self.all_submobs

    def compile(self, ctx, clock):
        if not clock.affine:
            return False
        n = len(self.all_submobs)
        start, end = clock.span()
        times = np.linspace(0, 1, 513)
        values = np.array([self.rate_func(t) for t in times])
        if self.reverse_rate_function:
            values = 1 - values
        counts = self.int_func(values * n).astype(int)
        z = ctx.z_of(self.mobject)
        ctx.claim(self.mobject)
        leaves_before = 0
        total = len(self.mobject.family_members_with_points())
        for j, sub in enumerate(self.all_submobs):
            shown = counts > j
            leaves = sub.family_members_with_points()
            edges = np.flatnonzero(np.diff(np.concatenate([[0], shown.astype(int), [0]])))
            for a, b in zip(edges[0::2], edges[1::2]):  # shown[a:b]
                t0 = start + times[a] * (end - start)
                t1 = end if b == len(times) else start + times[b] * (end - start)
                for k, leaf in enumerate(leaves):
                    ctx.show(leaf, z(leaves_before + k, total), t0, t1)
            leaves_before += len(leaves)
        return True


class AddTextLetterByLetter(ShowIncreasingSubsets):
    def __init__(self, text, rate_func=linear, time_per_char: float = 0.1, run_time: float | None = None, **kwargs):
        if run_time is None:
            run_time = max(0.06, time_per_char) * len(text)
        super().__init__(text, int_func=np.ceil, rate_func=rate_func, run_time=run_time, **kwargs)


class RemoveTextLetterByLetter(AddTextLetterByLetter):
    introducer = False

    def __init__(self, text, **kwargs):
        super().__init__(text, reverse_rate_function=True, remover=True, **kwargs)


# --- indications and motion (baked unless noted) --------------------------------------------

class Indicate(Transform):
    def __init__(self, mobject, scale_factor: float = 1.2, color=YELLOW, rate_func=there_and_back, **kwargs):
        self.color, self.scale_factor = color, scale_factor
        super().__init__(mobject, rate_func=rate_func, **kwargs)

    def create_target(self):
        return self.mobject.copy().scale(self.scale_factor).set_color(self.color)


class Flash(AnimationGroup):
    def __init__(self, point, line_length: float = 0.2, num_lines: int = 12, flash_radius: float = 0.1,
                 line_stroke_width: float = 3, color=YELLOW, time_width: float = 1, run_time: float = 1.0, **kwargs):
        from .shapes import Line
        if isinstance(point, Mobject):
            point = point.get_center()
        point = np.asarray(point, dtype=float)
        lines = []
        for k in range(num_lines):
            angle = k * TAU / num_lines
            d = np.array([np.cos(angle), np.sin(angle), 0])
            lines.append(Line(point + flash_radius * d, point + (flash_radius + line_length) * d,
                              stroke_width=line_stroke_width, color=color))
        flashes = [ShowPassingFlash(l, time_width=time_width, run_time=run_time) for l in lines]
        super().__init__(*flashes, group=Group(*lines), run_time=run_time, **kwargs)


class Circumscribe(Succession):
    def __init__(self, mobject, shape=None, fade_in: bool = False, fade_out: bool = False, time_width: float = 0.3,
                 buff: float = 0.1, color=YELLOW, run_time: float = 1, stroke_width: float = 4, **kwargs):
        from .shapes import Rectangle
        shape = shape or Rectangle
        frame = shape(color=color, stroke_width=stroke_width)
        frame.stretch_to_fit_width(mobject.width + 2 * buff).stretch_to_fit_height(mobject.height + 2 * buff)
        frame.move_to(mobject)
        if fade_in or fade_out:
            parts = [Create(frame, run_time=run_time / 2), FadeOut(frame, run_time=run_time / 2)]
        else:
            parts = [ShowPassingFlash(frame, time_width=time_width, run_time=run_time)]
        super().__init__(*parts, **kwargs)


class Wiggle(Animation):
    def __init__(self, mobject, scale_value: float = 1.1, rotation_angle: float = 0.01 * TAU, n_wiggles: int = 6,
                 scale_about_point=None, rotate_about_point=None, run_time: float = 2, **kwargs):
        self.scale_value, self.rotation_angle, self.n_wiggles = scale_value, rotation_angle, n_wiggles
        self.scale_about_point, self.rotate_about_point = scale_about_point, rotate_about_point
        super().__init__(mobject, run_time=run_time, **kwargs)

    def interpolate_submobject(self, submobject, starting, alpha):
        submobject.points = starting.points.copy()
        center = self.mobject.get_center() if self.scale_about_point is None else self.scale_about_point
        submobject.scale(1 + (self.scale_value - 1) * there_and_back(alpha), about_point=center)
        submobject.rotate(wiggle(alpha, self.n_wiggles) * self.rotation_angle,
                          about_point=center if self.rotate_about_point is None else self.rotate_about_point)


class Rotate(Animation):
    def __init__(self, mobject, angle: float = PI, axis=OUT, about_point=None, about_edge=None, **kwargs):
        self.angle, self.axis, self.about_point, self.about_edge = angle, axis, about_point, about_edge
        super().__init__(mobject, **kwargs)

    def interpolate_mobject(self, alpha):
        self.mobject.become(self.starting_mobject)
        about = self.about_point
        if about is None:
            about = self.starting_mobject.get_critical_point(self.about_edge if self.about_edge is not None else ORIGIN)
        self.mobject.rotate(self.rate_func(alpha) * self.angle, axis=self.axis, about_point=about)


Rotating = Rotate


class MoveAlongPath(Animation):
    def __init__(self, mobject, path, suspend_mobject_updating: bool = False, **kwargs):
        self.path = path
        super().__init__(mobject, suspend_mobject_updating=suspend_mobject_updating, **kwargs)

    def interpolate_mobject(self, alpha):
        self.mobject.move_to(self.path.point_from_proportion(self.rate_func(alpha)))


class UpdateFromFunc(Animation):
    def __init__(self, mobject, update_function, suspend_mobject_updating: bool = False, **kwargs):
        self.update_function = update_function
        super().__init__(mobject, suspend_mobject_updating=suspend_mobject_updating, **kwargs)

    def interpolate_mobject(self, alpha):
        self.update_function(self.mobject)


class UpdateFromAlphaFunc(UpdateFromFunc):
    def interpolate_mobject(self, alpha):
        self.update_function(self.mobject, self.rate_func(alpha))


# --- .animate ----------------------------------------------------------------------------

class _AnimationBuilder:
    """mob.animate.shift(UP).scale(2): a Transform to the result."""

    def __init__(self, mobject):
        self.mobject = mobject
        self.methods = []
        self.anim_args = {}

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self.methods.append((name, args, kwargs))
            return self
        return record

    def __call__(self, **anim_args):
        self.anim_args.update(anim_args)
        return self

    def set_anim_args(self, **anim_args):
        self.anim_args.update(anim_args)
        return self

    def build(self):
        return _MethodAnimation(self.mobject, self.methods, **self.anim_args)


# --- alignment ---------------------------------------------------------------------

def _align_families(a: Mobject, b: Mobject):
    """Give a and b the same number of drawn parts, each pair with the same
    number of curves (manim's align_data, flattened to the drawn parts)."""
    la, lb = a.family_members_with_points(), b.family_members_with_points()
    if not la and not lb:
        return
    if not la:
        a.add(_null_like(lb[0], a.get_center()))
        la = a.family_members_with_points()
    if not lb:
        b.add(_null_like(la[0], b.get_center()))
        lb = b.family_members_with_points()
    if len(la) < len(lb):
        _add_parts(a, la, len(lb))
    elif len(lb) < len(la):
        _add_parts(b, lb, len(la))
    for x, y in zip(a.family_members_with_points(), b.family_members_with_points()):
        if isinstance(x, VMobject) and isinstance(y, VMobject) and not isinstance(x, ImageMobject):
            _align_points(x, y)


def _null_like(m, center):
    null = m.copy()
    null.submobjects = []
    null.points = np.repeat(np.asarray(center, dtype=float)[None], max(len(m.points), 4), axis=0)
    if isinstance(null, VMobject):
        null.fade(1)
    return null


def _add_parts(group: Mobject, leaves, target: int):
    """Repeat drawn parts (manim's add_n_more_submobjects) until there are
    `target`; the copies lie on their originals."""
    n = len(leaves)
    repeat = (np.arange(target) * n) // target
    counts = np.bincount(repeat, minlength=n)
    for leaf, count in zip(leaves, counts):
        for _ in range(count - 1):
            copy = leaf.copy()
            copy.submobjects = []
            group.add(copy)


def _align_points(a: VMobject, b: VMobject):
    """manim's VMobject.align_points: as many subpaths, each with as many curves."""
    for m in (a, b):
        whole = 4 * (len(m.points) // 4)
        if whole == 0:
            center = m.points[0] if len(m.points) else m.get_center()
            m.points = np.repeat(np.asarray(center, dtype=float)[None], 4, axis=0)
        else:
            m.points = m.points[:whole]
    if len(a.points) == len(b.points):
        return
    sa, sb = a.get_subpaths(), b.get_subpaths()
    n = max(len(sa), len(sb))

    def nth(paths, i):
        if i >= len(paths):
            return np.repeat(paths[-1][-1:], 4, axis=0)
        path = paths[i]
        while len(path) > 4 and np.allclose(path[-4:], path[-5]):
            path = path[:-4]
        return path
    new_a, new_b = [], []
    for i in range(n):
        pa, pb = nth(sa, i), nth(sb, i)
        ca, cb = len(pa) // 4, len(pb) // 4
        new_a.append(_insert_curves(pa, max(0, cb - ca)))
        new_b.append(_insert_curves(pb, max(0, ca - cb)))
    a.points = np.concatenate(new_a)
    b.points = np.concatenate(new_b)


def _insert_curves(points: np.ndarray, n: int) -> np.ndarray:
    """manim's bezier_remap: split the curves evenly until there are n more."""
    if n == 0:
        return points
    from .mobject import sub_bezier
    curves = points.reshape(-1, 4, 3)
    current = len(curves)
    target = current + n
    repeat = (np.arange(target) * current) // target
    split = np.bincount(repeat, minlength=current)
    out = []
    for curve, k in zip(curves, split):
        for j in range(k):
            out.append(sub_bezier(curve, j / k, (j + 1) / k))
    return np.concatenate(out)


def _interpolate(m, a, b, alpha, path_arc: float = 0, axis=OUT):
    if path_arc:
        from .mobject import _rotation
        # points move along arcs (manim's path_along_arc)
        vects = b.points - a.points
        centers = a.points + 0.5 * vects
        if abs(path_arc) > 1e-6:
            rot = _rotation(alpha * path_arc, axis)
            centers = centers + np.cross(np.asarray(axis, dtype=float), vects / 2.0) / np.tan(path_arc / 2)
            m.points = centers + (a.points - centers) @ rot.T
        else:
            m.points = a.points + alpha * vects
        if isinstance(m, VMobject) and isinstance(a, VMobject) and isinstance(b, VMobject):
            saved = m.points
            m.interpolate(a, b, alpha)
            m.points = saved
        return
    m.interpolate(a, b, alpha)


def _become_keeping_parts(mobject: Mobject, state: Mobject):
    """mobject takes state's points and styles; its parts stay the same
    objects when the structure matches."""
    mine, theirs = mobject.get_family(), state.get_family()
    if len(mine) == len(theirs) and all(type(x) is type(y) for x, y in zip(mine, theirs)):
        for x, y in zip(mine, theirs):
            subs = x.submobjects
            for key, value in y.__dict__.items():
                if key != "submobjects":
                    setattr(x, key, value.copy() if isinstance(value, np.ndarray) else value)
            x.submobjects = subs
    else:
        mobject.become(state)
