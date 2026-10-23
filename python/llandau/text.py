"""Text in Linux Biolinum, all of it typeset with Typst (math too).

    text("plain"), bold("strong"), italic("aside")
    text("cost $O(N^3)$")            # contains $ -> the whole string is Typst markup
    eq("arrow(F) = m arrow(a)")       # display math, Typst syntax
    e = eq("{F} = m a", F="arrow(F)")  # e["F"]: the glyphs of a part, to colour, bracket, animate
    title("slide title"), bullets([...])
"""
import hashlib
import re
import subprocess
from functools import cache, partial
from pathlib import Path

from .mobject import SVGMobject, ManimColor, VGroup, LEFT

from . import theme
from .theme import FONT
from .layout import R, place, stack, S
from .media import CACHE_DIR

__all__ = ["text", "texts", "bold", "italic", "eq", "typst", "title", "bullets", "NORMAL", "BOLD", "ITALIC"]

NORMAL, BOLD, ITALIC = "NORMAL", "BOLD", "ITALIC"

# first one installed wins; Libertinus Math matches Libertine/Biolinum
MATH_FONTS = ("Libertinus Math", "New Computer Modern Math")
# extra Typst code prepended to every snippet, e.g. '#import "@preview/physica:0.9.5": *'
TYPST_PREAMBLE = ""
# Typst renders in pt; 1/72 makes size=N match Text(font_size=N) (measured: identical cap height)
TYPST_SCALE = 1 / 72

_MATH = re.compile(r"(?<!\\)\$")


def _minus(s: str) -> str:
    """A number's leading hyphen as a minus sign."""
    return "−" + s[1:] if s.startswith("-") else s


def text(s: str, size: float | None = None, color=None, weight=NORMAL, slant=NORMAL,
         typst_markup: bool | None = None, font: str | None = None, **kwargs):
    """Single text object, typeset with Typst. Contains unescaped $...$ -> the string is Typst
    markup (so *, _, #, ~ have their Typst meaning there; force either way with typst_markup=).
    A literal dollar is \\$. Lines break only at explicit newlines. font: other than the deck's."""
    if typst_markup is None:
        typst_markup = bool(_MATH.search(s))
    if typst_markup:
        return typst(s, size=size, color=color, weight=weight, slant=slant, font=font, **kwargs)
    mob = typst(_plain(s), size=size, color=color, weight=weight, slant=slant, font=font, **kwargs)
    mob.plain_text = True
    return mob


def _typst_string(line: str) -> str:
    """A line as a Typst string: nothing in it is markup."""
    escaped = line.replace("\\", "\\\\").replace('"', '\\"')
    return f'#"{escaped}"'


bold = partial(text, weight=BOLD)
italic = partial(text, slant=ITALIC)


_PART = re.compile(r"\{(\w+)\}")


def eq(s: str, size: float | None = None, **kwargs):
    """Display math in Typst syntax (no $ needed). Glyphs: eq(...).family_members_with_points().

    Parts: a {name} in s with a keyword name=... is that Typst math (balanced: an operand, not
    "sqrt("), typeset in place as a group; eq(...)["name"] is its glyphs, a VGroup:
        e = eq("cal(S) = {length} - epsilon.alt T", length="cal(L)_g [gamma]")
        e["length"].set_color(BLUE)
        bracket(e["length"], "geodesic\nlength")
    Keep operators outside the parts: a part is spaced as one operand. Other kwargs go to SVGMobject."""
    parts = {name: kwargs.pop(name) for name in dict.fromkeys(_PART.findall(s)) if name in kwargs}
    if not parts:
        return typst(f"$ {s} $", size=size, **kwargs)
    tags = {(1, k, 254): name for k, name in enumerate(parts, start=1)}  # colours of nothing else
    fills = {name: f"rgb{tag}" for tag, name in tags.items()}

    def substitute(tagged: bool) -> str:
        def part(m):
            if m[1] not in parts:
                return m[0]
            fill = f"(fill: {fills[m[1]]})" if tagged else ""
            return f"#text{fill}[$ {parts[m[1]]} $]"  # the same element on both pages: the same layout
        return f"$ {_PART.sub(part, s)} $"

    # page 1 as shown, page 2 with each part in its own colour: path i of one is path i of the other
    doc = _document(substitute(False) + "\n#pagebreak()\n" + substitute(True), size, kwargs.pop("color", None),
                    kwargs.pop("weight", NORMAL), kwargs.pop("slant", NORMAL), kwargs.pop("font", None))
    shown, tagged = _compile_pages(doc, 2)
    mob = SVGMobject(str(shown), height=None, **kwargs).scale(TYPST_SCALE)
    paths = SVGMobject(str(tagged), height=None).submobjects
    if len(paths) != len(mob.submobjects):
        raise RuntimeError(f"eq: the parts of {s!r} changed its layout")
    for glyph, path in zip(mob.submobjects, paths):
        name = tags.get(tuple(int(round(255 * c)) for c in path.get_color().to_rgb()))
        if name is not None:
            mob.id_to_vgroup_dict.setdefault(name, VGroup()).add(glyph)
    missing = [name for name in parts if name not in mob.id_to_vgroup_dict]
    if missing:
        raise ValueError(f"eq: parts {missing} of {s!r} draw nothing")
    return mob


def typst(src: str, size: float | None = None, color=None, weight=NORMAL, slant=NORMAL, font: str | None = None,
          **kwargs) -> SVGMobject:
    """Arbitrary Typst markup as a vector mobject, in the deck font and colour.
    Colours set inside the markup are kept; kwargs go to SVGMobject."""
    doc = _document(src, size, color, weight, slant, font)
    return SVGMobject(str(_compile_pages(doc, 1)[0]), height=None, **kwargs).scale(TYPST_SCALE)


def texts(strings, size: float | None = None, color=None, weight=NORMAL, slant=NORMAL, font: str | None = None,
          **kwargs) -> list[SVGMobject]:
    """text() of each string, typeset in one Typst run (e.g. tick labels)."""
    sources = [s if _MATH.search(s) else _plain(s) for s in strings]
    if not sources:
        return []
    doc = _document("\n#pagebreak()\n".join(sources), size, color, weight, slant, font)
    out = []
    for s, path in zip(strings, _compile_pages(doc, len(sources))):
        mob = SVGMobject(str(path), height=None, **kwargs).scale(TYPST_SCALE)
        mob.plain_text = not _MATH.search(s)
        out.append(mob)
    return out


def _plain(s: str) -> str:
    return " \\\n".join(_typst_string(line) for line in s.replace(r"\$", "$").split("\n"))


def _document(src: str, size, color, weight, slant, font=None) -> str:
    """size, color None: the theme's body size and foreground."""
    fonts = ", ".join(f'"{f}"' for f in _math_fonts())
    size = size if size is not None else theme.current().body
    return "\n".join([
        "#set page(width: auto, height: auto, margin: 0pt, fill: none)",
        f'#set text(font: "{font or FONT}", size: {size}pt, fill: rgb("{ManimColor(color or theme.current().fg).to_hex()}"),'
        f' weight: "{"bold" if weight == BOLD else "regular"}", style: "{"italic" if slant == ITALIC else "normal"}")',
        f"#show math.equation: set text(font: ({fonts},))",
        TYPST_PREAMBLE,
        src,
    ])


@cache
def _math_fonts() -> list[str]:
    try:
        listing = subprocess.run(["typst", "fonts"], capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        raise RuntimeError("math needs the typst CLI (https://typst.app, or `brew install typst`)") from None
    available = {line.strip().lower() for line in listing.splitlines()}
    return [f for f in MATH_FONTS if f.lower() in available] or [MATH_FONTS[-1]]


def _compile_pages(doc: str, pages: int) -> list[Path]:
    """One SVG per page of a document (cached by its text)."""
    folder = CACHE_DIR / "typst" / hashlib.sha1(doc.encode()).hexdigest()[:16]
    paths = [folder / f"{k}.svg" for k in range(1, pages + 1)]
    if not all(p.exists() for p in paths):
        folder.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(["typst", "compile", "-", str(folder / "{p}.tmp.svg")], input=doc, text=True,
                                capture_output=True)
        if result.returncode != 0:
            raise ValueError(f"typst failed on:\n{doc}\n\n{result.stderr}")
        for p in paths:
            p.with_suffix(".tmp.svg").replace(p)  # atomic: a crashed run never leaves half a page
    return paths


def title(s: str, size: float | None = None, **kwargs):
    """Text placed left-aligned in the title band (shrunk if it doesn't fit); size: the theme's title."""
    return place(text(s, size=size if size is not None else theme.current().title, **kwargs), R.title, anchor=LEFT)


def bullets(items, gap: float = S, marker: str | None = None, **kwargs):
    """Left-aligned stack of text lines; marker e.g. "–" or "•"."""
    lines = [text(f"{marker} {item}" if marker else item, **kwargs) for item in items]
    return stack(*lines, gap=gap, align=LEFT)
