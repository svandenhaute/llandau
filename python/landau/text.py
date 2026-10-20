"""Text in Linux Biolinum, all of it typeset with Typst (math too).

    text("plain"), bold("strong"), italic("aside")
    text("cost $tilde N^3$")          # contains $ -> the whole string is Typst markup
    eq("arrow(F) = m arrow(a)")       # display math, Typst syntax
    title("slide title"), bullets([...])
"""
import hashlib
import re
import subprocess
from functools import cache, partial
from pathlib import Path

from .mobject import SVGMobject, ManimColor, LEFT

from . import theme
from .theme import FONT, TITLE, BODY
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


def text(s: str, size: float = BODY, color=None, weight=NORMAL, slant=NORMAL,
         typst_markup: bool | None = None, **kwargs):
    """Single text object, typeset with Typst. Contains unescaped $...$ -> the string is Typst
    markup (so *, _, #, ~ have their Typst meaning there; force either way with typst_markup=).
    A literal dollar is \\$. Lines break only at explicit newlines."""
    if typst_markup is None:
        typst_markup = bool(_MATH.search(s))
    if typst_markup:
        return typst(s, size=size, color=color, weight=weight, slant=slant, **kwargs)
    mob = typst(_plain(s), size=size, color=color, weight=weight, slant=slant, **kwargs)
    mob.plain_text = True
    return mob


def _typst_string(line: str) -> str:
    """A line as a Typst string: nothing in it is markup."""
    escaped = line.replace("\\", "\\\\").replace('"', '\\"')
    return f'#"{escaped}"'


bold = partial(text, weight=BOLD)
italic = partial(text, slant=ITALIC)


def eq(s: str, size: float = BODY, **kwargs):
    """Display math in Typst syntax (no $ needed). Glyphs: eq(...).family_members_with_points()."""
    return typst(f"$ {s} $", size=size, **kwargs)


def typst(src: str, size: float = BODY, color=None, weight=NORMAL, slant=NORMAL, **kwargs) -> SVGMobject:
    """Arbitrary Typst markup as a vector mobject, in the deck font and colour.
    Colours set inside the markup are kept; kwargs go to SVGMobject."""
    return SVGMobject(str(_compile(_document(src, size, color, weight, slant))), height=None, **kwargs).scale(TYPST_SCALE)


def texts(strings, size: float = BODY, color=None, weight=NORMAL, slant=NORMAL, **kwargs) -> list[SVGMobject]:
    """text() of each string, typeset in one Typst run (e.g. tick labels)."""
    sources = [s if _MATH.search(s) else _plain(s) for s in strings]
    if not sources:
        return []
    doc = _document("\n#pagebreak()\n".join(sources), size, color, weight, slant)
    out = []
    for s, path in zip(strings, _compile_pages(doc, len(sources))):
        mob = SVGMobject(str(path), height=None, **kwargs).scale(TYPST_SCALE)
        mob.plain_text = not _MATH.search(s)
        out.append(mob)
    return out


def _plain(s: str) -> str:
    return " \\\n".join(_typst_string(line) for line in s.replace(r"\$", "$").split("\n"))


def _document(src: str, size, color, weight, slant) -> str:
    fonts = ", ".join(f'"{f}"' for f in _math_fonts())
    return "\n".join([
        "#set page(width: auto, height: auto, margin: 0pt, fill: none)",
        f'#set text(font: "{FONT}", size: {size}pt, fill: rgb("{ManimColor(color or theme.current().fg).to_hex()}"),'
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


def _compile(doc: str) -> Path:
    out = CACHE_DIR / "typst" / f"{hashlib.sha1(doc.encode()).hexdigest()[:16]}.svg"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.svg")
        result = subprocess.run(["typst", "compile", "-", str(tmp)], input=doc, text=True, capture_output=True)
        if result.returncode != 0:
            raise ValueError(f"typst failed on:\n{doc}\n\n{result.stderr}")
        tmp.replace(out)
    return out


def _compile_pages(doc: str, pages: int) -> list[Path]:
    """One SVG per page of a document."""
    folder = CACHE_DIR / "typst" / hashlib.sha1(doc.encode()).hexdigest()[:16]
    paths = [folder / f"{k}.svg" for k in range(1, pages + 1)]
    if not all(p.exists() for p in paths):
        folder.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(["typst", "compile", "-", str(folder / "{p}.svg")], input=doc, text=True,
                                capture_output=True)
        if result.returncode != 0:
            raise ValueError(f"typst failed on:\n{doc}\n\n{result.stderr}")
    return paths


def title(s: str, size: float = TITLE, **kwargs):
    """Text placed left-aligned in the title band (shrunk if it doesn't fit)."""
    return place(text(s, size=size, **kwargs), R.title, anchor=LEFT)


def bullets(items, gap: float = S, marker: str | None = None, **kwargs):
    """Left-aligned stack of text lines; marker e.g. "–" or "•"."""
    lines = [text(f"{marker} {item}" if marker else item, **kwargs) for item in items]
    return stack(*lines, gap=gap, align=LEFT)
