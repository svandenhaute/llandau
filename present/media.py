"""Images, SVGs and layered Inkscape SVGs.

Layered SVGs are split at render time (and cached), so the .svg is the only source of truth.

Layer conventions:
  - steps are ordered by the first number in the layer label ("Layer 3", "2 arrows");
    layers sharing a number appear together. order="document" uses stacking order instead.
  - visibility saved in the file is ignored: every layer is shown eventually.
  - layers whose label starts with "_" are skipped (guides, drafts).
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .mobject import ImageMobject, SVGMobject, Group, VGroup
from .shapes import Rectangle

from .layout import Region, place

__all__ = ["image", "svg", "layered_svg", "svg_layers", "CACHE_DIR"]

SVG_NS = "http://www.w3.org/2000/svg"
INK_NS = "http://www.inkscape.org/namespaces/inkscape"
for _prefix, _uri in {
    "": SVG_NS,
    "inkscape": INK_NS,
    "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
    "xlink": "http://www.w3.org/1999/xlink",
}.items():
    ET.register_namespace(_prefix, _uri)

GROUPMODE = f"{{{INK_NS}}}groupmode"
LABEL = f"{{{INK_NS}}}label"
PAGE_ID = "present-page"
NON_DRAWING = {"defs", "metadata", "namedview", "style", "title", "desc", "script"}

CACHE_DIR = Path(os.environ.get("PRESENT_CACHE", ".present_cache"))
_INKSCAPE_MAC = "/Applications/Inkscape.app/Contents/MacOS/inkscape"


# A media file's unscaled size is arbitrary (pixels, dpi, SVGMobject's default height), so given a
# region, media fills it by default (fit="fill"); text uses place()'s shrink-only default instead.

def image(path, region: Region | None = None, fit="fill", dpi: int = 300, renderer: str | None = None,
          **place_kwargs) -> ImageMobject:
    """Raster image; an .svg is rasterized first (cached), keeping full fidelity for e.g. matplotlib output."""
    path = Path(path)
    if path.suffix.lower() == ".svg":
        renderer = _renderer(renderer)
        source = path.read_bytes()
        out = _cached([hashlib.sha1(source).hexdigest(), dpi, renderer], ".png")
        if not out.exists():
            _rasterize(source, out, dpi, renderer)
        path = out
    mob = ImageMobject(str(path))
    return place(mob, region, fit=fit, **place_kwargs) if region is not None else mob


def svg(path, region: Region | None = None, **kwargs) -> SVGMobject:
    """Plain vector SVG (no layering). Extra kwargs go to place() or SVGMobject."""
    place_kwargs = {"fit": "fill"} | {k: kwargs.pop(k) for k in ("anchor", "fit", "buff", "scale") if k in kwargs}
    mob = SVGMobject(str(path), **kwargs)
    return place(mob, region, **place_kwargs) if region is not None else mob


# --- layers ------------------------------------------------------------------

@dataclass(frozen=True)
class Layer:
    id: str
    label: str
    index: int  # position in the document (= stacking order)


def svg_layers(path) -> list[Layer]:
    """Top-level Inkscape layers of an SVG, in document order."""
    root = ET.parse(str(path)).getroot()
    return [
        Layer(el.get("id") or f"_layer{i}", el.get(LABEL) or "", i)
        for i, el in enumerate(_top_layers(root))
    ]


def _top_layers(el):
    """Layer elements not nested inside another layer."""
    for child in el:
        if child.get(GROUPMODE) == "layer":
            yield child
        else:
            yield from _top_layers(child)


def _steps(layers: list[Layer], order: str, select) -> list[list[Layer]]:
    layers = [l for l in layers if not l.label.startswith("_")]
    if select is not None:
        wanted = list(select)
        layers = [l for i, l in enumerate(layers) if i in wanted or l.label in wanted or l.id in wanted]
    if order == "document":
        return [[l] for l in layers]
    if order != "label":
        raise ValueError(f"order must be 'label' or 'document', not {order!r}")
    keyed = {}
    for l in layers:
        m = re.search(r"\d+", l.label)
        if m is None:
            raise ValueError(
                f"layer {l.label!r} has no number in its label; number it or use order='document'"
            )
        keyed.setdefault(int(m.group()), []).append(l)
    return [keyed[k] for k in sorted(keyed)]


def _visible_only(path, visible: set[str], keep_loose: bool, page_marker: bool = False) -> bytes:
    """SVG bytes with only the given layers shown (in their original stacking order).
    keep_loose   keep drawing elements that sit outside any layer
    page_marker  add an invisible rectangle covering the page (to locate the page after parsing)"""
    tree = ET.parse(str(path))
    root = tree.getroot()
    if page_marker:
        x, y, w, h = _page_box(root)
        marker = ET.SubElement(root, f"{{{SVG_NS}}}g", id=PAGE_ID)  # only <g> ids are kept by SVGMobject
        ET.SubElement(marker, f"{{{SVG_NS}}}rect", x=str(x), y=str(y), width=str(w), height=str(h),
                      style="fill:#000000;fill-opacity:0;stroke:none")
    for i, el in enumerate(_top_layers(root)):
        style = re.sub(r"display\s*:\s*[^;]+;?", "", el.get("style", "")).strip("; ")
        shown = (el.get("id") or f"_layer{i}") in visible
        el.set("style", ";".join(filter(None, [style, "display:inline" if shown else "display:none"])))
        el.attrib.pop("display", None)
    if not keep_loose:
        for child in list(root):
            tag = child.tag.split("}")[-1]
            if child.get(GROUPMODE) != "layer" and tag not in NON_DRAWING:
                root.remove(child)
    return ET.tostring(root, xml_declaration=True, encoding="utf-8")


def _page_box(root) -> tuple[float, ...]:
    """Page rectangle in user units: the viewBox, or width/height if there is none."""
    if root.get("viewBox"):
        return tuple(float(v) for v in re.split(r"[\s,]+", root.get("viewBox").strip()))
    size = [float(re.match(r"[\d.]+", root.get(k, "0")).group()) for k in ("width", "height")]
    return (0.0, 0.0, *size)


def _cached(key_parts, suffix: str) -> Path:
    h = hashlib.sha1("\0".join(map(str, key_parts)).encode()).hexdigest()[:16]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{h}{suffix}"


def _renderer(name: str | None) -> str:
    name = name or os.environ.get("PRESENT_SVG_RENDERER")
    candidates = [name] if name else ["rsvg-convert", "inkscape", "cairosvg"]
    for c in candidates:
        if c == "cairosvg":
            try:
                import cairosvg  # noqa: F401
                return c
            except ImportError:
                continue
        if c == "inkscape" and (shutil.which("inkscape") or os.path.exists(_INKSCAPE_MAC)):
            return c
        if shutil.which(c):
            return c
    raise RuntimeError(f"no SVG rasterizer found (tried {candidates}); install librsvg, inkscape or cairosvg")


def _rasterize(svg_bytes: bytes, out: Path, dpi: int, renderer: str) -> None:
    tmp = out.with_suffix(".tmp.png")
    if renderer == "rsvg-convert":
        subprocess.run(["rsvg-convert", "-z", str(dpi / 96), "-o", str(tmp)], input=svg_bytes, check=True)
    elif renderer == "inkscape":
        exe = shutil.which("inkscape") or _INKSCAPE_MAC
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
            f.write(svg_bytes)
        try:
            subprocess.run(
                [exe, f.name, "--export-type=png", f"--export-dpi={dpi}", "--export-area-page",
                 f"--export-filename={tmp}"],
                check=True, capture_output=True,
            )
        finally:
            os.unlink(f.name)
    else:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_bytes, write_to=str(tmp), scale=dpi / 96)
    tmp.replace(out)  # atomic: a crashed render never leaves a half-written cache entry


def layered_svg(
    path,
    region: Region | None = None,
    mode: str = "raster",
    order: str = "label",
    layers=None,
    accumulate: bool = True,
    dpi: int = 300,
    renderer: str | None = None,
    fit="fill",
    **place_kwargs,
) -> Group:
    """One mobject per reveal step of an Inkscape SVG, already positioned (Group of steps).

    mode        "raster": each step rendered to PNG (full fidelity: text, filters, embedded images);
                "vector": SVGMobject submobjects (crisp, animatable). Manim's SVG support is limited:
                <text> is dropped (Inkscape: Path > Object to Path first), no filters/embedded images
    order       "label" (number in layer name) or "document" (stacking order)
    layers      optional subset: indices (after filtering), labels or ids
    accumulate  True: layers build up; False: each step shows only its own layers

    Showing step k means: if result.replace_previous, show it *instead of* step k-1 (raster steps
    are complete pictures, rendered together so stacking matches Inkscape exactly); otherwise
    add it (vector steps hold only their new layers). Deck.show_svg takes care of this.
    """
    path = Path(path)
    steps = _steps(svg_layers(path), order, layers)
    if not steps:
        raise ValueError(f"{path} has no (selected) Inkscape layers")
    digest = hashlib.sha1(path.read_bytes()).hexdigest()

    if mode == "raster":
        renderer = _renderer(renderer)
        mobs, visible = [], set()
        for i, step in enumerate(steps):
            ids = {l.id for l in step}
            visible = visible | ids if accumulate else ids
            keep_loose = accumulate or i == 0  # content outside layers belongs to the first step
            out = _cached([digest, sorted(visible), keep_loose, dpi, renderer], ".png")
            if not out.exists():
                _rasterize(_visible_only(path, visible, keep_loose), out, dpi, renderer)
            mobs.append(ImageMobject(str(out)))
        result = Group(*mobs)  # all PNGs share the page size, so they overlap exactly
        result.replace_previous = True
    elif mode == "vector":
        # SVGMobject drops display:none elements, so unhide every layer first
        unhidden = _cached([digest, "vector"], ".svg")
        if not unhidden.exists():
            all_ids = {l.id for l in svg_layers(path)}
            unhidden.write_bytes(_visible_only(path, all_ids, keep_loose=True, page_marker=True))
        full = SVGMobject(str(unhidden), use_svg_cache=False)  # cached copies break id_to_vgroup_dict
        groups = full.id_to_vgroup_dict
        if region is not None:
            # place by the page (as raster does), not by the content: objects lying off the page
            # would otherwise shrink the drawing; then move the whole drawing along with it
            page = groups[PAGE_ID]
            proxy = place(Rectangle(width=page.width, height=page.height), region, fit=fit, **place_kwargs)
            full.scale(proxy.width / page.width, about_point=page.get_center())
            full.shift(proxy.get_center() - page.get_center())
        missing = [l.id for step in steps for l in step if l.id not in groups]
        if missing:
            raise KeyError(f"layers {missing} not found in {path} (layers need an id attribute)")
        result = Group(*[VGroup(*[groups[l.id] for l in step]) for step in steps])
        result.replace_previous = not accumulate
        return result
    else:
        raise ValueError(f"mode must be 'raster' or 'vector', not {mode!r}")

    if region is not None:
        place(result, region, fit=fit, **place_kwargs)
    return result
