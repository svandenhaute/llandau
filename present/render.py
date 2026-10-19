"""Render a deck and convert it, using the SLIDES list from the scene file.

    present render scene.py                    # render all, write scene.html
    present render scene.py --only Masses QM   # render a few (no conversion)
    present render scene.py --present          # render all, then present
    present render scene.py -q l -- --disable_caching   # after --: passed to manim
    present live scene.py                      # present with live polyscope steps (see present.live)
"""
import argparse
import ast
import collections
import runpy
import subprocess
import sys
from pathlib import Path


def duplicate_classes(path: Path) -> list[str]:
    """Top-level classes defined more than once: the later definition silently wins."""
    tree = ast.parse(path.read_text())
    names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    return [name for name, count in collections.Counter(names).items() if count > 1]


def slide_names(path: Path) -> list[str]:
    sys.path.insert(0, str(path.parent.resolve()))
    module = runpy.run_path(str(path))
    if "SLIDES" not in module:
        sys.exit(f"{path} defines no SLIDES list (use SLIDES = order(...))")
    return [cls if isinstance(cls, str) else cls.__name__ for cls in module["SLIDES"]]


# reveal.js only pauses a step's background video when leaving it, so on a later visit the video
# still shows its last frame until the restart (currentTime = 0) has finished seeking: the
# finished animation flashes before it plays again. Rewind on leaving instead.
REWIND_ON_LEAVE = """
<script data-present="rewind-on-leave">
Reveal.on("slidechanged", (event) => {
  const video = event.previousSlide && Reveal.getSlideBackground(event.previousSlide)?.querySelector("video");
  if (video) { video.pause(); video.currentTime = 0; }
});
</script>
"""


def inject(html: Path, script: str) -> None:
    source = html.read_text()
    if script.strip() not in source:
        i = source.rindex("</body>")
        html.write_text(source[:i] + script + source[i:])


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]

    if argv[:1] == ["live"]:
        from .live import main as live
        sys.exit(live(argv[1:]))
    if argv[:1] != ["render"]:
        sys.exit("usage: present render|live SLIDES.py [options]   (present render --help)")
    argv = argv[1:]
    parser = argparse.ArgumentParser(prog="present render", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", type=Path)
    parser.add_argument("--only", nargs="+", metavar="SLIDE", help="render only these slides")
    parser.add_argument("-q", "--quality", default="h", choices=list("lmhpk"),
                        help="manim quality: l=480p15 m=720p30 h=1080p60 (default) p=1440p60 k=4k60")
    parser.add_argument("--html", type=Path, help="output file (default: <file>.html)")
    parser.add_argument("--present", action="store_true", help="present instead of converting")
    parser.add_argument("--no-render", action="store_true", help="only convert/present")
    args = parser.parse_args(argv)

    dups = duplicate_classes(args.file)
    if dups:
        sys.exit(f"classes defined more than once in {args.file}: {', '.join(dups)}")

    slides = slide_names(args.file)
    targets = args.only or slides
    if not args.no_render:
        run(["manim", "render", f"-q{args.quality}", *extra, str(args.file), *dict.fromkeys(targets)])

    if args.present:
        run(["manim-slides", "present", *targets])
    elif args.only is None or args.html is not None:
        html = args.html or args.file.with_suffix(".html")
        run(["manim-slides", "convert", "--to", "html", *targets, str(html)])
        inject(html, REWIND_ON_LEAVE)


if __name__ == "__main__":
    main()
