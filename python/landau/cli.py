"""Present a deck, or export it, using the SLIDES list from the scene file.
Nothing is rendered ahead: the deck is drawn live, animations included.

    landau play slides.py                  # present (F: full screen, arrows: steps)
    landau play slides.py --only Masses QM # a few slides
    landau play slides.py --start 12       # from step 12
    landau pdf slides.py                   # one page per step: slides.pdf
    landau png slides.py OUT_DIR           # one PNG per step
"""
import argparse
import ast
import collections
import runpy
import sys
from pathlib import Path


def duplicate_classes(path: Path) -> list[str]:
    """Top-level classes defined more than once: the later definition silently wins."""
    tree = ast.parse(path.read_text())
    names = [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    return [name for name, count in collections.Counter(names).items() if count > 1]


def load_slides(path: Path, only=None) -> list:
    sys.path.insert(0, str(path.parent.resolve()))
    module = runpy.run_path(str(path))
    if "SLIDES" not in module:
        sys.exit(f"{path} defines no SLIDES list (use SLIDES = order(...))")
    slides = [module[s] if isinstance(s, str) else s for s in module["SLIDES"]]
    if only:
        slides = [s for s in slides if s.__name__ in only]
        missing = set(only) - {s.__name__ for s in slides}
        if missing:
            sys.exit(f"not in SLIDES: {', '.join(sorted(missing))}")
    return slides


def stills(deck, size=(1920, 1080)):
    """The end of every step, as RGBA images."""
    for time, _, _ in deck.marks:
        yield deck.render(time - 1e-6, *size)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="landau", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["play", "pdf", "png"])
    parser.add_argument("file", type=Path)
    parser.add_argument("out", type=Path, nargs="?", help="pdf: output file; png: output folder")
    parser.add_argument("--only", nargs="+", metavar="SLIDE", help="only these slides")
    parser.add_argument("--start", type=int, default=0, help="play: the step to start at")
    parser.add_argument("--fullscreen", "-f", action="store_true", help="play: start in full screen")
    parser.add_argument("--size", type=int, nargs=2, default=(1920, 1080), metavar=("W", "H"),
                        help="pdf, png: pixels per page")
    args = parser.parse_args(argv)

    dups = duplicate_classes(args.file)
    if dups:
        sys.exit(f"classes defined more than once in {args.file}: {', '.join(dups)}")
    from .scene import build
    deck = build(load_slides(args.file, args.only))

    if args.command == "play":
        deck.play(start=args.start, fullscreen=args.fullscreen, title=args.file.stem)
    elif args.command == "pdf":
        from PIL import Image
        out = args.out or args.file.with_suffix(".pdf")
        pages = [Image.fromarray(image).convert("RGB") for image in stills(deck, tuple(args.size))]
        pages[0].save(out, save_all=True, append_images=pages[1:], resolution=args.size[1] / 7.5)
        print(f"{out}: {len(pages)} pages")
    elif args.command == "png":
        from PIL import Image
        out = args.out or args.file.with_suffix("")
        out.mkdir(parents=True, exist_ok=True)
        for k, image in enumerate(stills(deck, tuple(args.size))):
            Image.fromarray(image).save(out / f"{k:03d}.png")
        print(f"{out}/: {len(deck.marks)} images")


if __name__ == "__main__":
    main()
