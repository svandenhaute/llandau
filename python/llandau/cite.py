"""Citations for slides, from a DOI, an arXiv id or an article's web page.

    cite("10.1038/s41586-021-03819-2")           # Jumper et al., Nature 596, 583 (2021)
    cite("arXiv:1706.03762", title=True)         # Vaswani et al., “Attention Is All You Need”, arXiv:1706.03762 (2017)
    cite("https://www.nature.com/articles/s41586-021-03819-2", authors="all")
    self.cite("10.1038/...")                      # in a Deck: in the footer strip, left

The metadata comes from doi.org (CSL-JSON; arXiv papers have DOIs 10.48550/arXiv.<id>), a web
page is read for its citation_doi meta tag. Every lookup is kept in .llandau_cache/references.json,
so a deck built once plays offline. Journal names are abbreviated by llandau.journals where it
knows the journal, else as Crossref does. Edit an entry in the cache to change a citation (e.g. a
"container-title-short" Crossref lacks), delete it to look it up again.
"""
from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from . import theme
from .text import text
from .media import CACHE_DIR
from .journals import abbreviation

__all__ = ["cite", "citation", "reference"]

CACHE_FILE = "references.json"
TIMEOUT = 15  # seconds per request
_USER_AGENT = "llandau (https://github.com/svandenhaute/llandau)"
# the fields a citation needs: Crossref's CSL also lists every reference of the paper
_FIELDS = ("type", "title", "author", "container-title", "container-title-short", "publisher", "volume",
           "issue", "page", "article-number", "issued", "published-print", "published-online", "DOI")
_NAME_FIELDS = ("family", "given", "literal", "name", "non-dropping-particle")

_DOI = re.compile(r"10\.\d{4,9}/[^\s\"<>?#]+")
# new-style ids (1706.03762) and old-style ones (cond-mat/0211034), with an optional version
_ARXIV_ID = r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?"
_ARXIV = re.compile(r"(?:arxiv[:./]\s*|arxiv\.org/(?:abs|pdf)/)" + _ARXIV_ID, re.IGNORECASE)
_BARE_ARXIV = re.compile(_ARXIV_ID + r"(?:\.pdf)?", re.IGNORECASE)


def cite(*sources: str, size: float | None = None, sep: str = "; ", title: bool = False, authors: int | str = 1,
         **kwargs):
    """Citation(s) as one text, e.g. "Jumper et al., Nature 596, 583 (2021)" (see citation());
    several are joined by sep ("\n": one per line). kwargs go to text(): color, slant, ..."""
    line = sep.join(citation(s, title=title, authors=authors) for s in sources)
    return text(line, size=size if size is not None else theme.current().label, typst_markup=False, **kwargs)


def citation(source: str, title: bool = False, authors: int | str = 1) -> str:
    """A short citation of a DOI, arXiv id or article URL.

    title    True: the title in quotes after the authors
    authors  family names shown before "et al."; two authors are always both named. "all": every one
    """
    ref = reference(source)
    names = [_name(a) for a in ref.get("author", [])]
    shown = len(names) if authors == "all" else max(int(authors), 2)
    if len(names) > shown:
        who = ", ".join(names[:int(authors)]) + " et al."
    else:
        who = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else "".join(names)
    parts = [who] if who else []
    if title and ref.get("title"):
        parts.append(f"“{_clean(_first(ref['title']))}”")
    parts.append(_venue(ref))
    year = _year(ref)
    return ", ".join(p for p in parts if p) + (f" ({year})" if year else "")


def reference(source: str) -> dict:
    """The CSL-JSON metadata of a DOI, arXiv id or article URL (only the fields a citation needs),
    from the cache or looked up and cached."""
    key = _key(source)
    cache = _load_cache()
    if key in cache:
        return cache[key]
    doi = _doi(source)
    if doi is None:
        doi = _doi_of_page(source)
    ref = cache.get(doi.lower()) or _fetch(doi)
    cache[doi.lower()] = cache[key] = ref
    _save_cache(cache)
    return ref


# --- finding the DOI ---------------------------------------------------------

def _key(source: str) -> str:
    doi = _doi(source)
    return doi.lower() if doi else source.strip()


def _doi(source: str) -> str | None:
    """The DOI in a string (a bare DOI, a doi.org or publisher URL, an arXiv id or URL), or None."""
    s = urllib.parse.unquote(source.strip())
    if m := _ARXIV.search(s) or _BARE_ARXIV.fullmatch(s):
        return f"10.48550/arXiv.{m.group(1)}"
    if m := _DOI.search(s):
        doi = m.group(0).rstrip(".,;)")
        return re.sub(r"(\.pdf|/(full|abstract|pdf|epdf))$", "", doi)
    return None


class _Meta(HTMLParser):
    """The content of a page's <meta name=...> tags, first one of each name."""

    def __init__(self):
        super().__init__()
        self.meta = {}

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            a = dict(attrs)
            name = (a.get("name") or a.get("property") or "").lower()
            if name and a.get("content"):
                self.meta.setdefault(name, a["content"])


def _doi_of_page(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"not a DOI, arXiv id or URL: {url!r}")
    try:
        page = _get(url, "text/html").decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(f"could not read {url} ({e}); cite its DOI instead") from None
    parser = _Meta()
    parser.feed(page)
    for name in ("citation_doi", "dc.identifier", "prism.doi", "citation_arxiv_id"):
        if (value := parser.meta.get(name)) and (doi := _doi(value) or _doi(f"arXiv:{value}")):
            return doi
    raise ValueError(f"no DOI found on {url}; cite its DOI instead")


# --- looking it up -----------------------------------------------------------

def _get(url: str, accept: str) -> bytes:
    request = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read()


def _fetch(doi: str) -> dict:
    url = "https://doi.org/" + urllib.parse.quote(doi, safe="/:;()")
    try:
        csl = json.loads(_get(url, "application/vnd.citationstyles.csl+json"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError(f"unknown DOI: {doi}") from None
        raise RuntimeError(f"looking up {doi} failed ({e}) and it isn't cached") from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(f"looking up {doi} needs the network ({e}); it isn't cached yet") from None
    ref = {k: csl[k] for k in _FIELDS if k in csl}
    if "author" in ref:  # not their affiliations, ORCIDs, ...
        ref["author"] = [{k: v for k, v in a.items() if k in _NAME_FIELDS} for a in ref["author"]]
    return ref


def _load_cache() -> dict:
    try:
        return json.loads((CACHE_DIR / CACHE_FILE).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / CACHE_FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False, sort_keys=True))
    tmp.replace(path)  # atomic: a crash never leaves a half-written cache


# --- formatting --------------------------------------------------------------

def _first(value) -> str:
    """CSL strings are sometimes lists of strings."""
    return (value[0] if value else "") if isinstance(value, list) else (value or "")


def _clean(s: str) -> str:
    """Crossref titles carry HTML (<i>, <sub>, &amp;) and line breaks."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", s)).split())


def _name(author: dict) -> str:
    """Family name with its particle ("van der Waals"), or a consortium's literal name."""
    if "family" not in author:
        return _clean(author.get("literal", author.get("name", "")))
    particle = author.get("non-dropping-particle", "")
    return _clean(f"{particle} {author['family']}" if particle else author["family"])


def _venue(ref: dict) -> str:
    doi = ref.get("DOI", "")
    if doi.lower().startswith("10.48550/arxiv."):
        arxiv_id = doi[len("10.48550/arxiv."):]
        return "arXiv:" + (arxiv_id.lower() if "/" in arxiv_id else arxiv_id)  # DataCite: COND-MAT/0211034
    full = _clean(_first(ref.get("container-title")))
    journal = abbreviation(full) or _clean(_first(ref.get("container-title-short"))) or full
    if not journal:
        return _clean(_first(ref.get("publisher")))
    page = str(ref.get("page", "")).split("-")[0].split("–")[0] or ref.get("article-number", "")
    return " ".join(str(p) for p in (journal, ref.get("volume", "")) if p) + (f", {page}" if page else "")


def _year(ref: dict) -> int | None:
    for k in ("issued", "published-print", "published-online"):
        parts = ref.get(k, {}).get("date-parts") or [[None]]
        if parts[0] and parts[0][0]:
            return parts[0][0]
    return None
