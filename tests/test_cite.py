import json
import urllib.error
from importlib import import_module

import pytest

from llandau import LIGHT, Deck, R, build
from llandau.cite import _doi, citation, cite
from llandau.journals import abbreviation

cite_module = import_module("llandau.cite")  # llandau.cite is the function

REAL_FETCH = cite_module._fetch

ALPHAFOLD = {
    "DOI": "10.1038/s41586-021-03819-2", "type": "article-journal",
    "title": "Highly accurate protein structure prediction with <i>AlphaFold</i>",
    "author": [{"family": "Jumper", "given": "John"}, {"family": "Evans", "given": "Richard"},
               {"family": "Pritzel", "given": "Alexander"}, {"family": "Green", "given": "Tim"}],
    "container-title": "Nature", "container-title-short": "Nature", "volume": "596", "page": "583-589",
    "issued": {"date-parts": [[2021, 7, 15]]},
}
ATTENTION = {
    "DOI": "10.48550/ARXIV.1706.03762", "title": "Attention Is All You Need", "publisher": "arXiv",
    "author": [{"family": "Vaswani", "given": "Ashish"}, {"family": "Shazeer", "given": "Noam"},
               {"family": "Parmar", "given": "Niki"}],
    "issued": {"date-parts": [[2017]]},
}
WAALS = {
    "DOI": "10.1000/waals", "author": [{"family": "Waals", "non-dropping-particle": "van der"},
                                       {"literal": "The Consortium"}],
    "container-title": ["Annalen der Physik"], "article-number": "042001", "issued": {"date-parts": [[1873]]},
}
REFS = {"10.1038/s41586-021-03819-2": ALPHAFOLD, "10.48550/arxiv.1706.03762": ATTENTION, "10.1000/waals": WAALS}


@pytest.fixture
def lookups(tmp_path, monkeypatch):
    """doi.org stubbed out, the cache in tmp_path; returns the DOIs looked up."""
    seen = []

    def fetch(doi):
        seen.append(doi)
        return dict(REFS[doi.lower()])

    monkeypatch.setattr(cite_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cite_module, "_fetch", fetch)
    return seen


@pytest.mark.parametrize("source, doi", [
    ("10.1038/s41586-021-03819-2", "10.1038/s41586-021-03819-2"),
    ("doi:10.1038/s41586-021-03819-2.", "10.1038/s41586-021-03819-2"),
    ("https://doi.org/10.1103/PhysRevLett.98.146401", "10.1103/PhysRevLett.98.146401"),
    ("https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.9b00143", "10.1021/acs.jctc.9b00143"),
    ("https://onlinelibrary.wiley.com/doi/full/10.1002/anie.201703114", "10.1002/anie.201703114"),
    ("https://link.springer.com/article/10.1007%2Fs10822-016-9917-4", "10.1007/s10822-016-9917-4"),
    ("arXiv:1706.03762", "10.48550/arXiv.1706.03762"),
    ("1706.03762v7", "10.48550/arXiv.1706.03762"),
    ("https://arxiv.org/pdf/1706.03762v7.pdf", "10.48550/arXiv.1706.03762"),
    ("cond-mat/0211034", "10.48550/arXiv.cond-mat/0211034"),
    ("https://arxiv.org/abs/hep-th/9711200v3", "10.48550/arXiv.hep-th/9711200"),
    ("https://www.nature.com/articles/s41586-021-03819-2", None),  # read from the page
])
def test_doi_of_a_source(source, doi):
    assert _doi(source) == doi


def test_citation_formats(lookups):
    doi = "10.1038/s41586-021-03819-2"
    assert citation(doi) == "Jumper et al., Nature 596, 583 (2021)"
    assert citation(doi, authors=2) == "Jumper, Evans et al., Nature 596, 583 (2021)"
    assert citation(doi, authors="all") == "Jumper, Evans, Pritzel and Green, Nature 596, 583 (2021)"
    assert citation(doi, title=True) == ("Jumper et al., “Highly accurate protein structure prediction with "
                                         "AlphaFold”, Nature 596, 583 (2021)")
    assert citation("arXiv:1706.03762") == "Vaswani et al., arXiv:1706.03762 (2017)"
    assert citation("10.1000/waals") == "van der Waals and The Consortium, Annalen der Physik, 042001 (1873)"


def test_lookups_are_cached(lookups, tmp_path, monkeypatch):
    citation("https://arxiv.org/abs/1706.03762")
    citation("arXiv:1706.03762v2")
    assert lookups == ["10.48550/arXiv.1706.03762"]  # one lookup for both spellings
    assert "10.48550/arxiv.1706.03762" in json.loads((tmp_path / "references.json").read_text())

    url = "https://www.nature.com/articles/s41586-021-03819-2"
    monkeypatch.setattr(cite_module, "_get", lambda url, accept:
                        b'<meta content="10.1038/s41586-021-03819-2" name="citation_doi">')
    assert citation(url).startswith("Jumper et al.")

    def offline(url, accept):
        raise urllib.error.URLError("no network")

    # offline: what was looked up before still resolves, page URLs too; the rest says it isn't cached
    monkeypatch.setattr(cite_module, "_get", offline)
    monkeypatch.setattr(cite_module, "_fetch", REAL_FETCH)
    assert citation(url) == citation("10.1038/s41586-021-03819-2")
    with pytest.raises(RuntimeError, match="isn't cached"):
        citation("10.1000/not-cached")


def test_deck_cites_in_the_footer(lookups):
    shown = []

    class Cited(Deck):
        theme = LIGHT

        def construct(self):
            shown.append(self.cite("10.1038/s41586-021-03819-2", "arXiv:1706.03762"))
            self.step()

    deck = build([Cited])
    assert len(deck.marks) == 1
    mob, footer = shown[0], R.footer
    assert footer.x0 - 1e-6 <= mob.get_left()[0] and mob.get_right()[0] <= footer.x1 - 0.8  # the number's room
    assert mob.get_bottom()[1] == pytest.approx(footer.y0)  # on the slide number's line, at its size
    assert mob.height == pytest.approx(cite("10.1038/s41586-021-03819-2", "arXiv:1706.03762").height)  # not shrunk


@pytest.mark.parametrize("title, short", [
    ("Journal of Chemical Theory and Computation", "J. Chem. Theory Comput."),  # Crossref: often none
    ("Angewandte Chemie International Edition", "Angew. Chem. Int. Ed."),        # Crossref: "Angew Chem Int Ed"
    ("The Journal of Chemical Physics", "J. Chem. Phys."),                       # "The" dropped
    ("Journal of Chemical Physics", "J. Chem. Phys."),
    ("Nature Structural &amp; Molecular Biology", "Nat. Struct. Mol. Biol."),   # HTML-escaped, as Crossref has it
    ("Chemistry - A European Journal", "Chem. Eur. J."),                        # a hyphen for the dash
    ("Chem. Sci.", "Chem. Sci."),                                               # RSC files some under the abbreviation
    ("EPL (Europhysics Letters)", "EPL"),                                       # an alias
    ("Journal of Unlisted Results", None),
])
def test_journal_abbreviations(title, short):
    assert abbreviation(title) == short


def test_journals_override_crossref(lookups, monkeypatch):
    jctc = {"DOI": "10.1021/acs.jctc.9b00143", "author": [{"family": "Grimme"}], "volume": "15", "page": "2847-2862",
            "container-title": "Journal of Chemical Theory and Computation", "issued": {"date-parts": [[2019]]}}
    monkeypatch.setitem(REFS, "10.1021/acs.jctc.9b00143", jctc)  # no short title: as Crossref has it for one ACS article in ten
    assert citation("10.1021/acs.jctc.9b00143") == "Grimme, J. Chem. Theory Comput. 15, 2847 (2019)"
    monkeypatch.setitem(REFS, "10.1000/unlisted", dict(jctc, DOI="10.1000/unlisted", **{
        "container-title": "Journal of Unlisted Results", "container-title-short": "J. Unlisted Res."}))
    assert citation("10.1000/unlisted").startswith("Grimme, J. Unlisted Res. 15")  # unknown: Crossref's


GOLDSTEIN = {"ISBN:9788131758915": {
    "title": "Classical Mechanics", "publish_date": "Nov 01, 2011",
    "authors": [{"name": "Herbert Goldstein, Charles Poole, John Safko"}, {"name": "Herbert Goldstein"},
                {"name": "Charles P. Poole"}, {"name": "John L. Safko"}],
}}


def test_books_by_isbn(tmp_path, monkeypatch):
    seen = []

    def get(url, accept):
        seen.append(url)
        return json.dumps(GOLDSTEIN if "9788131758915" in url else {}).encode()

    monkeypatch.setattr(cite_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(cite_module, "_get", get)
    # the combined author entry is left out; a book's title is its venue, title=True adds nothing
    assert citation("isbn:978-81-317-5891-5") == "Goldstein et al., Classical Mechanics (2011)"
    assert citation("ISBN 9788131758915", authors="all", title=True) == \
        "Goldstein, Poole and Safko, Classical Mechanics (2011)"
    assert len(seen) == 1  # cached under its digits
    with pytest.raises(ValueError, match="Open Library"):
        citation("isbn:0000000000")
    with pytest.raises(ValueError, match="ISBN-10 or ISBN-13"):
        citation("isbn:12345")
