"""Journal abbreviations for citations, by full title.

Crossref's short titles (what doi.org returns) can't be used as they come: Springer Nature's have
no dots ("Nat Commun"), Elsevier's and many of Wiley's are the full title, AIP has none, APS
spells out its newer journals ("Phys. Rev. Research"), ACS leaves it out on about one article in
ten, and several journals mix styles between years. So the journals below always get the
abbreviation listed here: ISO 4 (CASSI for ACS journals), except where a field writes its
journals otherwise (astronomy: ApJ, MNRAS, A&A; JHEP; mathematics: MathSciNet's forms). Other
journals keep Crossref's short title.

    abbreviation("The Journal of Chemical Physics")   # "J. Chem. Phys."
    JOURNALS["Journal of Obscure Results"] = "J. Obscure Res."   # add your own
"""
from __future__ import annotations

import html
import re

__all__ = ["JOURNALS", "abbreviation"]

JOURNALS = {
    # multidisciplinary
    "Nature": "Nature",
    "Science": "Science",
    "Science Advances": "Sci. Adv.",
    "Proceedings of the National Academy of Sciences": "Proc. Natl. Acad. Sci. U.S.A.",
    "Nature Communications": "Nat. Commun.",
    "Scientific Reports": "Sci. Rep.",
    "Scientific Data": "Sci. Data",
    "PLOS ONE": "PLoS One",
    "Cell": "Cell",
    "Joule": "Joule",
    "Chem": "Chem",
    "Matter": "Matter",
    "Nature Physics": "Nat. Phys.",
    "Nature Chemistry": "Nat. Chem.",
    "Nature Materials": "Nat. Mater.",
    "Nature Nanotechnology": "Nat. Nanotechnol.",
    "Nature Methods": "Nat. Methods",
    "Nature Machine Intelligence": "Nat. Mach. Intell.",
    "Nature Computational Science": "Nat. Comput. Sci.",
    "Nature Reviews Physics": "Nat. Rev. Phys.",
    "Nature Reviews Chemistry": "Nat. Rev. Chem.",
    "Nature Reviews Materials": "Nat. Rev. Mater.",
    "Nature Reviews Drug Discovery": "Nat. Rev. Drug Discov.",
    "Nature Catalysis": "Nat. Catal.",
    "Nature Energy": "Nat. Energy",
    "Nature Photonics": "Nat. Photonics",
    "Nature Biotechnology": "Nat. Biotechnol.",
    "Nature Structural & Molecular Biology": "Nat. Struct. Mol. Biol.",
    "Communications Chemistry": "Commun. Chem.",
    "Communications Physics": "Commun. Phys.",
    "Communications Materials": "Commun. Mater.",
    "npj Computational Materials": "npj Comput. Mater.",
    # physics
    "Physical Review Letters": "Phys. Rev. Lett.",
    "Physical Review A": "Phys. Rev. A",
    "Physical Review B": "Phys. Rev. B",
    "Physical Review C": "Phys. Rev. C",
    "Physical Review D": "Phys. Rev. D",
    "Physical Review E": "Phys. Rev. E",
    "Physical Review X": "Phys. Rev. X",
    "Physical Review Research": "Phys. Rev. Res.",
    "Physical Review Materials": "Phys. Rev. Mater.",
    "Physical Review Applied": "Phys. Rev. Appl.",
    "Physical Review Fluids": "Phys. Rev. Fluids",
    "Physical Review Accelerators and Beams": "Phys. Rev. Accel. Beams",
    "PRX Quantum": "PRX Quantum",
    "Reviews of Modern Physics": "Rev. Mod. Phys.",
    "The Journal of Chemical Physics": "J. Chem. Phys.",
    "Applied Physics Letters": "Appl. Phys. Lett.",
    "Journal of Applied Physics": "J. Appl. Phys.",
    "Journal of Mathematical Physics": "J. Math. Phys.",
    "Physics of Fluids": "Phys. Fluids",
    "Chaos: An Interdisciplinary Journal of Nonlinear Science": "Chaos",
    "Physics Today": "Phys. Today",
    "Journal of Physics: Condensed Matter": "J. Phys.: Condens. Matter",
    "Journal of Physics A: Mathematical and Theoretical": "J. Phys. A: Math. Theor.",
    "New Journal of Physics": "New J. Phys.",
    "Reports on Progress in Physics": "Rep. Prog. Phys.",
    "Machine Learning: Science and Technology": "Mach. Learn.: Sci. Technol.",
    "Journal of Statistical Mechanics: Theory and Experiment": "J. Stat. Mech.",
    "Europhysics Letters": "EPL",
    "Physics Reports": "Phys. Rep.",
    "Physics Letters A": "Phys. Lett. A",
    "Physics Letters B": "Phys. Lett. B",
    "Nuclear Physics B": "Nucl. Phys. B",
    "Annals of Physics": "Ann. Phys.",
    "Computer Physics Communications": "Comput. Phys. Commun.",
    "Journal of Computational Physics": "J. Comput. Phys.",
    "Physica D: Nonlinear Phenomena": "Physica D",
    "Journal of High Energy Physics": "JHEP",
    "Journal of Statistical Physics": "J. Stat. Phys.",
    "Communications in Mathematical Physics": "Commun. Math. Phys.",
    "SciPost Physics": "SciPost Phys.",
    "Quantum": "Quantum",
    "The Astrophysical Journal": "ApJ",
    "Monthly Notices of the Royal Astronomical Society": "MNRAS",
    "Astronomy & Astrophysics": "A&A",
    "Journal of Fluid Mechanics": "J. Fluid Mech.",
    "Optica": "Optica",
    "Optics Express": "Opt. Express",
    "Advances in Physics": "Adv. Phys.",
    "Annual Review of Condensed Matter Physics": "Annu. Rev. Condens. Matter Phys.",
    "Biophysical Journal": "Biophys. J.",
    # materials
    "Advanced Materials": "Adv. Mater.",
    "Advanced Functional Materials": "Adv. Funct. Mater.",
    "Advanced Energy Materials": "Adv. Energy Mater.",
    "Advanced Science": "Adv. Sci.",
    "Small": "Small",
    "Nano Letters": "Nano Lett.",
    "ACS Nano": "ACS Nano",
    "Carbon": "Carbon",
    "Microporous and Mesoporous Materials": "Microporous Mesoporous Mater.",
    # chemistry
    "Journal of the American Chemical Society": "J. Am. Chem. Soc.",
    "JACS Au": "JACS Au",
    "Journal of Chemical Theory and Computation": "J. Chem. Theory Comput.",
    "The Journal of Physical Chemistry A": "J. Phys. Chem. A",
    "The Journal of Physical Chemistry B": "J. Phys. Chem. B",
    "The Journal of Physical Chemistry C": "J. Phys. Chem. C",
    "The Journal of Physical Chemistry Letters": "J. Phys. Chem. Lett.",
    "Journal of Chemical Information and Modeling": "J. Chem. Inf. Model.",
    "Chemical Reviews": "Chem. Rev.",
    "Accounts of Chemical Research": "Acc. Chem. Res.",
    "ACS Catalysis": "ACS Catal.",
    "ACS Central Science": "ACS Cent. Sci.",
    "ACS Energy Letters": "ACS Energy Lett.",
    "ACS Materials Letters": "ACS Mater. Lett.",
    "ACS Applied Materials & Interfaces": "ACS Appl. Mater. Interfaces",
    "Chemistry of Materials": "Chem. Mater.",
    "Inorganic Chemistry": "Inorg. Chem.",
    "Organic Letters": "Org. Lett.",
    "The Journal of Organic Chemistry": "J. Org. Chem.",
    "Journal of Medicinal Chemistry": "J. Med. Chem.",
    "Journal of Chemical Education": "J. Chem. Educ.",
    "Macromolecules": "Macromolecules",
    "Langmuir": "Langmuir",
    "Analytical Chemistry": "Anal. Chem.",
    "Biochemistry": "Biochemistry",
    "Angewandte Chemie International Edition": "Angew. Chem. Int. Ed.",
    "Angewandte Chemie": "Angew. Chem.",
    "Chemistry – A European Journal": "Chem. Eur. J.",
    "ChemPhysChem": "ChemPhysChem",
    "ChemSusChem": "ChemSusChem",
    "ChemCatChem": "ChemCatChem",
    "Journal of Computational Chemistry": "J. Comput. Chem.",
    "International Journal of Quantum Chemistry": "Int. J. Quantum Chem.",
    "WIREs Computational Molecular Science": "WIREs Comput. Mol. Sci.",
    "Chemical Science": "Chem. Sci.",
    "Chemical Society Reviews": "Chem. Soc. Rev.",
    "Chemical Communications": "Chem. Commun.",
    "Physical Chemistry Chemical Physics": "Phys. Chem. Chem. Phys.",
    "Journal of Materials Chemistry A": "J. Mater. Chem. A",
    "Energy & Environmental Science": "Energy Environ. Sci.",
    "Faraday Discussions": "Faraday Discuss.",
    "Dalton Transactions": "Dalton Trans.",
    "Green Chemistry": "Green Chem.",
    "RSC Advances": "RSC Adv.",
    "Nanoscale": "Nanoscale",
    "Digital Discovery": "Digital Discovery",
    "Molecular Physics": "Mol. Phys.",
    "Chemical Physics Letters": "Chem. Phys. Lett.",
    "Theoretical Chemistry Accounts": "Theor. Chem. Acc.",
    "Journal of Catalysis": "J. Catal.",
    "Electrochimica Acta": "Electrochim. Acta",
    "Journal of Molecular Liquids": "J. Mol. Liq.",
    "Coordination Chemistry Reviews": "Coord. Chem. Rev.",
    "Annual Review of Physical Chemistry": "Annu. Rev. Phys. Chem.",
    # biology, for chemists and physicists
    "Journal of Molecular Biology": "J. Mol. Biol.",
    "Nucleic Acids Research": "Nucleic Acids Res.",
    "Bioinformatics": "Bioinformatics",
    "Proteins: Structure, Function, and Bioinformatics": "Proteins",
    # mathematics and statistics (MathSciNet)
    "Annals of Mathematics": "Ann. of Math.",
    "Inventiones mathematicae": "Invent. Math.",
    "Acta Mathematica": "Acta Math.",
    "Journal of the American Mathematical Society": "J. Amer. Math. Soc.",
    "Duke Mathematical Journal": "Duke Math. J.",
    "Mathematische Annalen": "Math. Ann.",
    "Advances in Mathematics": "Adv. Math.",
    "Geometric and Functional Analysis": "Geom. Funct. Anal.",
    "Publications mathématiques de l'IHÉS": "Publ. Math. IHÉS",
    "Journal of Functional Analysis": "J. Funct. Anal.",
    "Journal of Differential Equations": "J. Differential Equations",
    "Archive for Rational Mechanics and Analysis": "Arch. Ration. Mech. Anal.",
    "Communications on Pure and Applied Mathematics": "Comm. Pure Appl. Math.",
    "SIAM Review": "SIAM Rev.",
    "SIAM Journal on Numerical Analysis": "SIAM J. Numer. Anal.",
    "SIAM Journal on Scientific Computing": "SIAM J. Sci. Comput.",
    "SIAM Journal on Matrix Analysis and Applications": "SIAM J. Matrix Anal. Appl.",
    "SIAM Journal on Applied Mathematics": "SIAM J. Appl. Math.",
    "SIAM Journal on Optimization": "SIAM J. Optim.",
    "SIAM Journal on Computing": "SIAM J. Comput.",
    "Multiscale Modeling & Simulation": "Multiscale Model. Simul.",
    "Numerische Mathematik": "Numer. Math.",
    "Mathematics of Computation": "Math. Comp.",
    "Acta Numerica": "Acta Numer.",
    "Foundations of Computational Mathematics": "Found. Comput. Math.",
    "Journal of Computational and Applied Mathematics": "J. Comput. Appl. Math.",
    "Journal of Computational Dynamics": "J. Comput. Dyn.",
    "Mathematical Programming": "Math. Program.",
    "Probability Theory and Related Fields": "Probab. Theory Related Fields",
    "The Annals of Probability": "Ann. Probab.",
    "The Annals of Applied Probability": "Ann. Appl. Probab.",
    "The Annals of Statistics": "Ann. Statist.",
    "Journal of the American Statistical Association": "J. Amer. Statist. Assoc.",
    "Journal of the Royal Statistical Society Series B: Statistical Methodology": "J. R. Stat. Soc. B",
    "Biometrika": "Biometrika",
    "Bernoulli": "Bernoulli",
    # computer science (IEEE's own abbreviations for its journals)
    "Communications of the ACM": "Commun. ACM",
    "Journal of the ACM": "J. ACM",
    "ACM Computing Surveys": "ACM Comput. Surv.",
    "ACM Transactions on Graphics": "ACM Trans. Graph.",
    "ACM Transactions on Mathematical Software": "ACM Trans. Math. Softw.",
    "IEEE Transactions on Pattern Analysis and Machine Intelligence": "IEEE Trans. Pattern Anal. Mach. Intell.",
    "IEEE Transactions on Information Theory": "IEEE Trans. Inf. Theory",
    "IEEE Transactions on Neural Networks and Learning Systems": "IEEE Trans. Neural Netw. Learn. Syst.",
    "IEEE Transactions on Signal Processing": "IEEE Trans. Signal Process.",
    "IEEE Transactions on Image Processing": "IEEE Trans. Image Process.",
    "IEEE Transactions on Visualization and Computer Graphics": "IEEE Trans. Vis. Comput. Graph.",
    "Proceedings of the IEEE": "Proc. IEEE",
    "Computing in Science & Engineering": "Comput. Sci. Eng.",
    "Neural Computation": "Neural Comput.",
    "Neural Networks": "Neural Netw.",
    "Neurocomputing": "Neurocomputing",
    "Machine Learning": "Mach. Learn.",
    "Artificial Intelligence": "Artif. Intell.",
    "Journal of Artificial Intelligence Research": "J. Artif. Intell. Res.",
    "International Journal of Computer Vision": "Int. J. Comput. Vis.",
    "Pattern Recognition": "Pattern Recognit.",
    "Computer Graphics Forum": "Comput. Graph. Forum",
    "Journal of Open Source Software": "J. Open Source Softw.",
    "SoftwareX": "SoftwareX",
    "The International Journal of High Performance Computing Applications": "Int. J. High Perform. Comput. Appl.",
    "Parallel Computing": "Parallel Comput.",
    "Journal of Parallel and Distributed Computing": "J. Parallel Distrib. Comput.",
}

# other titles Crossref files these journals under (older names, reordered ones)
ALIASES = {
    "EPL (Europhysics Letters)": "Europhysics Letters",
    "Europhysics Letters (EPL)": "Europhysics Letters",
    "Astronomy and Astrophysics": "Astronomy & Astrophysics",
    "Proteins: Structure, Function, and Genetics": "Proteins: Structure, Function, and Bioinformatics",
    "Theoretical Chemistry Accounts: Theory, Computation, and Modeling (Theoretica Chimica Acta)":
        "Theoretical Chemistry Accounts",
    "GAFA Geometric And Functional Analysis": "Geometric and Functional Analysis",
}


def _key(title: str) -> str:
    """Case, "The", "&", dashes and punctuation don't matter: "Chem. Sci." is "Chem Sci"."""
    t = html.unescape(title).lower().replace("&", " and ")
    t = " ".join(re.sub(r"[^\w]+", " ", t).split())
    return t.removeprefix("the ")


def abbreviation(title: str) -> str | None:
    """The abbreviation of a journal's full title (or of an abbreviation of its own, as RSC
    sometimes files them: "Chem. Sci."), None if the journal isn't listed."""
    if not title:
        return None
    key = _key(title)
    for full, short in JOURNALS.items():  # a scan, not an index: JOURNALS may have gained entries
        if key in (_key(full), _key(short)):
            return short
    full = next((f for a, f in ALIASES.items() if _key(a) == key), None)
    return JOURNALS.get(full) if full else None
