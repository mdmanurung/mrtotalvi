# Configuration file for the Sphinx documentation builder.
# `scverse_doc` sets the theme and shared defaults, see https://scverse-doc.readthedocs.io
# For the full list of options, see https://www.sphinx-doc.org/page/usage/configuration.html

from datetime import datetime
from importlib.metadata import metadata

from scverse_doc.registry import intersphinx

# NOTE: If you installed your project in editable mode, this might be stale.
#       If this is the case, reinstall it to refresh the metadata
info = metadata("mrtotalvi")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}."
version = release = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
source_repository = urls["Source"]

extensions = [
    "scverse_doc",
    "scverse_misc.sphinx_ext",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "IPython.sphinxext.ipython_console_highlighting",
]

bibtex_bibfiles = ["references.bib"]
typehints_defaults = "braces"
always_use_bars_union = True  # use `|` instead of `Union` in types even when building with Python ≤3.14

templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# Review notes and their scripts are working records, not part of the site.
exclude_patterns = ["_build", "reviews/**"]
# ADR-0008 links a local .scratch file that is not in the repo.
suppress_warnings = ["myst.xref_missing"]

# Core scverse packages (incl. scvi-tools) plus the libraries MrTotalVI's signatures name.
intersphinx_mapping = intersphinx() | {
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
}

# Links that cannot resolve: docstrings name private helpers (``_components``, ``_stats``),
# and inherited scvi-tools docstrings link to packages or bib keys this site does not load.
nitpick_ignore_regex = [
    ("py:.*", r"(mrtotalvi\.)?_(components|stats)\..*"),
    ("py:.*", r"EncoderUZ|EncoderXU_TotalVI|_setup_hierarchy|u_prior_mixture|super\(\)\..*"),
    ("py:.*", r"annbatch\..*|scvi\.settings\..*"),
]

xref_overrides = {
    # If building the documentation fails because a type annotation links to a private path
    # (e.g. `igraph._igraph.GraphBase`), map it to the documented name here.
    # For a missing link that can’t be fixed this way, add it to `nitpick_ignore` instead.
    #     "igraph._igraph.GraphBase": "igraph.GraphBase",
    "xarray.core.dataarray.DataArray": "xarray.DataArray",
    "xarray.core.dataset.Dataset": "xarray.Dataset",
}
